import torch
from torch.nn import Linear, ReLU
import torch.nn.functional as F
from torch_geometric.nn import PointNetConv, global_max_pool
from torch_geometric.utils import farthest_point_sample, knn

import torch
from torch_geometric.data import Data

def parse_txt_to_data(txt_path):
    with open(txt_path, 'r') as f:
        lines = f.readlines()

    points = []
    semantic_labels = []
    instance_labels = []
    current_instance = -1

    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Check for header line (e.g., "[0] SRF :plane (1, 1)")
        if line.startswith('['):
            parts = line.split()
            current_instance = int(parts[0][1:-1])  # Extract [0] → 0
            semantic_type = parts[2][1:]  # ":plane" → "plane"
            semantic_label = 0 if semantic_type == "plane" else 1  # Map to class ID
        else:
            # Parse point data: x,y,z,nx,ny,nz,curvature
            data = list(map(float, line.split(',')))
            points.append(data)
            semantic_labels.append(semantic_label)
            instance_labels.append(current_instance)

    # Convert to tensors
    pos = torch.tensor([p[:3] for p in points], dtype=torch.float)  # (N, 3)
    x = torch.tensor([p[3:] for p in points], dtype=torch.float)    # (N, 4: nx, ny, nz, curvature)
    y_semantic = torch.tensor(semantic_labels, dtype=torch.long)   # (N,)
    y_instance = torch.tensor(instance_labels, dtype=torch.long)    # (N,)

    return Data(pos=pos, x=x, y_semantic=y_semantic, y_instance=y_instance)

class PointNetPlusPlus(torch.nn.Module):
    def __init__(self, in_channels=7, num_classes=3, embed_dim=64):
        super().__init__()
        
        # Encoder (Set Abstraction)
        self.sa1 = PointNetConv(
            local_nn=torch.nn.Sequential(
                Linear(in_channels, 64),
                ReLU(),
                Linear(64, 64)
            ),
            global_nn=torch.nn.Sequential(
                Linear(64, 128),
                ReLU(),
                Linear(128, 128)
            )
        )
        
        self.sa2 = PointNetConv(
            local_nn=torch.nn.Sequential(
                Linear(128 + 3, 128),  # +3 for xyz coordinates
                ReLU(),
                Linear(128, 256)
            ),
            global_nn=torch.nn.Sequential(
                Linear(256, 256),
                ReLU(),
                Linear(256, 256)
            )
        )

        # Decoder (Feature Propagation)
        self.fp2 = FeaturePropagation(in_channels=256 + 128, out_channels=256)
        self.fp1 = FeaturePropagation(in_channels=256 + in_channels, out_channels=256)

        # Heads
        self.semantic_head = Linear(256, num_classes)
        self.instance_head = Linear(256, embed_dim)

    def forward(self, x, pos, batch):
        # --- Encoder ---
        # SA1: Downsample and extract local features
        idx_sa1 = farthest_point_sample(pos, ratio=0.5, batch=batch)
        sa1_pos, sa1_batch = pos[idx_sa1], batch[idx_sa1]
        edge_index_sa1 = knn(pos, sa1_pos, k=32, batch_x=batch, batch_y=sa1_batch)
        x_sa1 = self.sa1(x=(x, None), pos=(pos, sa1_pos), edge_index=edge_index_sa1)

        # SA2: Further downsample
        idx_sa2 = farthest_point_sample(sa1_pos, ratio=0.25, batch=sa1_batch)
        sa2_pos, sa2_batch = sa1_pos[idx_sa2], sa1_batch[idx_sa2]
        edge_index_sa2 = knn(sa1_pos, sa2_pos, k=64, batch_x=sa1_batch, batch_y=sa2_batch)
        x_sa2 = self.sa2(x=(x_sa1, None), pos=(sa1_pos, sa2_pos), edge_index=edge_index_sa2)

        # --- Decoder ---
        # FP2: Upsample from SA2 to SA1
        x_fp2 = self.fp2(x=(x_sa1, x_sa2), pos=(sa1_pos, sa2_pos))
        
        # FP1: Upsample to original resolution
        x_fp1 = self.fp1(x=(x, x_fp2), pos=(pos, sa1_pos))

        # --- Heads ---
        semantic = self.semantic_head(x_fp1)
        embeddings = self.instance_head(x_fp1)
        return semantic, embeddings

class FeaturePropagation(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            Linear(in_channels, out_channels),
            ReLU(),
            Linear(out_channels, out_channels)
        )

    def forward(self, x, pos):
        x_skip, x = x
        pos_skip, pos = pos
        
        # Interpolate features using inverse distance weighting
        dist = torch.cdist(pos, pos_skip)
        k = 3
        knn_indices = dist.topk(k, dim=1, largest=False).indices
        knn_weights = 1.0 / (dist.gather(1, knn_indices) + 1e-8)
        knn_weights /= knn_weights.sum(dim=1, keepdim=True)
        
        x_interpolated = (x_skip[knn_indices] * knn_weights.unsqueeze(-1)).sum(dim=1)
        x_combined = torch.cat([x, x_interpolated], dim=-1)
        return self.mlp(x_combined)
    


from torch_geometric.data import Dataset, Data

class SurfaceDataset(Dataset):
    def __init__(self, root, transform=None):
        super().__init__(root, transform)
        # Load preprocessed .pt files containing Data objects
        self.paths = [os.path.join(root, f) for f in os.listdir(root)]

    def len(self):
        return len(self.paths)

    def get(self, idx):
        data = torch.load(self.paths[idx])  # Load Data object
        return data
    

def discriminative_loss(embeddings, instance_labels, delta_var=0.5, delta_dist=1.5):
    unique_instances = torch.unique(instance_labels)
    if len(unique_instances) == 0:
        return torch.tensor(0.0, device=embeddings.device)

    loss_var = 0.0
    loss_dist = 0.0
    loss_reg = 0.0

    for instance in unique_instances:
        mask = (instance_labels == instance)
        emb_instance = embeddings[mask]
        if emb_instance.shape[0] == 0:
            continue
        mean_emb = emb_instance.mean(dim=0)
        # Variance term (pull points to mean)
        loss_var += torch.mean(torch.clamp(torch.norm(emb_instance - mean_emb, dim=1) - delta_var, min=0)**2)
        # Regularization term
        loss_reg += torch.mean(torch.norm(mean_emb, dim=0))

    # Distance term (push clusters apart)
    num_instances = len(unique_instances)
    if num_instances > 1:
        means = torch.stack([embeddings[instance_labels == i].mean(dim=0) for i in unique_instances])
        for i in range(num_instances):
            for j in range(i + 1, num_instances):
                loss_dist += torch.clamp(2 * delta_dist - torch.norm(means[i] - means[j]), min=0)**2

    return (loss_var + 0.1 * loss_dist + 0.001 * loss_reg) / len(unique_instances)


from torch_geometric.loader import DataLoader

dataset = SurfaceDataset(root='path/to/data')
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

model = PointNetPlusPlus(in_channels=7, num_classes=3, embed_dim=64)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(100):
    for batch in train_loader:
        optimizer.zero_grad()
        # Forward pass
        semantic_pred, embeddings = model(batch.x, batch.pos, batch.batch)
        # Losses
        loss_sem = F.cross_entropy(semantic_pred, batch.y_semantic)
        loss_inst = discriminative_loss(embeddings, batch.y_instance)
        total_loss = loss_sem + loss_inst
        # Backward pass
        total_loss.backward()
        optimizer.step()

# Save model checkpoint
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}, 'model_checkpoint.pth')

# inference & postprocessing
model.eval()
with torch.no_grad():
    x, pos, batch = batch.x, batch.pos, batch.batch  # Extract data from the batch
    semantic_pred, embeddings = model(x, pos, batch)

# Semantic labels
semantic_labels = torch.argmax(semantic_pred, dim=1).cpu().numpy()

# Instance clustering (example for "flat" surfaces)
flat_mask = (semantic_labels == FLAT_CLASS_ID)
flat_embeddings = embeddings[flat_mask].cpu().numpy()

# DBSCAN clustering
from sklearn.cluster import DBSCAN
clustering = DBSCAN(eps=0.3, min_samples=10)
instance_ids = clustering.fit_predict(flat_embeddings)


# visualization
# import open3d as o3d

# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(pos.cpu().numpy())
# pcd.colors = o3d.utility.Vector3dVector(instance_colors)  # Map instance IDs to colors
# o3d.visualization.draw_geometries([pcd])