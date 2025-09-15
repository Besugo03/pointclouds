import torch
from torch.nn import Linear, ReLU
import torch.nn.functional as F
from torch_geometric.nn import PointNetConv, fps, knn_interpolate
from torch_geometric.nn import knn
from torch_geometric.nn.pool import fps
import os
from torch_geometric.data import Data
import torch
from torch.nn import Linear, ReLU, Sequential
import torch.nn.functional as F
from torch_geometric.nn import PointNetConv, fps, knn_interpolate


CHECKPOINT_DIR = "checkpoints"
SAVE_EVERY = 5  # Save every 5 epochs
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def parse_txt_to_data(txt_path):
    # Mapping from semantic type string to class ID
    # Adjust this mapping based on your actual dataset classes
    # plane: 29912 times
    # cylinder: 10508 times
    # cone: 3496 times
    # sphere: 2305 times
    # torus: 3275 times
    # one: 306 times
    # ellipsoid: 56 times
    # untrimmed: 8 times
    semantic_map = {
        "plane": 0,
        "cylinder": 1,
        "cone": 2,
        "sphere": 3,
        "ellipsoid": 4,
        "torus": 5,
        "one-sheeted hyperboloid" : 6,
        "ellipsoid" : 7,
        "untrimmed surface": 8,
        # Add other types if present, mapping them to appropriate IDs
    }
    # Use a default ID (e.g., max_id + 1) for unknown types, or raise an error
    default_semantic_id = max(semantic_map.values()) + 1 if semantic_map else 0

    with open(txt_path, 'r') as f:
        lines = f.readlines()

    points = []
    semantic_labels = []
    instance_labels = []
    current_instance = -1
    current_semantic_label = -1 # Initialize semantic label

    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Check for header line (e.g., "[0] SRF :plane (1, 1)")
        if line.startswith('['):
            parts = line.split()
            current_instance = int(parts[0][1:-1])  # Extract [0] → 0
            # Handle potential colon in semantic type like ":plane"
            semantic_type = parts[2].lstrip(':')
            # Get semantic label from map, use default if not found
            current_semantic_label = semantic_map.get(semantic_type, default_semantic_id)
        elif current_instance != -1: # Ensure we only parse points after a header
            # Parse point data: x,y,z,nx,ny,nz,curvature
            try:
                data = list(map(float, line.split(',')))
                if len(data) == 7: # Check for expected number of values
                    points.append(data)
                    semantic_labels.append(current_semantic_label)
                    instance_labels.append(current_instance)
                else:
                    print(f"Warning: Skipping malformed data line in {txt_path}: {line}")
            except ValueError:
                print(f"Warning: Skipping line with non-float data in {txt_path}: {line}")


    if not points:
         print(f"Warning: No valid points found in {txt_path}")
         # Return an empty Data object or handle appropriately
         return Data(pos=torch.empty((0, 3), dtype=torch.float),
                     x=torch.empty((0, 4), dtype=torch.float),
                     y_semantic=torch.empty((0,), dtype=torch.long),
                     y_instance=torch.empty((0,), dtype=torch.long))


    # Convert to tensors
    pos = torch.tensor([p[:3] for p in points], dtype=torch.float)  # (N, 3)
    x = torch.tensor([p[3:] for p in points], dtype=torch.float)    # (N, 4: nx, ny, nz, curvature)
    y_semantic = torch.tensor(semantic_labels, dtype=torch.long)   # (N,)
    y_instance = torch.tensor(instance_labels, dtype=torch.long)    # (N,)

    # Determine number of classes based on the mapping used
    num_classes = max(semantic_map.values()) + 1 if semantic_map else 1 # +1 because IDs are 0-based

    # Add num_classes to the data object if needed downstream, though not strictly required by the loader
    # data.num_classes = num_classes

    return Data(pos=pos, x=x, y_semantic=y_semantic, y_instance=y_instance)

# --- Preprocessing Script ---
def dir_to_dataset(raw_txt_dir, processed_dir):
    os.makedirs(processed_dir, exist_ok=True)
    print(f"Starting preprocessing from '{raw_txt_dir}' to '{processed_dir}'")
    processed_count = 0
    for txt_file in os.listdir(raw_txt_dir):
        if txt_file.endswith(".txt"):
            txt_path = os.path.join(raw_txt_dir, txt_file)
            print(f"Processing: {txt_file}")
            data = parse_txt_to_data(txt_path)
            # Only save if data is valid (contains points)
            if (
                data is not None
                and hasattr(data, 'pos')
                and data.pos is not None
                and hasattr(data.pos, 'shape')
                and data.pos.shape[0] > 0
            ):
                 save_path = os.path.join(processed_dir, f"{txt_file[:-4]}.pt")
                 torch.save(data, save_path)
                 print(f"Saved: {save_path}")
                 processed_count += 1
            else:
                 print(f"Skipped saving empty data from: {txt_file}")
    print(f"Preprocessing finished. Processed {processed_count} files.")

class PointNetPlusPlus(torch.nn.Module):
    def __init__(self, num_semantic_classes=8, embed_dim=64): # Adjusted num_classes for your parser
        super().__init__()
        
        # <<< FIX: We are building a standard PointNet++ "U-Net" style architecture.
        # This involves an encoder (Set Abstraction) and a decoder (Feature Propagation).

        # --- ENCODER (ZOOMING OUT) ---

        # SA1: First level of abstraction.
        # Input features are 7 (pos=3, normals+curvature=4). local_nn gets these +3 for relative pos.
        self.sa1_module = PointNetConv(
            local_nn=Sequential(Linear(7 + 3, 64), ReLU(), Linear(64, 64)),
            global_nn=Sequential(Linear(64, 128))
        )
        
        # SA2: Second level of abstraction.
        # Input is 128 features from SA1. local_nn gets these +3 for relative pos.
        self.sa2_module = PointNetConv(
            local_nn=Sequential(Linear(128 + 3, 128), ReLU(), Linear(128, 256)),
            global_nn=Sequential(Linear(256, 512)) # This is the "bottleneck" with the most abstract features.
        )

        # --- DECODER (ZOOMING BACK IN) ---
        # <<< FIX: These were the missing parts. They propagate features from coarse to fine.

        # FP2: Propagates features from SA2 back to SA1's resolution.
        # Input channels = (SA2 features) + (SA1 features) = 512 + 128
        self.fp2_module = self.create_fp_module(in_channels=512 + 128, mlp_channels=[256, 256])

        # FP1: Propagates features from FP2 back to the original point resolution.
        # Input channels = (FP2 features) + (Initial features) = 256 + 7
        self.fp1_module = self.create_fp_module(in_channels=256 + 7, mlp_channels=[128, 128])
        
        # --- PREDICTION HEADS ---
        # <<< FIX: The heads now operate on the final, full-resolution feature map from the decoder.
        
        self.semantic_head = Sequential(Linear(128, 64), ReLU(), Linear(64, num_semantic_classes))
        self.instance_head = Sequential(Linear(128, 64), ReLU(), Linear(64, embed_dim))

    def create_fp_module(self, in_channels, mlp_channels):
        # Helper to create a Feature Propagation module
        return Sequential(
            Linear(in_channels, mlp_channels[0]),
            ReLU(),
            Linear(mlp_channels[0], mlp_channels[1])
        )

    def forward(self, x, pos, batch):
        # <<< FIX: Combined initial features are created here.
        initial_features = torch.cat([x, pos], dim=1)
        
        # --- ENCODER ---
        # SA1 Layer
        idx_sa1 = fps(pos, batch, ratio=0.5) # Downsample to 50% of points
        x_sa1, pos_sa1, batch_sa1 = self.sa1_module(
            (initial_features, None), (pos, pos[idx_sa1]), batch=(batch, batch[idx_sa1])
        )
        
        # SA2 Layer
        idx_sa2 = fps(pos_sa1, batch_sa1, ratio=0.25) # Downsample to 25% of the previous set
        x_sa2, pos_sa2, batch_sa2 = self.sa2_module(
            (x_sa1, None), (pos_sa1, pos_sa1[idx_sa2]), batch=(batch_sa1, batch_sa1[idx_sa2])
        )
        
        # --- DECODER ---
        # <<< FIX: This is the full, corrected data flow.

        # FP2: Upsample from SA2 to SA1 resolution.
        # Interpolate SA2 features to SA1's point locations.
        x_interp2 = knn_interpolate(x_sa2, pos_sa2, pos_sa1, batch_sa2, batch_sa1, k=3)
        # Combine with SA1's original features (skip connection).
        x_fp2 = self.fp2_module(torch.cat([x_interp2, x_sa1], dim=1))
        
        # FP1: Upsample from SA1 to original resolution.
        # Interpolate FP2 features to the original point locations.
        x_interp1 = knn_interpolate(x_fp2, pos_sa1, pos, batch_sa1, batch, k=3)
        # Combine with the very first features (skip connection).
        x_fp1 = self.fp1_module(torch.cat([x_interp1, initial_features], dim=1))
        
        # --- HEADS ---
        # Now we have a feature vector for EVERY original point.
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

# --- Dataset Class ---
class SurfaceDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        # root is expected to be the directory containing the processed .pt files
        super().__init__(root, transform, pre_transform)
        self.paths = [os.path.join(root, f) for f in os.listdir(root) if f.endswith('.pt')]
        if not self.paths:
             print(f"Warning: No '.pt' files found in the dataset directory: {root}")


    # These properties replace raw_file_names and processed_file_names logic
    # if you load pre-processed files directly.

    def len(self):
        return len(self.paths)

    def get(self, idx):
        try:
            data = torch.load(self.paths[idx])
            return data
        except Exception as e:
            print(f"Error loading data file: {self.paths[idx]}")
            print(e)
            # Return None or raise error, depending on desired handling
            return Data()
    

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

    # Add epsilon to prevent division by zeroc
    return (loss_var + 0.1 * loss_dist + 0.001 * loss_reg) / (len(unique_instances) + 1e-8)

from torch_geometric.loader import DataLoader

# --- Define Paths ---
RAW_DATA_DIR = "C:/Users/besugo/Downloads/MODEL_analog-20250329T150651Z-001/training_dataset"
PROCESSED_DATA_DIR = "C:/Users/besugo/Downloads/MODEL_analog-20250329T150651Z-001/processed"
training_dataset_dir = PROCESSED_DATA_DIR # Use the same variable

# --- Run Preprocessing ---
print("--- Running Preprocessing ---")
dir_to_dataset(RAW_DATA_DIR, PROCESSED_DATA_DIR)
print("--- Preprocessing Done ---")

# --- Now Initialize Dataset and DataLoader ---
print(f"Initializing dataset from: {training_dataset_dir}")
dataset = SurfaceDataset(root=training_dataset_dir)

if len(dataset) == 0:
     raise ValueError(f"Dataset is empty! No '.pt' files found in {training_dataset_dir}. Check preprocessing output and paths.")

print(f"Dataset loaded with {len(dataset)} samples.")
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

model = PointNetPlusPlus(num_semantic_classes=7, embed_dim=64)
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

        # saving the epoch periodically to avoid overfitting
        if (epoch + 1) % SAVE_EVERY == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': total_loss / len(train_loader),
            }, os.path.join(CHECKPOINT_DIR, f'model_epoch_{epoch+1}.pth'))

# Save model checkpoint
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}, 'model_checkpoint.pth')

from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.pyplot as plt

# pointcloud to test inference on
test_pointcloud_path = "C:/Users/besugo/Downloads/MODEL_analog-20250329T150651Z-001/A_03.25.25_0050_pts.txt"
output_ply_path = "output_instance_segmented.ply"

print(f"\nStarting inference on: {test_pointcloud_path}")

# Load the test data
test_data = parse_txt_to_data(test_pointcloud_path)
if test_data is None or not hasattr(test_data, "pos") or test_data.pos is None or test_data.pos.shape[0] == 0:
    print("Could not load or parse test data, skipping inference.")
else:
    test_loader = DataLoader([test_data], batch_size=1) # Batch size 1 for single file inference

    # Load the trained model weights (if not already loaded)
    # Example: If running inference separately from training
    # checkpoint = torch.load('model_checkpoint.pth')
    # model.load_state_dict(checkpoint['model_state_dict'])
    # print("Loaded trained model weights.")

    model.eval() # Set model to evaluation mode (disables dropout, etc.)

    with torch.no_grad(): # Disable gradient calculations for inference
        for batch in test_loader: # Will loop only once for batch_size=1
            print(f"Processing point cloud with {batch.num_points} points...")
            # Ensure data is on the same device as the model (if using GPU)
            # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            # model.to(device)
            # batch = batch.to(device)

            # --- Model Prediction ---
            semantic_logits, embeddings = model(batch.x, batch.pos, batch.batch)

            # Get semantic predictions
            semantic_pred = semantic_logits.argmax(dim=1) # Shape: (N,)

            # --- Instance Segmentation using Clustering ---
            print("Performing clustering on embeddings...")
            # Move embeddings to CPU and convert to NumPy for scikit-learn
            embeddings_np = embeddings.cpu().numpy()

            # Apply DBSCAN clustering
            # ** IMPORTANT: Tune DBSCAN parameters (eps, min_samples) **
            # These values heavily depend on your embedding space and data.
            # Good values require experimentation. Start with these and adjust.
            dbscan = DBSCAN(eps=0.5, min_samples=10)
            instance_pred = dbscan.fit_predict(embeddings_np) # Shape: (N,)
            # instance_pred contains cluster labels (0, 1, 2, ...)
            # Points deemed noise by DBSCAN get label -1.

            print(f"Clustering found {len(np.unique(instance_pred[instance_pred >= 0]))} instances (excluding noise).")

            # --- Visualization Preparation ---
            # Get positions as NumPy array
            pos_np = batch.pos.cpu().numpy() # Shape: (N, 3)

            # Map instance IDs (cluster labels) to colors
            unique_instance_ids = np.unique(instance_pred)
            num_instances = len(unique_instance_ids[unique_instance_ids >= 0]) # Count non-noise instances

            # Generate distinct colors using a colormap (e.g., 'viridis', 'tab20')
            # Using tab20 provides up to 20 distinct colors
            if num_instances > 0:
                 # Generate N distinct colors + 1 for noise
                 colors = plt.get_cmap('tab20', num_instances)
                 instance_colors_map = {inst_id: colors(i)[:3] # Get RGB tuple (0-1 range)
                                         for i, inst_id in enumerate(unique_instance_ids[unique_instance_ids >= 0])}
            else:
                 instance_colors_map = {}

            # Assign a default color (e.g., gray) for noise points (-1)
            instance_colors_map[-1] = (0.5, 0.5, 0.5) # Gray

            # Create color array for the point cloud
            point_colors_np = np.array([instance_colors_map[inst_id] for inst_id in instance_pred]) # Shape: (N, 3)

            # --- Save the colored point cloud as PLY ---
            print(f"Saving instance-colored point cloud to {output_ply_path}...")
            try:
                with open(output_ply_path, 'w') as f:
                    f.write("ply\n")
                    f.write("format ascii 1.0\n")
                    f.write(f"element vertex {len(pos_np)}\n")
                    f.write("property float x\n")
                    f.write("property float y\n")
                    f.write("property float z\n")
                    f.write("property uchar red\n")
                    f.write("property uchar green\n")
                    f.write("property uchar blue\n")
                    f.write("end_header\n")
                    for i in range(len(pos_np)):
                        # Scale colors from 0-1 range to 0-255 integer range
                        r, g, b = (point_colors_np[i] * 255).astype(np.uint8)
                        f.write(f"{pos_np[i, 0]:.6f} {pos_np[i, 1]:.6f} {pos_np[i, 2]:.6f} {r} {g} {b}\n")
                print(f"Successfully saved colored point cloud to {output_ply_path}")
            except Exception as e:
                print(f"Error saving PLY file: {e}")

# visualization
# import open3d as o3d

# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(pos.cpu().numpy())
# pcd.colors = o3d.utility.Vector3dVector(instance_colors)  # Map instance IDs to colors
# o3d.visualization.draw_geometries([pcd])


# TODO : Things that could be implemented
# 2. implement early stopping
# 3. implement model evaluation
# 4. implement model dropout