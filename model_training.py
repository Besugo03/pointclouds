import torch
from torch.nn import Linear, ReLU, Sequential
import torch.nn.functional as F
from torch_geometric.nn import PointNetConv, fps, knn_interpolate, radius
import os
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
from typing import cast


CHECKPOINT_DIR = "checkpoints"
SAVE_EVERY = 5  # Save every 5 epochs
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def parse_txt_to_data(txt_path):
    # data from semantic crawler :
    # plane: 29912 times
    # cylinder: 10508 times
    # cone: 3496 times
    # sphere: 2305 times
    # torus: 3275 times
    # one-sheeted hyperboloid: 306 times
    # ellipsoid: 56 times
    # untrimmed surface: 8 times
    semantic_map = {
        "plane": 0,
        "cylinder": 1,
        "cone": 2,
        "sphere": 3,
        "ellipsoid": 4,
        "torus": 5,
        "one-sheeted hyperboloid": 6,
        "untrimmed surface": 7,
    }
    # Use a default ID (e.g., max_id + 1) for unknown types, or raise an error
    default_semantic_id = max(semantic_map.values()) + 1 if semantic_map else 0

    with open(txt_path, "r") as f:
        lines = f.readlines()

    points = []

    semantic_labels = []
    instance_labels = []
    current_instance = -1
    current_semantic_label = -1  # Initialize semantic label

    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Check for header line (e.g., "[0] SRF :plane (1, 1)")
        if line.startswith("["):
            parts = line.split()
            current_instance = int(parts[0][1:-1])  # Extract [0] → 0
            # Handle potential colon in semantic type like ":plane"
            semantic_type = parts[2].lstrip(":")
            # Get semantic label from map, use default if not found
            current_semantic_label = semantic_map.get(
                semantic_type, default_semantic_id
            )
        elif current_instance != -1:  # Ensure we only parse points after a header
            # Parse point data: x,y,z,nx,ny,nz,curvature
            try:
                data = list(map(float, line.split(",")))
                if len(data) == 7:  # Check for expected number of values
                    points.append(data)
                    semantic_labels.append(current_semantic_label)
                    instance_labels.append(current_instance)
                else:
                    print(
                        f"Warning: Skipping malformed data line in {txt_path}: {line}"
                    )
            except ValueError:
                print(
                    f"Warning: Skipping line with non-float data in {txt_path}: {line}"
                )

    if not points:
        print(f"Warning: No valid points found in {txt_path}")
        # Return an empty Data object or handle appropriately
        return Data(
            pos=torch.empty((0, 3), dtype=torch.float),
            x=torch.empty((0, 4), dtype=torch.float),
            y_semantic=torch.empty((0,), dtype=torch.long),
            y_instance=torch.empty((0,), dtype=torch.long),
        )

    # Convert to tensors
    pos = torch.tensor([p[:3] for p in points], dtype=torch.float)  # (N, 3)
    x = torch.tensor(
        [p[3:] for p in points], dtype=torch.float
    )  # (N, 4: nx, ny, nz, curvature)
    y_semantic = torch.tensor(semantic_labels, dtype=torch.long)  # (N,)
    y_instance = torch.tensor(instance_labels, dtype=torch.long)  # (N,)

    # Determine number of classes based on the mapping used
    num_classes = (
        max(semantic_map.values()) + 1 if semantic_map else 1
    )  # +1 because IDs are 0-based

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
                and hasattr(data, "pos")
                and data.pos is not None
                and hasattr(data.pos, "shape")
                and data.pos.shape[0] > 0
            ):
                save_path = os.path.join(processed_dir, f"{txt_file[:-4]}.pt")
                torch.save(data, save_path)
                print(f"Saved: {save_path}")
                processed_count += 1
            else:
                print(f"Skipped saving empty data from: {txt_file}")
    print(f"Preprocessing finished. Processed {processed_count} files.")


class SAModule(torch.nn.Module):
    def __init__(self, ratio, r, nn):
        super().__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointNetConv(nn, add_self_loops=False)

    def forward(self, x, pos, batch):
        # 1. Campiona i punti (es. prende il 25% dei punti)
        idx = fps(pos, batch, ratio=self.ratio)
        # 2. Trova i vicini entro un raggio 'r'
        row, col = radius(
            pos, pos[idx], self.r, batch, batch[idx], max_num_neighbors=64
        )
        edge_index = torch.stack([col, row], dim=0)

        # 3. Applica la convoluzione
        x_dst = None if x is None else x[idx]
        x = self.conv((x, x_dst), (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch


class PointNetPlusPlus(torch.nn.Module):
    def __init__(
        self, num_semantic_classes=8, embed_dim=64
    ):  # Adjusted num_classes for your parser
        super().__init__()

        # ENCODER (4 Livelli)
        # Input features: 4 (nx, ny, nz, curv). +3 di pos aggiunti automaticamente
        self.sa1 = SAModule(
            0.25, 0.1, Sequential(Linear(4 + 3, 64), ReLU(), Linear(64, 64))
        )
        self.sa2 = SAModule(
            0.25, 0.2, Sequential(Linear(64 + 3, 128), ReLU(), Linear(128, 128))
        )
        self.sa3 = SAModule(
            0.25, 0.4, Sequential(Linear(128 + 3, 256), ReLU(), Linear(256, 256))
        )
        self.sa4 = SAModule(
            0.25, 0.8, Sequential(Linear(256 + 3, 512), ReLU(), Linear(512, 512))
        )

        # DECODER (4 Livelli)
        self.fp4 = FPModule(
            Sequential(Linear(512 + 256, 256), ReLU(), Linear(256, 256))
        )
        self.fp3 = FPModule(
            Sequential(Linear(256 + 128, 128), ReLU(), Linear(128, 128))
        )
        self.fp2 = FPModule(Sequential(Linear(128 + 64, 64), ReLU(), Linear(64, 64)))
        self.fp1 = FPModule(Sequential(Linear(64 + 4, 64), ReLU(), Linear(64, 64)))

        # HEADS
        self.semantic_head = Sequential(
            Linear(64, 64), ReLU(), Linear(64, num_semantic_classes)
        )
        self.instance_head = Sequential(Linear(64, 64), ReLU(), Linear(64, embed_dim))

    def forward(self, x, pos, batch):
        # Encoder
        x1, pos1, batch1 = self.sa1(x, pos, batch)
        x2, pos2, batch2 = self.sa2(x1, pos1, batch1)
        x3, pos3, batch3 = self.sa3(x2, pos2, batch2)
        x4, pos4, batch4 = self.sa4(x3, pos3, batch3)

        # Decoder
        x = self.fp4(x4, pos4, batch4, x3, pos3, batch3)
        x = self.fp3(x, pos3, batch3, x2, pos2, batch2)
        x = self.fp2(x, pos2, batch2, x1, pos1, batch1)
        x = self.fp1(
            x, pos1, batch1, x, pos, batch
        )  # Ritorna alla risoluzione originale

        return self.semantic_head(x), self.instance_head(x)


class FPModule(torch.nn.Module):
    def __init__(self, nn):
        super().__init__()
        self.nn = nn

    def forward(self, x, pos, batch, x_skip, pos_skip, batch_skip):
        # Interpola i punti dal livello meno denso a quello più denso
        x = knn_interpolate(x, pos, pos_skip, batch, batch_skip, k=3)
        if x_skip is not None:
            x = torch.cat([x, x_skip], dim=1)
        x = self.nn(x)
        return x


# --- Dataset Class ---
class SurfaceDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        # root is expected to be the directory containing the processed .pt files
        super().__init__(root, transform, pre_transform)
        self.paths = [
            os.path.join(root, f) for f in os.listdir(root) if f.endswith(".pt")
        ]
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


def discriminative_loss(
    embeddings, instance_labels, batch_index, delta_var=0.5, delta_dist=1.5
):
    total_loss = 0.0
    num_graphs = batch_index.max().item() + 1
    valid_graphs = 0

    # Calcoliamo la loss separatamente per ogni nuvola di punti nel batch
    for b in range(num_graphs):
        # Estrai solo i punti di QUESTA specifica nuvola
        mask = batch_index == b
        emb_b = embeddings[mask]
        inst_b = instance_labels[mask]

        unique_instances = torch.unique(inst_b)
        num_instances = len(unique_instances)

        if num_instances == 0:
            continue

        valid_graphs += 1

        means = torch.stack([emb_b[inst_b == i].mean(dim=0) for i in unique_instances])

        # 1. VARIANCE LOSS
        loss_var = 0.0
        for i, instance in enumerate(unique_instances):
            emb_instance = emb_b[inst_b == instance]
            dist_to_mean = torch.norm(emb_instance - means[i], dim=1)
            loss_var += torch.mean(torch.clamp(dist_to_mean - delta_var, min=0) ** 2)
        loss_var /= num_instances

        # 2. DISTANCE LOSS
        loss_dist = 0.0
        if num_instances > 1:
            dist_matrix = torch.cdist(means, means)
            mask_triu = torch.triu(torch.ones_like(dist_matrix), diagonal=1).bool()
            distances = dist_matrix[mask_triu]
            loss_dist = torch.mean(torch.clamp(2 * delta_dist - distances, min=0) ** 2)

        # 3. REGOLARIZZAZIONE
        loss_reg = torch.mean(torch.norm(means, dim=1))

        # Sommiamo la loss di questa nuvola al totale del batch
        total_loss += loss_var + 0.1 * loss_dist + 0.001 * loss_reg

    if valid_graphs == 0:
        return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

    return total_loss / valid_graphs


# -------------------------------------
#   TRAINING
# -------------------------------------
import numpy as np
from torch.optim.lr_scheduler import StepLR
import math


# --- METRICS FUNCTIONS ---
def compute_accuracy(pred, target):
    pred_classes = pred.argmax(dim=1)
    correct = (pred_classes == target).sum().item()
    return correct / target.shape[0]


def compute_miou(pred, target, num_classes):
    pred_classes = pred.argmax(dim=1)
    ious = []
    for cls in range(num_classes):
        pred_inds = pred_classes == cls
        target_inds = target == cls
        intersection = (pred_inds & target_inds).sum().item()
        union = (pred_inds | target_inds).sum().item()
        if union == 0:
            continue  # Ignora le classi non presenti nel batch
        ious.append(float(intersection) / float(max(union, 1)))
    return sum(ious) / len(ious) if ious else 0.0


# --- Define Paths ---
RAW_DATA_DIR = (
    "C:/Users/besugo/Downloads/MODEL_analog-20250329T150651Z-001/training_dataset"
)
PROCESSED_DATA_DIR = (
    "C:/Users/besugo/Downloads/MODEL_analog-20250329T150651Z-001/processed"
)
training_dataset_dir = PROCESSED_DATA_DIR  # Use the same variable

# --- Run Preprocessing ---
# print("--- Running Preprocessing ---")
# dir_to_dataset(RAW_DATA_DIR, PROCESSED_DATA_DIR)
# print("--- Preprocessing Done ---")

# --- Now Initialize Dataset and DataLoader ---

print(f"Initializing dataset from: {training_dataset_dir}")

# trasformazione per normalizzazione dei dati
transform = T.Compose(
    [
        T.Center(),  # Sposta il centroide in 0,0,0
        T.NormalizeScale(),  # Scala i punti tra -1 e 1
    ]
)
dataset = SurfaceDataset(root=training_dataset_dir, transform=transform)

if len(dataset) == 0:
    raise ValueError(
        f"Dataset is empty! No '.pt' files found in {training_dataset_dir}. Check preprocessing output and paths."
    )
print(f"Dataset loaded with {len(dataset)} samples.")


# --- SPLIT TRAIN/VAL (Modo nativo PyG) ---
dataset_size = len(dataset)

# Crea un array di indici mescolati casualmente (da 0 a dataset_size - 1)
indices = torch.randperm(dataset_size).tolist()
# ho provato con random_split di torch e si lamentava del tipo di dato, quindi oh well faremo cosi'

train_size = int(0.8 * dataset_size)

# Suddividi gli indici
train_indices = indices[:train_size]
val_indices = indices[train_size:]

# Creiamo i subset e diciamo a Pyright di trattarli esplicitamente come Dataset
# i know this is some jank-ass shit, its just to make pyright shut the hell up
train_dataset = cast(Dataset, dataset[train_indices])
val_dataset = cast(Dataset, dataset[val_indices])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

model = PointNetPlusPlus(num_semantic_classes=8, embed_dim=64)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Spostamento del training su GPU se disponibile
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {device}")
model = model.to(device)

# SCHEDULER (Dimezza il Learning Rate ogni 20 epoche)
scheduler = StepLR(optimizer, step_size=20, gamma=0.5)

# --- TRAINING LOOP --
num_epochs = 100
num_classes = 8

for epoch in range(num_epochs):
    # FASE DI TRAINING
    model.train()
    train_loss, train_acc, train_miou = 0, 0, 0

    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        sem_pred, inst_embeddings = model(batch.x, batch.pos, batch.batch)

        loss_sem = F.cross_entropy(sem_pred, batch.y_semantic)
        loss_inst = discriminative_loss(inst_embeddings, batch.y_instance, batch.batch)
        total_loss = loss_sem + loss_inst

        total_loss.backward()
        optimizer.step()

        train_loss += total_loss.item()
        train_acc += compute_accuracy(sem_pred, batch.y_semantic)
        train_miou += compute_miou(sem_pred, batch.y_semantic, num_classes)

    scheduler.step()  # Aggiorna il learning rate alla fine dell'epoca

    # FASE DI VALIDATION
    model.eval()
    val_loss, val_acc, val_miou = 0, 0, 0
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            sem_pred, inst_embeddings = model(batch.x, batch.pos, batch.batch)

            loss_sem = F.cross_entropy(sem_pred, batch.y_semantic)
            loss_inst = discriminative_loss(
                inst_embeddings, batch.y_instance, batch.batch
            )
            val_loss += (loss_sem + loss_inst).item()
            val_acc += compute_accuracy(sem_pred, batch.y_semantic)
            val_miou += compute_miou(sem_pred, batch.y_semantic, num_classes)

    # STAMPA RISULTATI
    print(f"Epoch {epoch + 1}/{num_epochs} [LR: {scheduler.get_last_lr()[0]:.5f}]")
    print(
        f"TRAIN | Loss: {train_loss / len(train_loader):.4f} | Acc: {train_acc / len(train_loader):.4f} | mIoU: {train_miou / len(train_loader):.4f}"
    )
    print(
        f"VAL   | Loss: {val_loss / len(val_loader):.4f} | Acc: {val_acc / len(val_loader):.4f} | mIoU: {val_miou / len(val_loader):.4f}"
    )
    print("-" * 50)

    # Salva il best model
    # (Inserisci qui la logica di salvataggio checkpoint come nel tuo codice originale)
# Save model checkpoint
torch.save(
    {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    },
    "model_checkpoint.pth",
)

from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.pyplot as plt

# pointcloud to test inference on
test_pointcloud_path = "C:/Users/besugo/Downloads/MODEL_analog-20250329T150651Z-001/A_03.25.25_0050_pts.txt"
output_ply_path = "output_instance_segmented.ply"

print(f"\nStarting inference on: {test_pointcloud_path}")

# Load the test data
test_data = parse_txt_to_data(test_pointcloud_path)

if (
    test_data is None
    or not hasattr(test_data, "pos")
    or test_data.pos is None
    or test_data.pos.shape[0] == 0
):
    print("Could not load or parse test data, skipping inference.")
else:
    # IMPORTANTE: Salva le coordinate originali prima di normalizzarle
    # così quando salvi il .ply alla fine sarà nella scala/posizione reale!
    original_pos_np = test_data.pos.clone().numpy()

    # IMPORTANTE: Applica la stessa trasformazione usata nel training!
    test_data = transform(test_data)

    test_loader = DataLoader([test_data], batch_size=1)

    model.eval()
    model = model.to(device)  # Sposta il modello su GPU

    with torch.no_grad():
        for batch in test_loader:
            print(f"Processing point cloud with {batch.num_points} points...")
            batch = batch.to(device)  # Sposta i dati su GPU

            # --- Model Prediction ---
            semantic_logits, embeddings = model(batch.x, batch.pos, batch.batch)

            semantic_pred = semantic_logits.argmax(dim=1)

            # --- Instance Segmentation using Clustering ---
            print("Performing clustering on embeddings...")
            embeddings_np = embeddings.cpu().numpy()

            dbscan = DBSCAN(eps=0.5, min_samples=10)
            instance_pred = dbscan.fit_predict(embeddings_np)

            print(
                f"Clustering found {len(np.unique(instance_pred[instance_pred >= 0]))} instances (excluding noise)."
            )

            # --- Visualization Preparation ---
            unique_instance_ids = np.unique(instance_pred)
            num_instances = len(unique_instance_ids[unique_instance_ids >= 0])

            if num_instances > 0:
                colors = plt.get_cmap("tab20", num_instances)
                instance_colors_map = {
                    inst_id: colors(i)[:3]
                    for i, inst_id in enumerate(
                        unique_instance_ids[unique_instance_ids >= 0]
                    )
                }
            else:
                instance_colors_map = {}

            instance_colors_map[-1] = (0.5, 0.5, 0.5)

            point_colors_np = np.array(
                [instance_colors_map[inst_id] for inst_id in instance_pred]
            )

            # --- Save the colored point cloud come PLY ---
            print(f"Saving instance-colored point cloud to {output_ply_path}...")
            try:
                with open(output_ply_path, "w") as f:
                    f.write("ply\n")
                    f.write("format ascii 1.0\n")
                    f.write(
                        f"element vertex {len(original_pos_np)}\n"
                    )  # Usiamo original_pos_np!
                    f.write("property float x\n")
                    f.write("property float y\n")
                    f.write("property float z\n")
                    f.write("property uchar red\n")
                    f.write("property uchar green\n")
                    f.write("property uchar blue\n")
                    f.write("end_header\n")
                    for i in range(len(original_pos_np)):
                        r, g, b = (point_colors_np[i] * 255).astype(np.uint8)
                        # Salviamo con le coordinate REALI, non quelle normalizzate
                        f.write(
                            f"{original_pos_np[i, 0]:.6f} {original_pos_np[i, 1]:.6f} {original_pos_np[i, 2]:.6f} {r} {g} {b}\n"
                        )
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
