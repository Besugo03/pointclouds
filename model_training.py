import torch
from torch.nn import Linear, ReLU, Sequential, BatchNorm1d
import torch.nn.functional as F
from torch_geometric.nn import PointNetConv, fps, knn_interpolate, radius
import os
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
from typing import cast
import numpy as np
from torch.optim.lr_scheduler import StepLR
import argparse


CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# ==========================================
#               PARSER & DATASET
# ==========================================
def parse_txt_to_data(txt_path):
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
    default_semantic_id = max(semantic_map.values()) + 1 if semantic_map else 0

    with open(txt_path, "r") as f:
        lines = f.readlines()

    points, semantic_labels, instance_labels = [], [], []
    current_instance, current_semantic_label = -1, -1

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if line.startswith("["):
            parts = line.split()
            current_instance = int(parts[0][1:-1])
            semantic_type = parts[2].lstrip(":")
            current_semantic_label = semantic_map.get(
                semantic_type, default_semantic_id
            )
        elif current_instance != -1:
            try:
                data = list(map(float, line.split(",")))
                if len(data) == 7:  # CONTROLLO CRITICO: Servono 7 valori!
                    points.append(data)
                    semantic_labels.append(current_semantic_label)
                    instance_labels.append(current_instance)
                else:
                    pass  # Silenziamo l'avviso per non intasare il terminale
            except ValueError:
                pass

    if not points:
        return Data(
            pos=torch.empty((0, 3), dtype=torch.float),
            x=torch.empty((0, 4), dtype=torch.float),
        )

    pos = torch.tensor([p[:3] for p in points], dtype=torch.float)
    x = torch.tensor([p[3:] for p in points], dtype=torch.float)
    y_semantic = torch.tensor(semantic_labels, dtype=torch.long)
    y_instance = torch.tensor(instance_labels, dtype=torch.long)

    return Data(pos=pos, x=x, y_semantic=y_semantic, y_instance=y_instance)


def dir_to_dataset(raw_txt_dir, processed_dir):
    os.makedirs(processed_dir, exist_ok=True)
    processed_count = 0
    for txt_file in os.listdir(raw_txt_dir):
        if txt_file.endswith(".txt"):
            txt_path = os.path.join(raw_txt_dir, txt_file)
            data = parse_txt_to_data(txt_path)
            if data is not None and data.pos.shape[0] > 0:
                save_path = os.path.join(processed_dir, f"{txt_file[:-4]}.pt")
                torch.save(data, save_path)
                processed_count += 1
    print(f"[*] Preprocessing finished. Processed {processed_count} files.")


class SurfaceDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.paths = [
            os.path.join(root, f) for f in os.listdir(root) if f.endswith(".pt")
        ]

    def len(self):
        return len(self.paths)

    def get(self, idx):
        return torch.load(self.paths[idx], weights_only=False)


# ==========================================
#               MODEL ARCHITECTURE
# ==========================================
class SAModule(torch.nn.Module):
    def __init__(self, ratio, r, nn):
        super().__init__()
        self.ratio, self.r = ratio, r
        self.conv = PointNetConv(nn, add_self_loops=False)

    def forward(self, x, pos, batch):
        idx = fps(pos, batch, ratio=self.ratio)
        row, col = radius(
            pos, pos[idx], self.r, batch, batch[idx], max_num_neighbors=64
        )
        edge_index = torch.stack([col, row], dim=0)
        x_dst = None if x is None else x[idx]
        x = self.conv((x, x_dst), (pos, pos[idx]), edge_index)
        return x, pos[idx], batch[idx]


class FPModule(torch.nn.Module):
    def __init__(self, nn):
        super().__init__()
        self.nn = nn

    def forward(self, x, pos, batch, x_skip, pos_skip, batch_skip):
        x = knn_interpolate(x, pos, pos_skip, batch, batch_skip, k=3)
        if x_skip is not None:
            x = torch.cat([x, x_skip], dim=1)
        return self.nn(x)


class PointNetPlusPlus(torch.nn.Module):
    def __init__(self, num_semantic_classes=8, embed_dim=64):
        super().__init__()

        # Helper per creare i blocchi in modo pulito con BatchNorm
        def make_mlp(in_channels, mlp_channels):
            layers = []
            for out_channels in mlp_channels:
                layers.append(Linear(in_channels, out_channels))
                layers.append(BatchNorm1d(out_channels))  # LA MAGIA ANTICOLLASSO
                layers.append(ReLU())
                in_channels = out_channels
            return Sequential(*layers)

        # ENCODER
        self.sa1 = SAModule(0.25, 0.1, make_mlp(4 + 3, [64, 64]))
        self.sa2 = SAModule(0.25, 0.2, make_mlp(64 + 3, [128, 128]))
        self.sa3 = SAModule(0.25, 0.4, make_mlp(128 + 3, [256, 256]))
        self.sa4 = SAModule(0.25, 0.8, make_mlp(256 + 3, [512, 512]))

        # DECODER
        self.fp4 = FPModule(make_mlp(512 + 256, [256, 256]))
        self.fp3 = FPModule(make_mlp(256 + 128, [128, 128]))
        self.fp2 = FPModule(make_mlp(128 + 64, [64, 64]))
        self.fp1 = FPModule(make_mlp(64 + 4, [64, 64]))

        # HEADS
        # (Nelle teste finali di solito non si mette BatchNorm sull'ultimo layer)
        self.semantic_head = Sequential(
            Linear(64, 64), BatchNorm1d(64), ReLU(), Linear(64, num_semantic_classes)
        )
        self.instance_head = Sequential(
            Linear(64, 64), BatchNorm1d(64), ReLU(), Linear(64, embed_dim)
        )

    def forward(self, x, pos, batch):
        x0 = x
        x1, pos1, batch1 = self.sa1(x, pos, batch)
        x2, pos2, batch2 = self.sa2(x1, pos1, batch1)
        x3, pos3, batch3 = self.sa3(x2, pos2, batch2)
        x4, pos4, batch4 = self.sa4(x3, pos3, batch3)

        d3 = self.fp4(x4, pos4, batch4, x3, pos3, batch3)
        d2 = self.fp3(d3, pos3, batch3, x2, pos2, batch2)
        d1 = self.fp2(d2, pos2, batch2, x1, pos1, batch1)
        out = self.fp1(d1, pos1, batch1, x0, pos, batch)

        # Normalizzazione L2 sugli Embeddings!
        # Questo costringe i vettori delle istanze a stare su una sfera perfetta
        # rendendo MOLTO più facile il lavoro della Loss Repulsiva
        embeddings = self.instance_head(out)
        # embeddings = F.normalize(embeddings, p=2, dim=1)

        return self.semantic_head(out), embeddings


# ==========================================
#               LOSS & METRICS
# ==========================================
def discriminative_loss(
    embeddings, instance_labels, batch_index, delta_var=0.5, delta_dist=1.5
):
    total_loss = 0.0
    num_graphs = batch_index.max().item() + 1
    valid_graphs = 0

    for b in range(num_graphs):
        mask = batch_index == b
        emb_b = embeddings[mask]
        inst_b = instance_labels[mask]

        unique_instances = torch.unique(inst_b)
        num_instances = len(unique_instances)
        if num_instances == 0:
            continue
        valid_graphs += 1

        means = torch.stack([emb_b[inst_b == i].mean(dim=0) for i in unique_instances])

        loss_var = 0.0
        for i, instance in enumerate(unique_instances):
            emb_instance = emb_b[inst_b == instance]
            loss_var += torch.mean(
                torch.clamp(
                    torch.norm(emb_instance - means[i], dim=1) - delta_var, min=0
                )
                ** 2
            )
        loss_var /= num_instances

        loss_dist = 0.0
        if num_instances > 1:
            dist_matrix = torch.cdist(means, means)
            mask_triu = torch.triu(torch.ones_like(dist_matrix), diagonal=1).bool()
            loss_dist = torch.mean(
                torch.clamp(2 * delta_dist - dist_matrix[mask_triu], min=0) ** 2
            )

        loss_reg = torch.mean(torch.norm(means, dim=1))
        # ricordare che il termine che sta fra la somma e loss_dist sarebbe la repulsive force.
        total_loss += loss_var + 1 * loss_dist + 0.001 * loss_reg

    return (
        total_loss / valid_graphs
        if valid_graphs > 0
        else torch.tensor(0.0, device=embeddings.device, requires_grad=True)
    )


def compute_accuracy(pred, target):
    return (pred.argmax(dim=1) == target).sum().item() / max(target.shape[0], 1)


def compute_miou(pred, target, num_classes):
    pred_classes = pred.argmax(dim=1)
    ious = []
    for cls in range(num_classes):
        intersection = ((pred_classes == cls) & (target == cls)).sum().item()
        union = ((pred_classes == cls) | (target == cls)).sum().item()
        if union > 0:
            ious.append(intersection / union)
    return sum(ious) / len(ious) if ious else 0.0


# ==========================================
#               MAIN SCRIPT
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Point Cloud Instance Segmentation")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "inference"],
        default="train",
        help="Train or Inference.",
    )

    # Path Arguments
    parser.add_argument(
        "--raw_dir",
        type=str,
        default="/home/besugo/Downloads/MODEL_param/extracted_model/MODEL_param/processed/Bolt_500_processed/",
        help="Dir with 7-coord TXT files for training",
    )
    parser.add_argument(
        "--pt_dir",
        type=str,
        default="/home/besugo/Downloads/MODEL_param/extracted_model/MODEL_param/pt_tensors/",
        help="Dir to save/load .pt tensors",
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default="P_05.16.25_0001_pts_processed.txt",
        help="Path to the 7-coord .txt file for inference.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="model_checkpoint.pth",
        help="Path to the saved model weights.",
    )

    # Inference Arguments
    parser.add_argument(
        "--eps", type=float, default=0.5, help="DBSCAN epsilon radius (tune this!)"
    )
    parser.add_argument(
        "--min_samples", type=int, default=10, help="DBSCAN min points per cluster"
    )
    parser.add_argument(
        "--debug_embeddings",
        action="store_true",
        help="Saves a PLY of the 64D embedding space compressed to 3D via PCA",
    )

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Using device: {device}")

    if args.mode == "train":
        print("\n--- Starting Training Mode ---")
        if len(os.listdir(args.pt_dir)) == 0:
            print("--- Running Preprocessing ---")
            dir_to_dataset(args.raw_dir, args.pt_dir)

        transform = T.Compose([T.Center(), T.NormalizeScale()])
        dataset = SurfaceDataset(root=args.pt_dir, transform=transform)
        if len(dataset) == 0:
            raise ValueError("Dataset is empty!")

        indices = torch.randperm(len(dataset)).tolist()
        train_size = int(0.8 * len(dataset))

        # Modalita' originale. Di sotto vi e' il test.
        train_dataset = cast(Dataset, dataset[indices[:train_size]])
        # Test per vedere se traina davvero
        # train_dataset = cast(Dataset, dataset[:2])
        val_dataset = cast(Dataset, dataset[indices[train_size:]])

        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

        torch.cuda.empty_cache()
        model = PointNetPlusPlus(num_semantic_classes=8, embed_dim=64).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = StepLR(optimizer, step_size=20, gamma=0.5)

        for epoch in range(100):
            model.train()
            train_loss_sem, train_loss_inst, train_acc, train_miou = (
                0,
                0,
                0,
                0,
            )  # Modifica qui

            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                sem_pred, inst_embeddings = model(batch.x, batch.pos, batch.batch)

                loss_sem = F.cross_entropy(sem_pred, batch.y_semantic)
                loss_inst = discriminative_loss(
                    inst_embeddings, batch.y_instance, batch.batch
                )
                loss = loss_sem + loss_inst

                loss.backward()
                optimizer.step()

                # Salviamo le loss separate!
                train_loss_sem += loss_sem.item()
                train_loss_inst += loss_inst.item()
                train_acc += compute_accuracy(sem_pred, batch.y_semantic)
                train_miou += compute_miou(sem_pred, batch.y_semantic, 8)

            scheduler.step()

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
                    val_miou += compute_miou(sem_pred, batch.y_semantic, 8)

            # STAMPA CORRETTA (valori mediati per numero di batch)
            print(
                f"Epoch {epoch + 1:03d} | "
                f"TR: L_Sem={train_loss_sem / len(train_loader):.3f} L_Inst={train_loss_inst / len(train_loader):.3f} mIoU={train_miou / len(train_loader):.3f} | "
            )
            print(
                f"VL: L={val_loss / len(val_loader):.3f} A={val_acc / len(val_loader):.3f} mIoU={val_miou / len(val_loader):.3f}"
            )

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            args.weights,
        )
        print(f"[*] Model saved to {args.weights}")

    elif args.mode == "inference":
        from sklearn.cluster import DBSCAN
        from sklearn.decomposition import PCA
        import matplotlib.pyplot as plt

        print(f"\n--- Starting Inference Mode on {args.test_file} ---")
        model = PointNetPlusPlus(num_semantic_classes=8, embed_dim=64).to(device)
        model.load_state_dict(
            torch.load(args.weights, map_location=device, weights_only=False)[
                "model_state_dict"
            ]
        )
        model.eval()

        test_data = parse_txt_to_data(args.test_file)
        if test_data.pos.shape[0] == 0:
            print(
                "[!] ERRORE: La nuvola in input è vuota o manca di normali/curvature (Serve un file con 7 coordinate per riga!)"
            )
            exit()

        original_pos_np = test_data.pos.clone().numpy()
        test_data = T.Compose([T.Center(), T.NormalizeScale()])(test_data)

        with torch.no_grad():
            batch = next(iter(DataLoader([test_data], batch_size=1))).to(device)
            print(
                f"[*] Processing {batch.pos.shape[0]} valid points with 7-features..."
            )

            # CORREZIONE: Estraiamo sia i logits semantici che gli embeddings!
            semantic_logits, embeddings = model(batch.x, batch.pos, batch.batch)
            semantic_pred = semantic_logits.argmax(dim=1)

            embeddings_np = embeddings.cpu().numpy()

            print(
                f"[*] Clustering con DBSCAN (eps={args.eps}, min_samples={args.min_samples})..."
            )
            instance_pred = DBSCAN(
                eps=args.eps, min_samples=args.min_samples
            ).fit_predict(embeddings_np)
            unique_instances = np.unique(instance_pred)
            num_instances = len(unique_instances[unique_instances >= 0])
            print(f"[*] Trovate {num_instances} istanze valide.")

            # PCA Debug
            if args.debug_embeddings:
                print(
                    "[*] Esporto la visualizzazione dello spazio degli Embeddings (PCA)..."
                )
                pca = PCA(n_components=3)
                pca_emb = pca.fit_transform(embeddings_np)
                # Normalizziamo le coordinate PCA per visualizzarle bene in MeshLab
                pca_emb = (pca_emb - pca_emb.min(axis=0)) / (
                    pca_emb.max(axis=0) - pca_emb.min(axis=0)
                )

                with open("debug_embeddings_space.ply", "w") as f:
                    f.write(
                        f"ply\nformat ascii 1.0\nelement vertex {len(pca_emb)}\nproperty float x\nproperty float y\nproperty float z\nend_header\n"
                    )
                    for pt in pca_emb:
                        f.write(f"{pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f}\n")
                print(
                    "[*] File 'debug_embeddings_space.ply' salvato! Aprilo per vedere se il modello ha raggruppato bene i punti."
                )

            # Coloring & Saving
            color_map = {-1: (0.5, 0.5, 0.5)}
            if num_instances > 0:
                colors = plt.get_cmap("tab20", num_instances)
                for i, inst_id in enumerate(unique_instances[unique_instances >= 0]):
                    color_map[inst_id] = colors(i)[:3]

            point_colors = np.array([color_map[id] for id in instance_pred])

            # Save PLY con Semantica e Istanze come campi custom!
            # Crea il nome basandosi sul file originale
            base_name = os.path.splitext(os.path.basename(args.test_file))[0]

            # Salva nella stessa cartella del file di input
            # input_dir = os.path.dirname(args.test_file)
            input_dir = "."

            # Oppure, se vuoi salvarli nella cartella in cui stai lavorando ma col nome giusto:
            # input_dir = "."

            output_ply = os.path.join(input_dir, f"{base_name}_segmented.ply")
            debug_ply = os.path.join(input_dir, f"{base_name}_debug_PCA.ply")
            with open(output_ply, "w") as f:
                f.write("ply\n")
                f.write("format ascii 1.0\n")
                f.write(f"element vertex {len(original_pos_np)}\n")
                f.write("property float x\n")
                f.write("property float y\n")
                f.write("property float z\n")
                f.write("property uchar red\n")
                f.write("property uchar green\n")
                f.write("property uchar blue\n")

                # --- I NOSTRI CAMPI MAGICI ---
                f.write(
                    "property int semantic_class\n"
                )  # Che cos'è? (0=plane, 1=cylinder...)
                f.write(
                    "property int instance_id\n"
                )  # Quale numero è? (Cluster DBSCAN)

                f.write("end_header\n")

                # Attenzione: semantic_pred è un tensore PyTorch sulla GPU, lo portiamo su CPU e Numpy
                semantic_np = semantic_pred.cpu().numpy()

                for i in range(len(original_pos_np)):
                    r, g, b = (point_colors[i] * 255).astype(np.uint8)

                    # Estraiamo la classe (es. 0) e l'istanza (es. 2) per QUESTO singolo punto
                    sem_val = semantic_np[i]
                    inst_val = instance_pred[
                        i
                    ]  # DBSCAN label (può essere -1 se è rumore)

                    # Scriviamo la riga completa: X Y Z R G B Semantica Istanza
                    f.write(
                        f"{original_pos_np[i, 0]:.6f} {original_pos_np[i, 1]:.6f} {original_pos_np[i, 2]:.6f} "
                        f"{r} {g} {b} {sem_val} {inst_val}\n"
                    )

            print(f"[*] Saved output to {output_ply}")

            # --- STAMPA UN PICCOLO RECAP A SCHERMO PER COMODITÀ ---
            semantic_map_reverse = {
                0: "plane",
                1: "cylinder",
                2: "cone",
                3: "sphere",
                4: "ellipsoid",
                5: "torus",
                6: "hyperboloid",
                7: "untrimmed",
            }
            print("\n[*] --- INFERENCE SUMMARY ---")
            for inst_id in unique_instances:
                if inst_id == -1:
                    continue  # Salta il rumore

                # Trova tutti i punti che appartengono a questa istanza
                points_in_instance = instance_pred == inst_id

                sem_in_instance = semantic_np[points_in_instance]

                # CORREZIONE 2: Cast a int per risolvere il problema Numpy intp vs Python int
                majority_sem = int(np.bincount(sem_in_instance).argmax())

                surface_name = semantic_map_reverse.get(majority_sem, "Unknown")
                print(
                    f"    - Instance {inst_id}: {points_in_instance.sum()} points -> Classified as {surface_name.upper()}"
                )
