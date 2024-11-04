import numpy as np
import open3d as o3d
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from scipy.spatial import KDTree
import matplotlib.pyplot as plt

# Parametri manipolabili
K_NEIGHBORS = 10  # Numero di vicini per il calcolo della normale
EPS_CURVATURE = 0.05  # Threshold di curvatura per DBSCAN
MIN_SAMPLES = 10  # Numero minimo di punti per un cluster in DBSCAN
VOXEL_SIZE = 0.2  # Dimensione del voxel per il downsampling

# Step 1: Caricamento e downsampling della nuvola di punti
def load_and_downsample_point_cloud(file_path, voxel_size=VOXEL_SIZE):
    print("Caricamento della nuvola di punti...")
    points = np.loadtxt(file_path, delimiter=",")
    print(f"Nuvola di punti caricata con {points.shape[0]} punti.")
    
    # Converti i punti in PointCloud Open3D e applica il downsampling
    pcl = o3d.geometry.PointCloud()
    pcl.points = o3d.utility.Vector3dVector(points)
    pcl = pcl.voxel_down_sample(voxel_size=voxel_size)
    
    downsampled_points = np.asarray(pcl.points)
    print(f"Nuvola di punti ridotta a {downsampled_points.shape[0]} punti dopo il downsampling.")
    return downsampled_points

# Step 2: Calcolo delle normali tramite PCA locale
def calculate_normals(points, k_neighbors=K_NEIGHBORS):
    print("Calcolo delle normali...")
    tree = KDTree(points)
    normals = []
    for i, point in enumerate(points):
        _, idx = tree.query(point, k=k_neighbors)
        neighbors = points[idx]
        
        pca = PCA(n_components=3)
        pca.fit(neighbors)
        
        normal = pca.components_[-1]  # Il vettore normale è l'ultimo componente
        normals.append(normal)
        
        if i % 1000 == 0:
            print(f"Normale calcolata per {i} punti su {points.shape[0]}")
    
    print("Calcolo delle normali completato.")
    return np.array(normals)

# Step 3: Calcolo della curvatura basata sulla variazione delle normali (versione normale)
def calculate_curvature(points, normals, k_neighbors=K_NEIGHBORS):
    print("Calcolo della curvatura...")
    tree = KDTree(points)
    curvatures = []
    for i, point in enumerate(points):
        _, idx = tree.query(point, k=k_neighbors)
        local_normals = normals[idx]
        
        # Variazione angolare delle normali
        angles = [np.arccos(np.clip(np.dot(normals[i], n), -1.0, 1.0)) for n in local_normals]
        curvature = np.mean(angles)
        curvatures.append(curvature)
        
        if i % 1000 == 0:
            print(f"Curvatura calcolata per {i} punti su {points.shape[0]}")
    
    print("Calcolo della curvatura completato.")
    return np.array(curvatures)

# Calcolo della curvatura in blocchi (versione alternativa)
def calculate_curvature_in_blocks(points, normals, k_neighbors=K_NEIGHBORS, block_size=10000):
    print("Calcolo della curvatura in blocchi...")
    tree = KDTree(points)
    curvatures = np.zeros(points.shape[0])
    
    for i in range(0, len(points), block_size):
        block_end = min(i + block_size, len(points))
        print(f"Calcolo della curvatura per il blocco {i} - {block_end} su {len(points)}")
        
        for j in range(i, block_end):
            _, idx = tree.query(points[j], k=k_neighbors)
            local_normals = normals[idx]
            angles = [np.arccos(np.clip(np.dot(normals[j], n), -1.0, 1.0)) for n in local_normals]
            curvatures[j] = np.mean(angles)
            
    print("Calcolo della curvatura completato.")
    return curvatures

# Step 4: Clusterizzazione in base alla curvatura
def cluster_surfaces(points, curvatures, eps_curvature=EPS_CURVATURE, min_samples=MIN_SAMPLES):
    print("Clusterizzazione delle superfici...")
    clustering = DBSCAN(eps=eps_curvature, min_samples=min_samples).fit(curvatures.reshape(-1, 1))
    labels = clustering.labels_
    unique_labels = np.unique(labels)
    print(f"Clusterizzazione completata. Trovati {len(unique_labels) - 1} cluster validi.")
    return labels

# Step 5: Visualizzazione e colorazione
def visualize_clusters(points, labels):
    print("Visualizzazione della nuvola di punti clusterizzata...")
    unique_labels = np.unique(labels)
    colors = plt.cm.jet(np.linspace(0, 1, len(unique_labels)))[:, :3]

    pcl = o3d.geometry.PointCloud()
    pcl.points = o3d.utility.Vector3dVector(points)
    
    point_colors = np.array([colors[label] if label != -1 else [0, 0, 0] for label in labels])
    pcl.colors = o3d.utility.Vector3dVector(point_colors)
    
    o3d.visualization.draw_geometries([pcl])
    print("Visualizzazione completata.")

# Step 6: Lista delle superfici trovate
def list_surfaces(labels):
    print("Elenco delle superfici trovate:")
    surfaces = {}
    for label in np.unique(labels):
        if label != -1:
            surfaces[label] = np.sum(labels == label)
    for surface_id, count in surfaces.items():
        print(f"Superficie {surface_id}: {count} punti")

# Main
file_path = "/home/besughino/Downloads/pointclouds_24.10.21/C4.txt"
points = load_and_downsample_point_cloud(file_path)
normals = calculate_normals(points)
curvatures = calculate_curvature(points, normals)
labels = cluster_surfaces(points, curvatures)
visualize_clusters(points, labels)
list_surfaces(labels)
