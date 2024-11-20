import numpy as np
import open3d as o3d
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from scipy.spatial import KDTree
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt
from matplotlib import cm

o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)

# ==============================
# Adjustable Parameters
# ==============================
# Point cloud processing parameters
K_NEIGHBORS = 12      # Number of neighbors for normal and curvature calculation
VOXEL_SIZE = 1.0      # Downsampling voxel size (increase for larger point clouds)

# Curvature clustering parameters
EPS_CURVATURE = 0.02  # Curvature threshold for DBSCAN clustering
MIN_SAMPLES = 15      # Minimum samples in a cluster for DBSCAN

# Arrow and sphere visualization parameters
ARROW_LENGTH = 2.0    # Length of normal arrows
SPHERE_RADIUS = 0.2   # Radius of the spheres at arrow tips
STEP = 100            # Step size for sampling points for normal visualization

# Surface type classification thresholds
FLAT_CURVATURE_THRESH = 0.01     # Average curvature threshold for flat surfaces
EDGE_CURVATURE_VAR_THRESH = 0.1  # Variance threshold to classify edges
# ==============================

# Load and downsample the point cloud
def load_and_downsample_point_cloud(file_path, voxel_size=VOXEL_SIZE):
    print("Loading point cloud...")
    points = np.loadtxt(file_path, delimiter=",")
    print(f"Loaded {points.shape[0]} points.")
    
    pcl = o3d.geometry.PointCloud()
    pcl.points = o3d.utility.Vector3dVector(points)
    pcl = pcl.voxel_down_sample(voxel_size=voxel_size)
    
    downsampled_points = np.asarray(pcl.points)
    print(f"Downsampled to {downsampled_points.shape[0]} points.")
    return downsampled_points

# Compute normals using PCA on neighbors
def calculate_normals(points, k_neighbors=K_NEIGHBORS):
    print("Calculating normals...")
    tree = KDTree(points)
    normals = []

    for i in range(len(points)):
        # neighbor points and their indices
        _, idx = tree.query(points[i], k=k_neighbors)
        neighbors = points[idx]
        
        pca = PCA(n_components=3)
        pca.fit(neighbors)
        normal = pca.components_[-1]
        normals.append(normal)
    
    print("Normals calculated.")
    # In calculate_normals
    if np.any(np.isnan(normals)) or np.any(np.isinf(normals)):
        print("[WARNING] : NaN or Inf found in normals! Some normals may be invalid.")
    else:
        print("No NaN or Inf found in normals")
    return np.array(normals)

# Calculate curvature based on angular difference of normals
def calculate_curvature(points, normals, k_neighbors=K_NEIGHBORS):
    print("Calculating curvature...")
    tree = KDTree(points)
    curvatures = []

    for i in range(len(points)):
        _, idx = tree.query(points[i], k=k_neighbors)
        local_normals = normals[idx]
        
        # Calculate mean angular change between normals
        # calculated by dot product of normals, clamped to [-1, 1]
        # then arccos to get the angle of separation between normals in radians
        angles = [np.arccos(np.clip(np.dot(normals[i], n), -1.0, 1.0)) for n in local_normals]
        curvatures.append(np.mean(angles))

    print("Curvature calculated.")
    if np.any(np.isnan(curvatures)) or np.any(np.isinf(curvatures)):
        print("[WARNING] : NaN or Inf found in curvatures")
    else:
        print("No NaN or Inf found in curvatures")
    return np.array(curvatures)

# Cluster surfaces using DBSCAN based on curvature
def cluster_surfaces(points, curvatures, eps_curvature=EPS_CURVATURE, min_samples=MIN_SAMPLES):
    print("Clustering surfaces...")
    clustering = DBSCAN(eps=eps_curvature, min_samples=min_samples).fit(curvatures.reshape(-1, 1))
    labels = clustering.labels_
    unique_labels = np.unique(labels)
    print(f"Found {len(unique_labels) - 1} clusters.")
    return labels

# Classify surfaces by curvature and variance
def classify_surface(curvatures, labels):
    surface_types = {}
    for label in np.unique(labels):
        if label == -1:  # Ignore noise points
            continue

        cluster_curvatures = curvatures[labels == label]
        avg_curvature = np.mean(cluster_curvatures)
        curvature_var = np.var(cluster_curvatures)
        
        # Classify based on average curvature and variance
        if curvature_var < EDGE_CURVATURE_VAR_THRESH:
            if avg_curvature < FLAT_CURVATURE_THRESH:
                surface_type = "Flat"
            else:
                surface_type = "Curved"
        else:
            surface_type = "Edge"

        surface_types[label] = surface_type
    return surface_types

# Create arrows and spheres for normal visualization
def create_arrows_and_spheres(points, normals, step=STEP):
    geometries = []
    print(len(points))
    for i in range(0, len(points), step):
        try:
            origin = points[i]
            end = origin + normals[i] * ARROW_LENGTH

            # Create arrow
            print(f"Creating arrow {i} with origin {origin}...")
            arrow = o3d.geometry.LineSet()
            arrow.points = o3d.utility.Vector3dVector([origin, end])
            arrow.lines = o3d.utility.Vector2iVector([[0, 1]])
            arrow.paint_uniform_color([1, 0, 0])  # Red color

            print("Appending arrow...")
            geometries.append(arrow)

            # Create sphere at arrow tip
            print("Creating sphere...")
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=SPHERE_RADIUS)
            sphere.paint_uniform_color([1, 0, 0])
            sphere.translate(end)
            geometries.append(sphere)
        except Exception as e:
            print(f"Error creating geometry at index {i}: {e}")
            continue

    print("Arrows and spheres calulate. Returning geometries...")
    return geometries

# Visualize point cloud with clustered surfaces
def visualize_clusters(points, labels, normals):
    print("Visualizing clusters...")

    print("Calculating surface types...")
    unique_labels = np.unique(labels)
    colors = plt.cm.jet(np.linspace(0, 1, len(unique_labels)))[:, :3]

    print("Classifying surface types...")
    pcl = o3d.geometry.PointCloud()
    pcl.points = o3d.utility.Vector3dVector(points)
    
    print("Creating point cloud colors...")
    point_colors = np.array([colors[label] if label != -1 else [0, 0, 0] for label in labels])
    pcl.colors = o3d.utility.Vector3dVector(point_colors)

    # non so che cazzo sia successo al codice per le frecce, ma non funziona
    print("Creating normal arrows and spheres...")
    geometries = create_arrows_and_spheres(points, normals)
    print("arrows and spheres created.")

    print("Visualizing point cloud with clusters...")
    o3d.visualization.draw_geometries([pcl] + geometries)
    o3d.visualization.draw_geometries([pcl])
    print("Visualization complete.")

# Visualize point cloud with colors based on curvature and add a legend
def visualize_curvature_with_legend(points, curvatures, curvature_threshold=0.05):

    # Clip and normalize curvature values for better contrast
    print("Clipping and normalizing curvatures...")
    curvatures_clipped = np.clip(curvatures, 0, curvature_threshold)
    curvatures_normalized = curvatures_clipped / curvature_threshold
    
    # Map normalized curvature to colors using 'jet' colormap
    print("Mapping curvature to colors...")
    colormap = plt.get_cmap('jet')
    colors = colormap(curvatures_normalized)[:, :3]

    # Create a point cloud and assign colors based on curvature
    print("Creating point cloud with curvature colors...")
    pcl = o3d.geometry.PointCloud()
    print("using utility piece of shit")
    # using utility crashes execution
    pcl.points = o3d.utility.Vector3dVector(points)
    pcl.colors = o3d.utility.Vector3dVector(colors)
    print("utility code ends")
    

    # Open3D visualization for the point cloud
    print("Visualizing curvature with legend...")
    o3d.visualization.draw_geometries([pcl], window_name="Curvature Visualization with Legend")
    print("Visualization complete.")

    # Create a colorbar for the legend using Matplotlib
    fig, ax = plt.subplots(figsize=(6, 1))
    fig.subplots_adjust(bottom=0.5)

    # Configure colorbar settings
    norm = plt.Normalize(vmin=0, vmax=curvature_threshold)
    cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=colormap), cax=ax, orientation='horizontal')
    cbar.set_label("Curvature Value")
    
    # Display the legend as a separate Matplotlib figure
    plt.show()
    print("Curvature visualization with legend complete.")

# Main execution with additional curvature visualization
file_path = "pr_01_fixed.txt"
points = load_and_downsample_point_cloud(file_path)
normals = calculate_normals(points)
curvatures = calculate_curvature(points, normals)
labels = cluster_surfaces(points, curvatures)

# Main execution
# file_path = "C:\\Users\\besugo\\Documents\\pointclouds\\C4.txt"
# points = load_and_downsample_point_cloud(file_path)
# normals = calculate_normals(points)
# curvatures = calculate_curvature(points, normals)
# labels = cluster_surfaces(points, curvatures)


# debug
print("Points shape:", points.shape)
print("Normals shape:", normals.shape)
print("Labels shape:", labels.shape)
print("Unique labels:", np.unique(labels))
# New curvature-based visualization
print("visualizing curvature...")
visualize_clusters(points, labels, normals)
visualize_curvature_with_legend(points, curvatures)

# another curvature-based visualization but this time with the arrows and spheres aswell.