import numpy as np
import open3d as o3d
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from scipy.spatial import KDTree
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib

matplotlib.use('TkAgg')

# ==============================
# Adjustable Parameters
# ==============================
# Point cloud processing parameters
K_NEIGHBORS = 20      # Number of neighbors for normal and curvature calculation
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
        _, idx = tree.query(points[i], k=k_neighbors)
        neighbors = points[idx]
        
        pca = PCA(n_components=3)
        pca.fit(neighbors)
        normal = pca.components_[-1]
        normals.append(normal)
    
    print("Normals calculated.")
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
        angles = [np.arccos(np.clip(np.dot(normals[i], n), -1.0, 1.0)) for n in local_normals]
        curvatures.append(np.mean(angles))

    print("Curvature calculated.")
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
    for i in range(0, len(points), step):
        origin = points[i]
        end = origin + normals[i] * ARROW_LENGTH

        # Create arrow
        arrow = o3d.geometry.LineSet()
        arrow.points = o3d.utility.Vector3dVector([origin, end])
        arrow.lines = o3d.utility.Vector2iVector([[0, 1]])
        arrow.paint_uniform_color([1, 0, 0])  # Red color
        geometries.append(arrow)

        # Create sphere at arrow tip
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=SPHERE_RADIUS)
        sphere.paint_uniform_color([1, 0, 0])
        from matplotlib import cm
        sphere.translate(end)
        geometries.append(sphere)

    return geometries

# Visualize point cloud with clustered surfaces
def visualize_clusters(points, labels, normals):
    print("Visualizing clusters...")
    unique_labels = np.unique(labels)
    colors = plt.cm.jet(np.linspace(0, 1, len(unique_labels)))[:, :3]

    pcl = o3d.geometry.PointCloud()
    pcl.points = o3d.utility.Vector3dVector(points)
    
    point_colors = np.array([colors[label] if label != -1 else [0, 0, 0] for label in labels])
    pcl.colors = o3d.utility.Vector3dVector(point_colors)

    geometries = create_arrows_and_spheres(points, normals)

    o3d.visualization.draw_geometries([pcl] + geometries)
    print("Visualization complete.")

# Visualize point cloud with colors based on curvature and add a legend
def visualize_curvature_with_legend(points, curvatures, curvature_threshold=0.05):
    print("Visualizing curvature-based coloring with legend...")

    # Clip and normalize curvature values for better contrast
    curvatures_clipped = np.clip(curvatures, 0, curvature_threshold)
    curvatures_normalized = curvatures_clipped / curvature_threshold
    
    # Map normalized curvature to colors using 'jet' colormap
    colormap = plt.get_cmap('jet')
    colors = colormap(curvatures_normalized)[:, :3]

    # Create a point cloud and assign colors based on curvature
    pcl = o3d.geometry.PointCloud()
    pcl.points = o3d.utility.Vector3dVector(points)
    pcl.colors = o3d.utility.Vector3dVector(colors)

    # Open3D visualization for the point cloud
    o3d.visualization.draw_geometries([pcl], window_name="Curvature Visualization with Legend")

    # Create a colorbar for the legend using Matplotlib
    fig, ax = plt.subplots(figsize=(6, 1))
    fig.subplots_adjust(bottom=0.5)

    # Configure colorbar settings
    norm = plt.Normalize(vmin=0, vmax=curvature_threshold)
    cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=colormap), cax=ax, orientation='horizontal')
    cbar.set_label("Curvature Value")
    
    # Display the legend as a separate Matplotlib figure
    plt.show(block=True)
    print("Curvature visualization with legend complete.")

# Main execution with additional curvature visualization
file_path = "C:\\Users\\besugo\\Documents\\pointclouds\\C4.txt"
points = load_and_downsample_point_cloud(file_path)
normals = calculate_normals(points)
curvatures = calculate_curvature(points, normals)
labels = cluster_surfaces(points, curvatures)

# # Main execution
# file_path = "C:\\Users\\besugo\\Documents\\pointclouds\\C4.txt"
# points = load_and_downsample_point_cloud(file_path)
# normals = calculate_normals(points)
# curvatures = calculate_curvature(points, normals)
# labels = cluster_surfaces(points, curvatures)

# visualize_clusters(points, labels, normals)
# New curvature-based visualization
visualize_curvature_with_legend(points, curvatures)