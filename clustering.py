import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN
from scipy.spatial import KDTree

def load_point_cloud(filename):
    try:
        # Load point cloud from a .txt file with comma-separated values
        points = np.loadtxt(filename, delimiter=',')
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("Point cloud data must have three columns.")
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        return points, pcd
    except Exception as e:
        print(f"Error loading point cloud: {e}")
        raise

def estimate_normals(pcd):
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    return np.asarray(pcd.normals)

def dbscan_clustering(points, eps=0.1, min_samples=50):
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(points)
    return labels

def region_growing_clustering(points, normals, angle_threshold=np.deg2rad(3), distance_threshold=0.01):
    kdtree = KDTree(points)
    labels = -np.ones(len(points), dtype=int)
    current_label = 0

    for i in range(len(points)):
        if labels[i] != -1:
            continue
        neighbors = kdtree.query_ball_point(points[i], distance_threshold)
        cluster = []
        
        for neighbor in neighbors:
            if labels[neighbor] == -1 and np.dot(normals[i], normals[neighbor]) > np.cos(angle_threshold):
                labels[neighbor] = current_label
                cluster.append(neighbor)

        if len(cluster) > 10:  # Minimum points in a curved cluster
            current_label += 1
    
    return labels

def classify_point_cloud(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # Estimate normals for the point cloud
    normals = estimate_normals(pcd)
    
    # DBSCAN for flat surfaces
    dbscan_labels = dbscan_clustering(points, eps=0.1, min_samples=50)
    flat_face_indices = (dbscan_labels >= 0) & (np.abs(np.dot(normals, normals.mean(axis=0))) > 0.95)
    
    # Curvature for edge detection
    kdtree = o3d.geometry.KDTreeFlann(pcd)
    curvatures = np.zeros(len(points))
    for i in range(len(points)):
        [_, idx, _] = kdtree.search_knn_vector_3d(pcd.points[i], 10)
        covariance_matrix = np.cov(points[idx].T)
        eigvals = np.linalg.eigvalsh(covariance_matrix)
        curvatures[i] = eigvals[0] / np.sum(eigvals) if np.sum(eigvals) > 0 else 0
    edge_indices = curvatures > 0.05
    
    # Region growing for curved surfaces
    non_flat_indices = ~flat_face_indices
    rg_labels = region_growing_clustering(points[non_flat_indices], normals[non_flat_indices],
                                          angle_threshold=np.deg2rad(3), distance_threshold=0.01)
    curved_surface_indices = np.zeros(len(points), dtype=bool)
    curved_surface_indices[non_flat_indices] = rg_labels >= 0

    # Initialize colors array and apply classifications
    colors = np.full((len(points), 3), 0.7)  # Gray for unclassified points
    colors[flat_face_indices] = [0, 0, 1]    # Blue for flat faces
    colors[edge_indices] = [1, 0, 0]         # Red for edges
    colors[curved_surface_indices] = [0, 1, 0]  # Green for curved surfaces
    
    # Debugging: print counts for each category
    print(f"Flat faces: {np.sum(flat_face_indices)}")
    print(f"Edges: {np.sum(edge_indices)}")
    print(f"Curved surfaces: {np.sum(curved_surface_indices)}")

    # Apply colors to the point cloud and display
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd], window_name="Classified Point Cloud")

# Load the point cloud file and classify it
filename = "/home/besughino/Downloads/pointclouds_24.10.21/C4.txt"
points, _ = load_point_cloud(filename)
classify_point_cloud(points)

