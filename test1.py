import open3d as o3d
import numpy as np
from sklearn.cluster import DBSCAN

# Load point cloud from .txt file with comma-separated values
def load_point_cloud_from_txt(file_path):
    points = []
    with open(file_path, 'r') as f:
        for line in f:
            x, y, z = map(float, line.strip().split(','))  # Split by comma
            points.append([x, y, z])
    return np.array(points)

# Load point cloud data
points = load_point_cloud_from_txt("/home/besughino/Downloads/pointclouds_24.10.21/C4.txt")
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

# Preprocessing: Downsampling, outlier removal
pcd = pcd.voxel_down_sample(voxel_size=0.005)
pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=5, std_ratio=20.0)

# Estimate normals for segmentation
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=50))

# Plane Segmentation: Find multiple planes
planes = []
remaining_cloud = pcd
for _ in range(3):  # Attempt to extract multiple planes
    plane_model, inliers = remaining_cloud.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)
    plane_cloud = remaining_cloud.select_by_index(inliers)
    plane_cloud.paint_uniform_color([0, 0, 1])  # Blue color for planes
    planes.append(plane_cloud)
    remaining_cloud = remaining_cloud.select_by_index(inliers, invert=True)

# Calculate curvature to separate flat and curved surfaces in remaining points
curvatures = np.asarray([np.linalg.norm(np.cross(n, [0, 0, 1])) for n in remaining_cloud.normals])
flat_points = np.asarray(remaining_cloud.points)[curvatures <= 0.1]
curved_points = np.asarray(remaining_cloud.points)[curvatures > 0.1]

# DBSCAN clustering on flat points
dbscan_flat = DBSCAN(eps=0.025, min_samples=10).fit(flat_points)
flat_cloud = o3d.geometry.PointCloud()
flat_cloud.points = o3d.utility.Vector3dVector(flat_points)
flat_cloud.paint_uniform_color([0, 0, 1])  # Blue color for flat surfaces

# DBSCAN clustering on curved points
dbscan_curved = DBSCAN(eps=0.01, min_samples=5).fit(curved_points)
curved_cloud = o3d.geometry.PointCloud()
curved_cloud.points = o3d.utility.Vector3dVector(curved_points)
curved_cloud.paint_uniform_color([1, 0, 0])  # Red color for curved surfaces

# Combine planes, flat surfaces, and curved surfaces for final visualization
all_clouds = planes + [flat_cloud, curved_cloud]
o3d.visualization.draw_geometries(all_clouds)

