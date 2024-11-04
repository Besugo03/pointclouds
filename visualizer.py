import open3d as o3d
import numpy as np

# Load point cloud data from a text file
chosen_file = "C4"
point_cloud = np.loadtxt(f"/home/besughino/Downloads/pointclouds_24.10.21/{chosen_file}.txt", delimiter=",")

# Create an Open3D PointCloud object and assign points
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(point_cloud)

# Visualize
o3d.visualization.draw_geometries([pcd])
