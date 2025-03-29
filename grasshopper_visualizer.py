import open3d as o3d
import numpy as np
from matplotlib.colors import to_rgb

def parse_point_cloud_data(data):
    surfaces = []
    current_surface = None
    current_points = []
    
    for line in data.split('\n'):
        line = line.strip()
        if not line:
            continue
            
        if line.startswith('[') and 'SRF' in line:
            # New surface found
            if current_surface is not None:
                surfaces.append((current_surface, np.array(current_points)))
                current_points = []
            current_surface = line
        else:
            # Point data line
            if current_surface is not None:
                coords = list(map(float, line.split(',')))
                current_points.append(coords)
    
    # Add the last surface
    if current_surface is not None and current_points:
        surfaces.append((current_surface, np.array(current_points)))
    
    return surfaces

def visualize_with_open3d(surfaces):
    """
    Script to visualize data in the grasshopper-generated format using Open3D.
    """
    # Create a list to hold all point clouds
    pcd_list = []
    
    # Generate distinct colors for each surface
    colors = [
        to_rgb('red'), to_rgb('green'), to_rgb('blue'),
        to_rgb('cyan'), to_rgb('magenta'), to_rgb('yellow'),
        to_rgb('orange'), to_rgb('purple'), to_rgb('lime'),
        to_rgb('pink'), to_rgb('teal'), to_rgb('lavender')
    ]
    
    for i, (surface_info, points) in enumerate(surfaces):
        if points.shape[0] == 0:
            continue
            
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # Assign color to all points in this surface
        color = colors[i % len(colors)]
        pcd.colors = o3d.utility.Vector3dVector([color for _ in range(len(points))])
        
        # Add to list
        pcd_list.append(pcd)
    
    # Visualize all point clouds together
    o3d.visualization.draw_geometries(pcd_list)

# Your input data
filepath = "C:/Users/besugo/Downloads/MODEL_analog-20250329T150651Z-001/MODEL_analog/A_03.24.25_0040_pts.txt"
data = open(filepath, 'r').read()

# Parse and visualize
surfaces = parse_point_cloud_data(data)
visualize_with_open3d(surfaces)