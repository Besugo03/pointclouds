import numpy as np
import open3d as o3d
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from scipy.spatial import KDTree
from multiprocessing import Pool, cpu_count, freeze_support
import matplotlib.pyplot as plt
from matplotlib import cm
import pyvista as pv
import time
import timeit


o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)

# ==============================
# Adjustable Parameters
# ==============================
# Point cloud processing parameters
K_NEIGHBORS = 100      # Number of neighbors for normal and curvature calculation
VOXEL_SIZE = 1.5      # Downsampling voxel size (increase for larger point clouds)

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

def visualize_curvature_with_pyvista(points, curvatures):
    # Calcola la soglia di curvatura dinamicamente (80° percentile)
    curvature_threshold = np.percentile(curvatures, 60)
    print(f"Soglia di curvatura (80° percentile): {curvature_threshold}")
    
    # Clip e normalizza i valori di curvatura
    curvatures_clipped = np.clip(curvatures, 0, curvature_threshold)
    curvatures_normalized = curvatures_clipped / curvature_threshold
    
    # Crea una nuvola di punti in PyVista
    cloud = pv.PolyData(points)
    
    # Aggiungi i valori di curvatura come scalari
    cloud.point_data["curvature"] = curvatures_clipped
    
    # Crea un plotter con una barra dei colori
    plotter = pv.Plotter()
    plotter.add_points(
        cloud,
        scalars="curvature",
        cmap="jet",
        render_points_as_spheres=True,
        point_size=5,
        clim=[0, curvature_threshold]  # Imposta i limiti del colore
    )
    
    # Aggiungi una barra dei colori
    # plotter.add_scalar_bar(
    #     title="Curvature",
    #     n_labels=5,
    #     italic=False,
    #     fmt="%.2f",
    #     font_family="arial"
    # )
    
    # Mostra il visualizzatore
    print("Apertura del visualizzatore. Usa il mouse per ruotare/zoomare.")
    plotter.show()
    
    return plotter

# Load and downsample the point cloud
def load_and_downsample_point_cloud(file_path, voxel_size=VOXEL_SIZE):

    print("Loading point cloud...")
    points = np.loadtxt(file_path, delimiter=",")
    print(f"Loaded {points.shape[0]} points.")
    
    # calculate the voxel size dynamically based on the point cloud size.
    # the target number of points is 30k
    target_points = 30000
    # tried 1/3 but it undershot too much
    voxel_size = np.power(points.shape[0] / target_points, 1/3)
    print(f"Voxel size: {voxel_size}")

    pcl = o3d.geometry.PointCloud()
    pcl.points = o3d.utility.Vector3dVector(points)
    pcl = pcl.voxel_down_sample(voxel_size=voxel_size)
    
    downsampled_points = np.asarray(pcl.points)
    print(f"Downsampled to {downsampled_points.shape[0]} points.")
    return downsampled_points

def load_and_downsample_point_cloud_new(
    file_path, 
    target_points=30000, 
    max_iter=10, 
    tolerance=0.1
):
    print("Loading point cloud...")
    points = np.loadtxt(file_path, delimiter=",")
    print(f"Loaded {points.shape[0]} points.")
    
    if points.shape[0] <= target_points:
        print("Point cloud already smaller than target, returning original.")
        return points
    
    pcl = o3d.geometry.PointCloud()
    pcl.points = o3d.utility.Vector3dVector(points)
    
    # Step 1: Compute an initial voxel size guess based on point density
    bbox = pcl.get_axis_aligned_bounding_box()
    bbox_volume = bbox.volume()
    point_density = points.shape[0] / bbox_volume
    
    # Estimate voxel size needed to get ~target_points (assuming uniform distribution)
    # Since N_downsampled ≈ (bbox_volume / voxel_size³) * occupancy_factor
    # We approximate occupancy_factor as 0.2 (adjust if needed)
    initial_voxel_size = (bbox_volume * 0.2 / target_points) ** (1/3)
    
    # Step 2: Binary search refinement to hit target accurately
    min_voxel = initial_voxel_size * 0.1  # Try smaller voxels if needed
    max_voxel = initial_voxel_size * 10   # Try larger voxels if needed
    
    best_voxel = initial_voxel_size
    best_num_points = 0
    
    for _ in range(max_iter):
        current_voxel = (min_voxel + max_voxel) / 2
        downsampled = pcl.voxel_down_sample(current_voxel)
        num_points = len(downsampled.points)
        
        # Check if within tolerance (default ±10%)
        if abs(num_points - target_points) / target_points <= tolerance:
            print(f"Found optimal voxel size: {current_voxel:.5f} → {num_points} points")
            return np.asarray(downsampled.points)
        
        # Update search bounds
        if num_points < target_points:
            max_voxel = current_voxel  # Need smaller voxels (more points)
        else:
            min_voxel = current_voxel  # Need larger voxels (fewer points)
        
        # Track best solution in case we don't converge exactly
        if abs(num_points - target_points) < abs(best_num_points - target_points):
            best_voxel = current_voxel
            best_num_points = num_points
    
    # If not converged, return the best result
    print(f"Using best voxel size: {best_voxel:.5f} → {best_num_points} points (target: {target_points})")
    downsampled = pcl.voxel_down_sample(best_voxel)
    return np.asarray(downsampled.points)

def load_and_downsample_point_cloud_by_skip(surfaces : list, surfacePoints : list, step_size : int):
    global lastpoints, lastpointsPosition
    lastpoints = []
    lastpointsPosition = []
    # "downsamples" the point cloud by skipping points.
    # if we encounter a double \n\n in the txt file, or if it was in the lines we skipped, 
    # we re-insert it before appending the desired point.
    
    # now we have a list of surfaces, we can skip the points for each one
    for i in range(len(surfacePoints)):
        # Downsample the points
        surfacePoints[i] = [surfacePoints[i][j] for j in range(len(surfacePoints[i])) if j % step_size == 0]
        print("last point is : ",surfacePoints[i][-1])
        lastpoints.append(surfacePoints[i][-1])
        # must also sum the previous lengths of lastpointIdx
        # to get the correct index of the last point
        if i == 0:
            lastpointsPosition.append(len(surfacePoints[i]))
        else:
            lastpointsPosition.append(len(surfacePoints[i]) + lastpointsPosition[-1])
        print(f"Downsampled surface to {len(surfacePoints[i])} points.")
        print(lastpoints)
        print(lastpointsPosition)
    
    # now we join the surfaces back together and convert to a numpy array
    
    final_points = []
    for pointsList in surfacePoints:
        # now we can convert to a numpy array.
        # each subarray in pointslist is a string. There's one for each line.
        # we need to convert it to a float array
        points = np.array([list(map(float, point[0].split(","))) for point in pointsList])
        final_points.append(points)

    # now we can concatenate the points
    final_points = np.concatenate(final_points, axis=0)
    print(f"Downsampled to {final_points.shape[0]} points.")
    return final_points


# Compute normals using PCA on neighbors
def calculate_normals(points, KDtree, k_neighbors=K_NEIGHBORS):
    print("Calculating normals...")
    normals = []

    for i in range(len(points)):
        # neighbor points and their indices
        _, idx = KDtree.query(points[i], k=k_neighbors)
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

# Definisci la funzione process_point al di fuori della funzione calculate_normals_parallel
def process_point(args):
    # Disimballare gli argomenti
    i, points, tree, k_neighbors = args
    _, idx = tree.query(points[i], k=k_neighbors)
    neighbors = points[idx]
    pca = PCA(n_components=3)
    pca.fit(neighbors)
    return pca.components_[-1]

def calculate_normals_parallel(points, k_neighbors=K_NEIGHBORS):
    tree = KDTree(points)
    
    # Preparare gli argomenti per ogni punto
    args_list = [(i, points, tree, k_neighbors) for i in range(len(points))]
    
    with Pool(processes=cpu_count()) as pool:
        normals = pool.map(process_point, args_list)
    
    return np.array(normals)

def calculate_normals_open3d(points, k_neighbors=K_NEIGHBORS):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # calcolo delle normali con Open3D
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k_neighbors))
    
    normals = np.asarray(pcd.normals)
    
    return normals

def calculate_normals_vectorized(points, k_neighbors=K_NEIGHBORS):
    tree = KDTree(points)
    _, indices = tree.query(points, k=k_neighbors)
    
    # Raccoglie i vicini per tutti i punti
    neighbors = np.array([points[np.atleast_1d(idx)] for idx in np.atleast_1d(indices)])
    
    normals = np.zeros((len(points), 3))
    for i in range(len(points)):
        # Center the neighbors (PCA should do this, but just in case)
        centered = neighbors[i] - np.mean(neighbors[i], axis=0)
        
        # Compute covariance matrix
        cov = np.dot(centered.T, centered) / (k_neighbors - 1)
        
        # Get eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        
        # Use the eigenvector corresponding to the smallest eigenvalue (last index in eigh)
        normals[i] = eigenvectors[:, 0]  # eigh sorts eigenvalues in ascending order, so last index is smallest
    
    return normals

# Calculate caurvature based on angular difference of normals
def calculate_curvature(points, KDtree, normals, k_neighbors=K_NEIGHBORS):
    print("Calculating curvature...")
    curvatures = []

    for i in range(len(points)):
        _, idx = KDtree.query(points[i], k=k_neighbors)
        local_normals = normals[idx]
        
        # Calculate mean angular change between normals
        # calculated by dot product of normals, clamped to [-1, 1] to be safe (numerical errors caused issues)
        # then arccos to get the angle of separation between normals in radians
        angles = [np.arccos(np.clip(np.dot(normals[i], n), -1.0, 1.0)) for n in local_normals]
        curvatures.append(np.mean(angles))

    print("Curvature calculated.")
    if np.any(np.isnan(curvatures)) or np.any(np.isinf(curvatures)):
        print("[WARNING] : NaN or Inf found in curvatures")
    else:
        print("No NaN or Inf found in curvatures")
    return np.array(curvatures)

def calculate_curvature_vectorized(points, normals, tree, k_neighbors=K_NEIGHBORS):
    # Ottiene gli indici dei k vicini più prossimi per tutti i punti in una sola chiamata
    _, indices = tree.query(points, k=k_neighbors)
    
    # Estrae le normali per tutti i vicini di tutti i punti
    neighbor_normals = normals[indices]  # shape: (n_points, k_neighbors, 3)
    
    # Calcola il prodotto scalare tra ogni normale e i suoi vicini
    # Espande la dimensione delle normali principali per il broadcasting
    main_normals = normals[:, np.newaxis, :]  # shape: (n_points, 1, 3)
    
    # Calcola prodotti scalari
    dot_products = np.sum(main_normals * neighbor_normals, axis=2)  # shape: (n_points, k_neighbors)
    # prendo il valore assoluto di dot_products per tenere conto sia delle normali che puntano  dentro che quelle fuori al solido.
    dot_products = np.abs(dot_products)

    # Applica clip e arccos per ottenere gli angoli
    dot_products = np.clip(dot_products, -1.0, 1.0)
    angles = np.arccos(dot_products)
    
    # Calcola la media degli angoli per ogni punto
    curvatures = np.mean(angles, axis=1)

    return curvatures

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
    try :
        print("Visualizing clusters...")

        print("Calculating surface types...")
        unique_labels = np.unique(labels)
        colors = cm.get_cmap('jet')(np.linspace(0, 1, len(unique_labels)))[:, :3]

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
        # o3d.visualization.draw_geometries([pcl] + geometries)
        # o3d.visualization.draw_geometries([pcl])
        print("Visualization complete.")
    except Exception as e:
        print(f"Error visualizing clusters: {e}")

# Visualize point cloud with colors based on curvature and add a legend
# this uses Open3D for visualization, which is buggy and may crash
# def visualize_curvature_with_legend(points, curvatures, curvature_threshold=0.5):
#     # calculate the curvature threshold dynamically by taking the 95th percentile
#     # of curvature values
#     curvature_threshold = np.percentile(curvatures, 95)
#     try :
#         # Clip and normalize curvature values for better contrast
#         print("Clipping and normalizing curvatures...")
#         curvatures_clipped = np.clip(curvatures, 0, curvature_threshold)
#         curvatures_normalized = curvatures_clipped / curvature_threshold
        
#         # Map normalized curvature to colors using 'jet' colormap
#         print("Mapping curvature to colors...")
#         colormap = plt.get_cmap('jet')
#         colors = colormap(curvatures_normalized)[:, :3]

#         # Create a point cloud and assign colors based on curvature
#         print("Creating point cloud with curvature colors...")
#         pcl = o3d.geometry.PointCloud()
#         print("using utility code")
#         # using utility crashes execution
#         pcl.points = o3d.utility.Vector3dVector(points)
#         pcl.colors = o3d.utility.Vector3dVector(colors)
#         print("utility code ends")
        

#         # Open3D visualization for the point cloud
#         print("Visualizing curvature with legend...")
#         o3d.visualization.draw_geometries([pcl], window_name="Curvature Visualization with Legend")
#         print("Visualization complete.")

#         # Create a colorbar for the legend using Matplotlib
#         fig, ax = plt.subplots(figsize=(6, 1))
#         fig.subplots_adjust(bottom=0.5)

#         # Configure colorbar settings
#         norm = plt.Normalize(vmin=0, vmax=curvature_threshold)
#         cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=colormap), cax=ax, orientation='horizontal')
#         cbar.set_label("Curvature Value")
        
#         # Display the legend as a separate Matplotlib figure
#         plt.show()
#         print("Curvature visualization with legend complete.")
#     except Exception as e:
#         print(f"Error visualizing curvature with legend: {e}")

# file_path = "pr_01_fixed.txt"
# file_path = "C:/Users/besugo/Downloads/MODEL_analog-20250329T150651Z-001/MODEL_analog_cleaned/A_03.24.25_0022_pts.txt"

# def benchmark_functions():
#     # Carica i dati
#     points = load_and_downsample_point_cloud_by_skip(file_path, step_size=10)
    
#     # Crea una KDTree per il riutilizzo
#     tree = KDTree(points)
    
#     # benchmark per il calcolo delle normali
#     start_time = time.time()
#     normals = calculate_normals(points, tree)
#     normal_time = time.time() - start_time
#     print(f"Calcolo delle normali: {normal_time:.2f} secondi")

#     # con metodo parallelo
#     start_time = time.time()
#     normals_parallel = calculate_normals_parallel(points)
#     parallel_time = time.time() - start_time
#     print(f"Calcolo delle normali parallelo: {parallel_time:.2f} secondi")
#     print(f"Speedup: {normal_time / parallel_time:.2f}x")

#     # con metodo open3d
#     start_time = time.time()
#     normals_open3d = calculate_normals_open3d(points, k_neighbors=25)
#     open3d_time = time.time() - start_time
#     print(f"Calcolo delle normali Open3D: {open3d_time:.2f} secondi")
#     print(f"Speedup: {normal_time / open3d_time:.2f}x")

#     # con metodo vettorizzato
#     start_time = time.time()
#     normals_vectorized = calculate_normals_vectorized(points)
#     vectorized_time = time.time() - start_time
#     print(f"Calcolo delle normali vettorizzato: {vectorized_time:.2f} secondi")
#     print(f"Speedup: {normal_time / vectorized_time:.2f}x")

#     # con metodo vettorizzato esatto
#     # start_time = time.time()
#     # normals_vectorized_exact = calculate_normals_vectorized_exact(points)
#     # vectorized_exact_time = time.time() - start_time
#     # print(f"Calcolo delle normali vettorizzato esatto: {vectorized_exact_time:.2f} secondi")
#     # print(f"Speedup: {normal_time / vectorized_exact_time:.2f}x")


#     # Verifica che i risultati siano simili
#     print("Differenza media tra normali originali e parallele:", 
#           np.mean(np.abs(normals - normals_parallel)))
#     print("Differenza media tra normali originali e Open3D:",
#             np.mean(np.abs(normals - normals_open3d)))
#     print("Differenza media tra normali originali e vettorizzate:",
#             np.mean(np.abs(normals - normals_vectorized)))
#     # print("Differenza media tra normali originali e vettorizzate esatte:",
#     #         np.mean(np.abs(normals - normals_vectorized_exact)))
    
#     # Benchmark per il metodo originale
#     start_time = time.time()
#     curvatures_original = calculate_curvature(points, normals=normals, KDtree=tree)
#     original_time = time.time() - start_time
#     print(f"Metodo originale: {original_time:.2f} secondi")
    
#     # Benchmark per il metodo vettorizzato
#     start_time = time.time()
#     curvatures_vectorized = calculate_curvature_vectorized(points, normals=normals, tree=tree)
#     vectorized_time = time.time() - start_time
#     print(f"Metodo vettorizzato: {vectorized_time:.2f} secondi")
#     print(f"Speedup: {original_time / vectorized_time:.2f}x")

#     # Verifica sulla similaritiá dei risultati
#     print("Differenza media tra curvature originali e vettorizzate:", 
#           np.mean(np.abs(curvatures_original - curvatures_vectorized)))
    
#     # Visualizza la curvatura con legenda
#     curvatures2 = calculate_curvature_vectorized(points, normals_open3d, tree=tree)
#     visualize_curvature_with_pyvista(points, curvatures2)
#     visualize_curvature_with_pyvista(points, curvatures_original)

def calculate_and_export_pointcloud(surfacepoints : list,
                                    surfaces : list,
                                    input_file_path : str,
                                    output_file_dir : str,
                                    voxel_size : float = VOXEL_SIZE,
                                    k_neighbors : int = K_NEIGHBORS):
    # calculate the pointcloud using :
    # 1. load_and_downsample_point_cloud
    # 2. calculate_normals (using the parallel version)
    # 3. calculate_curvature (using the parallel version)
    # 4. save the pointcloud to a file in output_file_dir (name of the file = input_file_path + "_processed.txt")
    # the pointcloud will be saved in format x,y,z,nx,ny,nz,curvature

    # load and downsample the point cloud
    points = load_and_downsample_point_cloud_by_skip(surfaces, surfacepoints, step_size=10)
    print("LastPoints :")
    print(lastpoints)
    print("LastPointsPosition :")
    print(lastpointsPosition)
    # create a KDTree for the point cloud
    tree = KDTree(points)
    # calculate normals using the parallel version
    normals = calculate_normals_parallel(points)
    # calculate curvature using the parallel version
    curvatures = calculate_curvature_vectorized(points, normals, tree=tree)
    import os
    # save the point cloud to a new file which has the same name as the input file with "_processed.txt" appended
    # since it has "." in the name, we need to split the name and append "_processed" to the first part
    # get the name of the file without the extension
    filename = os.path.splitext(os.path.basename(input_file_path))[0]
    # create the output file path
    output_file_path = os.path.join(output_file_dir, filename + "_processed.txt")
    with open(output_file_path, 'w') as f:
        f.write(surfaces[0] + "\n")
        for i in range(len(points)):
            f.write(f"{points[i][0]},{points[i][1]},{points[i][2]},{normals[i][0]},{normals[i][1]},{normals[i][2]},{curvatures[i]}\n")
            if i in lastpointsPosition:
                print("i in lastpoints position : ",surfaces[lastpointsPosition.index(i)+1])
                f.write(surfaces[lastpointsPosition.index(i)+1] + "\n")
    print(f"Point cloud saved to {output_file_path}")
    return points, normals, curvatures

# Esegui il benchmark
# if __name__ == '__main__':
#     freeze_support()  # This is needed on Windows
#     benchmark_functions()