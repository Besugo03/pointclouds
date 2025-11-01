from new_clustering_and_benchmarks import calculate_and_export_pointcloud
import os
from threading import Thread
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import time

from sematic_map_crawler import extract_surface_type


base_dir = "C:/Users/user/Downloads/MODEL_param"
subfolders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]

# Define a wrapper function to run the multithreaded task
def threaded_task(txt_file_path, output_dir, surfacePoints, surfaces):
    calculate_and_export_pointcloud(
                                    surfacepoints=surfacePoints,
                                    surfaces=surfaces,
                                    input_file_path=txt_file_path, 
                                    output_file_dir=output_dir,
                                    )

# nuvola di test originale uscita da grasshopper
# test_ghCloud = "C:/Users/besugo/Downloads/MODEL_analog-20250329T150651Z-001/MODEL_analog/A_03.24.25_0001_pts.txt"

# nuvola di test che include la nuvola con le normali e curvatura calcolate
# test_computedCloud = "C:/Users/besugo/Downloads/MODEL_analog-20250329T150651Z-001/processed/A_03.24.25_0001_pts_processed.txt"



if __name__ == '__main__':
    print("test")
    # Count total files to process for progress bar
    total_files = 0
    folder_files = {}
    for folder in subfolders:
        folder_path = os.path.join(base_dir, folder)
        files = [f for f in os.listdir(folder_path) if f.endswith("_pts.txt")]
        folder_files[folder] = files
        total_files += len(files)
        print(f"total files updated to {total_files}")
    print(f"total files to process : {total_files}")
    

    with tqdm(total=total_files, desc="Processing files") as pbar:
        for folder in subfolders:
            coords_only_dir = os.path.join(base_dir, folder)
            output_dir = coords_only_dir + "_processed"
            os.makedirs(output_dir, exist_ok=True)
            files = folder_files[folder]
            for file in files:
                surfacePoints = []
                surfaces = []
                surfacePointsIdx = -1
                print(f"Processing {file} in {folder}")
                with open(os.path.join(coords_only_dir, file), 'r') as f:
                    lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    if line == "":
                        continue
                    elif line.startswith("["):
                        if extract_surface_type(line) == "untrimmed surface":
                            continue
                        surfaces.append(line.strip())
                        surfacePoints.append([])  # Create a new list for the new surface
                        surfacePointsIdx += 1
                    elif (line[0].isdigit() or line[0] == '-') and surfacePointsIdx != -1: 
                        surfacePoints[surfacePointsIdx].append([line])
                txt_file_path = os.path.join(coords_only_dir, file)
                thread = Thread(target=threaded_task, args=(txt_file_path, output_dir, surfacePoints, surfaces))
                thread.start()
                thread.join()
                pbar.update(1)