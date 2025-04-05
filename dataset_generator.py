from new_clustering_and_benchmarks import calculate_and_export_pointcloud
import os
from threading import Thread

coords_only_dir = "C:/Users/besugo/Downloads/MODEL_analog-20250329T150651Z-001/MODEL_analog_cleaned"
output_dir = "C:/Users/besugo/Downloads/MODEL_analog-20250329T150651Z-001/processed/"

# Define a wrapper function to run the multithreaded task
def threaded_task(txt_file_path, output_dir, surfacePoints, surfaces):
    calculate_and_export_pointcloud(
                                    surfacepoints=surfacePoints,
                                    surfaces=surfaces,
                                    input_file_path=txt_file_path, 
                                    output_file_dir=output_dir,
                                    )

# nuvola di test originale uscita da grasshopper
test_ghCloud = "C:/Users/besugo/Downloads/MODEL_analog-20250329T150651Z-001/MODEL_analog/A_03.24.25_0001_pts.txt"

# nuvola di test che include la nuvola con le normali e curvatura calcolate
test_computedCloud = "C:/Users/besugo/Downloads/MODEL_analog-20250329T150651Z-001/processed/A_03.24.25_0001_pts_processed.txt"


ghCloudsDir = "C:/Users/besugo/Downloads/MODEL_analog-20250329T150651Z-001/"
if __name__ == '__main__':
    files = os.listdir(ghCloudsDir)
    print(files)
    for file in files:
        # we have the corresponding points for each surface based on index.
        # eg. in surfacepoints[0] there will be the points of the surface described in surfaces[0].
        surfacePoints = []
        surfaces = []
        surfacePointsIdx = -1
        print(file)
        if file.endswith("_pts.txt"):
            with open(os.path.join(ghCloudsDir,file), 'r') as f:
                lines = f.readlines()
            for line in lines:
                line = line.strip()
                if line == "":
                    continue
                elif line.startswith("["):
                    surfaces.append(line.strip())
                    surfacePointsIdx += 1
                elif line[0].isdigit() or line[0] == '-':
                    try : surfacePoints[surfacePointsIdx]
                    except:
                        surfacePoints.append([])
                    surfacePoints[surfacePointsIdx].append([line])
                f.close()
            txt_file_path = os.path.join(ghCloudsDir, file)
            thread = Thread(target=threaded_task, args=(txt_file_path, output_dir, surfacePoints, surfaces))
            thread.start()
            thread.join()  # Wait for the thread to finish before proceeding
