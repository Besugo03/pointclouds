from new_clustering_and_benchmarks import calculate_and_export_pointcloud
import os
from threading import Thread

coords_only_dir = "C:/Users/besugo/Downloads/MODEL_analog-20250329T150651Z-001/MODEL_analog_cleaned"
output_dir = "C:/Users/besugo/Downloads/MODEL_analog-20250329T150651Z-001/processed/"

# Define a wrapper function to run the multithreaded task
def threaded_task(txt_file_path, output_dir):
    calculate_and_export_pointcloud(input_file_path=txt_file_path, 
                                    output_file_dir=output_dir)

test_path = "C:/Users/besugo/Downloads/MODEL_analog-20250329T150651Z-001/MODEL_analog/A_03.24.25_0001_pts.txt"
computed_testpath = "C:/Users/besugo/Downloads/MODEL_analog-20250329T150651Z-001/processed/A_03.24.25_0001_pts_processed.txt"
lines = []
surfaces = []
computed_surfaces_points = []

with open(test_path, 'r') as f:
    lines = f.readlines()
for line in lines:
    if line.startswith("["):
        surfaces.append(line.strip())

with open(computed_testpath, 'r') as f:
    lines = f.read()
    computed_surfaces_str = lines.split('\n\n')


print(surfaces)
print(len(computed_surfaces_str))


if __name__ == '__main__':
    files = os.listdir(coords_only_dir)
    for file in files:
        if file.endswith(".txt"):
            # Get the file name without the extension
            filename = os.path.splitext(file)[0]
            # Define the path to the .txt file
            txt_file_path = os.path.join(coords_only_dir, file)
            # Call the function to convert .txt to .ply
            thread = Thread(target=threaded_task, args=(txt_file_path, output_dir))
            thread.start()
            thread.join()  # Wait for the thread to finish before proceeding