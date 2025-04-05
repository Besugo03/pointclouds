import os 


files_directory = "C:\\Users\\besugo\\Downloads\\MODEL_analog-20250329T150651Z-001\\MODEL_analog"

saved_directory = "C:\\Users\\besugo\\Downloads\\MODEL_analog-20250329T150651Z-001\\MODEL_analog_cleaned"

# Grasshopper-generated format : 
# [0] SRF :plane (1, 1)

# 4.353275, -39.420604, 57.691206
# 3.49419, -38.982253, 56.942181
# 2.71226, -38.808911, 56.642542
# 2.02894, -38.453603, 56.023966

# desired format :
# 4.353275, -39.420604, 57.691206
# 3.49419, -38.982253, 56.942181
# 2.71226, -38.808911, 56.642542
# 2.02894, -38.453603, 56.023966
def clean_grasshopper_file(file_path):
    file_contents = []
    surfaces = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # if the first character is a number, then it is a point data line
            if line[0].isdigit() or line[0] == '-':
                file_contents.append([line])
            if line[0] == "[":
                file_contents.append("\n")
                surfaces.append(line)
            else:
                continue

    # save the file
    output_file = os.path.join(saved_directory, os.path.basename(file_path))
    with open(output_file, 'w') as f:
        for line in file_contents:
            f.write(line[0] + "\n")
    print(f"File saved to: {output_file}")

    print("Surfaces:")
    for surface in surfaces:
        print(surface.strip())
    return (file_contents, surfaces)

for file in os.listdir(files_directory):
    if file.endswith(".txt") and "pts" in file:
        clean_grasshopper_file(os.path.join(files_directory, file))

clean_grasshopper_file("C:/Users/besugo/Downloads/MODEL_analog-20250329T150651Z-001/MODEL_analog/A_03.24.25_0001_pts.txt")