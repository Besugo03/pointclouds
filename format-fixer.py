# open file pr_01.txt
with open("pr_02.txt", "r") as file:
    lines = file.readlines()
    # for each line, formatted as "x", "y", "z", "r", "g", "b"
    # format it as x,y,z, removing the quotes
    for i in range(len(lines)):
        line = lines[i]
        line = line.replace("\"", "")
        line = line.replace("\n", "")
        # split the line by commas and take only the first 3 elements
        linearray = line.split(",")
        line = ",".join(linearray[:3])
        lines[i] = line + "\n"
    # write the formatted lines to the file
    with open("pr_02_fixed.txt", "w") as file:
        file.writelines(lines)
