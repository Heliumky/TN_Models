import os
import shutil

# path
base_dir = os.getcwd()

# source code folder name (ex:org)
src_folder = os.path.join(base_dir, "org")

# generate the R folder value from 5.2~10
r_values = [round(i * 0.4, 1) for i in range(13, 26)]  # 0.1~10.0
#r_values = [round(i * 0.1, 1) for i in range(1, 51)]
# new_ .. is replaced value
new_N = 6
new_shift = -20


# iterated all r values to created the different r values
for r in r_values:
    dest_folder = os.path.join(base_dir, f"R{r}")
    # copy folder
    if os.path.exists(dest_folder):
        shutil.rmtree(dest_folder)
    shutil.copytree(src_folder, dest_folder)

    # modify varibles in the file 
    file_path = os.path.join(dest_folder, "two_particles.py")

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # replace the values for what ever you want
    content = content.replace(
        "N = 5",
        f"N = {new_N}"
    )
    content = content.replace(
        "shift = -20",
        f"shift = {new_shift}"
    )
    content = content.replace(
        "Rad = 0.2",
        f"Rad = {r}"
    )


    # save
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)

print("All the folders are created successfully.")

