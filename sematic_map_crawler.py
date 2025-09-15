import os
from threading import Thread, Lock
from tqdm import tqdm
import re

base_dir = "/home/besugo/Downloads/MODEL_param/extracted_model/MODEL_param"
subfolders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]

surfaces = {}

def extract_surface_type(line):
    match = re.search(r'SRF\s*:\s*([a-zA-Z\- ]+)', line)
    if match:
        return match.group(1).strip().lower()
    return "unknown"

def process_folder(folder, pbar, desc_lock):
    folder_path = os.path.join(base_dir, folder)
    files = [f for f in os.listdir(folder_path) if f.endswith("_pts.txt")]
    for file in files:
        with desc_lock:
            pbar.set_description(f"Processing: {folder}/{file}")
        with open(os.path.join(folder_path, file), 'r') as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line.startswith("["):
                surface_type = extract_surface_type(line)
                with surfaces_lock:
                    if surface_type not in surfaces:
                        surfaces[surface_type] = 0
                    surfaces[surface_type] += 1
        with desc_lock:
            pbar.update(1)

surfaces_lock = Lock()
desc_lock = Lock()
threads = []
total_files = sum(
    len([f for f in os.listdir(os.path.join(base_dir, folder)) if f.endswith("_pts.txt")])
    for folder in subfolders
)

with tqdm(total=total_files, desc="Processing folders") as pbar:
    for folder in subfolders:
        t = Thread(target=process_folder, args=(folder, pbar, desc_lock))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()

for surface, count in surfaces.items():
    print(f"{surface}: {count} times")