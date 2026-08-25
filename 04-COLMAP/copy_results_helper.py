#!/usr/bin/env python3
"""
Helper script to copy COLMAP reconstruction results to output directory
"""
import shutil
from pathlib import Path

SOURCE_DIR = "/data/fj/04-COLMAP"
OUTPUT_DIR = "/data/fj/11-COLMAP-fuse"

def copy_plant_results(plant_name):
    """Copy completed plant results to output directory"""
    plant_dir = Path(SOURCE_DIR) / plant_name

    # Check if sparse_txt exists (indicates reconstruction complete)
    sparse_txt = plant_dir / "sparse_txt"
    if not sparse_txt.exists():
        return False

    # Create output directories
    output_dir = Path(OUTPUT_DIR) / plant_name
    output_sparse = output_dir / "sparse"
    output_sparse.mkdir(parents=True, exist_ok=True)
    output_txt = output_dir / "sparse_txt"
    output_txt.mkdir(parents=True, exist_ok=True)

    # Copy sparse binary files
    sparse_0 = plant_dir / "sparse" / "0"
    if sparse_0.exists():
        for file in sparse_0.glob("*"):
            if file.is_file() and file.suffix in ['.bin', '.ini']:
                dest = output_sparse / file.name
                if not dest.exists():
                    shutil.copy2(file, dest)
                    print(f"  [COPY] {plant_name}/{file.name}")

    # Copy sparse text files
    for file in sparse_txt.glob("*.txt"):
        dest = output_txt / file.name
        if not dest.exists():
            shutil.copy2(file, dest)
            print(f"  [COPY] {plant_name}/sparse_txt/{file.name}")

    return True

# Copy all completed plants
print("Copying completed reconstruction results...")
print()

completed = []
for plant in sorted(Path(SOURCE_DIR).iterdir()):
    if plant.is_dir() and not plant.name.startswith('.'):
        if copy_plant_results(plant.name):
            completed.append(plant.name)

print()
print(f"✓ Copied {len(completed)} plants to {OUTPUT_DIR}")
for plant in completed:
    print(f"  - {plant}")
