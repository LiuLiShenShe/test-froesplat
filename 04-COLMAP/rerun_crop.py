"""
Re-run CaoMei2 and ChangShouHua3 with crop images.
These folders degraded when run with originals, need to restore crop-based results.
"""

import os, sys, shutil, glob, subprocess, time

SAM_DIR   = r"D:\CAAS\03-SAM"
COLMAP_DIR = r"D:\CAAS\04-COLMAP"
CONVERT_PY = r"D:\CAAS\05-2d-gaussian-splatting-great-again-dev\convert.py"
PYTHON_EXE = r"D:\CAAS\05-2d-gaussian-splatting-great-again-dev\.venv_uv\Scripts\python.exe"
COLMAP_EXE = r"D:\CAAS\04-COLMAP\colmap_bin\bin\colmap.exe"

FOLDERS = ["CaoMei2", "ChangShouHua3"]

for folder_name in FOLDERS:
    sam_folder    = os.path.join(SAM_DIR, folder_name)
    colmap_folder = os.path.join(COLMAP_DIR, folder_name)
    input_folder  = os.path.join(colmap_folder, "input")

    print(f"\n{'='*60}")
    print(f"Re-running: {folder_name}")
    print(f"{'='*60}")

    # 1. Clear old results
    for sub in ["sparse", "distorted", "images", "input", "stereo"]:
        p = os.path.join(colmap_folder, sub)
        if os.path.exists(p):
            shutil.rmtree(p)
            print(f"  Removed {sub}/")
    # Remove database
    db = os.path.join(colmap_folder, "distorted", "database.db")
    if os.path.exists(db):
        os.remove(db)
    db2 = os.path.join(colmap_folder, "database.db")
    if os.path.exists(db2):
        os.remove(db2)
        print("  Removed database.db")

    # 2. Copy crop_*.png to input/
    os.makedirs(input_folder, exist_ok=True)
    crop_files = sorted(glob.glob(os.path.join(sam_folder, "crop_*.png")))
    print(f"  Copying {len(crop_files)} crop images to input/ ...")
    for src in crop_files:
        shutil.copy2(src, os.path.join(input_folder, os.path.basename(src)))

    # 3. Run convert.py
    cmd = [
        PYTHON_EXE, CONVERT_PY,
        "-s", colmap_folder,
        "--colmap_executable", COLMAP_EXE,
        "--camera", "OPENCV",
    ]
    print(f"  Running COLMAP pipeline ...")
    t0 = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        elapsed = time.time() - t0
        if result.returncode == 0:
            print(f"  SUCCESS in {elapsed:.1f}s")
        else:
            print(f"  FAILED (code {result.returncode}) in {elapsed:.1f}s")
            print(f"  stderr: {result.stderr[-500:]}")
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT after 1800s")
        continue
    except Exception as e:
        print(f"  ERROR: {e}")
        continue

    # 4. Check registration rate
    sparse_dir = os.path.join(colmap_folder, "sparse", "0")
    if os.path.exists(sparse_dir):
        images_dir = os.path.join(colmap_folder, "images")
        if os.path.exists(images_dir):
            n_reg = len(os.listdir(images_dir))
            n_total = len(crop_files)
            pct = 100.0 * n_reg / n_total if n_total > 0 else 0
            print(f"  Registration: {n_reg}/{n_total} = {pct:.1f}%")
        else:
            print(f"  WARNING: images/ not created")
    else:
        print(f"  WARNING: sparse/0 not created")

print(f"\nDone!")
