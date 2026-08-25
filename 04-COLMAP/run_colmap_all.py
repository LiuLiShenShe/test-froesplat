"""
Batch COLMAP processing script.
For each folder in 03-SAM, prepare input and run convert.py for COLMAP.
"""

import os
import sys
import shutil
import glob
import subprocess
import time
import json

# Paths
SAM_DIR = r"D:\CAAS\03-SAM"
COLMAP_DIR = r"D:\CAAS\04-COLMAP"
CONVERT_PY = r"D:\CAAS\05-2d-gaussian-splatting-great-again-dev\convert.py"
PYTHON_EXE = r"D:\CAAS\05-2d-gaussian-splatting-great-again-dev\.venv_uv\Scripts\python.exe"
COLMAP_EXE = r"D:\CAAS\04-COLMAP\colmap_bin\bin\colmap.exe"

# Get all subdirectories in 03-SAM (skip files)
folders = sorted([
    d for d in os.listdir(SAM_DIR)
    if os.path.isdir(os.path.join(SAM_DIR, d))
])

print(f"Found {len(folders)} folders to process:")
for f in folders:
    print(f"  - {f}")
print()

# Log file
log_file = os.path.join(COLMAP_DIR, "colmap_batch_log.json")
log_data = {}

for i, folder_name in enumerate(folders):
    sam_folder = os.path.join(SAM_DIR, folder_name)
    colmap_folder = os.path.join(COLMAP_DIR, folder_name)
    input_folder = os.path.join(colmap_folder, "input")

    print(f"\n{'='*60}")
    print(f"[{i+1}/{len(folders)}] Processing: {folder_name}")
    print(f"{'='*60}")

    # Step 1: Create output directory and input subfolder
    os.makedirs(input_folder, exist_ok=True)

    # Step 2: Copy crop_*.png files to input folder
    crop_files = sorted(glob.glob(os.path.join(sam_folder, "crop_*.png")))
    if not crop_files:
        print(f"  WARNING: No crop_*.png files found in {sam_folder}, skipping.")
        log_data[folder_name] = {"status": "skipped", "reason": "no crop files"}
        continue

    print(f"  Found {len(crop_files)} crop images.")

    # Copy files (skip if already copied)
    copied = 0
    for src in crop_files:
        dst = os.path.join(input_folder, os.path.basename(src))
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
            copied += 1
    print(f"  Copied {copied} new files to input/ ({len(crop_files) - copied} already existed).")

    # Step 3: Check if COLMAP already processed (sparse/0 exists)
    sparse_dir = os.path.join(colmap_folder, "sparse", "0")
    if os.path.exists(sparse_dir) and os.listdir(sparse_dir):
        print(f"  COLMAP output already exists in sparse/0, skipping.")
        log_data[folder_name] = {"status": "already_done", "images": len(crop_files)}
        continue

    # Step 4: Run convert.py
    cmd = [
        PYTHON_EXE,
        CONVERT_PY,
        "-s", colmap_folder,
        "--colmap_executable", COLMAP_EXE,
        "--camera", "OPENCV",
    ]

    print(f"  Running: {' '.join(cmd)}")
    start_time = time.time()

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1800  # 30 minute timeout per folder
        )
        elapsed = time.time() - start_time

        if result.returncode == 0:
            print(f"  SUCCESS! Completed in {elapsed:.1f}s")
            log_data[folder_name] = {
                "status": "success",
                "images": len(crop_files),
                "time_seconds": round(elapsed, 1)
            }
        else:
            print(f"  FAILED with code {result.returncode} after {elapsed:.1f}s")
            print(f"  STDOUT: {result.stdout[-500:] if result.stdout else 'empty'}")
            print(f"  STDERR: {result.stderr[-500:] if result.stderr else 'empty'}")
            log_data[folder_name] = {
                "status": "failed",
                "return_code": result.returncode,
                "images": len(crop_files),
                "time_seconds": round(elapsed, 1),
                "stderr": result.stderr[-1000:] if result.stderr else ""
            }
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT after 1800s!")
        log_data[folder_name] = {
            "status": "timeout",
            "images": len(crop_files)
        }
    except Exception as e:
        print(f"  ERROR: {e}")
        log_data[folder_name] = {
            "status": "error",
            "message": str(e)
        }

    # Save log after each folder
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)

print(f"\n{'='*60}")
print("All folders processed!")
print(f"Log saved to: {log_file}")

# Summary
success = sum(1 for v in log_data.values() if v.get("status") == "success")
failed = sum(1 for v in log_data.values() if v.get("status") == "failed")
skipped = sum(1 for v in log_data.values() if v.get("status") in ("skipped", "already_done"))
print(f"  Success: {success}, Failed: {failed}, Skipped: {skipped}")
