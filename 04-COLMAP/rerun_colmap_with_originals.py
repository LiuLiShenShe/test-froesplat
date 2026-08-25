"""
rerun_colmap_with_originals.py
==========================================================
Phase 1: Re-run COLMAP using original (full) images for failed/warned folders
Phase 2: Filter sparse 3D points using SAM masks (keep only plant points)

Strategy:
- The crop images lost background texture, causing COLMAP to fail on some folders.
- By using original images, COLMAP can find enough feature matches.
- After sparse reconstruction, we use SAM masks to filter out background 3D points.
- The camera poses (from original images) remain valid since crop/orig share identical geometry.

File mapping:
- Original frames:    01-FFmepg/<folder>/XXXX.jpg  (full video-extracted frames)
- FFT-kept frames:    02-FFT/<folder>/XXXX.jpg      (blur-filtered subset, same names)
- SAM mask/crop:      03-SAM/<folder>/mask_XXXX.png  (XXXX matches original frame number)
- COLMAP output:      04-COLMAP/<folder>/             (sparse/0/, images/, etc.)
"""

import struct
import os
import json
import shutil
import subprocess
import time
import sys
import numpy as np
from PIL import Image

# ============ Configuration ============
FFMPEG_DIR  = r"D:\CAAS\01-FFmepg"
FFT_DIR     = r"D:\CAAS\02-FFT"
SAM_DIR     = r"D:\CAAS\03-SAM"
COLMAP_DIR  = r"D:\CAAS\04-COLMAP"
CONVERT_PY  = r"D:\CAAS\05-2d-gaussian-splatting-great-again-dev\convert.py"
PYTHON_EXE  = r"D:\CAAS\05-2d-gaussian-splatting-great-again-dev\.venv_uv\Scripts\python.exe"
COLMAP_EXE  = r"D:\CAAS\04-COLMAP\colmap_bin\bin\colmap.exe"

# Folders to reprocess (FAIL + WARN from first run)
RERUN_FOLDERS = [
    # FAIL (<30% registration with crop images)
    "CaoMei1",
    "ChangShouHua1",
    "ChangShouHua2",
    "DouBanLv2",
    "HongZhang",
    "WanNianQing1",
    "WanNianQing2",
    "WangWenCao2",
    "XianKeLai2",
    # WARN (30-70% registration with crop images)
    "CaoMei2",
    "ChangShouHua3",
    "DouBanLv1",
    # Added after diagnosis
    "XiangPiShu2",
]

# A 3D point is kept if >= this fraction of its observations fall within SAM mask
MASK_THRESHOLD_RATIO = 0.3

# COLMAP timeout per folder (seconds)
COLMAP_TIMEOUT = 3600  # 1 hour

LOG_FILE = os.path.join(COLMAP_DIR, "rerun_log.json")


# ============ COLMAP Binary I/O ============

def read_images_bin(path):
    """Read COLMAP images.bin → dict of {image_id: {qvec, tvec, camera_id, name, points2D}}"""
    images = {}
    with open(path, 'rb') as f:
        num_images = struct.unpack('<Q', f.read(8))[0]
        for _ in range(num_images):
            image_id = struct.unpack('<I', f.read(4))[0]
            qvec = struct.unpack('<4d', f.read(32))
            tvec = struct.unpack('<3d', f.read(24))
            camera_id = struct.unpack('<I', f.read(4))[0]
            name_bytes = b''
            while True:
                c = f.read(1)
                if c == b'\x00':
                    break
                name_bytes += c
            name = name_bytes.decode('utf-8')
            num_points2D = struct.unpack('<Q', f.read(8))[0]
            points2D = []
            for _ in range(num_points2D):
                x, y = struct.unpack('<2d', f.read(16))
                point3D_id = struct.unpack('<q', f.read(8))[0]
                points2D.append([x, y, point3D_id])
            images[image_id] = {
                'qvec': qvec, 'tvec': tvec, 'camera_id': camera_id,
                'name': name, 'points2D': points2D
            }
    return images


def write_images_bin(path, images):
    """Write COLMAP images.bin"""
    with open(path, 'wb') as f:
        f.write(struct.pack('<Q', len(images)))
        for image_id in sorted(images.keys()):
            img = images[image_id]
            f.write(struct.pack('<I', image_id))
            f.write(struct.pack('<4d', *img['qvec']))
            f.write(struct.pack('<3d', *img['tvec']))
            f.write(struct.pack('<I', img['camera_id']))
            f.write(img['name'].encode('utf-8') + b'\x00')
            f.write(struct.pack('<Q', len(img['points2D'])))
            for pt in img['points2D']:
                f.write(struct.pack('<2d', pt[0], pt[1]))
                f.write(struct.pack('<q', pt[2]))


def read_points3D_bin(path):
    """Read COLMAP points3D.bin → dict of {point3D_id: {xyz, rgb, error, track}}"""
    points = {}
    with open(path, 'rb') as f:
        num_points = struct.unpack('<Q', f.read(8))[0]
        for _ in range(num_points):
            point3D_id = struct.unpack('<Q', f.read(8))[0]
            xyz = struct.unpack('<3d', f.read(24))
            rgb = struct.unpack('<3B', f.read(3))
            error = struct.unpack('<d', f.read(8))[0]
            track_length = struct.unpack('<Q', f.read(8))[0]
            track = []
            for _ in range(track_length):
                img_id, pt2d_idx = struct.unpack('<II', f.read(8))
                track.append((img_id, pt2d_idx))
            points[point3D_id] = {
                'xyz': xyz, 'rgb': rgb, 'error': error, 'track': track
            }
    return points


def write_points3D_bin(path, points):
    """Write COLMAP points3D.bin"""
    with open(path, 'wb') as f:
        f.write(struct.pack('<Q', len(points)))
        for point3D_id in sorted(points.keys()):
            pt = points[point3D_id]
            f.write(struct.pack('<Q', point3D_id))
            f.write(struct.pack('<3d', *pt['xyz']))
            f.write(struct.pack('<3B', *pt['rgb']))
            f.write(struct.pack('<d', pt['error']))
            f.write(struct.pack('<Q', len(pt['track'])))
            for img_id, pt2d_idx in pt['track']:
                f.write(struct.pack('<II', img_id, pt2d_idx))


# ============ Mask Filtering ============

def filter_points_with_masks(folder_name, sparse_dir):
    """
    Filter 3D points using SAM masks.
    For each 3D point, check its 2D observations against masks.
    Keep only points where enough observations fall within the plant mask.
    """
    print(f"  [Filter] Loading sparse model from {sparse_dir}")
    images = read_images_bin(os.path.join(sparse_dir, 'images.bin'))
    points = read_points3D_bin(os.path.join(sparse_dir, 'points3D.bin'))
    print(f"  [Filter] {len(images)} images, {len(points)} 3D points")

    # Build mapping: image_name (e.g. '0042.jpg') → mask file path
    # mask files are named mask_XXXX.png where XXXX = frame number
    sam_folder = os.path.join(SAM_DIR, folder_name)
    mask_map = {}
    for img_id, img in images.items():
        name = img['name']  # e.g. '0042.jpg'
        stem = os.path.splitext(name)[0]  # '0042'
        mask_path = os.path.join(sam_folder, f"mask_{stem}.png")
        if os.path.exists(mask_path):
            mask_map[img_id] = mask_path

    print(f"  [Filter] Found masks for {len(mask_map)}/{len(images)} registered images")

    # Cache loaded masks: {mask_path: numpy_array}
    mask_cache = {}

    def get_mask(img_id):
        """Load and cache mask for an image."""
        if img_id not in mask_map:
            return None
        path = mask_map[img_id]
        if path not in mask_cache:
            mask_cache[path] = np.array(Image.open(path).convert('L'))
        return mask_cache[path]

    # Filter each 3D point
    kept_ids = set()
    removed_ids = set()

    for p3d_id, pt in points.items():
        in_mask_count = 0
        total_checked = 0

        for img_id, pt2d_idx in pt['track']:
            if img_id not in images:
                continue
            img = images[img_id]
            if pt2d_idx >= len(img['points2D']):
                continue

            x, y, _ = img['points2D'][pt2d_idx]
            mask = get_mask(img_id)
            if mask is None:
                continue

            total_checked += 1
            px, py = int(round(x)), int(round(y))
            h, w = mask.shape
            if 0 <= px < w and 0 <= py < h and mask[py, px] > 127:
                in_mask_count += 1

        # Keep point if enough observations are within mask
        if total_checked > 0 and (in_mask_count / total_checked) >= MASK_THRESHOLD_RATIO:
            kept_ids.add(p3d_id)
        else:
            removed_ids.add(p3d_id)

    print(f"  [Filter] Kept {len(kept_ids)} points, removed {len(removed_ids)} points")

    # Build filtered points3D
    filtered_points = {pid: points[pid] for pid in kept_ids}

    # Update images.bin: set point3D_id = -1 for removed points
    for img_id, img in images.items():
        for i, (x, y, p3d_id) in enumerate(img['points2D']):
            if p3d_id >= 0 and p3d_id in removed_ids:
                img['points2D'][i][2] = -1

    # Write back
    print(f"  [Filter] Writing filtered sparse model...")
    write_points3D_bin(os.path.join(sparse_dir, 'points3D.bin'), filtered_points)
    write_images_bin(os.path.join(sparse_dir, 'images.bin'), images)

    return len(kept_ids), len(removed_ids)


# ============ Main Processing ============

def clear_old_results(colmap_folder):
    """Remove old COLMAP results to allow clean re-run."""
    for subdir in ['distorted', 'images', 'sparse', 'stereo', 'input']:
        path = os.path.join(colmap_folder, subdir)
        if os.path.exists(path):
            shutil.rmtree(path)
    # Also remove database if exists
    db = os.path.join(colmap_folder, 'database.db')
    if os.path.exists(db):
        os.remove(db)


def get_fft_kept_files(folder_name):
    """Get list of FFT-kept frame filenames for a folder."""
    fft_log = os.path.join(FFT_DIR, folder_name, 'filter_log.json')
    with open(fft_log, 'r') as f:
        data = json.load(f)
    return [p['file'] for p in data['per_frame'] if p['kept']]


def process_folder(folder_name, log_data):
    """Process a single folder: re-run COLMAP with originals + mask filter."""
    colmap_folder = os.path.join(COLMAP_DIR, folder_name)
    input_folder = os.path.join(colmap_folder, "input")

    print(f"\n{'='*60}")
    print(f"Processing: {folder_name}")
    print(f"{'='*60}")

    # Step 1: Get FFT-kept filenames
    kept_files = get_fft_kept_files(folder_name)
    print(f"  FFT-kept frames: {len(kept_files)}")

    # Step 2: Clear old results
    print(f"  Clearing old COLMAP results...")
    clear_old_results(colmap_folder)

    # Step 3: Copy original images (not crops) to input/
    os.makedirs(input_folder, exist_ok=True)
    orig_folder = os.path.join(FFMPEG_DIR, folder_name)
    copied = 0
    for fname in kept_files:
        src = os.path.join(orig_folder, fname)
        dst = os.path.join(input_folder, fname)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            copied += 1
        else:
            print(f"  WARNING: Original frame not found: {src}")
    print(f"  Copied {copied} original images to input/")

    # Step 4: Run convert.py (COLMAP pipeline)
    cmd = [
        PYTHON_EXE, CONVERT_PY,
        "-s", colmap_folder,
        "--colmap_executable", COLMAP_EXE,
        "--camera", "OPENCV",
    ]
    print(f"  Running COLMAP (timeout={COLMAP_TIMEOUT}s)...")
    start_time = time.time()

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=COLMAP_TIMEOUT
        )
        elapsed = time.time() - start_time

        if result.returncode != 0:
            print(f"  COLMAP FAILED (code {result.returncode}) after {elapsed:.1f}s")
            if result.stderr:
                print(f"  STDERR (last 300 chars): {result.stderr[-300:]}")
            log_data[folder_name] = {
                "phase1": "colmap_failed",
                "return_code": result.returncode,
                "time_seconds": round(elapsed, 1),
            }
            return False

        print(f"  COLMAP completed in {elapsed:.1f}s")

    except subprocess.TimeoutExpired:
        print(f"  COLMAP TIMEOUT after {COLMAP_TIMEOUT}s!")
        log_data[folder_name] = {"phase1": "timeout"}
        return False

    # Step 5: Verify COLMAP output
    sparse_dir = os.path.join(colmap_folder, "sparse", "0")
    if not os.path.exists(os.path.join(sparse_dir, "images.bin")):
        print(f"  ERROR: No sparse/0/images.bin found!")
        log_data[folder_name] = {"phase1": "no_sparse_output"}
        return False

    # Read registration stats
    with open(os.path.join(sparse_dir, 'images.bin'), 'rb') as f:
        num_registered = struct.unpack('<Q', f.read(8))[0]
    with open(os.path.join(sparse_dir, 'points3D.bin'), 'rb') as f:
        num_points = struct.unpack('<Q', f.read(8))[0]

    rate = num_registered / copied * 100 if copied > 0 else 0
    print(f"  Registration: {num_registered}/{copied} = {rate:.1f}%")
    print(f"  3D points: {num_points}")

    # Step 6: Filter point cloud with SAM masks
    kept, removed = filter_points_with_masks(folder_name, sparse_dir)

    # Also filter distorted/sparse/0 if it exists
    dist_sparse = os.path.join(colmap_folder, "distorted", "sparse", "0")
    if os.path.exists(os.path.join(dist_sparse, "images.bin")):
        print(f"  [Filter] Also filtering distorted/sparse/0...")
        filter_points_with_masks(folder_name, dist_sparse)

    log_data[folder_name] = {
        "phase1": "success",
        "colmap_time": round(elapsed, 1),
        "input_images": copied,
        "registered": num_registered,
        "registration_rate": round(rate, 1),
        "points3d_before_filter": num_points,
        "points3d_after_filter": kept,
        "points3d_removed": removed,
    }

    print(f"  DONE! Final: {kept} plant points, {removed} background points removed")
    return True


def main():
    print("=" * 60)
    print("COLMAP Re-run with Original Images + SAM Mask Filtering")
    print("=" * 60)
    print(f"Folders to reprocess: {len(RERUN_FOLDERS)}")
    for f in RERUN_FOLDERS:
        print(f"  - {f}")

    log_data = {}
    success_count = 0
    fail_count = 0

    for i, folder_name in enumerate(RERUN_FOLDERS):
        print(f"\n[{i+1}/{len(RERUN_FOLDERS)}]", end="")
        ok = process_folder(folder_name, log_data)
        if ok:
            success_count += 1
        else:
            fail_count += 1

        # Save log after each folder
        with open(LOG_FILE, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)

    # Final summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Success: {success_count}")
    print(f"Failed:  {fail_count}")
    print(f"Log: {LOG_FILE}")

    for name, info in log_data.items():
        status = info.get('phase1', '?')
        if status == 'success':
            print(f"  {name}: {info['registration_rate']}% reg, "
                  f"{info['points3d_after_filter']} plant pts")
        else:
            print(f"  {name}: {status}")


if __name__ == "__main__":
    main()
