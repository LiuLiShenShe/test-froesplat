"""
Remove stray/waste light (废光) from 3DGS point clouds — V2 aggressive version.
Adapted from 05-2DGS-GA/clean_ply_v2.py for standard 3DGS (3 scale dims).

Filters:
  1. Low opacity (<0.01)
  2. Oversized splats (max scale > 0.5)
  3. Dark Gaussians (brightness < 0.10)
  4. Dark + moderately large combo
  5. Dark + desaturated (gray/black waste, not dark-green plants)
  6. Statistical spatial outlier (>3σ)

Saves cleaned PLY as point_cloud_clean.ply alongside original,
AND replaces the original with the cleaned version (backup as _original.ply).
"""
import os
import sys
import gc
import time
import shutil
import numpy as np
from plyfile import PlyData, PlyElement

# ─── V2 Thresholds (more aggressive) ───
OPACITY_MIN = 0.01
BRIGHTNESS_MIN = 0.10
SCALE_MAX = 0.5
OUTLIER_STD_FACTOR = 3.0
DARK_SCALE_BRIGHTNESS = 0.20
DARK_SCALE_THRESHOLD = 0.05
DESAT_BRIGHTNESS = 0.25
DESAT_RANGE = 0.06

C0 = 0.28209479177387814  # SH basis constant


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def clean_point_cloud_3dgs_v2(ply_path, output_path=None, replace_original=True, verbose=True):
    """Clean a 3DGS point cloud with aggressive dark Gaussian removal.
    
    Args:
        ply_path: Path to the original point_cloud.ply
        output_path: Path for cleaned PLY (default: same dir with _clean_v2 suffix)
        replace_original: If True, backup original and replace with cleaned version
        verbose: Print details
    
    Returns:
        (n_original, n_remaining) tuple
    """
    ply = PlyData.read(ply_path)
    v = ply['vertex']
    n_original = len(v)

    if verbose:
        print(f"  Original: {n_original} points")

    # Extract properties
    xyz = np.vstack([v['x'], v['y'], v['z']]).T
    opacity = sigmoid(np.array(v['opacity']))

    # SH DC -> approximate RGB
    r = np.array(v['f_dc_0']) * C0 + 0.5
    g = np.array(v['f_dc_1']) * C0 + 0.5
    b = np.array(v['f_dc_2']) * C0 + 0.5
    brightness = (r + g + b) / 3.0

    # Color range (saturation proxy)
    rgb = np.vstack([r, g, b]).T
    color_range = rgb.max(axis=1) - rgb.min(axis=1)

    # Scale: 3DGS has 3 scales (scale_0, scale_1, scale_2)
    s0 = np.exp(np.array(v['scale_0']))
    s1 = np.exp(np.array(v['scale_1']))
    s2 = np.exp(np.array(v['scale_2']))
    scale_max = np.maximum(np.maximum(s0, s1), s2)

    # ─── Filter 1: Low opacity ───
    mask_opacity = opacity >= OPACITY_MIN
    n_low_opacity = (~mask_opacity).sum()

    # ─── Filter 2: Oversized splats ───
    mask_scale = scale_max <= SCALE_MAX
    n_large_scale = (~mask_scale).sum()

    # ─── Filter 3: Dark Gaussians (brightness < 0.10) ───
    mask_bright = brightness >= BRIGHTNESS_MIN
    n_dark = (~mask_bright).sum()

    # ─── Filter 4: Dark + moderately large ───
    mask_dark_large = ~((brightness < DARK_SCALE_BRIGHTNESS) & (scale_max > DARK_SCALE_THRESHOLD))
    n_dark_large = (~mask_dark_large).sum()

    # ─── Filter 5: Dark + desaturated (gray/black waste, not dark-green plants) ───
    mask_desat = ~((brightness < DESAT_BRIGHTNESS) & (color_range < DESAT_RANGE))
    n_desat = (~mask_desat).sum()

    # ─── Filter 6: Statistical outlier removal (spatial) ───
    median_pos = np.median(xyz, axis=0)
    dist = np.linalg.norm(xyz - median_pos, axis=1)
    dist_mean = dist.mean()
    dist_std = dist.std()
    dist_threshold = dist_mean + OUTLIER_STD_FACTOR * dist_std
    mask_spatial = dist <= dist_threshold
    n_outlier = (~mask_spatial).sum()

    # ─── Combine all filters ───
    mask = mask_opacity & mask_scale & mask_bright & mask_dark_large & mask_desat & mask_spatial
    n_removed = (~mask).sum()
    n_remaining = mask.sum()

    if verbose:
        print(f"  Removed: {n_removed} ({100 * n_removed / n_original:.1f}%)")
        print(f"    Low opacity (<{OPACITY_MIN}): {n_low_opacity}")
        print(f"    Large scale (>{SCALE_MAX}): {n_large_scale}")
        print(f"    Dark (<{BRIGHTNESS_MIN}): {n_dark}")
        print(f"    Dark+large (b<{DARK_SCALE_BRIGHTNESS} & s>{DARK_SCALE_THRESHOLD}): {n_dark_large}")
        print(f"    Dark+desaturated (b<{DESAT_BRIGHTNESS} & range<{DESAT_RANGE}): {n_desat}")
        print(f"    Spatial outlier (>{OUTLIER_STD_FACTOR}σ): {n_outlier}")
        print(f"  Remaining: {n_remaining} ({100 * n_remaining / n_original:.1f}%)")

    # Build cleaned vertex data
    kept_indices = np.where(mask)[0]
    props = v.properties
    dtype_list = [(p.name, v[p.name].dtype) for p in props]

    new_data = np.empty(n_remaining, dtype=dtype_list)
    for p in props:
        new_data[p.name] = v[p.name][kept_indices]

    # Release PlyData file references before any file operations (Windows fix)
    del ply, v
    gc.collect()

    new_vertex = PlyElement.describe(new_data, 'vertex')

    if output_path is None:
        base, ext = os.path.splitext(ply_path)
        output_path = base + "_clean_v2" + ext

    # Ensure absolute paths to avoid copy issues
    output_path = os.path.abspath(output_path)
    ply_path_abs = os.path.abspath(ply_path)

    PlyData([new_vertex]).write(output_path)
    if verbose:
        print(f"  Saved: {output_path}")

    # Optionally replace the original file
    if replace_original:
        backup_path = os.path.splitext(ply_path_abs)[0] + "_original.ply"
        # Small delay to ensure Windows releases file handles
        time.sleep(0.2)
        if not os.path.exists(backup_path):
            # Use os.rename instead of shutil.copy2 to avoid file lock issues on Windows
            os.rename(ply_path_abs, backup_path)
            if verbose:
                print(f"  Backup original -> {backup_path}")
        else:
            os.remove(ply_path_abs)
            if verbose:
                print(f"  Removed original (backup already exists)")
        # Rename clean file to original name
        os.rename(output_path, ply_path_abs)
        if verbose:
            print(f"  Replaced original with cleaned version")

    return n_original, n_remaining


def clean_scene_checkpoint(gs_model_dir, iteration=7000, verbose=True):
    """Clean the point cloud in a vanilla GS checkpoint directory.
    
    Args:
        gs_model_dir: e.g. SuGaR-main/output/vanilla_gs/BaiZhang
        iteration: GS training iteration
        verbose: Print details
    
    Returns:
        (n_original, n_remaining) or None if PLY not found
    """
    ply_path = os.path.join(gs_model_dir, "point_cloud", f"iteration_{iteration}", "point_cloud.ply")
    if not os.path.exists(ply_path):
        if verbose:
            print(f"  PLY not found: {ply_path}")
        return None

    if verbose:
        print(f"  PLY: {ply_path}")

    return clean_point_cloud_3dgs_v2(ply_path, replace_original=True, verbose=verbose)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Clean 3DGS point cloud (V2 aggressive)")
    parser.add_argument("--gs_dir", type=str, required=True,
                        help="Path to vanilla_gs scene dir (e.g. output/vanilla_gs/BaiZhang)")
    parser.add_argument("--iteration", type=int, default=7000)
    parser.add_argument("--no_replace", action="store_true",
                        help="Don't replace original, only save _clean_v2 copy")
    args = parser.parse_args()

    print(f"=== clean_3dgs_ply_v2 ===")
    result = clean_point_cloud_3dgs_v2(
        os.path.join(args.gs_dir, "point_cloud", f"iteration_{args.iteration}", "point_cloud.ply"),
        replace_original=not args.no_replace,
    )
    if result:
        n_orig, n_clean = result
        print(f"\nTotal: {n_orig} -> {n_clean} ({100 * n_clean / n_orig:.1f}%)")


if __name__ == "__main__":
    main()
