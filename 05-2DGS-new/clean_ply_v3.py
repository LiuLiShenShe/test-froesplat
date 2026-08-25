"""
Remove stray/waste light (废光) from point clouds.

Primary mode (2DGS V3):
  Uses opacity/SH/scale-aware filters designed for 2DGS Gaussians.

Fallback mode (generic PLY):
  Uses statistical outlier removal for regular XYZ/RGB point clouds
  (e.g., COLMAP fused.ply or nerfstudio exported point clouds).

Output naming remains: <input>_clean_v3.ply
"""
import os
import sys
import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree

try:
    import open3d as o3d
    HAS_OPEN3D = True
except Exception:
    HAS_OPEN3D = False

# ─── V3 Thresholds ───
# (V2 base thresholds, unchanged)
OPACITY_MIN = 0.01
BRIGHTNESS_MIN = 0.10
SCALE_MAX = 0.5
OUTLIER_STD_FACTOR = 3.0
DARK_SCALE_BRIGHTNESS = 0.20
DARK_SCALE_THRESHOLD = 0.05
DESAT_BRIGHTNESS = 0.25
DESAT_RANGE = 0.06

# V3 new: Scale-dependent brightness floor
# If scale > SCALE_DARK_MULT * median_scale, require brightness >= threshold
SCALE_BRIGHT_TIERS = [
    # (scale_multiplier_of_median, min_brightness)
    (1.0, 0.15),    # Gaussians > 1.0x median scale must have brightness >= 0.15
    (1.5, 0.20),    # Gaussians > 1.5x median scale must have brightness >= 0.20
    (2.5, 0.28),    # Gaussians > 2.5x median scale must have brightness >= 0.28
]

# V3 new: KNN local brightness outlier
KNN_K = 20                    # Number of neighbors to check
KNN_BRIGHTNESS_RATIO = 0.40   # Point brightness < ratio * median(neighbor brightness) = outlier
KNN_MAX_BRIGHTNESS = 0.35     # Only check points darker than this (skip bright points for speed)

C0 = 0.28209479177387814
REQUIRED_2DGS_FIELDS = {"opacity", "f_dc_0", "f_dc_1", "f_dc_2", "scale_0", "scale_1"}


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def clean_point_cloud_v3(ply_path, output_path=None, verbose=True):
    """Clean a 2DGS point cloud — V3 with scale-dependent + KNN filters."""
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

    # Scale (2DGS has scale_0, scale_1)
    s0 = np.exp(np.array(v['scale_0']))
    s1 = np.exp(np.array(v['scale_1']))
    scale_max = np.maximum(s0, s1)

    # ─── Filter 1: Low opacity ───
    mask_opacity = opacity >= OPACITY_MIN
    n_low_opacity = (~mask_opacity).sum()

    # ─── Filter 2: Oversized splats ───
    mask_scale = scale_max <= SCALE_MAX
    n_large_scale = (~mask_scale).sum()

    # ─── Filter 3: Dark Gaussians ───
    mask_bright = brightness >= BRIGHTNESS_MIN
    n_dark = (~mask_bright).sum()

    # ─── Filter 4: Dark + moderately large ───
    mask_dark_large = ~((brightness < DARK_SCALE_BRIGHTNESS) & (scale_max > DARK_SCALE_THRESHOLD))
    n_dark_large = (~mask_dark_large).sum()

    # ─── Filter 5: Dark + desaturated ───
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

    # ─── Filter 7 [V3 NEW]: Scale-dependent brightness floor ───
    # Larger Gaussians must be brighter. Dark large blobs = waste.
    median_scale = np.median(scale_max)
    mask_scale_bright = np.ones(n_original, dtype=bool)
    for scale_mult, min_br in SCALE_BRIGHT_TIERS:
        tier_threshold = median_scale * scale_mult
        tier_mask = ~((scale_max > tier_threshold) & (brightness < min_br))
        mask_scale_bright &= tier_mask
    n_scale_bright = (~mask_scale_bright).sum()

    # ─── Filter 8 [V3 NEW]: KNN local brightness outlier ───
    # If a point is much darker than its local neighbors, it's waste.
    mask_knn = np.ones(n_original, dtype=bool)
    # Only check dark-ish points (brightness < KNN_MAX_BRIGHTNESS) for efficiency
    candidates = np.where(brightness < KNN_MAX_BRIGHTNESS)[0]

    if len(candidates) > 0 and n_original > KNN_K + 1:
        if verbose:
            print(f"    KNN check: {len(candidates)} candidates (brightness < {KNN_MAX_BRIGHTNESS})")
        tree = cKDTree(xyz)
        # Query K+1 neighbors (first is self)
        _, nn_indices = tree.query(xyz[candidates], k=KNN_K + 1)
        # Remove self (first column)
        nn_indices = nn_indices[:, 1:]
        # Median brightness of neighbors
        nn_brightness = brightness[nn_indices]
        nn_median_br = np.median(nn_brightness, axis=1)
        # Mark as outlier if point brightness < ratio * median neighbor brightness
        is_outlier = brightness[candidates] < KNN_BRIGHTNESS_RATIO * nn_median_br
        mask_knn[candidates[is_outlier]] = False
    n_knn = (~mask_knn).sum()

    # ─── Combine all filters ───
    mask = (mask_opacity & mask_scale & mask_bright & mask_dark_large &
            mask_desat & mask_spatial & mask_scale_bright & mask_knn)
    n_removed = (~mask).sum()
    n_remaining = mask.sum()

    if verbose:
        print(f"  Removed: {n_removed} ({100*n_removed/n_original:.1f}%)")
        print(f"    [1] Low opacity (<{OPACITY_MIN}): {n_low_opacity}")
        print(f"    [2] Large scale (>{SCALE_MAX}): {n_large_scale}")
        print(f"    [3] Dark (<{BRIGHTNESS_MIN}): {n_dark}")
        print(f"    [4] Dark+large (b<{DARK_SCALE_BRIGHTNESS} & s>{DARK_SCALE_THRESHOLD}): {n_dark_large}")
        print(f"    [5] Dark+desaturated (b<{DESAT_BRIGHTNESS} & range<{DESAT_RANGE}): {n_desat}")
        print(f"    [6] Spatial outlier (>{OUTLIER_STD_FACTOR}σ): {n_outlier}")
        print(f"    [7] Scale-bright floor (V3): {n_scale_bright}  [median_scale={median_scale:.4f}]")
        print(f"    [8] KNN local dark outlier (V3): {n_knn}")
        print(f"  Remaining: {n_remaining} ({100*n_remaining/n_original:.1f}%)")

    # Build cleaned vertex data
    kept_indices = np.where(mask)[0]
    props = v.properties
    dtype_list = [(p.name, v[p.name].dtype) for p in props]

    new_data = np.empty(n_remaining, dtype=dtype_list)
    for p in props:
        new_data[p.name] = v[p.name][kept_indices]

    new_vertex = PlyElement.describe(new_data, 'vertex')

    if output_path is None:
        base, ext = os.path.splitext(ply_path)
        output_path = base + "_clean_v3" + ext

    PlyData([new_vertex]).write(output_path)
    if verbose:
        print(f"  Saved: {output_path}")

    return n_original, n_remaining


def clean_generic_point_cloud(ply_path, output_path=None, verbose=True, nb_neighbors=30, std_ratio=2.0):
    """Fallback cleaner for standard XYZ/RGB PLY files."""
    if not HAS_OPEN3D:
        raise RuntimeError("open3d is required for generic PLY cleaning but is not available")

    pcd = o3d.io.read_point_cloud(ply_path)
    n_original = len(pcd.points)
    if n_original == 0:
        raise RuntimeError(f"Empty or unreadable point cloud: {ply_path}")

    if verbose:
        print("  Mode: generic statistical outlier removal")
        print(f"  Original: {n_original} points")

    pcd_clean, _ = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors,
        std_ratio=std_ratio,
    )
    n_remaining = len(pcd_clean.points)

    if output_path is None:
        base, ext = os.path.splitext(ply_path)
        output_path = base + "_clean_v3" + ext

    o3d.io.write_point_cloud(output_path, pcd_clean)
    if verbose:
        removed = n_original - n_remaining
        print(f"  Removed: {removed} ({100*removed/max(n_original,1):.1f}%)")
        print(f"  Remaining: {n_remaining} ({100*n_remaining/max(n_original,1):.1f}%)")
        print(f"  Saved: {output_path}")

    return n_original, n_remaining


def clean_any_ply_v3(ply_path, output_path=None, verbose=True):
    """Dispatch to 2DGS V3 cleaner or generic cleaner based on vertex fields."""
    ply = PlyData.read(ply_path)
    vertex = ply["vertex"]
    fields = {p.name for p in vertex.properties}

    if REQUIRED_2DGS_FIELDS.issubset(fields):
        if verbose:
            print("  Mode: 2DGS V3")
        return clean_point_cloud_v3(ply_path, output_path=output_path, verbose=verbose)

    if verbose:
        missing = sorted(REQUIRED_2DGS_FIELDS - fields)
        print(f"  2DGS fields missing: {missing}")
    return clean_generic_point_cloud(ply_path, output_path=output_path, verbose=verbose)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Clean 2DGS point cloud (V3)")
    parser.add_argument("--base_dir", type=str, default=r"D:\CAAS\05-2DGS-new")
    parser.add_argument("--scenes", nargs="+", default=None)
    parser.add_argument("--iterations", nargs="+", type=int, default=[30000])
    parser.add_argument("--ply_paths", nargs="+", default=None, help="Clean explicit PLY paths directly")
    args = parser.parse_args()

    base_dir = args.base_dir

    print(f"=== clean_ply_v3 ===")
    if args.ply_paths:
        print(f"Mode: direct PLY paths")
        print(f"PLY files: {len(args.ply_paths)}")
    else:
        print(f"Mode: base_dir scenes")
        print(f"Base: {base_dir}")
        if args.scenes:
            scenes = args.scenes
        else:
            scenes = []
            for d in sorted(os.listdir(base_dir)):
                pc_dir = os.path.join(base_dir, d, "point_cloud")
                if os.path.isdir(pc_dir):
                    scenes.append(d)
        print(f"Scenes: {len(scenes)}")
        print(f"Iterations: {args.iterations}")
    print(f"V3 filters:")
    print(f"  Scale-bright tiers (2DGS mode): {SCALE_BRIGHT_TIERS}")
    print(f"  KNN (2DGS mode): K={KNN_K}, ratio={KNN_BRIGHTNESS_RATIO}, max_br={KNN_MAX_BRIGHTNESS}")
    print("=" * 60)

    total_orig = 0
    total_clean = 0

    if args.ply_paths:
        for i, ply_path in enumerate(args.ply_paths, start=1):
            print(f"\n[{i}/{len(args.ply_paths)}] {ply_path}")
            if not os.path.exists(ply_path):
                print("  SKIP (not found)")
                continue
            try:
                n_orig, n_clean = clean_any_ply_v3(ply_path)
                total_orig += n_orig
                total_clean += n_clean
            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()
    else:
        for scene in scenes:
            print(f"\n[{scene}]")
            for it in args.iterations:
                ply_path = os.path.join(base_dir, scene, "point_cloud", f"iteration_{it}", "point_cloud.ply")
                if not os.path.exists(ply_path):
                    print(f"  iter {it}: SKIP (not found)")
                    continue

                print(f"  iter {it}:")
                try:
                    n_orig, n_clean = clean_any_ply_v3(ply_path)
                    total_orig += n_orig
                    total_clean += n_clean
                except Exception as e:
                    print(f"  ERROR: {e}")
                    import traceback
                    traceback.print_exc()

    print("\n" + "=" * 60)
    print(f"Total: {total_orig} -> {total_clean} ({100*total_clean/max(total_orig,1):.1f}%)")


if __name__ == "__main__":
    main()
