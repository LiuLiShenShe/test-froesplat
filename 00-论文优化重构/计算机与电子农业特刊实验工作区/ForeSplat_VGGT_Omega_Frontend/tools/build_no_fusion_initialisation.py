#!/usr/bin/env python3
"""Build no-fusion GGGS initialisation datasets from raw VGGT-Omega per-frame depth.

Two targets, both bypassing the ForeSplat multi-view fusion module entirely:

* ``E2_Omega_RawDepth_NoFusion``  - per-view stratified sampling of the raw depth
  back-projection points, total = N_E1 (E1's actual init count), seed 0.
* ``E3_Omega_OfficialCOLMAP_100K`` - deterministic global sampling to ~100,000 points
  (official VGGT demo_colmap.py semantics), seed 0.

Every output 3-D point is the unprojection of one frame's VGGT-Omega depth map
(single-view, single COLMAP observation).  No averaging / voxel merging / moving.
Filters are exactly the E1 export filters (reused from ``visual_util``):
foreground & finite geometry & foreground-percentile-20 confidence & valid depth.

Usage::

    python tools/build_no_fusion_initialisation.py --target E2 --seed 0
    python tools/build_no_fusion_initialisation.py --target E3 --seed 0
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GGGS_DIR = os.path.join(ROOT, "gggs_community")
for p in (ROOT, GGGS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from foresplat_adapter.inference import unproject_depth_map_to_point_map  # noqa: E402
from visual_util import (  # noqa: E402
    _apply_affine,
    _build_valid_mask,
    _resolve_preprocess_transforms,
    get_scene_alignment_transform,
)

N_E1_DEFAULT = 1_252_383  # len(E1 init cloud points3D.ply); re-read from disk if available
E3_TARGET = 100_000


def _load_stack(per_frame_dir: str, prefix: str, stems: list[str]) -> np.ndarray:
    # files are named <prefix><stem>.npy (stem e.g. "crop_0000"); quarantined
    # frames leave gaps in the numeric index, so never assume crop_{i:04d}.
    arrs = [np.load(os.path.join(per_frame_dir, f"{prefix}{stem}.npy")) for stem in stems]
    return np.stack(arrs, axis=0)


def _e1_init_count(dataset_dir: str) -> int:
    from plyfile import PlyData
    ply = os.path.join(dataset_dir, "sparse", "0", "points3D.ply")
    if os.path.isfile(ply):
        return int(len(PlyData.read(ply)["vertex"].data))
    return N_E1_DEFAULT


def _stratified_allocate(counts: np.ndarray, target: int) -> np.ndarray:
    """Largest-remainder allocation of ``target`` across bins with capacities ``counts``."""
    counts = counts.astype(np.int64)
    total = int(counts.sum())
    if total <= target:
        return counts.copy()
    ideal = target * counts.astype(np.float64) / total
    alloc = np.floor(ideal).astype(np.int64)
    alloc = np.minimum(alloc, counts)
    deficit = target - int(alloc.sum())
    if deficit > 0:
        # distribute remaining by largest fractional part among frames with spare capacity
        frac = ideal - np.floor(ideal)
        order = np.argsort(-frac, kind="stable")
        for idx in order:
            if deficit <= 0:
                break
            if alloc[idx] < counts[idx]:
                alloc[idx] += 1
                deficit -= 1
        # second pass in case capacity gaps blocked the first
        i = 0
        while deficit > 0 and i < len(alloc):
            room = counts[i] - alloc[i]
            if room > 0:
                add = min(room, deficit)
                alloc[i] += add
                deficit -= add
            i += 1
    return alloc


def build(target: str, batch_root: str, experiments_root: str, seed: int, resolution: int) -> dict:
    plant = "KongQueZhuYu"
    src_ds = os.path.join(batch_root, plant, "GGGS_Dataset")
    per_frame = os.path.join(src_ds, "per_frame")
    cams = json.load(open(os.path.join(per_frame, "cameras.json"), encoding="utf-8"))
    transforms_raw = json.load(open(os.path.join(src_ds, "metadata", "preprocess_transforms.json"), encoding="utf-8"))
    frames_meta = transforms_raw["frames"]
    cams_frames = cams["frames"]
    num_frames = int(cams["num_frames"])
    H, W = int(cams["model_height"]), int(cams["model_width"])
    # registry and transform records must describe the same 208 views in the same order
    if [f["image_name"] for f in frames_meta] != [f["image_name"] for f in cams_frames]:
        raise ValueError("preprocess_transforms frame order does not match cameras.json frame order")

    exp_id = "E2_Omega_RawDepth_NoFusion" if target == "E2" else "E3_Omega_OfficialCOLMAP_100K"
    exp_dir = os.path.join(experiments_root, exp_id)
    dataset_dir = os.path.join(exp_dir, "dataset")
    init_dir = os.path.join(exp_dir, "initial_pointcloud")
    for d in (dataset_dir, init_dir):
        os.makedirs(d, exist_ok=True)

    # ---- per-frame arrays (indexed by stem; quarantined frames leave gaps) --
    stems = [f["stem"] for f in cams_frames]
    depth = _load_stack(per_frame, "depth_", stems).astype(np.float32)          # (N,H,W)
    conf = _load_stack(per_frame, "confidence_", stems).astype(np.float32)      # (N,H,W)
    mask = _load_stack(per_frame, "model_mask_", stems).astype(bool)            # (N,H,W)
    rgb = _load_stack(per_frame, "model_rgb_", stems).astype(np.uint8)          # (N,H,W,3)

    intrinsic = np.stack([np.asarray(f["intrinsic_model"], dtype=np.float64) for f in cams_frames])
    extrinsic = np.stack([np.asarray(f["extrinsic_model_w2c"], dtype=np.float64) for f in cams_frames])

    predictions = {
        "depth": depth[..., None],
        "depth_conf": conf,
        "extrinsic": extrinsic,
        "intrinsic": intrinsic,
        "images": rgb.astype(np.float32) / 255.0,
    }
    points_3d = unproject_depth_map_to_point_map(depth[..., None], extrinsic, intrinsic)

    # ---- filters identical to the E1 export (percentile-20 over foreground) -
    rng = np.random.default_rng(seed)
    valid, conf_used, conf_thr = _build_valid_mask(
        predictions=predictions,
        points_3d=points_3d,
        depth_conf=conf,
        colors_rgb=rgb,
        num_frames=num_frames,
        height=H,
        width=W,
        foreground_masks=mask,
        conf_thres=20.0,
        min_conf=1e-5,
        mask_black_bg=False,
        mask_white_bg=False,
        filter_depth_edges=False,
        depth_edge_rtol=0.03,
        mask_sky=False,
        target_dir=None,
        rng=rng,
        max_points=0,  # no global limit here; sampling happens per target below
    )

    # ---- scene alignment (must equal the E1 alignment) ---------------------
    T_align = get_scene_alignment_transform(extrinsic)
    T_align_json = np.asarray(cams["scene_alignment_matrix"], dtype=np.float64)
    align_delta = float(np.abs(T_align - T_align_json).max())

    flat_points = points_3d.reshape(-1, 3)
    aligned = flat_points @ T_align[:3, :3].T + T_align[:3, 3]
    flat_rgb = rgb.reshape(-1, 3)
    flat_conf = conf.reshape(-1)
    flat_valid = valid.reshape(-1)

    # per-frame valid counts (unlimited)
    per_frame_valid = valid.reshape(num_frames, -1).sum(axis=1).astype(np.int64)
    total_valid = int(per_frame_valid.sum())

    # ---- sampling ----------------------------------------------------------
    if target == "E2":
        n_target = _e1_init_count(src_ds)
        alloc = _stratified_allocate(per_frame_valid, n_target)
        chosen = []
        rng_sel = np.random.default_rng(seed)
        valid_idx_by_frame = valid.reshape(num_frames, -1)
        for i in range(num_frames):
            idx = np.flatnonzero(valid_idx_by_frame[i])
            k = int(alloc[i])
            if k <= 0:
                continue
            if k < idx.size:
                idx = rng_sel.choice(idx, size=k, replace=False)
            chosen.append(idx + i * (H * W))  # flat index into (N*H*W)
        sel = np.sort(np.concatenate(chosen)) if chosen else np.zeros((0,), dtype=np.int64)
        sampling = "per_view_stratified_proportional_to_valid_fg"
    else:
        n_target = E3_TARGET
        all_idx = np.flatnonzero(flat_valid)
        rng_sel = np.random.default_rng(seed)
        if all_idx.size > n_target:
            sel = np.sort(rng_sel.choice(all_idx, size=n_target, replace=False))
        else:
            sel = all_idx
        sampling = "global_deterministic_random"

    sel = np.asarray(sel, dtype=np.int64)
    xyz_sel = aligned[sel]
    rgb_sel = flat_rgb[sel]
    conf_sel = flat_conf[sel]
    frame_of_sel = (sel // (H * W)).astype(np.int32)
    pix_of_sel = (sel % (H * W)).astype(np.int64)
    px_sel = (pix_of_sel % W).astype(np.int32)
    py_sel = (pix_of_sel // W).astype(np.int32)

    n_actual = int(xyz_sel.shape[0])

    # ---- build a fresh, self-consistent COLMAP model (single-obs tracks) ---
    import pycolmap

    transforms = _resolve_preprocess_transforms(
        [{"image_name": f["image_name"], "pixel_transform": f["pixel_transform"],
          "original_width": f["original_width"], "original_height": f["original_height"]}
         for f in frames_meta],
        [f["image_name"] for f in frames_meta], num_frames, W, H,
    )
    A_inv = np.stack([np.linalg.inv(np.asarray(t["pixel_transform"], dtype=np.float64).reshape(3, 3)) for t in transforms])
    original_sizes = [(int(t["original_width"]), int(t["original_height"])) for t in transforms]
    T_align_inv = np.linalg.inv(T_align)

    recon = pycolmap.Reconstruction()
    # cameras (identical formula to save_colmap) + images with only the selected observations
    sel_frame_counts = np.bincount(frame_of_sel, minlength=num_frames)
    # map flat sel -> order within its frame (point2D index)
    order_in_frame = np.zeros_like(pix_of_sel)
    run = np.zeros(num_frames, dtype=np.int64)
    for k in range(len(sel)):
        fi = frame_of_sel[k]
        order_in_frame[k] = run[fi]
        run[fi] += 1

    for fidx in range(num_frames):
        K_model = intrinsic[fidx]
        K_original = A_inv[fidx] @ K_model
        W0, H0 = original_sizes[fidx]
        recon.add_camera(pycolmap.Camera(
            model="PINHOLE", width=W0, height=H0,
            params=np.array([K_original[0, 0], K_original[1, 1], K_original[0, 2], K_original[1, 2]]),
            camera_id=fidx + 1,
        ))
        E_model = np.eye(4)
        E_model[:3, :4] = extrinsic[fidx]
        E_aligned = E_model @ T_align_inv
        image = pycolmap.Image(
            id=fidx + 1, name=os.path.basename(cams_frames[fidx]["image_name"]), camera_id=fidx + 1,
            cam_from_world=pycolmap.Rigid3d(pycolmap.Rotation3d(E_aligned[:3, :3]), E_aligned[:3, 3]),
        )
        pts2d = []
        for k in np.flatnonzero(frame_of_sel == fidx):
            xy_model = np.array([px_sel[k], py_sel[k]], dtype=np.float64)
            xy_orig = _apply_affine(A_inv[fidx], xy_model[None, :])[0]
            pts2d.append(pycolmap.Point2D(xy_orig))
        image.points2D = pycolmap.ListPoint2D(pts2d)
        image.registered = True
        recon.add_image(image)

    for k in range(n_actual):
        track = pycolmap.Track()
        track.add_element(int(frame_of_sel[k]) + 1, int(order_in_frame[k]))
        recon.add_point3D(xyz_sel[k], track, rgb_sel[k])

    sparse_dir = os.path.join(dataset_dir, "sparse", "0")
    os.makedirs(sparse_dir, exist_ok=True)
    recon.write(sparse_dir)
    try:
        recon.write_text(sparse_dir)
    except Exception as exc:  # pragma: no cover
        print(f"[build] write_text failed ({exc}); .txt skipped", flush=True)

    # ---- images + masks: identical 208 views (hard link, fall back to copy) --
    img_dst = os.path.join(dataset_dir, "images")
    msk_dst = os.path.join(dataset_dir, "masks")
    os.makedirs(img_dst, exist_ok=True)
    os.makedirs(msk_dst, exist_ok=True)

    def _link(src, dst):
        if os.path.exists(dst):
            return
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)

    for f in frames_meta:
        name = f["image_name"]
        _link(os.path.join(src_ds, "images", name), os.path.join(img_dst, name))
        _link(os.path.join(src_ds, "masks", name), os.path.join(msk_dst, name))

    # ---- points3D.ply (GGGS init cloud) with camera-facing normals ----------
    from foresplat_adapter.ply import write_point_cloud_ply
    cam_centers = np.stack([-extrinsic[i][:, :3].T @ extrinsic[i][:, 3] for i in range(num_frames)])
    centers_sel = cam_centers[frame_of_sel]
    direction = centers_sel - xyz_sel
    normals = direction / np.maximum(np.linalg.norm(direction, axis=1, keepdims=True), 1e-8)
    write_point_cloud_ply(
        os.path.join(sparse_dir, "points3D.ply"),
        xyz_sel, rgb_sel, normals=normals,
        comments=[f"VGGT-Omega raw per-frame depth back-projection (NO fusion), {sampling}, seed {seed}",
                  "Each point is a single-view depth unprojection; not SfM triangulation"],
    )

    # ---- initial_pointcloud/ (provenance + unlimited stats) ----------------
    write_point_cloud_ply(os.path.join(init_dir, "init_cloud.ply"), xyz_sel, rgb_sel, normals=normals,
                          comments=["E2/E3 initialisation cloud (selected raw depth points)"])
    np.savez_compressed(
        os.path.join(init_dir, "provenance.npz"),
        frame_id=frame_of_sel, px=px_sel, py=py_sel, confidence=conf_sel,
        xyz=xyz_sel.astype(np.float32), rgb=rgb_sel.astype(np.uint8),
    )
    stats = {
        "experiment": exp_id,
        "target": target,
        "sampling": sampling,
        "seed": seed,
        "num_frames": num_frames,
        "model_hw": [H, W],
        "original_sizes_sample": original_sizes[:2],
        "conf_percentile": 20.0,
        "conf_threshold_value": float(conf_thr),
        "filter_depth_edges": False,
        "total_valid_points_unlimited": total_valid,
        "per_frame_valid_counts_unlimited": per_frame_valid.tolist(),
        "target_points": int(n_target),
        "points_written": n_actual,
        "points_short_fallback": bool(n_actual < n_target),
        "alignment_max_abs_delta_vs_e1": align_delta,
        "xyz_range_min": xyz_sel.min(axis=0).tolist() if n_actual else None,
        "xyz_range_max": xyz_sel.max(axis=0).tolist() if n_actual else None,
    }
    json.dump(stats, open(os.path.join(init_dir, "init_stats.json"), "w", encoding="utf-8"), indent=2)
    json.dump(stats, open(os.path.join(exp_dir, "init_stats.json"), "w", encoding="utf-8"), indent=2)

    print(f"[build] {exp_id}: wrote {n_actual} init points "
          f"(target {n_target}, valid-unlimited {total_valid}, conf_thr {conf_thr:.4f}, "
          f"align_delta {align_delta:.2e}) -> {dataset_dir}", flush=True)
    return stats


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=["E2", "E3"], required=True)
    ap.add_argument("--plant", default="KongQueZhuYu")
    ap.add_argument("--batch-root", default=os.path.join(ROOT, "runs/full_sequence_batch"))
    ap.add_argument("--experiments-root", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resolution", type=int, default=2)
    args = ap.parse_args(argv)
    args.batch_root = os.path.abspath(args.batch_root)
    experiments_root = args.experiments_root or os.path.join(
        ROOT, "experiments", args.plant, "no_fusion_comparison")
    build(args.target, args.batch_root, experiments_root, args.seed, args.resolution)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
