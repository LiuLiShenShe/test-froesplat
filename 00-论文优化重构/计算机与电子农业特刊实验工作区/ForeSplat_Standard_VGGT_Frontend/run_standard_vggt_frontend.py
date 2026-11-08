#!/usr/bin/env python3
"""Run the standard VGGT frontend on KongQueZhuYu and export a GGGS-ready COLMAP dataset.

Pipeline (official demo_colmap.py semantics, mask-aware, no fusion):
  1. Read the E1 image + mask manifest (identical 208 views).
  2. Mask-aware square preprocessing (RGB bicubic black-bg, mask nearest, mask pad 0).
  3. Single global 208-frame inference: aggregator + camera_head + depth_head (bf16).
  4. World points = official unproject_depth_map_to_point_map(depth, extrinsic, intrinsic).
  5. Foreground filter: fg & depth>0 & conf>=p20(in-foreground) & finite.
  6. Per-view stratified sampling to N_E1 (seed 0); no point moved/averaged.
  7. COLMAP export with K_original = A^{-1} K_model and point2D = A^{-1} p_model
     (exact affine; avoids the official padded point2D shift bug).
  8. Reprojection verification.

Usage:
    CUDA_VISIBLE_DEVICES=1 python run_standard_vggt_frontend.py [--max-frames N] [--smoke]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "vggt"))

from vggt_foreground.inference import load_model, run_vggt_heads, world_points_from_depth  # noqa: E402
from vggt_foreground.preprocess import (  # noqa: E402
    apply_affine, build_affine, load_and_preprocess_images_square_with_masks,
)
from vggt_foreground.verify import test_affine_roundtrip, test_principal_point, test_reprojection  # noqa: E402

PLANT = "KongQueZhuYu"
OMEGA_ROOT = os.path.join(os.path.dirname(ROOT), "ForeSplat_VGGT_Omega_Frontend")
E1_DATASET = os.path.join(OMEGA_ROOT, "runs/full_sequence_batch", PLANT, "GGGS_Dataset")
E1_INIT_COUNT = 1_252_383


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stratified_allocate(counts: np.ndarray, target: int) -> np.ndarray:
    counts = counts.astype(np.int64)
    total = int(counts.sum())
    if total <= target:
        return counts.copy()
    ideal = target * counts.astype(np.float64) / total
    alloc = np.minimum(np.floor(ideal).astype(np.int64), counts)
    deficit = target - int(alloc.sum())
    if deficit > 0:
        frac = ideal - np.floor(ideal)
        for idx in np.argsort(-frac, kind="stable"):
            if deficit <= 0:
                break
            if alloc[idx] < counts[idx]:
                alloc[idx] += 1
                deficit -= 1
        i = 0
        while deficit > 0 and i < len(alloc):
            room = counts[i] - alloc[i]
            if room > 0:
                add = min(room, deficit)
                alloc[i] += add
                deficit -= add
            i += 1
    return alloc


def read_manifest(dataset_dir: str, max_frames: int | None):
    img_dir = os.path.join(dataset_dir, "images")
    msk_dir = os.path.join(dataset_dir, "masks")
    names = sorted(f for f in os.listdir(img_dir) if f.lower().endswith(".png"))
    if max_frames:
        names = names[:max_frames]
    return names, [os.path.join(img_dir, n) for n in names], \
           [os.path.join(msk_dir, os.path.splitext(n)[0] + ".png") for n in names]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=os.path.join(ROOT, "experiments", PLANT, "standard_vggt_comparison"))
    args = ap.parse_args(argv)

    device = "cuda"
    out_root = os.path.join(args.out, "Standard_VGGT_Frontend")
    dataset_dir = os.path.join(args.out, "Standard_VGGT_Dataset")
    for d in ("depth", "confidence", "colmap", "visualizations", "predictions", "metadata"):
        os.makedirs(os.path.join(out_root, d), exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, "sparse", "0"), exist_ok=True)

    names, img_paths, msk_paths = read_manifest(E1_DATASET, args.max_frames)
    print(f"[frontend] {len(names)} views from E1 dataset {E1_DATASET}", flush=True)

    # ---- coordinate tests (must pass before inference) ---------------------
    rt = test_affine_roundtrip(2160, 3840)
    pp = test_principal_point(2160, 3840)
    assert rt["ok"] and pp["ok"], f"coordinate tests failed: {rt} {pp}"
    print(f"[frontend] coordinate tests OK (roundtrip {rt['max_roundtrip_error']:.1e}, pp {pp['error']:.1e})", flush=True)

    # ---- mask-aware preprocessing -----------------------------------------
    images, masks518, affines518, sizes = load_and_preprocess_images_square_with_masks(
        img_paths, msk_paths, load_res=1024, model_res=518)
    images = images.to(device)
    print(f"[frontend] images {tuple(images.shape)}, masks {tuple(masks518.shape)}", flush=True)

    # ---- single global inference ------------------------------------------
    model = load_model(os.path.join(ROOT, "checkpoints", "model.pt"), device)
    res = run_vggt_heads(model, images, model_res=518, device=device)
    extrinsic, intrinsic = res["extrinsic"], res["intrinsic"]
    depth, conf = res["depth"], res["conf"]                      # depth (N,518,518,1)
    depth_hw = depth[..., 0]                                     # (N,518,518) for filtering/saving
    print(f"[frontend] inference done: extrinsic{extrinsic.shape} intrinsic{intrinsic.shape} depth{depth_hw.shape}", flush=True)

    # ---- world points (official unprojection, at 518) ---------------------
    world = world_points_from_depth(torch.from_numpy(depth).to(device), torch.from_numpy(extrinsic).to(device),
                                    torch.from_numpy(intrinsic).to(device))
    world = np.asarray(world).astype(np.float32)                 # (N,518,518,3)
    torch.cuda.empty_cache()

    # ---- foreground filter -------------------------------------------------
    fg = masks518.cpu().numpy().astype(bool)
    finite = np.isfinite(world).all(axis=-1) & np.isfinite(conf) & np.isfinite(depth_hw) & (depth_hw > 0)
    finite = np.asarray(finite, dtype=bool)
    valid_unfiltered = fg & finite                               # no conf filter (for stats)
    cand_conf = conf[valid_unfiltered]
    p20 = float(np.percentile(cand_conf[np.isfinite(cand_conf)], 20.0))
    valid = valid_unfiltered & np.isfinite(conf) & (conf >= p20)
    n_unfiltered = int(valid_unfiltered.sum())
    n_valid = int(valid.sum())
    print(f"[frontend] conf p20(in-foreground) = {p20:.4f}; valid unfiltered {n_unfiltered} -> after p20 {n_valid}", flush=True)

    # ---- per-view stratified sampling to N_E1 ------------------------------
    per_frame_valid = valid.reshape(len(names), -1).sum(axis=1)
    alloc = stratified_allocate(per_frame_valid, E1_INIT_COUNT)
    rng = np.random.default_rng(0)
    chosen_flat = []
    H, W = 518, 518
    for i in range(len(names)):
        idx = np.flatnonzero(valid[i].reshape(-1))
        k = int(alloc[i])
        if k <= 0:
            continue
        if k < idx.size:
            idx = rng.choice(idx, size=k, replace=False)
        chosen_flat.append(idx + i * (H * W))
    sel = np.sort(np.concatenate(chosen_flat)) if chosen_flat else np.zeros((0,), dtype=np.int64)
    sel_flat = sel
    sel_world = world.reshape(-1, 3)[sel_flat]
    sel_conf = conf.reshape(-1)[sel_flat]
    sel_depth = depth_hw.reshape(-1)[sel_flat]
    frame_of = (sel_flat // (H * W)).astype(np.int64)
    pix = (sel_flat % (H * W)).astype(np.int64)
    px_m, py_m = (pix % W).astype(np.float64), (pix // W).astype(np.float64)
    rgb_model = images.cpu().numpy().transpose(0, 2, 3, 1)       # (N,1024,1024,3) - only need 518 colors
    n_actual = int(sel_world.shape[0])
    print(f"[frontend] sampled {n_actual} init points (target {E1_INIT_COUNT})", flush=True)

    # ---- reprojection verification ----------------------------------------
    # per-frame world points + model pixels (all valid points, capped for speed)
    wpts, mpix, ext_l, intr_l, aff_l, sz_l = [], [], [], [], [], []
    for i in range(len(names)):
        idx = np.flatnonzero(valid[i].reshape(-1))
        if idx.size > 2000:
            idx = rng.choice(idx, size=2000, replace=False)
        wpts.append(world[i].reshape(-1, 3)[idx])
        mpix.append(np.stack([idx % W, idx // W], axis=1))
        ext_l.append(extrinsic[i]); intr_l.append(intrinsic[i])
        aff_l.append(affines518[i].numpy()); sz_l.append(sizes[i])
    rep = test_reprojection(wpts, mpix, ext_l, intr_l, aff_l, np.asarray(sz_l))
    print(f"[frontend] reprojection: mean {rep.get('mean_px', float('nan')):.4f} px, p95 {rep.get('p95_px', float('nan')):.4f} px, ok={rep['ok']}", flush=True)

    # ---- build COLMAP with original-resolution cameras ---------------------
    import pycolmap
    from PIL import Image
    recon = pycolmap.Reconstruction()
    # point2D index within its frame (only sampled points are exported)
    run = np.zeros(len(names), dtype=np.int64)
    order_in_frame = np.zeros(n_actual, dtype=np.int64)
    for k in range(n_actual):
        order_in_frame[k] = run[frame_of[k]]
        run[frame_of[k]] += 1
    for i in range(len(names)):
        A518 = affines518[i].numpy()
        A_inv = np.linalg.inv(A518)
        K_orig = A_inv @ np.asarray(intrinsic[i], dtype=np.float64)
        W0, H0 = int(sizes[i, 0]), int(sizes[i, 1])
        recon.add_camera(pycolmap.Camera(model="PINHOLE", width=W0, height=H0,
                                         params=np.array([K_orig[0, 0], K_orig[1, 1], K_orig[0, 2], K_orig[1, 2]]),
                                         camera_id=i + 1))
        E = np.eye(4); E[:3, :4] = extrinsic[i]
        img = pycolmap.Image(id=i + 1, name=names[i], camera_id=i + 1,
                             cam_from_world=pycolmap.Rigid3d(pycolmap.Rotation3d(E[:3, :3]), E[:3, 3]))
        pts2d = []
        for k in np.flatnonzero(frame_of == i):
            p_orig = apply_affine(A_inv, np.array([[px_m[k], py_m[k]]]))[0]
            pts2d.append(pycolmap.Point2D(p_orig))
        img.points2D = pycolmap.ListPoint2D(pts2d)
        img.registered = True
        recon.add_image(img)
    # colors: sample from the model-resolution (518) RGB by nearest pixel
    rgb518 = F.interpolate(images, size=(518, 518), mode="bilinear", align_corners=False).cpu().numpy().transpose(0, 2, 3, 1)
    rgb518 = (np.clip(rgb518, 0, 1) * 255).round().astype(np.uint8)
    for k in range(n_actual):
        fi = frame_of[k]
        c = rgb518[fi, int(py_m[k]), int(px_m[k])]
        track = pycolmap.Track(); track.add_element(int(fi) + 1, int(order_in_frame[k]))
        recon.add_point3D(sel_world[k], track, c)
    recon.write(os.path.join(dataset_dir, "sparse", "0"))
    try:
        recon.write_text(os.path.join(dataset_dir, "sparse", "0"))
    except Exception as exc:
        print(f"[frontend] write_text failed ({exc})", flush=True)

    # points3D.ply (GGGS init cloud)
    centers = np.stack([-extrinsic[i][:, :3].T @ extrinsic[i][:, 3] for i in range(len(names))])
    cam_sel = centers[frame_of]
    d = cam_sel - sel_world
    nrm = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-8)
    cols = rgb518[frame_of, np.clip(py_m, 0, 517).astype(int), np.clip(px_m, 0, 517).astype(int)]
    write_point_cloud_ply(os.path.join(dataset_dir, "sparse", "0", "points3D.ply"),
                          sel_world, cols, normals=nrm,
                          comments=[f"Standard VGGT-1B raw depth back-projection, no fusion, seed 0, conf p20={p20:.4f}"])

    # images + masks (hard link)
    for sub, src_dir in (("images", os.path.join(E1_DATASET, "images")), ("masks", os.path.join(E1_DATASET, "masks"))):
        dst = os.path.join(dataset_dir, sub)
        os.makedirs(dst, exist_ok=True)
        for n in names:
            src = os.path.join(src_dir, n)
            dstp = os.path.join(dst, n)
            if not os.path.exists(dstp):
                try:
                    os.link(src, dstp)
                except OSError:
                    import shutil; shutil.copy2(src, dstp)

    # ---- save depth / confidence / predictions ---------------------------
    np.save(os.path.join(out_root, "predictions", "world_points_sampled.npy"), sel_world.astype(np.float32))
    np.savez_compressed(os.path.join(out_root, "predictions", "sampled_provenance.npz"),
                        frame=frame_of, px_model=px_m, py_model=py_m, confidence=sel_conf,
                        depth=sel_depth, xyz=sel_world.astype(np.float32))
    for i, n in enumerate(names):
        np.save(os.path.join(out_root, "depth", os.path.splitext(n)[0] + ".npy"), depth_hw[i].astype(np.float32))
        np.save(os.path.join(out_root, "confidence", os.path.splitext(n)[0] + ".npy"), conf[i].astype(np.float32))

    # ---- metadata ---------------------------------------------------------
    meta = {
        "num_frames": len(names), "image_names": names,
        "conf_percentile": 20.0, "conf_threshold_value": p20,
        "total_valid_points_unfiltered": n_unfiltered,
        "total_valid_points_after_p20": n_valid,
        "target_init_points": E1_INIT_COUNT, "points_written": n_actual,
        "per_frame_valid_counts_after_p20": per_frame_valid.tolist(),
        "affine_s": float(518 / 3840),
        "reprojection": rep,
        "coordinate_tests": {"roundtrip": rt, "principal_point": pp},
    }
    json.dump(meta, open(os.path.join(out_root, "metadata.json"), "w"), indent=2)
    json.dump({"frames": [{"image_name": n, "affine518": affines518[i].numpy().tolist(),
                           "original_size": [int(sizes[i, 0]), int(sizes[i, 1])],
                           "rgb_sha256": sha256(img_paths[i]), "mask_sha256": sha256(msk_paths[i])}
                          for i, n in enumerate(names)]},
              open(os.path.join(out_root, "metadata", "input_manifest.json"), "w"), indent=2)
    print(f"[frontend] DONE: wrote {n_actual} init points; reprojection mean {rep.get('mean_px', float('nan')):.4f} px", flush=True)
    return 0


def write_point_cloud_ply(path, xyz, colors, normals=None, comments=None):
    from plyfile import PlyData, PlyElement
    xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    colors = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
    n = len(xyz)
    props = [PlyElement.describe(np.zeros(n, dtype=[("x", "f8"), ("y", "f8"), ("z", "f8")]), "vertex")]
    data = np.zeros(n, dtype=[("x", "f8"), ("y", "f8"), ("z", "f8"),
                              ("nx", "f8"), ("ny", "f8"), ("nz", "f8"),
                              ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    data["x"], data["y"], data["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    if normals is not None:
        nrm = np.asarray(normals).reshape(-1, 3)
        data["nx"], data["ny"], data["nz"] = nrm[:, 0], nrm[:, 1], nrm[:, 2]
    data["red"], data["green"], data["blue"] = colors[:, 0], colors[:, 1], colors[:, 2]
    el = PlyElement.describe(data, "vertex")
    PlyData([el], text=False).write(path)


if __name__ == "__main__":
    raise SystemExit(main())
