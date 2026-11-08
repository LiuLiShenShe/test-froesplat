#!/usr/bin/env python3
"""Geometric diagnostics for a ForeSplat GGGS export (read-only).

Answers three questions about the exported COLMAP model:

1. Are the per-view focal lengths consistent? (VGGT-Omega predicts FoV per frame)
2. Are the camera baselines sensible? (pairwise distances + viewing angles)
3. Do the single-view depth back-projections agree across views?  For sampled
   points observed by camera ``i`` we project them into camera ``j``, look up
   the nearest 2D observation of camera ``j`` (within 3 px) and compare depths.

Usage::

    python tools/diagnose_reconstruction.py --dataset-dir GGGS_Dataset
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pycolmap
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

INVALID_POINT3D = 2**64 - 1


def _camera_arrays(image):
    R = image.cam_from_world.rotation.matrix()
    t = np.asarray(image.cam_from_world.translation, dtype=np.float64)
    return R, t, -R.T @ t


def diagnostics(sparse_dir: str, sample_per_pair: int = 1500, seed: int = 0) -> dict:
    rec = pycolmap.Reconstruction(sparse_dir)
    cameras = dict(rec.cameras)
    images = dict(rec.images)
    rng = np.random.default_rng(seed)

    # --- flatten the point table once (vectorised lookups below) ------------
    all_pids = np.array(sorted(rec.points3D), dtype=np.int64)
    xyz_all = np.stack([np.asarray(rec.points3D[int(p)].xyz, dtype=np.float64) for p in all_pids])
    pid2row = {int(p): k for k, p in enumerate(all_pids)}

    cam_meta = {}
    obs2d, obs_row = {}, {}
    depth_of = {}
    for image_id, image in images.items():
        camera = cameras[image.camera_id]
        fx, fy, cx, cy = (float(v) for v in np.asarray(camera.params))
        R, t, center = _camera_arrays(image)
        cam_meta[image_id] = {
            "name": image.name,
            "width": int(camera.width),
            "height": int(camera.height),
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
            "center": center.tolist(),
            "forward": (R.T @ np.array([0.0, 0.0, 1.0])).tolist(),
        }

        xy_list, row_list = [], []
        for point2D in image.points2D:
            pid = int(point2D.point3D_id)
            if pid < INVALID_POINT3D and pid in pid2row:
                xy_list.append(np.asarray(point2D.xy, dtype=np.float64))
                row_list.append(pid2row[pid])
        if xy_list:
            obs2d[image_id] = np.asarray(xy_list).reshape(-1, 2)
            obs_row[image_id] = np.asarray(row_list, dtype=np.int64)
            depth_of[image_id] = (xyz_all[obs_row[image_id]] @ R.T + t)[:, 2]
        else:
            obs2d[image_id] = np.zeros((0, 2))
            obs_row[image_id] = np.zeros((0,), dtype=np.int64)
            depth_of[image_id] = np.zeros((0,))

    # --- 1. focal consistency ------------------------------------------------
    fx = np.array([m["fx"] for m in cam_meta.values()])
    fy = np.array([m["fy"] for m in cam_meta.values()])
    focal = {
        "fx_min": float(fx.min()),
        "fx_max": float(fx.max()),
        "fx_max_over_min": float(fx.max() / max(fx.min(), 1e-12)),
        "fx_median": float(np.median(fx)),
        "fy_max_over_min": float(fy.max() / max(fy.min(), 1e-12)),
        "per_camera_fx_over_width": {
            m["name"]: m["fx"] / m["width"] for m in cam_meta.values()
        },
    }

    # --- 2. camera baselines -------------------------------------------------
    ids = sorted(cam_meta)
    centers = np.stack([np.asarray(cam_meta[i]["center"]) for i in ids])
    fwd = np.stack([np.asarray(cam_meta[i]["forward"]) for i in ids])
    dist = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=-1)
    angle = np.degrees(np.arccos(np.clip(fwd @ fwd.T, -1.0, 1.0)))
    off_diag = dist[np.triu_indices(len(ids), k=1)]
    baselines = {
        "camera_centers": {cam_meta[i]["name"]: cam_meta[i]["center"] for i in ids},
        "pairwise_distance_min": float(off_diag.min()) if off_diag.size else None,
        "pairwise_distance_max": float(off_diag.max()) if off_diag.size else None,
        "pairwise_distance_mean": float(off_diag.mean()) if off_diag.size else None,
        "pairwise_angle_deg_max": float(angle[np.triu_indices(len(ids), k=1)].max()) if len(ids) > 1 else None,
        "pairwise_distance_matrix": dist.tolist(),
    }

    # --- 3. cross-view depth agreement ---------------------------------------
    trees = {j: cKDTree(obs2d[j]) for j in ids if obs2d[j].shape[0] > 0}
    pairs = []
    rel_all, inlier5, inlier10 = [], [], []
    evaluated = 0
    for i in ids:
        if obs2d[i].shape[0] == 0:
            continue
        pick = rng.choice(obs2d[i].shape[0], size=min(sample_per_pair, obs2d[i].shape[0]), replace=False)
        X = xyz_all[obs_row[i][pick]]
        for j in ids:
            if j == i or j not in trees:
                continue
            image_j = images[j]
            camera = cameras[image_j.camera_id]
            fx, fy, cx, cy = (float(v) for v in np.asarray(camera.params))
            R_j, t_j, _ = _camera_arrays(image_j)

            cam = X @ R_j.T + t_j
            pix = np.stack(
                [fx * cam[:, 0] / np.maximum(cam[:, 2], 1e-9) + cx,
                 fy * cam[:, 1] / np.maximum(cam[:, 2], 1e-9) + cy],
                axis=1,
            )
            inside = (
                (cam[:, 2] > 1e-6)
                & (pix[:, 0] >= 0) & (pix[:, 0] < camera.width)
                & (pix[:, 1] >= 0) & (pix[:, 1] < camera.height)
            )
            if not inside.any():
                continue
            dist2, idx = trees[j].query(pix[inside], k=1)
            near = dist2 < 3.0
            if not near.any():
                continue
            z_proj = cam[inside, 2][near]
            z_nn = depth_of[j][idx][near]
            rel = np.abs(z_proj - z_nn) / np.maximum(z_proj, 1e-9)
            rel_all.append(rel)
            inlier5.append(float(np.mean(rel < 0.05)))
            inlier10.append(float(np.mean(rel < 0.10)))
            evaluated += int(near.sum())
            pairs.append(
                {
                    "from": cam_meta[i]["name"],
                    "to": cam_meta[j]["name"],
                    "projected_inside": int(inside.sum()),
                    "evaluated": int(near.sum()),
                    "median_rel_depth_error": float(np.median(rel)),
                    "inlier_frac_5pct": inlier5[-1],
                    "inlier_frac_10pct": inlier10[-1],
                }
            )

    all_rel = np.concatenate(rel_all) if rel_all else np.zeros((0,))
    consistency = {
        "evaluated_observations": evaluated,
        "median_rel_depth_error": float(np.median(all_rel)) if all_rel.size else None,
        "p90_rel_depth_error": float(np.percentile(all_rel, 90)) if all_rel.size else None,
        "mean_inlier_frac_5pct": float(np.mean(inlier5)) if inlier5 else None,
        "mean_inlier_frac_10pct": float(np.mean(inlier10)) if inlier10 else None,
        "pairs": pairs,
    }

    return {
        "num_cameras": len(cam_meta),
        "num_points3d": len(all_pids),
        "focal": focal,
        "baselines": baselines,
        "cross_view_consistency": consistency,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="ForeSplat reconstruction diagnostics")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--gggs-repo", default=None, help="optional GGGS clone for readColmapSceneInfo re-check")
    parser.add_argument("--sample-per-pair", type=int, default=1500)
    parser.add_argument("--out", default=None, help="write JSON report here (default: <dataset>/debug/diagnostics.json)")
    args = parser.parse_args(argv)

    sparse_dir = os.path.join(args.dataset_dir, "sparse", "0")
    report = diagnostics(sparse_dir, sample_per_pair=args.sample_per_pair)

    if args.gggs_repo:
        from foresplat_adapter.verify import verify_gggs_dataset

        try:
            verification = verify_gggs_dataset(args.dataset_dir, args.gggs_repo)
            report["gggs_readback"] = {
                "all_checks_passed": verification["all_checks_passed"],
                "num_train_cameras": verification["num_train_cameras"],
                "point_cloud_num_points": verification["point_cloud"]["num_points"],
            }
        except Exception as exc:  # noqa: BLE001
            report["gggs_readback"] = {"error": repr(exc)}

    out_path = args.out or os.path.join(args.dataset_dir, "debug", "diagnostics.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else float(o))

    print(f"== diagnostics for {args.dataset_dir} ==")
    print(f"cameras={report['num_cameras']} points3D={report['num_points3d']}")
    print(
        "focal fx min/med/max = {:.1f}/{:.1f}/{:.1f}  (max/min = {:.3f})".format(
            report["focal"]["fx_min"], report["focal"]["fx_median"], report["focal"]["fx_max"],
            report["focal"]["fx_max_over_min"],
        )
    )
    b = report["baselines"]
    print(
        "baselines: min {:.3f} mean {:.3f} max {:.3f}, max viewing angle {:.1f} deg".format(
            b["pairwise_distance_min"] or 0, b["pairwise_distance_mean"] or 0,
            b["pairwise_distance_max"] or 0, b["pairwise_angle_deg_max"] or 0,
        )
    )
    c = report["cross_view_consistency"]
    print(
        "cross-view depth: evaluated {} obs, median rel err {:.4f}, p90 {:.4f}, inlier@5% {:.3f}, inlier@10% {:.3f}".format(
            c["evaluated_observations"], c["median_rel_depth_error"] or 0, c["p90_rel_depth_error"] or 0,
            c["mean_inlier_frac_5pct"] or 0, c["mean_inlier_frac_10pct"] or 0,
        )
    )
    print(f"full report: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
