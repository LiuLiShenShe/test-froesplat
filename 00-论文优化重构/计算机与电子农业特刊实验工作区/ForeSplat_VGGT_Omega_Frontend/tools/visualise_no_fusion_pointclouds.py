#!/usr/bin/env python3
"""Point-cloud comparison for the no-fusion study (E1 fused / E2 raw depth / E3 100K).

Renders 8 fixed views per cloud with identical camera placement, point size and
colour mapping so E1 vs E2 vs E3 are directly comparable.  This is CPU-only and
offline (matplotlib), GPU training continues in parallel.

Usage::

    python tools/visualise_no_fusion_pointclouds.py
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from plyfile import PlyData

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NF_ROOT = os.path.join(ROOT, "experiments", "KongQueZhuYu", "no_fusion_comparison")
DISPLAY_CAP = 400_000  # per-view display subsample (identical procedure for all clouds)

CLOUDS = [
    ("E1", "E1 fused (ForeSplat)", os.path.join(ROOT, "runs/full_sequence_batch/KongQueZhuYu/GGGS_Dataset/sparse/0/points3D.ply")),
    ("E2", "E2 raw depth (no fusion)", os.path.join(NF_ROOT, "E2_Omega_RawDepth_NoFusion/initial_pointcloud/init_cloud.ply")),
    ("E3", "E3 official 100K", os.path.join(NF_ROOT, "E3_Omega_OfficialCOLMAP_100K/initial_pointcloud/init_cloud.ply")),
]

VIEWS = [
    ("01_front", 0, 0, 1.0),
    ("02_back", 0, 180, 1.0),
    ("03_left", 0, 90, 1.0),
    ("04_right", 0, -90, 1.0),
    ("05_top", 85, 0, 1.0),
]


def load_cloud(path: str):
    ply = PlyData.read(path)
    v = ply["vertex"].data
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)
    rgb = np.stack([v["red"], v["green"], v["blue"]], axis=1).astype(np.uint8)
    return xyz, rgb


def sub(xyz: np.ndarray, rgb: np.ndarray, cap: int = DISPLAY_CAP) -> tuple[np.ndarray, np.ndarray]:
    if xyz.shape[0] <= cap:
        return xyz, rgb
    rng = np.random.default_rng(0)
    idx = rng.choice(xyz.shape[0], size=cap, replace=False)
    return xyz[idx], rgb[idx]


def draw(ax, xyz, rgb, elev, azim, center, radius, title, psize):
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=psize, c=rgb / 255.0, rasterized=True, linewidths=0)
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))
    ax.set_title(title, fontsize=8)
    ax.set_axis_off()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiments-root", default=NF_ROOT)
    args = ap.parse_args(argv)
    nf = args.experiments_root

    clouds = {}
    for key, label, _ in CLOUDS:
        if key == "E1":
            path = os.path.join(ROOT, "runs/full_sequence_batch/KongQueZhuYu/GGGS_Dataset/sparse/0/points3D.ply")
        else:
            path = os.path.join(nf, dict(E2="E2_Omega_RawDepth_NoFusion", E3="E3_Omega_OfficialCOLMAP_100K")[key],
                                "initial_pointcloud", "init_cloud.ply")
        xyz, rgb = load_cloud(path)
        clouds[key] = (xyz, rgb, label)
        print(f"[viz] {key}: {len(xyz)} points loaded", flush=True)

    # robust framing: centre = geometric median-ish (percentile), radius = 98.5th pct of
    # per-point distance from centre, so a handful of far outliers don't shrink the plant.
    allxyz = np.concatenate([clouds[k][0] for k in ("E1", "E2", "E3")])
    center = np.percentile(allxyz, 50, axis=0)
    dist = np.linalg.norm(allxyz - center, axis=1)
    radius = float(np.percentile(dist, 99.5)) * 1.05
    inner = center.copy()

    # zoom regions (identical 3D box across clouds)
    rad = np.linalg.norm(allxyz[:, :2] - center[:2], axis=1)
    r_edge = np.percentile(rad, 90)
    edge_pt = np.array([center[0] + r_edge * 0.85, center[1] + r_edge * 0.35, center[2]])
    edge_r = radius * 0.20
    ov_pt = center + np.array([0.30 * radius, 0.30 * radius, 0.10 * radius])
    ov_r = radius * 0.16
    int_r = radius * 0.20

    orbit_psize = 0.25   # dense clouds -> small dots
    zoom_psize = 2.0     # zoomed-in -> visible dots

    summary = {}
    # ---- orbit views ------------------------------------------------------
    for name, elev, azim, _ in VIEWS:
        fig = plt.figure(figsize=(11, 4.0), dpi=160)
        for col, key in enumerate(("E1", "E2", "E3")):
            xyz, rgb, label = clouds[key]
            xs, rs = sub(xyz, rgb)
            ax = fig.add_subplot(1, 3, col + 1, projection="3d")
            draw(ax, xs, rs, elev, azim, center, radius, f"{label}\n{len(xyz)} pts", orbit_psize)
        fig.suptitle(f"Initialisation point cloud - {name} (fixed camera, colour = point RGB)", fontsize=10)
        out = os.path.join(nf, "initialization_comparison", f"{name}.png")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print(f"[viz] wrote {out}", flush=True)
        summary[name] = {"type": "orbit", "elev": elev, "azim": azim}

    # ---- zoom views -------------------------------------------------------
    zooms = [
        ("06_leaf_edge_zoom", edge_pt, edge_r),
        ("07_adjacent_leaf_overlap", ov_pt, ov_r),
        ("08_plant_interior", inner, int_r),
    ]
    for name, zc, zr in zooms:
        fig = plt.figure(figsize=(11, 4.0), dpi=160)
        for col, key in enumerate(("E1", "E2", "E3")):
            xyz, rgb, label = clouds[key]
            mask = np.all(np.abs(xyz - zc) <= zr, axis=1)
            ax = fig.add_subplot(1, 3, col + 1, projection="3d")
            draw(ax, xyz[mask], rgb[mask], 15, -60, zc, zr, f"{label}\n{int(mask.sum())} pts in box", zoom_psize)
        fig.suptitle(f"Initialisation point cloud - {name} (identical 3D box, fixed camera)", fontsize=10)
        out = os.path.join(nf, "initialization_comparison", f"{name}.png")
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print(f"[viz] wrote {out}", flush=True)
        summary[name] = {"type": "zoom", "center": zc.tolist(), "radius": float(zr)}

    json.dump(summary, open(os.path.join(nf, "initialization_comparison", "viz_config.json"), "w", encoding="utf-8"), indent=2)
    print("[viz] all initialisation comparison figures written", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
