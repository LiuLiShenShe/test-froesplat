#!/usr/bin/env python3
"""Unified render comparison: GT | VGGT-Omega+GGGS (E1) | Standard-VGGT+GGGS (E5) | 2DGS (E6).

Eight named views shared by both experiments (same image files on both sides, so the
view indices match one-to-one), rendered from each experiment's own trained Gaussians
and camera. E6 2DGS lives in a different venv (diff_surfel_rasterization), so its
renders are pre-produced by render_e6_views.py into render_comparison/e6_renders/
and composed here. Plus four leaf-detail zoom crops. Plus foreground sharpness
analysis (gradient magnitude / Laplacian variance in the GT-mask region).

Writes under experiments/KongQueZhuYu/standard_vggt_comparison/render_comparison/:
  triptych_<view08d>.png        GT | Omega+GGGS | Std+GGGS | 2DGS
  leaf_zoom_<leaf>_<view08d>.png
  sharpness.csv                 per-view foreground gradient stats
"""

import argparse
import csv
import gc
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

OMEGA = "/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend"
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(OMEGA, "tools"))
sys.path.insert(0, os.path.join(OMEGA, "gggs_community"))

from evaluate_gggs_experiments import _args_for, _pipe, render_view, to_numpy  # noqa: E402
from utils.cuda_preload import preload_cuda_ext_deps  # noqa: E402

preload_cuda_ext_deps()
import skimage.color as skc
from scene import GaussianModel, Scene  # noqa: E402
from utils.general_utils import safe_state  # noqa: E402

PLANT = "KongQueZhuYu"
STD = os.path.join(ROOT, "experiments", PLANT, "standard_vggt_comparison")
E5_DATASET = os.path.join(STD, "Standard_VGGT_Dataset")
E5_CKPT = os.path.join(STD, "E5_Standard_VGGT_GGGS_Baseline", "checkpoints")
E1_DATASET = os.path.join(OMEGA, "runs", "full_sequence_batch", PLANT, "GGGS_Dataset")
E1_CKPT = os.path.join(OMEGA, "experiments", PLANT, "E1_GGGS_Baseline", "checkpoints")
OUT = os.path.join(STD, "render_comparison")
E6_RENDERS = os.path.join(OUT, "e6_renders")   # pre-rendered by render_e6_views.py (2DGS venv)
os.makedirs(OUT, exist_ok=True)

# 8 views sharing the same image files on both sides (step 26 over 208).
VIEWS = list(range(0, 208, 26))          # 0,26,52,78,104,130,156,182
LEAVES = [26, 78, 130, 156]              # 4 leaf-detail zoom views


def load(ckpt_dir, dataset_dir, iters):
    ds = _args_for(dataset_dir, ckpt_dir, 2)
    g = GaussianModel(3, 0)
    sc = Scene(ds, g, load_iteration=iters)
    return ds, g, sc, _pipe()


def shared_index_to_cam(cams, i):
    target = f"crop_{i:04d}"
    for cam in cams:
        if os.path.splitext(cam.image_name)[0] == target:
            return cam
    raise KeyError(target)


def fg_mask(cam):
    name = os.path.splitext(cam.image_name)[0] + ".png"
    path = os.path.join(E5_DATASET, "masks", name)
    with Image.open(path) as im:
        m = np.array(im.convert("L").resize((cam.image_width, cam.image_height), Image.NEAREST)) > 127
    return m


def gridded(panels, zoom=(0, 0, None, None)):
    """Horizontal grid; panels = list of HxWx3 float arrays. Optional crop (x0,y0,x1,y1)."""
    x0, y0, x1, y1 = zoom
    if x1 is None:
        x1, y1 = panels[0].shape[1], panels[0].shape[0]
    crops = [p[y0:y1, x0:x1] for p in panels]
    h, w = crops[0].shape[0], crops[0].shape[1]
    canvas = np.zeros((h, w * len(crops), 3), np.uint8)
    for k, c in enumerate(crops):
        canvas[:, k * w:(k + 1) * w] = (c * 255)
    return canvas


def load_e6_render(i):
    """E6 2DGS render from pre-generated PNG (2DGS venv run). Returns float HxWx3 or None."""
    p = os.path.join(E6_RENDERS, f"crop_{i:04d}.png")
    if not os.path.exists(p):
        return None
    return np.asarray(Image.open(p).convert("RGB"), np.uint8) / 255.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=30_000)
    args = ap.parse_args()

    safe_state(True)
    ds1, g1, s1, pipe1 = load(E1_CKPT, E1_DATASET, args.iters)
    ds5, g5, s5, pipe5 = load(E5_CKPT, E5_DATASET, args.iters)
    cams1, cams5 = s1.getTrainCameras(), s5.getTrainCameras()
    bg = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device="cuda")

    rows = []
    for i in VIEWS:
        cam1 = shared_index_to_cam(cams1, i)
        cam5 = shared_index_to_cam(cams5, i)
        img1 = to_numpy(torch.clamp(render_view(cam1, g1, pipe1, bg, ds1.kernel_size, False)[0], 0, 1))
        img5 = to_numpy(torch.clamp(render_view(cam5, g5, pipe5, bg, ds5.kernel_size, False)[0], 0, 1))
        img6 = load_e6_render(i)
        gt = to_numpy(torch.clamp(cam5.original_image, 0, 1))
        fg = fg_mask(cam5)
        # foreground sharpness: gradient magnitude on grayscale render
        def fg_grad(x, name):
            gray = skc.rgb2gray(x)
            gx = np.gradient(gray, axis=1); gy = np.gradient(gray, axis=0)
            gm = np.hypot(gx, gy)[fg]
            lv = np.var(gx[fg])  # Laplacian-ish proxy: gradient variance
            return float(gm.mean()), float(lv)
        row = dict(view=i, **{f"e1_{k}": v for k, v in zip(("grad", "gradvar"), fg_grad(img1, "e1"))},
                   **{f"e5_{k}": v for k, v in zip(("grad", "gradvar"), fg_grad(img5, "e5"))},
                   **{f"gt_{k}": v for k, v in zip(("grad", "gradvar"), fg_grad(gt, "gt"))})
        panels = [gt, img1, img5]
        if img6 is not None:
            e6_grad, e6_gv = fg_grad(img6, "e6")
            row["e6_grad"] = e6_grad
            row["e6_gradvar"] = e6_gv
            panels.append(img6)
        rows.append(row)
        Image.fromarray(gridded(panels)).save(os.path.join(OUT, f"triptych_{i:08d}.png"))
        gts = " | ".join(f"{p:>4}" for p in ["GT", "E1", "E5"] + (["E6"] if img6 is not None else []))
        if i in LEAVES:
            fg2 = fg
            ys, xs = np.nonzero(fg2)
            cx, cy = int(xs.mean()), int(ys.mean())
            half = 320
            x0, y0 = max(cx - half, 0), max(cy - half, 0)
            x1, y1 = min(cx + half, gt.shape[1]), min(cy + half, gt.shape[0])
            Image.fromarray(gridded(panels, (x0, y0, x1, y1))).save(
                os.path.join(OUT, f"leaf_zoom_v{i:08d}.png"))
        print(f"[rc] view {i}: GT grad {rows[-1]['gt_grad']:.4f} | E1 {rows[-1]['e1_grad']:.4f} | E5 {rows[-1]['e5_grad']:.4f}"
              + (f" | E6 {rows[-1]['e6_grad']:.4f}" if 'e6_grad' in rows[-1] else ""), flush=True)

    with open(os.path.join(OUT, "sharpness.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    json.dump(rows, open(os.path.join(OUT, "sharpness.json"), "w"), indent=2)
    print("render_comparison written to", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())