#!/usr/bin/env python3
"""Render the shared comparison views (E1/E5 views) from the trained E6 2DGS model.

Must run in the 2DGS venv (torch 2.5.1 + diff_surfel_rasterization). Saves full-frame
renders (1080x1920) for the 8 triptych views and 4 leaf-zoom views as PNGs under
render_comparison/e6_renders/, named crop_<i04d>.png, so render_comparison.py can
compose the quad-panel GT | E1 | E5 | E6.

Run:
  CUDA_VISIBLE_DEVICES=1 /data/fj/F2DMAS/2d-gaussian-splatting-main/venv/bin/python render_e6_views.py
"""

import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image

TWO_DGS_ROOT = "/data/fj/F2DMAS/2d-gaussian-splatting-main"
sys.path.insert(0, TWO_DGS_ROOT)

from arguments import ModelParams  # noqa: E402
from gaussian_renderer import GaussianModel, render  # noqa: E402
from scene import Scene  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
STD = os.path.join(ROOT, "experiments", "KongQueZhuYu", "standard_vggt_comparison")
E6_DATASET = os.path.join(STD, "Standard_VGGT_Dataset")
E6_CKPT = os.path.join(STD, "E6_2DGS_Standard_VGGT", "checkpoints")
OUT = os.path.join(STD, "render_comparison", "e6_renders")

VIEWS = list(range(0, 208, 26))
LEAVES = [26, 78, 130, 156]
TARGETS = sorted(set(VIEWS) | set(LEAVES))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=30_000)
    ap.add_argument("--resolution", type=int, default=2)
    args = ap.parse_args()

    ns = argparse.Namespace()
    ns.sh_degree = 3
    ns.source_path = E6_DATASET
    ns.model_path = E6_CKPT
    ns.images = "images"
    ns.dataset = ""
    ns.resolution = args.resolution
    ns.white_background = False
    ns.data_device = "cuda"
    ns.eval = False
    ns.disable_filter3D = False
    ns.kernel_size = 0.0

    pipe = argparse.Namespace()
    pipe.convert_SHs_python = False
    pipe.compute_cov3D_python = False
    pipe.debug = False
    pipe.depth_ratio = 0.0

    gaussians = GaussianModel(ns.sh_degree)
    scene = Scene(ns, gaussians, load_iteration=args.iters, shuffle=False)
    background = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device="cuda")
    cameras = scene.getTrainCameras()

    os.makedirs(OUT, exist_ok=True)
    name_to_cam = {os.path.splitext(cam.image_name)[0]: cam for cam in cameras}
    for i in TARGETS:
        key = f"crop_{i:04d}"
        cam = name_to_cam.get(key)
        if cam is None:
            print(f"[e6r] MISSING view {key}", flush=True)
            continue
        with torch.no_grad():
            pkg = render(cam, gaussians, pipe, background)
            image = torch.clamp(pkg["render"], 0.0, 1.0)
        arr = (image.detach().cpu().numpy().transpose(1, 2, 0) * 255).round().astype(np.uint8)
        Image.fromarray(arr).save(os.path.join(OUT, f"crop_{i:04d}.png"))
        print(f"[e6r] saved {key} {arr.shape[1]}x{arr.shape[0]}", flush=True)
    print("[e6r] done ->", OUT, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
