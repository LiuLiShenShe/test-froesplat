#!/usr/bin/env python3
"""Evaluate the E5 Standard-VGGT GGGS baseline using the EXACT same metric code path as E1.

Imports evaluate_experiment / select_fixed_views from the VGGT-Omega eval tool so the
PSNR/SSIM/LPIPS/FG-PSNR/FG-SSIM semantics, fixed-view selection and render pipeline are
identical to E1's. Only the dataset / checkpoint paths differ.
"""

import argparse
import gc
import json
import os
import sys

import numpy as np
import torch

OMEGA_ROOT = "/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend"

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(OMEGA_ROOT, "tools"))
sys.path.insert(0, os.path.join(OMEGA_ROOT, "gggs_community"))

from evaluate_gggs_experiments import (  # noqa: E402
    _args_for, _pipe, evaluate_experiment, select_fixed_views,
)
from utils.cuda_preload import preload_cuda_ext_deps  # noqa: E402

preload_cuda_ext_deps()

from scene import GaussianModel, Scene  # noqa: E402
from utils.general_utils import safe_state  # noqa: E402

PLANT = "KongQueZhuYu"
STD_ROOT = os.path.join(ROOT, "experiments", PLANT, "standard_vggt_comparison")
E5_DIR = os.path.join(STD_ROOT, "E5_Standard_VGGT_GGGS_Baseline")
E5_DATASET = os.path.join(STD_ROOT, "Standard_VGGT_Dataset")
E1_DIR = os.path.join(OMEGA_ROOT, "experiments", PLANT, "E1_GGGS_Baseline")
E1_DATASET = os.path.join(OMEGA_ROOT, "runs", "full_sequence_batch", PLANT, "GGGS_Dataset")
EXP_ID = "E5_Standard_VGGT_GGGS_Baseline"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=30_000)
    ap.add_argument("--resolution", type=int, default=2)
    ap.add_argument("--skip-mesh", action="store_true")
    args = ap.parse_args()

    safe_state(True)
    import lpips
    import skimage.metrics as skm

    lpips_model = lpips.LPIPS(net="alex", verbose=False).cuda().eval()

    masks_dir = os.path.join(E5_DATASET, "masks")
    # fixed view set from E5's own cameras (standard-VGGT world frame)
    ds_tmp = _args_for(E5_DATASET, os.path.join(E5_DIR, "checkpoints"), args.resolution)
    g_tmp = GaussianModel(3, 0)
    scene_tmp = Scene(ds_tmp, g_tmp, load_iteration=args.iters)
    cams_tmp = scene_tmp.getTrainCameras()
    fixed, crop = select_fixed_views(cams_tmp, masks_dir)
    del scene_tmp, g_tmp
    gc.collect()
    torch.cuda.empty_cache()

    print(f"[eval] {EXP_ID} on {E5_DATASET}", flush=True)
    summary, _ = evaluate_experiment(PLANT, EXP_ID, E5_DIR, E5_DATASET, masks_dir, args.iters,
                                     args.resolution, lpips_model, skm.structural_similarity,
                                     skm.peak_signal_noise_ratio, fixed, crop)
    json.dump(summary, open(os.path.join(E5_DIR, "metrics", "render_metrics.json"), "w"), indent=2)
    print("[eval] E5 summary:", json.dumps(summary, indent=2), flush=True)

    # unified comparison table against E1 (reads E1's render_metrics.json verbatim)
    e1 = json.load(open(os.path.join(E1_DIR, "metrics", "render_metrics.json")))
    def _train_time(d):
        try:
            with np.load(os.path.join(d, "gaussian", "gaussians.npz")) as z:
                return float(z["training_seconds"])
        except Exception:
            return float("nan")
    t1 = _train_time(E1_DIR); t5 = _train_time(E5_DIR)
    cmp = {
        "metric_scope": "training_views (all 208 views trained on)",
        "E1_VGGT_Omega_GGGS": {"PSNR": e1["psnr"], "SSIM": e1["ssim"], "LPIPS": e1["lpips"],
                                "FG-PSNR": e1["fg_psnr"], "FG-SSIM": e1["fg_ssim"],
                                "Gaussian Count": e1["num_gaussians"],
                                "Training Time (min)": t1 / 60},
        EXP_ID: {"PSNR": summary["psnr"], "SSIM": summary["ssim"], "LPIPS": summary["lpips"],
                  "FG-PSNR": summary["fg_psnr"], "FG-SSIM": summary["fg_ssim"],
                  "Gaussian Count": summary["num_gaussians"],
                  "Training Time (min)": t5 / 60},
    }
    json.dump(cmp, open(os.path.join(STD_ROOT, "comparison.json"), "w"), indent=2)
    print("[eval] wrote comparison.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
