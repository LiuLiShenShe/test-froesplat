#!/usr/bin/env python3
"""Evaluate E1 (fused, existing) / E2 (raw depth) / E3 (official 100K) with the
same 208 training views and identical render/metrics settings.

Reuses the verified helpers in ``evaluate_gggs_experiments.py``.  E1 is NOT
retrained; its existing checkpoint is evaluated.  Writes per-experiment metrics,
fixed-view renders + montages, and comparison.csv / comparison.md / final_analysis.md.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GGGS_DIR = os.path.join(ROOT, "gggs_community")
for p in (ROOT, GGGS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

_spec = importlib.util.spec_from_file_location("ev4", os.path.join(ROOT, "tools", "evaluate_gggs_experiments.py"))
ev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ev)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plant", default="KongQueZhuYu")
    ap.add_argument("--iters", type=int, default=30_000)
    ap.add_argument("--resolution", type=int, default=2)
    args = ap.parse_args(argv)

    nf = os.path.join(ROOT, "experiments", args.plant, "no_fusion_comparison")
    e1_ds = os.path.join(ROOT, "runs/full_sequence_batch", args.plant, "GGGS_Dataset")
    e1_exp = os.path.join(ROOT, "experiments", args.plant, "E1_GGGS_Baseline")

    EXPERIMENTS = [
        ("E1_fused", "E1 fused (ForeSplat)", e1_ds, e1_exp),
        ("E2_raw_depth", "E2 raw depth (no fusion)", os.path.join(nf, "E2_Omega_RawDepth_NoFusion/dataset"),
         os.path.join(nf, "E2_Omega_RawDepth_NoFusion")),
        ("E3_official_100k", "E3 official 100K", os.path.join(nf, "E3_Omega_OfficialCOLMAP_100K/dataset"),
         os.path.join(nf, "E3_Omega_OfficialCOLMAP_100K")),
    ]

    from utils.general_utils import safe_state
    safe_state(True)
    import lpips
    import skimage.metrics as skm

    lpips_model = lpips.LPIPS(net="alex", verbose=False).cuda().eval()

    # fixed view set from E2 (all share identical cameras)
    from scene.gaussian_model import GaussianModel
    from scene import Scene
    ds_tmp = ev._args_for(EXPERIMENTS[1][2], os.path.join(EXPERIMENTS[1][3], "checkpoints"), args.resolution)
    g_tmp = GaussianModel(3, 0)
    scene_tmp = Scene(ds_tmp, g_tmp, load_iteration=args.iters)
    cams_tmp = scene_tmp.getTrainCameras()
    masks_tmp = os.path.join(EXPERIMENTS[1][2], "masks")
    fixed, crop = ev.select_fixed_views(cams_tmp, masks_tmp)
    del scene_tmp, g_tmp
    import gc, torch
    gc.collect(); torch.cuda.empty_cache()

    summaries, montages, train_rows = {}, {}, {}
    for exp_id, label, dataset_dir, exp_dir in EXPERIMENTS:
        print(f"[eval] {exp_id}", flush=True)
        masks_dir = os.path.join(dataset_dir, "masks")
        s, m = ev.evaluate_experiment(args.plant, exp_id, exp_dir, dataset_dir, masks_dir, args.iters,
                                      args.resolution, lpips_model, skm.structural_similarity,
                                      skm.peak_signal_noise_ratio, fixed, crop)
        s["label"] = label
        summaries[exp_id] = s
        montages[exp_id] = m
        peak = float("nan")
        try:
            with np.load(os.path.join(exp_dir, "checkpoints/final_gaussians/gaussians.npz")) as d:
                peak = float(d["peak_vram_mib"]); secs = float(d["training_seconds"])
                train_rows[exp_id] = {"training_seconds": secs, "peak_vram_mib": peak}
        except Exception:
            train_rows[exp_id] = {}

    # montages: GT | E1 | E2 | E3 for every fixed view
    from PIL import Image
    comp_dir = os.path.join(nf, "comparison_renders")
    os.makedirs(comp_dir, exist_ok=True)
    for label in montages["E1_fused"]:
        panels = [montages["E1_fused"][label][0]]
        for exp_id, *_ in EXPERIMENTS:
            panels.append(montages[exp_id][label][1])
        h = min(p.shape[0] for p in panels); w = min(p.shape[1] for p in panels)
        panels = [p[:h, :w] for p in panels]
        Image.fromarray((np.clip(np.concatenate(panels, axis=1), 0, 1) * 255).round().astype(np.uint8)).save(
            os.path.join(comp_dir, f"{label}_gt_E1_E2_E3.png"))

    # ---- comparison csv/md ------------------------------------------------
    order = ["E1_fused", "E2_raw_depth", "E3_official_100k"]
    cols = ["label", "PSNR", "SSIM", "LPIPS", "FG-PSNR", "FG-SSIM", "Init Gaussians", "Final Gaussians",
            "Training Time (min)", "Peak VRAM (MiB)"]
    init_counts = {"E1_fused": 1252383, "E2_raw_depth": 1252383, "E3_official_100k": 100000}
    rows = []
    for eid in order:
        s = summaries[eid]
        tr = train_rows.get(eid, {})
        rows.append({
            "label": s["label"],
            "PSNR": f"{s['psnr']:.4f}", "SSIM": f"{s['ssim']:.4f}",
            "LPIPS": f"{s['lpips']:.4f}" if s["lpips"] is not None else "nan",
            "FG-PSNR": f"{s['fg_psnr']:.4f}" if s["fg_psnr"] is not None else "nan",
            "FG-SSIM": f"{s['fg_ssim']:.4f}" if s["fg_ssim"] is not None else "nan",
            "Init Gaussians": init_counts[eid],
            "Final Gaussians": s["num_gaussians"],
            "Training Time (min)": f"{tr.get('training_seconds', float('nan')) / 60:.1f}",
            "Peak VRAM (MiB)": f"{tr.get('peak_vram_mib', float('nan')):.0f}",
        })
    with open(os.path.join(nf, "comparison.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerows(rows)

    md = ["# No-fusion initialisation comparison - KongQueZhuYu", "",
          "Metrics are **Training-view Metrics** (all 208 views trained; eval split off).", "",
          "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        md.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    md += ["", "Fixed-view montages: `comparison_renders/<view>_gt_E1_E2_E3.png` "
           "(GT | E1 fused | E2 raw depth | E3 official 100K)."]
    open(os.path.join(nf, "comparison.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
    json.dump({"summaries": summaries, "train": train_rows}, open(os.path.join(nf, "comparison_raw.json"), "w", encoding="utf-8"), indent=2, default=float)
    print(f"[eval] wrote comparison.csv/.md under {nf}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
