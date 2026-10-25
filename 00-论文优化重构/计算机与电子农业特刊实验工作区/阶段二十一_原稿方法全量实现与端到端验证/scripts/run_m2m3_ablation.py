#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 21 — matched M0–M4 ablation for the foreground-object 2DGS + M2M3-GO.

All variants share an identical foreground-object core (manuscript §2.4–§2.5) and
differ only in the primitive-level refinement layer:

  M0  foreground-object 2DGS only                        (no M2M3-GO, no pruning)
  M1  M0 + mask-guided pruning (legacy topology capacity) — the pre-Phase21 proxy
  M2  M0 + M2M3-GO bidirectional scores -> utility prune (no Floor40, no densify)
  M3  M2 + Floor40 local capacity floor
  M4  M3 + deficit-driven densification                  (complete M2M3-GO)

Every variant is launched through run_pipeline.py, so each run gets a manifest
with the git commit, exact command, wall time and the SHA256 of the masks that
training consumed.  Evaluation uses the foreground metrics on the *same* masks.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path("/data/fj/F2DMAS")
PHASE = ROOT / "00-论文优化重构/计算机与电子农业特刊实验工作区/阶段二十一_原稿方法全量实现与端到端验证"
RUNNER = PHASE / "scripts/run_pipeline.py"
DATA = Path("/data/fj/F2DMAS_phase20e0_data/f53bf8f3_v1")

FOREGROUND_CORE = [
    "--images", "images_rgba",
    "--mask_mode", "alpha",
    "--mask_pattern", "mask_{stem}.png",
    "--init_pcd_mode", "foreground",
    "--init_pcd_min_observations", "3",
    "--init_pcd_dilate_mask_px", "2",
    "--use_foreground_rgb_loss",
    "--foreground_bg_rgb_weight", "0.0",
    "--use_mask_loss",
    "--mask_loss_type", "l1_dice",
    "--lambda_mask", "0.08",
    "--mask_ignore_boundary_px", "2",
    "--mask_loss_start_iter", "500",
    "--mask_loss_warmup_iters", "1500",
    "--use_bg_opacity_loss",
    "--lambda_bg", "0.02",
    "--view_weight_mode", "soft",
    "--view_weight_source", "computed",
]

VARIANTS = {
    "C0": {
        "label": "full-scene 2DGS control (no foreground-object components)",
        "args": ["--images", "images_rgba"],
        "skip_core": True,
    },
    "M0": {"label": "foreground-object 2DGS", "args": []},
    "M1": {
        "label": "M0 + mask-guided pruning (legacy topology capacity)",
        "args": ["--pruning_mode", "topology",
                 "--pruning_start_iter", "15000",
                 "--pruning_interval", "1000",
                 "--capacity_control_mode", "m2m3",
                 "--capacity_floor_ratio", "0.40",
                 "--capacity_floor_reference", "max_seen"],
    },
    "M2": {
        "label": "M0 + M2M3-GO utility pruning (no Floor40, no densify)",
        "args": ["--m2m3_go_mode", "prune",
                 "--m2m3_go_start_iter", "15000",
                 "--m2m3_go_interval", "1000",
                 "--m2m3_go_stop_iter", "24000",
                 "--m2m3_go_local_floor_ratio", "0.0",
                 "--m2m3_go_tau_mode", "quantile",
                 "--m2m3_go_tau_quantile", "0.01"],
    },
    "M3": {
        "label": "M2 + Floor40 local capacity floor",
        "args": ["--m2m3_go_mode", "prune",
                 "--m2m3_go_start_iter", "15000",
                 "--m2m3_go_interval", "1000",
                 "--m2m3_go_stop_iter", "24000",
                 "--m2m3_go_local_floor_ratio", "0.40",
                 "--m2m3_go_local_blocks", "6",
                 "--m2m3_go_tau_mode", "quantile",
                 "--m2m3_go_tau_quantile", "0.01"],
    },
    "M4": {
        "label": "M3 + deficit-driven densification (complete M2M3-GO)",
        "args": ["--m2m3_go_mode", "full",
                 "--m2m3_go_start_iter", "15000",
                 "--m2m3_go_interval", "1000",
                 "--m2m3_go_stop_iter", "24000",
                 "--m2m3_go_local_floor_ratio", "0.40",
                 "--m2m3_go_local_blocks", "6",
                 "--m2m3_go_densify_budget", "512",
                 "--m2m3_go_tau_mode", "quantile",
                 "--m2m3_go_tau_quantile", "0.01"],
    },
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", required=True)
    ap.add_argument("--group", required=True, help="E0 input group, e.g. C_R1")
    ap.add_argument("--variants", default="M0,M1,M2,M3,M4")
    ap.add_argument("--iterations", type=int, default=30000)
    ap.add_argument("--resolution", type=int, default=2)
    ap.add_argument("--run-name", default="m2m3_ablation")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing run manifest (e.g. after archiving "
                         "an interrupted run); the old manifest must be archived first")
    ap.add_argument("--port-base", type=int, default=6030,
                    help="network_gui port for the first variant; parallel drivers "
                         "on one host must use different bases")
    args = ap.parse_args()

    source = DATA / args.sample / args.group
    if not source.exists():
        raise SystemExit(f"missing frozen E0 dataset: {source}")
    mask_dir = source / "masks"
    if not mask_dir.exists():
        raise SystemExit(f"missing masks: {mask_dir}")
    cfg_dir = PHASE / "configs" / args.run_name
    cfg_dir.mkdir(parents=True, exist_ok=True)
    out_root = PHASE / "runs" / args.run_name

    for index, variant in enumerate([item for item in args.variants.split(",") if item]):
        if variant not in VARIANTS:
            raise SystemExit(f"unknown variant {variant}")
        spec = VARIANTS[variant]
        job_id = f"{args.sample}_{args.group}_{variant}"
        model_path = out_root / job_id / "model"
        job = {
            "job_id": job_id,
            "stages": ["train", "eval"],
            "train": {
                "source_path": str(source),
                "model_path": str(model_path),
                "iterations": args.iterations,
                "args": ([] if spec.get("skip_core") else FOREGROUND_CORE) + [
                    "--resolution", str(args.resolution),
                    "--eval",
                    # a distinct viewer port per variant: 2DGS' network_gui binds
                    # 6009 by default and parallel/left-over runs collide there
                    "--port", str(args.port_base + index),
                    "--mask_dir", str(mask_dir),
                    "--test_iterations", str(args.iterations),
                    "--save_iterations", str(args.iterations),
                ] + spec["args"],
            },
            "eval": {
                "model_path": str(model_path),
                "source_path": str(source),
                "iteration": args.iterations,
                "mask_dir": str(mask_dir),
                "output_json": str(out_root / job_id / "eval" / "foreground_test.json"),
                "args": ["--split", "test", "--boundary_ignore", "2",
                         "--with_lpips", "--quiet"],
            },
            "variant": variant,
            "variant_label": spec["label"],
            "seed": args.seed,
        }
        job_path = cfg_dir / f"{job_id}.json"
        job_path.write_text(json.dumps(job, indent=2, ensure_ascii=False), encoding="utf-8")
        cmd = [sys.executable, str(RUNNER), "--job", str(job_path)]
        if args.dry_run:
            cmd.append("--dry-run")
        if args.force:
            cmd.append("--force")
        print(f"[{variant}] {spec['label']}\n  -> {job_path}", flush=True)
        result = subprocess.run(cmd, cwd=str(ROOT))
        if result.returncode != 0:
            print(f"[{variant}] runner exited {result.returncode}", flush=True)
            return result.returncode
    return 0


if __name__ == "__main__":
    sys.exit(main())
