#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Full-RAP masks -> Foreground-Object 2DGS -> foreground metrics.

Replaces the previously frozen `C_R1` mask source with the *full* RAP-FSAM3
output (SG-MPS + CDR + SB-FBPR + residual repair + GCFC).

The training recipe is the established Foreground-Object 2DGS setting used by
the E4b mask-source experiment (`06-实验输出/KongQueZhuYu/E4b_*/command.txt`),
so a comparison against `C_R1` is apples-to-apples: same source path, same
resolution, same iterations, same losses, same pruning.

Mask provenance is written before training and re-verified against a manifest
pre-computed from the RAP run directory, so a run cannot silently train on a
different mask version than the one RAP-FSAM3 produced.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/data/fj/F2DMAS")
RECOVERY = ROOT / "00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_Original_Method_Recovery"
GS = ROOT / "2d-gaussian-splatting-m2m3-floor40"
GS_PY = GS / "venv/bin/python"
LOCKED = ROOT / "00-论文优化重构/数据管理/02-位姿COLMAP/03-final_locked"
OUT = RECOVERY / "runs/downstream"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_masks(scene: str, mask_dir: Path, model_dir: Path) -> dict:
    images = sorted((LOCKED / scene / "images").glob("*"))
    prepared = model_dir / "prepared_masks"
    prepared.mkdir(parents=True, exist_ok=True)
    entries, missing = [], []
    for image in images:
        source = mask_dir / f"mask_{image.stem}.png"
        if not source.exists():
            missing.append(image.name)
            entries.append({"image": image.name, "sha256": None, "status": "MISSING"})
            continue
        target = prepared / source.name
        if target.is_symlink() or target.exists():
            target.unlink()
        target.symlink_to(source.resolve())
        entries.append({"image": image.name, "mask_path": str(source),
                        "sha256": sha256(source), "status": "OK"})
    manifest = {
        "schema_version": 1,
        "source_label": f"RAP-FSAM3-full ({mask_dir})",
        "mask_dir": str(prepared),
        "mask_pattern": "mask_{stem}.png",
        "mask_threshold": 0.5,
        "image_count": len(images),
        "camera_count": len(entries),
        "missing_count": len(missing),
        "missing": missing,
        "entries": entries,
    }
    (model_dir / "mask_provenance_expected.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def train_and_eval(scene: str, model_dir: Path, iterations: int, resolution: int,
                   with_m2m3_go: bool, port: int, log_dir: Path,
                   view_weight_mode: str = "rgb_only") -> dict:
    source_path = LOCKED / scene
    prepared = model_dir / "prepared_masks"
    common = [
        "--source_path", str(source_path),
        "--model_path", str(model_dir),
        "--eval", "--resolution", str(resolution),
        "--view_quality_mode", "none",
        "--raw_gate_mode", "none", "--mask_gate_mode", "none", "--geo_gate_mode", "none",
        "--mask_mode", "alpha", "--mask_dir", str(prepared),
        "--mask_pattern", "mask_{stem}.png", "--mask_threshold", "0.5",
        "--mask_provenance_required",
        "--init_pcd_mode", "foreground_track",
        "--init_pcd_min_observations", "3",
        "--init_pcd_foreground_threshold", "0.9",
        "--init_pcd_dilate_mask_px", "0",
        "--use_foreground_rgb_loss", "--lambda_fg_rgb", "1.0",
        "--foreground_bg_rgb_weight", "0.0", "--foreground_rgb_crop_padding", "12",
        "--use_mask_loss", "--use_bg_opacity_loss",
        "--lambda_mask", "0.08", "--lambda_bg", "0.02",
        "--mask_loss_type", "l1_dice", "--mask_ignore_boundary_px", "2",
        "--mask_loss_start_iter", "500", "--mask_loss_warmup_iters", "1500",
        # ── Manuscript §2.6: mask-guided multi-cue pruning (P_keep) ──
        "--pruning_mode", "mask", "--pruning_start_iter", "18000",
        "--pruning_interval", "3000", "--pruning_opacity_threshold", "0.005",
        "--pruning_brightness_threshold", "0.01", "--pruning_mask_threshold", "0.45",
        "--pruning_mask_max_views", "12", "--pruning_max_remove_ratio", "0.03",
        "--pruning_mask_score_weight", "3.0", "--save_pruning_report",
        "--pruning_survival_mode", "multi_cue",
        "--pruning_cue_weights", "3.0,1.0,0.25,1.0,0.5",
        "--pruning_survival_threshold", "0.45",
        "--ip", "127.0.0.1", "--port", str(port),
    ]
    # ── Manuscript §2.5: view-quality-aware soft loss weighting ──
    # q_i = q_min + (q_max-q_min) sigmoid((s_i-mu_s)/(sigma_s+eps)) over mask
    # coverage / boundary sharpness / foreground RGB contrast, applied to the
    # foreground RGB term only.  `--view_weight_normalize` divides by the mean
    # weight, which is the minibatch form of the manuscript's sum-to-one q̄_i
    # (it keeps the mean loss scale unchanged).
    if view_weight_mode != "none":
        common += [
            "--view_weight_mode", view_weight_mode,
            "--view_weight_source", "computed",
            "--view_weight_q_min", "0.6", "--view_weight_q_max", "1.0",
            "--view_weight_normalize",
        ]
    if with_m2m3_go:
        common += [
            "--m2m3_go_mode", "full",
            "--m2m3_go_start_iter", "6000",
            "--m2m3_go_interval", "1000",
            "--m2m3_go_stop_iter", "24000",
            "--m2m3_go_tau_mode", "quantile",
            "--m2m3_go_tau_quantile", "0.01",
            "--m2m3_go_save_report",
        ]
    train_cmd = [str(GS_PY), "train.py"] + common + [
        "--iterations", str(iterations),
        "--test_iterations", "-1",
        "--save_iterations", str(iterations),
    ]
    log_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with (log_dir / "train.log").open("w", encoding="utf-8") as handle:
        rc_train = subprocess.run(train_cmd, cwd=str(GS), stdout=handle,
                                  stderr=subprocess.STDOUT).returncode
    wall_train = round(time.monotonic() - started, 1)
    if rc_train != 0:
        return {"train_returncode": rc_train, "train_wall_sec": wall_train,
                "train_cmd": " ".join(train_cmd), "status": "TRAIN_FAILED"}

    eval_cmd = [str(GS_PY), str(GS / "scripts/evaluate_foreground.py"),
                "-m", str(model_dir), "-s", str(source_path),
                "--images", "images", "--resolution", str(resolution), "--eval",
                "--iteration", str(iterations), "--mask_dir", str(prepared),
                "--split", "test", "--boundary_ignore", "2", "--with_lpips",
                "--output_json", str(model_dir / "eval/foreground_test.json")]
    (model_dir / "eval").mkdir(parents=True, exist_ok=True)
    with (log_dir / "eval.log").open("w", encoding="utf-8") as handle:
        rc_eval = subprocess.run(eval_cmd, cwd=str(GS), stdout=handle,
                                 stderr=subprocess.STDOUT).returncode
    return {"train_returncode": rc_train, "train_wall_sec": wall_train,
            "eval_returncode": rc_eval, "train_cmd": " ".join(train_cmd),
            "eval_cmd": " ".join(eval_cmd),
            "status": "COMPLETE" if rc_eval == 0 else "EVAL_FAILED"}


def extract_view_weight_evidence(log_path: Path) -> dict:
    """Parse the view-weight lines `train.py` prints, as run-level evidence."""
    evidence = {"computed": False, "loaded": False, "normalised": False,
                "computed_line": None, "loaded_line": None, "normalised_line": None}
    if not log_path.exists():
        return evidence
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "computed" in line and "soft view weights" in line:
            evidence["computed"] = True
            evidence["computed_line"] = line.strip()
        elif "loaded" in line and "soft view weights" in line:
            evidence["loaded"] = True
            evidence["loaded_line"] = line.strip()
        elif "normalised view weights" in line:
            evidence["normalised"] = True
            evidence["normalised_line"] = line.strip()
    return evidence


def main() -> int:
    import argparse as _ap
    ap = _ap.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--mask-run-dir", type=Path, required=True,
                    help="RAP output dir containing 最终掩膜/")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--iterations", type=int, default=30000)
    ap.add_argument("--resolution", type=int, default=2)
    ap.add_argument("--m2m3-go", action="store_true")
    ap.add_argument("--port", type=int, default=6030)
    ap.add_argument("--prepare-only", action="store_true")
    ap.add_argument("--view-weight-mode", default="rgb_only",
                    choices=["none", "rgb_only", "all_losses"],
                    help="Manuscript §2.5 view-quality-aware soft weighting. "
                         "none disables it (the historical setting); rgb_only "
                         "applies q_i to the foreground RGB term as written.")
    cli = ap.parse_args()

    model_dir = OUT / f"{cli.scene}_{cli.tag}"
    model_dir.mkdir(parents=True, exist_ok=True)
    mask_dir = cli.mask_run_dir / "最终掩膜"
    manifest = prepare_masks(cli.scene, mask_dir, model_dir)
    print(f"[prepare] {manifest['camera_count']} cameras, "
          f"{manifest['missing_count']} missing masks", flush=True)
    if manifest["missing_count"]:
        print("[prepare] MISSING:", manifest["missing"][:10], flush=True)
        return 2
    if cli.prepare_only:
        return 0

    result = train_and_eval(cli.scene, model_dir, cli.iterations, cli.resolution,
                            cli.m2m3_go, cli.port, model_dir / "logs",
                            view_weight_mode=cli.view_weight_mode)
    run = {
        "scene": cli.scene, "tag": cli.tag,
        "mask_run_dir": str(cli.mask_run_dir),
        "model_dir": str(model_dir),
        "iterations": cli.iterations, "resolution": cli.resolution,
        "m2m3_go": cli.m2m3_go,
        "view_weight_mode": cli.view_weight_mode,
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                     capture_output=True, text=True).stdout.strip(),
        "mask_manifest_sha256": sha256(model_dir / "mask_provenance_expected.json"),
        **result,
    }
    # §2.5 evidence: the weights the training process actually computed, read back
    # from its own stdout so a "view weighting was on" claim is checkable.
    run["view_weight_evidence"] = extract_view_weight_evidence(log_dir / "train.log")
    (model_dir / "run_record.json").write_text(
        json.dumps(run, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in run.items() if k not in ("train_cmd", "eval_cmd")},
                     indent=1, ensure_ascii=False))
    return 0 if run["status"] == "COMPLETE" else 1


if __name__ == "__main__":
    sys.exit(main())
