#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RAP-FSAM3 ablation on a CROSS-SCENE dev set (disjoint from the frozen test set).

Dev set (16 frames / 4 scenes) — deliberately NOT KongQueZhuYu, which is the
frozen evaluation scene of S21:

    CaoMei1        0000 0025 0050 0075 0100
    ChangShouHua2  0000 0025 0050 0075 0100
    DouBanLv1      0000 0025 0050 0075 0100
    XianKeLai1     0000                     (only frame with GT)

GT convention: `GT口径拆分审计/gt_masks_split_v2/<scene>/mask_potted_plant_*.png`
is the primary target (RAP-FSAM3 emits a potted-plant foreground prior);
`mask_plant_*` and `mask_pot_*` are reported alongside.

Per scene the candidate pool is produced once by the R6 run (real SAM3
inference, prompts P1..P5, per_instance) and then reused by every other
configuration, so the rungs differ ONLY in the selection/refinement rule.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path("/data/fj/F2DMAS")
RECOVERY = ROOT / "00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_Original_Method_Recovery"
SCRIPTS = ROOT / "00-论文优化重构/数据管理/07-运行脚本与超参/S20-RAP-FSAM3掩膜生成与验证/脚本"
PROD = SCRIPTS / "生成RAP-FSAM3掩膜.py"
FRAMES = ROOT / "00-论文优化重构/数据管理/01-输入图像/02-fft_frames"
GT_SPLIT = ROOT / "00-论文优化重构/数据管理/07-运行脚本与超参/S20-RAP-FSAM3掩膜生成与验证/GT口径拆分审计/gt_masks_split_v2"
COLMAP = ROOT / "04-COLMAP"
DEV_INPUT = RECOVERY / "dev_inputs"
RUNS = RECOVERY / "runs/dev_ablation_v2"
PYTHON = "/home/test/.venvs/sam3/bin/python"

SCENES = {
    "CaoMei1": ["0000", "0025", "0050", "0075", "0100"],
    "ChangShouHua2": ["0000", "0025", "0050", "0075", "0100"],
    "DouBanLv1": ["0000", "0025", "0050", "0075", "0100"],
    "XianKeLai1": ["0000"],
}

COMMON = [
    "--sam3_repo", str(ROOT / "第三方源码/SAM3-latest"),
    "--sam3_checkpoint", str(ROOT / "sam3/sam3.pt"),
    "--device", "cuda", "--force",
    "--candidate_mode", "per_instance",
    "--save_raw_instance_masks",
]

MULTI = ["--prompt_list", "P1,P2,P3,P4,P5", "--default_prompt_id", "P2",
         "--use_prompt_ensemble", "--prompt_selection_mode", "score_select"]
SINGLE = ["--prompt_list", "P2", "--default_prompt_id", "P2",
          "--prompt_selection_mode", "single"]
GATE = ["--use_semantic_gate", "--semantic_gate_mode", "gated_softmax",
        "--semantic_gate_temperature", "0.25"]
SPNP_BOXES = ["--use_spnp_refinement", "--spnp_backend", "sam3_if_supported",
              "--spnp_prompt_form", "boxes", "--spnp_negative_region_mode", "ring"]
SPNP_POINTS = ["--use_spnp_refinement", "--spnp_backend", "sam3_if_supported",
               "--spnp_prompt_form", "points", "--spnp_negative_region_mode", "manuscript"]
REPAIR = ["--use_residual_repair", "--thin_repair_mode", "adjacent",
          "--opening_kernel", "3"]


def gcfc(scene: str) -> list[str]:
    return ["--use_corrective_geometry", "--corrective_geometry_backend",
            "colmap_tracks_gaussian", "--colmap_dir", str(COLMAP / scene),
            "--strict_colmap_observations"]


CONFIGS = [
    ("R0_single_P2_base", SINGLE + ["--ranking_policy", "base"]),
    ("R1_single_P2_cdr", SINGLE + ["--ranking_policy", "r1"]),
    # R2/R3 gate ON vs R3n gate OFF isolates SG-MPS from the mere expansion of
    # the candidate set to five prompts: same pool, same ranking policy.
    ("R3n_multi_nogate_cdr", MULTI + ["--ranking_policy", "r1"]),
    # R3g = R3 + a gate commitment margin.  On these scenes the five prompt
    # confidences sit within ~0.03 of each other, so the pure-argmax gate (R3)
    # is effectively a coin flip; R3g measures what the gate buys once it is
    # only allowed to speak when it actually has a preference.
    ("R3g_multi_gate_margin", MULTI + GATE + ["--ranking_policy", "r1",
                                             "--semantic_gate_margin", "0.05"]),
    ("R2_multi_SGMPS_base", MULTI + GATE + ["--ranking_policy", "base"]),
    ("R3_multi_SGMPS_cdr", MULTI + GATE + ["--ranking_policy", "r1"]),
    ("R4b_R3_SPNP_boxes", MULTI + GATE + ["--ranking_policy", "r1"] + SPNP_BOXES),
    ("R4p_R3_SPNP_points", MULTI + GATE + ["--ranking_policy", "r1"] + SPNP_POINTS),
    ("R5_R4p_repair", MULTI + GATE + ["--ranking_policy", "r1"] + SPNP_POINTS + REPAIR),
]
# R6 = R5 + GCFC, and it is the pool producer (real SAM3 inference).  It runs
# FIRST in each scene so that every other rung reuses its candidate cache.
R6 = ("R6_R5_GCFC", None)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


_PROD_MODULE = None


def prod_module():
    global _PROD_MODULE
    if _PROD_MODULE is None:
        sys.path.insert(0, str(SCRIPTS))
        import importlib.util
        spec = importlib.util.spec_from_file_location("rap_metrics", PROD)
        module = importlib.util.module_from_spec(spec)
        sys.modules["rap_metrics"] = module
        spec.loader.exec_module(module)
        _PROD_MODULE = module
    return _PROD_MODULE


def fast_f1(pred: np.ndarray, gt: np.ndarray) -> float:
    inter = float(np.logical_and(pred, gt).sum())
    total = float(pred.sum() + gt.sum())
    return (2.0 * inter / total) if total else 1.0


def seg_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    from scipy import ndimage
    m = prod_module()
    out = m.segmentation_metrics(pred, gt)
    for tol in (5, 10):
        out[f"boundary_f1@{tol}"] = m.boundary_f1(pred, gt, tolerance=tol)
    pred_sum = int(pred.sum())
    if pred_sum == 0:
        out["leak_ratio"] = 0.0
    else:
        allowed = ndimage.binary_dilation(gt, np.ones((31, 31), dtype=bool))
        out["leak_ratio"] = float(np.logical_and(pred, ~allowed).sum()) / pred_sum
    return out


def prepare_dev_inputs() -> dict[str, list[Path]]:
    out = {}
    for scene, stems in SCENES.items():
        target = DEV_INPUT / scene
        target.mkdir(parents=True, exist_ok=True)
        made = []
        for stem in stems:
            src = FRAMES / scene / f"{stem}.jpg"
            if not src.exists():
                raise FileNotFoundError(src)
            dst = target / f"{stem}.jpg"
            if not dst.exists():
                dst.symlink_to(src.resolve())
            made.append(dst)
        out[scene] = made
    return out


def run_config(scene: str, name: str, extra: list[str], source_dir: Path | None) -> dict:
    out_dir = RUNS / scene / name
    report = out_dir / "逐帧报告.csv"
    if report.exists():
        print(f"[skip] {scene}/{name}", flush=True)
        return {"scene": scene, "config": name, "skipped": True}
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [PYTHON, str(PROD),
           "--input_dir", str(DEV_INPUT / scene),
           "--output_dir", str(out_dir)] + COMMON + extra
    if source_dir is not None:
        cmd += ["--reuse_existing_candidates", "--candidate_source_dir", str(source_dir)]
    log = RECOVERY / "logs" / f"dev_{scene}_{name}.log"
    started = time.monotonic()
    with log.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"command": cmd}, ensure_ascii=False) + "\n")
        handle.flush()
        env = os.environ.copy()
        env["PYTHONPATH"] = str(SCRIPTS) + os.pathsep + env.get("PYTHONPATH", "")
        rc = subprocess.run(cmd, cwd=str(ROOT), env=env, stdout=handle,
                            stderr=subprocess.STDOUT).returncode
    wall = round(time.monotonic() - started, 1)
    print(f"[{scene}/{name}] rc={rc} wall={wall}s", flush=True)
    return {"scene": scene, "config": name, "cmd": cmd, "returncode": rc,
            "wall_sec": wall, "log": str(log)}


def score_job(job: dict) -> dict:
    """Score one (scene, config, frame, GT convention) triple.

    Runs in a worker process.  `segmentation_metrics` dilates a boundary band and
    runs two distance transforms on a 2160x3840 image, which costs ~5 s per call
    -- serial scoring of the full grid takes well over an hour, so this is the
    unit of parallelism.
    """
    pred = np.array(Image.open(job["pred"]).convert("L")) > 0
    gt = np.array(Image.open(job["gt"]).convert("L")) > 0
    m = seg_metrics(pred, gt)
    rec, best = job["rec"], job["best"]
    sel = (rec.get("selected_prompt", ""), rec.get("selected_instance", ""))
    return {
        "_ord": job["ord"],
        "scene": job["scene"], "config": job["config"], "frame": job["frame"],
        "gt_convention": job["gt_convention"],
        "f1": round(m["f1"], 6), "iou": round(m["iou"], 6),
        "boundary_f1@2": round(m["boundary_f1"], 6),
        "boundary_f1@5": round(m["boundary_f1@5"], 6),
        "boundary_f1@10": round(m["boundary_f1@10"], 6),
        "hd95": round(m["hd95"], 3),
        "leak_ratio": round(m["leak_ratio"], 6),
        "pred_area": int(pred.sum()), "gt_area": int(gt.sum()),
        "selected_prompt": sel[0], "selected_instance": sel[1],
        "sg_mps_prompt": rec.get("sg_mps_prompt", ""),
        "cdr_policy": rec.get("cdr_policy", ""),
        "sb_fbpr_prompt_form": rec.get("sb_fbpr_prompt_form", ""),
        "sb_fbpr_accepted": rec.get("sb_fbpr_accepted", ""),
        "residual_restored_px": rec.get("residual_restored_px", ""),
        "gcfc_accepted": rec.get("gcfc_accepted", ""),
        "gcfc_reason": rec.get("gcfc_reason", ""),
        "gcfc_support_G": rec.get("gcfc_support_G", ""),
        "oracle_prompt": best["prompt"] if best else "",
        "oracle_instance": best["instance"] if best else "",
        "oracle_f1": best["f1"] if best else "",
        "selects_oracle_instance": int(
            best is not None and sel == (best["prompt"], str(best["instance"]))),
        "f1_gap_to_oracle": round(best["f1"] - m["f1"], 6) if best else "",
    }


def run_jobs(jobs: list[dict], workers: int) -> list[dict]:
    if not jobs:
        return []
    if workers <= 1:
        return [score_job(j) for j in jobs]
    import multiprocessing
    rows = []
    # fork, not spawn: the parent has already imported the production module and
    # has not touched CUDA (this script never does), so forking is both safe and
    # much cheaper than re-executing the 3400-line production script per worker.
    ctx = multiprocessing.get_context("fork")
    with ctx.Pool(processes=workers) as pool:
        for i, row in enumerate(pool.imap_unordered(score_job, jobs, chunksize=1), 1):
            rows.append(row)
            if i % 25 == 0 or i == len(jobs):
                print(f"[score] {i}/{len(jobs)}", flush=True)
    return rows


def score_all(manifest: dict, workers: int = 8) -> None:
    metrics_dir = RECOVERY / "metrics"
    metrics_dir.mkdir(exist_ok=True)
    gt_sets = {
        "potted_plant": "mask_potted_plant_{stem}.png",
        "plant_only": "mask_plant_{stem}.png",
        "pot": "mask_pot_{stem}.png",
    }
    jobs: list[dict] = []
    stages: list[dict] = []
    oracle_path = metrics_dir / "dev_oracle.json"
    oracle = json.loads(oracle_path.read_text(encoding="utf-8")) if oracle_path.exists() else {}

    for scene, stems in SCENES.items():
        pool = RUNS / scene / R6[0]
        # ---- oracle over the shared pool ----
        if scene not in oracle:
            scene_oracle = {}
            for stem in stems:
                per_gt = {}
                for tag, pattern in gt_sets.items():
                    gp = GT_SPLIT / scene / pattern.format(stem=stem)
                    if not gp.exists():
                        continue
                    gt = np.array(Image.open(gp).convert("L")) > 0
                    best = None
                    for raw_dir in sorted((pool / "候选掩膜").glob("raw_instance_*")):
                        prompt_id = raw_dir.name.replace("raw_instance_", "")
                        for mp in sorted(raw_dir.glob(f"mask_{stem}_*.png")):
                            inst = int(mp.stem.rsplit("_", 1)[1])
                            pred = np.array(Image.open(mp).convert("L")) > 0
                            if not pred.any():
                                continue
                            # F1/IoU only: the oracle ranking needs nothing
                            # else, and skipping boundary-band dilation +
                            # distance transforms is ~15x cheaper per candidate.
                            f1 = fast_f1(pred, gt)
                            if best is None or f1 > best["f1"]:
                                best = {"prompt": prompt_id, "instance": inst,
                                        "f1": round(f1, 6)}
                    if best is not None:
                        per_gt[tag] = best
                scene_oracle[stem] = per_gt
            oracle[scene] = scene_oracle
            oracle_path.write_text(json.dumps(oracle, indent=2, ensure_ascii=False),
                                   encoding="utf-8")
            print(f"[oracle] {scene} done", flush=True)

        for name, _ in [R6] + CONFIGS:
            out_dir = RUNS / scene / name
            report = out_dir / "逐帧报告.csv"
            if not report.exists():
                continue
            per_frame = {r["frame_id"]: r for r in csv.DictReader(report.open(encoding="utf-8"))}
            stages.append({
                "scene": scene, "config": name, "frames": len(per_frame),
                "sg_mps_executed": sum(int(r.get("sg_mps_executed") or 0) for r in per_frame.values()),
                "cdr_executed": sum(int(r.get("cdr_executed") or 0) for r in per_frame.values()),
                "sb_fbpr_executed": sum(int(r.get("sb_fbpr_executed") or 0) for r in per_frame.values()),
                "sb_fbpr_accepted": sum(int(r.get("sb_fbpr_accepted") or 0) for r in per_frame.values()),
                "residual_repair_executed": sum(int(r.get("residual_repair_executed") or 0) for r in per_frame.values()),
                "gcfc_executed": sum(int(r.get("gcfc_executed") or 0) for r in per_frame.values()),
                "gcfc_accepted": sum(int(r.get("gcfc_accepted") or 0) for r in per_frame.values()),
                "gcfc_reasons": ";".join(
                    f"{r['frame_id']}:{r.get('gcfc_reason','')}"
                    for r in per_frame.values() if str(r.get("gcfc_accepted")) == "0"),
            })
            for stem, rec in per_frame.items():
                pred_path = out_dir / "最终掩膜" / f"mask_{stem}.png"
                if not pred_path.exists():
                    continue
                for tag, pattern in gt_sets.items():
                    gp = GT_SPLIT / scene / pattern.format(stem=stem)
                    if not gp.exists():
                        continue
                    jobs.append({
                        "ord": len(jobs),
                        "scene": scene, "config": name, "frame": stem,
                        "gt_convention": tag, "pred": pred_path, "gt": gp,
                        "rec": rec,
                        "best": oracle.get(scene, {}).get(stem, {}).get(tag),
                    })

    rows = run_jobs(jobs, workers)
    rows.sort(key=lambda r: r.pop("_ord"))

    with (metrics_dir / "dev_ablation_per_frame.csv").open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with (metrics_dir / "dev_ablation_stage_counts.csv").open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=list(stages[0].keys()) if stages else ["config"])
        w.writeheader()
        w.writerows(stages)

    order = [R6[0]] + [name for name, _ in CONFIGS]
    summary = []
    for name in order:
        for tag in gt_sets:
            sub = [r for r in rows if r["config"] == name and r["gt_convention"] == tag]
            if not sub:
                continue
            gaps = [r["f1_gap_to_oracle"] for r in sub if isinstance(r["f1_gap_to_oracle"], float)]
            summary.append({
                "config": name, "gt_convention": tag, "frames": len(sub),
                "f1_mean": round(float(np.mean([r["f1"] for r in sub])), 6),
                "f1_min": round(float(np.min([r["f1"] for r in sub])), 6),
                "f1_std": round(float(np.std([r["f1"] for r in sub])), 6),
                "frames_f1_below_090": int(sum(1 for r in sub if r["f1"] < 0.90)),
                "frames_f1_below_095": int(sum(1 for r in sub if r["f1"] < 0.95)),
                "iou_mean": round(float(np.mean([r["iou"] for r in sub])), 6),
                "boundary_f1@2": round(float(np.mean([r["boundary_f1@2"] for r in sub])), 6),
                "boundary_f1@5": round(float(np.mean([r["boundary_f1@5"] for r in sub])), 6),
                "boundary_f1@10": round(float(np.mean([r["boundary_f1@10"] for r in sub])), 6),
                "hd95_mean": round(float(np.mean([r["hd95"] for r in sub])), 3),
                "hd95_median": round(float(np.median([r["hd95"] for r in sub])), 3),
                "leak_ratio_mean": round(float(np.mean([r["leak_ratio"] for r in sub])), 6),
                "oracle_selection_rate": round(float(np.mean([r["selects_oracle_instance"] for r in sub])), 4),
                "f1_gap_to_oracle_mean": round(float(np.mean(gaps)), 6) if gaps else "",
            })
    with (metrics_dir / "dev_ablation_summary.csv").open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)

    # per-scene breakdown
    per_scene = []
    for name in order:
        for scene in SCENES:
            sub = [r for r in rows if r["config"] == name and r["gt_convention"] == "potted_plant"
                   and r["scene"] == scene]
            if not sub:
                continue
            per_scene.append({
                "config": name, "scene": scene, "frames": len(sub),
                "f1_mean": round(float(np.mean([r["f1"] for r in sub])), 6),
                "f1_min": round(float(np.min([r["f1"] for r in sub])), 6),
                "boundary_f1@5": round(float(np.mean([r["boundary_f1@5"] for r in sub])), 6),
                "hd95_mean": round(float(np.mean([r["hd95"] for r in sub])), 3),
                "oracle_selection_rate": round(float(np.mean([r["selects_oracle_instance"] for r in sub])), 4),
            })
    with (metrics_dir / "dev_ablation_per_scene.csv").open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=list(per_scene[0].keys()))
        w.writeheader()
        w.writerows(per_scene)

    manifest["summary"] = summary
    manifest["stage_counts"] = stages
    manifest["per_scene"] = per_scene
    (metrics_dir / "dev_ablation_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    for r in summary:
        if r["gt_convention"] == "potted_plant":
            print(json.dumps(r, ensure_ascii=False))
    return summary


def main() -> int:
    import argparse as _ap
    ap = _ap.ArgumentParser()
    ap.add_argument("--scenes", default=",".join(SCENES))
    ap.add_argument("--configs", default="")
    ap.add_argument("--score-only", action="store_true")
    # Produce the runs only; leave metrics/ untouched.  Used to top up missing
    # rungs while a separate scoring process owns metrics/.
    ap.add_argument("--run-only", action="store_true")
    ap.add_argument("--workers", type=int, default=8,
                    help="parallel scoring processes (each metric call is ~5 s)")
    cli = ap.parse_args()
    scenes = [s for s in cli.scenes.split(",") if s]
    wanted = [c for c in cli.configs.split(",") if c]

    (RECOVERY / "logs").mkdir(parents=True, exist_ok=True)
    prepare_dev_inputs()

    manifest: dict = {
        "dev_set": {s: SCENES[s] for s in scenes},
        "frozen_eval_scene_excluded": "KongQueZhuYu",
        "gt_convention_primary": "potted_plant",
        "production_script_sha256": sha256(PROD),
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                     capture_output=True, text=True).stdout.strip(),
        "python": PYTHON,
        "runs": [],
    }

    if not cli.score_only:
        for scene in scenes:
            # 1. R6 first: real SAM3 candidate generation + full RAP-FSAM3
            run_config(scene, R6[0],
                       MULTI + GATE + ["--ranking_policy", "r1"] + SPNP_POINTS
                       + REPAIR + gcfc(scene), source_dir=None)
            pool = RUNS / scene / R6[0]
            # 2. every other rung reuses that exact candidate cache
            for name, extra in CONFIGS:
                if wanted and name not in wanted:
                    continue
                run_config(scene, name, extra, source_dir=pool)

    if cli.run_only:
        return 0
    score_all(manifest, workers=cli.workers)
    return 0


if __name__ == "__main__":
    sys.exit(main())
