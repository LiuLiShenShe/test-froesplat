#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RAP-FSAM3 manuscript-module ablation (R0..R6) on real SAM3 candidates.

Candidate pool policy
---------------------
All configurations read the *same* freshly-generated candidate pool produced by
`runs/R6_RAP_FULL_real/候选掩膜` (23 per-instance candidates for frame 0000 over
P1..P5).  Ranking-only configurations therefore differ **only** in the
selection rule, never in the candidate set.

Configurations (manuscript §2.3.2-§2.3.5)
-----------------------------------------
R0  single prompt P2, R0 ranking                 (legacy baseline)
R1  single prompt P2, R1/CDR ranking
R2  multi-prompt, SG-MPS gate, R0 ranking        (gate only)
R3  multi-prompt, SG-MPS gate, R1/CDR ranking    (gate + CDR)
R4b R3 + SB-FBPR with BOX prompts                (legacy prompt form)
R4p R3 + SB-FBPR with POINT prompts              (manuscript §2.3.4)
R5  R4p + residual/thin-structure repair
R6  R5 + GCFC (colmap_tracks_gaussian)           = full RAP-FSAM3
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
INPUT = ROOT / "00-论文优化重构/数据管理/05-评测结果/S21_KongQueZhuYu_E2_E3/selected_frames"
GT_MERGED = ROOT / "00-论文优化重构/数据管理/05-评测结果/S21_KongQueZhuYu_E2_E3/gt_masks"
GT_SPLIT = ROOT / "00-论文优化重构/数据管理/07-运行脚本与超参/S20-RAP-FSAM3掩膜生成与验证/GT口径拆分审计/gt_masks_split_v2/KongQueZhuYu"
COLMAP_DIR = ROOT / "04-COLMAP/KongQueZhuYu"
POOL = RECOVERY / "runs/R6_RAP_FULL_real"
RUNS = RECOVERY / "runs/ablation"
PYTHON = "/home/test/.venvs/sam3/bin/python"

COMMON = [
    "--input_dir", str(INPUT),
    "--sam3_repo", str(ROOT / "第三方源码/SAM3-latest"),
    "--sam3_checkpoint", str(ROOT / "sam3/sam3.pt"),
    "--device", "cuda",
    "--force",
    "--candidate_mode", "per_instance",
    "--score_weights", "area=1,comp=1,edge=1,temp=1,contrast=1,sam=0.5",
    "--reuse_existing_candidates",
    "--candidate_source_dir", str(POOL),
    "--gt_mask_dir", str(GT_MERGED),
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
GCFC = ["--use_corrective_geometry", "--corrective_geometry_backend",
        "colmap_tracks_gaussian", "--colmap_dir", str(COLMAP_DIR),
        "--strict_colmap_observations"]

CONFIGS = [
    ("R0_single_P2_base", SINGLE + ["--ranking_policy", "base"]),
    ("R1_single_P2_cdr", SINGLE + ["--ranking_policy", "r1"]),
    ("R2_multi_SGMPS_base", MULTI + GATE + ["--ranking_policy", "base"]),
    ("R3_multi_SGMPS_cdr", MULTI + GATE + ["--ranking_policy", "r1"]),
    ("R4b_R3_SPNP_boxes", MULTI + GATE + ["--ranking_policy", "r1"] + SPNP_BOXES),
    ("R4p_R3_SPNP_points", MULTI + GATE + ["--ranking_policy", "r1"] + SPNP_POINTS),
    ("R5_R4p_repair", MULTI + GATE + ["--ranking_policy", "r1"] + SPNP_POINTS + REPAIR),
    ("R6_R5_GCFC", MULTI + GATE + ["--ranking_policy", "r1"] + SPNP_POINTS + REPAIR + GCFC),
]


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


def seg_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    """F1 / IoU / Boundary-F1@2,5,10 / HD95 / background leakage.

    Leakage = fraction of predicted-foreground pixels farther than 15 px from
    the annotated object (spurious foreground, e.g. neighbouring plant parts
    or pot fragments dragged in by a bad instance).
    """
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


def oracle_candidates(pool: Path, frames: list[str], gt_sets: dict[str, tuple[Path, str]]) -> dict:
    """Per frame: the best candidate in the pool under each GT convention."""
    oracle = {}
    for stem in frames:
        per_gt = {}
        for tag, (gt_dir, pattern) in gt_sets.items():
            gp = gt_dir / pattern.format(stem=stem)
            if not gp.exists():
                continue
            gt = np.array(Image.open(gp).convert("L")) > 0
            best = None
            for raw_dir in sorted((pool / "候选掩膜").glob("raw_instance_*")):
                prompt_id = raw_dir.name.replace("raw_instance_", "")
                for mask_path in sorted(raw_dir.glob(f"mask_{stem}_*.png")):
                    inst = int(mask_path.stem.rsplit("_", 1)[1])
                    pred = np.array(Image.open(mask_path).convert("L")) > 0
                    m = prod_module().segmentation_metrics(pred, gt)
                    if best is None or m["f1"] > best["f1"]:
                        best = {"prompt": prompt_id, "instance": inst,
                                "f1": round(float(m["f1"]), 6),
                                "iou": round(float(m["iou"]), 6)}
            if best is not None:
                per_gt[tag] = best
        oracle[stem] = per_gt
    return oracle


def main() -> int:
    import argparse as _ap
    ap = _ap.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--gpu", default="0")
    cli = ap.parse_args()
    only = [x for x in cli.only.split(",") if x]
    RUNS.mkdir(parents=True, exist_ok=True)
    (RECOVERY / "logs").mkdir(parents=True, exist_ok=True)

    manifest: dict = {
        "candidate_pool": str(POOL),
        "candidate_pool_files_sha256": {
            str(p.relative_to(POOL)): sha256(p)
            for p in sorted((POOL / "候选掩膜").rglob("*.png"))
        },
        "input_images": {p.name: sha256(p) for p in sorted(INPUT.glob("*.jpg"))},
        "production_script_sha256": sha256(PROD),
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                     capture_output=True, text=True).stdout.strip(),
        "python": PYTHON,
        "runs": [],
    }
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = cli.gpu
    env["PYTHONPATH"] = str(SCRIPTS) + os.pathsep + env.get("PYTHONPATH", "")

    for name, extra in CONFIGS:
        if only and name not in only:
            continue
        out = RUNS / name
        if out.exists() and (out / "逐帧报告.csv").exists():
            print(f"[skip] {name} already complete", flush=True)
        else:
            out.mkdir(parents=True, exist_ok=True)
            cmd = [PYTHON, str(PROD), "--output_dir", str(out)] + COMMON + extra
            log = RECOVERY / "logs" / f"ablation_{name}.log"
            started = time.monotonic()
            with log.open("w", encoding="utf-8") as handle:
                handle.write(json.dumps({"command": cmd}, ensure_ascii=False) + "\n")
                handle.flush()
                result = subprocess.run(cmd, cwd=str(ROOT), env=env,
                                        stdout=handle, stderr=subprocess.STDOUT)
            wall = time.monotonic() - started
            manifest["runs"].append({"config": name, "cmd": cmd,
                                     "returncode": result.returncode,
                                     "wall_sec": round(wall, 1), "log": str(log)})
            print(f"[{name}] rc={result.returncode} wall={wall:.1f}s", flush=True)
            if result.returncode != 0:
                continue

    # ---- scoring against all three GT conventions + oracle selection audit ----
    GT_SETS = {
        "merged": (GT_MERGED, "mask_{stem}.png"),
        "potted_plant": (GT_SPLIT, "mask_potted_plant_{stem}.png"),
        "plant_only": (GT_SPLIT, "mask_plant_{stem}.png"),
    }
    frames = sorted(p.stem for p in INPUT.glob("*.jpg"))
    metrics_dir = RECOVERY / "metrics"
    metrics_dir.mkdir(exist_ok=True)
    oracle_cache = metrics_dir / "rap_ablation_oracle.json"
    if oracle_cache.exists():
        cached = json.loads(oracle_cache.read_text(encoding="utf-8"))
        if sorted(cached) == frames:
            oracle = cached
            print("[oracle] reused cached oracle table", flush=True)
        else:
            oracle = oracle_candidates(POOL, frames, GT_SETS)
    else:
        oracle = oracle_candidates(POOL, frames, GT_SETS)
    (metrics_dir / "rap_ablation_oracle.json").write_text(
        json.dumps(oracle, indent=2, ensure_ascii=False), encoding="utf-8")

    rows = []
    stage_rows = []
    for name, _ in CONFIGS:
        out = RUNS / name
        report = out / "逐帧报告.csv"
        if not report.exists():
            continue
        per_frame = {r["frame_id"]: r for r in csv.DictReader(report.open(encoding="utf-8"))}
        stage_rows.append({
            "config": name,
            "frames": len(per_frame),
            "sg_mps_executed": sum(int(r.get("sg_mps_executed") or 0) for r in per_frame.values()),
            "cdr_executed": sum(int(r.get("cdr_executed") or 0) for r in per_frame.values()),
            "sb_fbpr_executed": sum(int(r.get("sb_fbpr_executed") or 0) for r in per_frame.values()),
            "sb_fbpr_accepted": sum(int(r.get("sb_fbpr_accepted") or 0) for r in per_frame.values()),
            "residual_repair_executed": sum(int(r.get("residual_repair_executed") or 0) for r in per_frame.values()),
            "gcfc_executed": sum(int(r.get("gcfc_executed") or 0) for r in per_frame.values()),
            "gcfc_accepted": sum(int(r.get("gcfc_accepted") or 0) for r in per_frame.values()),
            "gcfc_failed_reasons": ";".join(
                f"{r['frame_id']}:{r.get('gcfc_reason','')}"
                for r in per_frame.values() if str(r.get("gcfc_accepted")) == "0"),
            "distinct_selected_prompts": ";".join(sorted({r.get("selected_prompt", "") for r in per_frame.values()})),
        })
        for stem, rec in per_frame.items():
            pred_path = out / "最终掩膜" / f"mask_{stem}.png"
            if not pred_path.exists():
                continue
            pred = np.array(Image.open(pred_path).convert("L")) > 0
            for gt_tag, (gt_dir, pattern) in GT_SETS.items():
                gp = gt_dir / pattern.format(stem=stem)
                if not gp.exists():
                    continue
                gt = np.array(Image.open(gp).convert("L")) > 0
                m = seg_metrics(pred, gt)
                best = oracle.get(stem, {}).get(gt_tag)
                sel = (rec.get("selected_prompt", ""), rec.get("selected_instance", ""))
                match = int(best is not None and sel == (best["prompt"], str(best["instance"])))
                rows.append({
                    "config": name, "frame": stem, "gt_convention": gt_tag,
                    "f1": round(m["f1"], 6), "iou": round(m["iou"], 6),
                    "boundary_f1@2": round(m["boundary_f1"], 6),
                    "boundary_f1@5": round(m["boundary_f1@5"], 6),
                    "boundary_f1@10": round(m["boundary_f1@10"], 6),
                    "hd95": round(m["hd95"], 3),
                    "leak_ratio": round(m["leak_ratio"], 6),
                    "pred_area": int(pred.sum()), "gt_area": int(gt.sum()),
                    "selected_prompt": rec.get("selected_prompt", ""),
                    "selected_instance": rec.get("selected_instance", ""),
                    "oracle_prompt": best["prompt"] if best else "",
                    "oracle_instance": best["instance"] if best else "",
                    "oracle_f1": best["f1"] if best else "",
                    "selects_oracle_instance": match,
                    "f1_gap_to_oracle": round(best["f1"] - m["f1"], 6) if best else "",
                    "sg_mps_prompt": rec.get("sg_mps_prompt", ""),
                    "cdr_policy": rec.get("cdr_policy", ""),
                    "sb_fbpr_prompt_form": rec.get("sb_fbpr_prompt_form", ""),
                    "sb_fbpr_accepted": rec.get("sb_fbpr_accepted", ""),
                    "residual_repair_accepted": rec.get("residual_repair_accepted", ""),
                    "gcfc_accepted": rec.get("gcfc_accepted", ""),
                    "gcfc_reason": rec.get("gcfc_reason", ""),
                    "final_mask_sha256": sha256(pred_path),
                })

    with (metrics_dir / "rap_ablation_per_frame.csv").open("w", newline="", encoding="utf-8") as h:
        writer = csv.DictWriter(h, fieldnames=list(rows[0].keys()) if rows else ["config"])
        writer.writeheader()
        writer.writerows(rows)
    with (metrics_dir / "rap_ablation_stage_counts.csv").open("w", newline="", encoding="utf-8") as h:
        writer = csv.DictWriter(h, fieldnames=list(stage_rows[0].keys()) if stage_rows else ["config"])
        writer.writeheader()
        writer.writerows(stage_rows)

    summary = []
    for name, _ in CONFIGS:
        for gt_tag in GT_SETS:
            sub = [r for r in rows if r["config"] == name and r["gt_convention"] == gt_tag]
            if not sub:
                continue
            gaps = [r["f1_gap_to_oracle"] for r in sub if isinstance(r["f1_gap_to_oracle"], float)]
            summary.append({
                "config": name, "gt_convention": gt_tag, "frames": len(sub),
                "f1_mean": round(float(np.mean([r["f1"] for r in sub])), 6),
                "f1_min": round(float(np.min([r["f1"] for r in sub])), 6),
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
    with (metrics_dir / "rap_ablation_summary.csv").open("w", newline="", encoding="utf-8") as h:
        writer = csv.DictWriter(h, fieldnames=list(summary[0].keys()) if summary else ["config"])
        writer.writeheader()
        writer.writerows(summary)
    manifest["summary"] = summary
    manifest["stage_counts"] = stage_rows
    manifest["oracle"] = oracle
    (metrics_dir / "rap_ablation_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    for row in summary:
        print(row)
    for row in stage_rows:
        print(row)
    return 0


if __name__ == "__main__":
    sys.exit(main())
