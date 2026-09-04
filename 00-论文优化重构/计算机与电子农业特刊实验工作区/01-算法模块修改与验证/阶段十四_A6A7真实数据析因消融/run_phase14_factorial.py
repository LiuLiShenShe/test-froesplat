#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 14 — 2×2 Factorial Ablation Runner

Runs V00 (Control), V10 (A6), V01 (A7), V11 (A6+A7) on the same 21-frame set.
Each variant writes to an isolated output directory.

Usage:
    python run_phase14_factorial.py [--variants V00 V10 V01 V11] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

PIPELINE = Path("/data/fj/F2DMAS/00-论文优化重构/数据管理/07-运行脚本与超参"
                "/S20-RAP-FSAM3掩膜生成与验证/脚本/生成RAP-FSAM3掩膜.py")
FRAME_INPUTS = Path("/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区"
                     "/01-算法模块修改与验证/阶段十二_GT_v2_QA与P6正式验收/frame_inputs")
PHASE14_DIR = Path("/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区"
                    "/01-算法模块修改与验证/阶段十四_A6A7真实数据析因消融")
PYTHON = "/home/test/biosoft/enter/envs/sam3/bin/python"

# Common args shared by all variants (frozen per protocol)
COMMON_ARGS = [
    "--prompt_list", "P6",
    "--default_prompt_id", "P6",
    "--candidate_mode", "per_instance",
    "--score_weights", "area=1,comp=1,edge=1,temp=1,contrast=1,sam=0.5",
    "--sam3_mask_threshold", "0.5",
    "--save_raw_instance_masks",
    "--save_candidate_masks",
    "--save_intermediate_masks",
    "--force",
    "--consensus_min_frames", "5",
]

# Variant-specific A6/A7 flags
VARIANTS = {
    "V00": {"a6": False, "a7": False, "label": "Control"},
    "V10": {"a6": True,  "a7": False, "label": "A6"},
    "V01": {"a6": False, "a7": True,  "label": "A7"},
    "V11": {"a6": True,  "a7": True,  "label": "A6+A7"},
}


def build_cmd(variant: str, dry_run: bool = False) -> list[str]:
    """Build CLI command for a single variant."""
    v = VARIANTS[variant]
    output_dir = PHASE14_DIR / f"{variant}_{v['label']}"

    cmd = [PYTHON, str(PIPELINE)]
    cmd += ["--input_dir", str(FRAME_INPUTS)]
    cmd += ["--output_dir", str(output_dir)]
    cmd += COMMON_ARGS

    if v["a6"]:
        cmd.append("--use_cross_view_consensus")
        cmd += ["--colmap_dir", str(PHASE14_DIR.parent.parent.parent / "数据管理/02-位姿COLMAP/03-final_locked")]

    if v["a7"]:
        cmd.append("--use_memory_propagation")

    if dry_run:
        cmd.append("--dry_run")

    return cmd


def run_variant(variant: str, dry_run: bool = False) -> dict:
    """Run a single variant and return timing/status info."""
    v = VARIANTS[variant]
    output_dir = PHASE14_DIR / f"{variant}_{v['label']}"
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = build_cmd(variant, dry_run)
    log_file = PHASE14_DIR / "logs" / f"{variant}_run.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Running {variant} ({v['label']})")
    print(f"Output: {output_dir}")
    print(f"{'='*60}")

    t0 = time.time()
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=7200,  # 2 hour timeout per variant
    )
    elapsed = time.time() - t0

    # Save logs
    log_file.write_text(
        f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}",
        encoding="utf-8",
    )

    status = {
        "variant": variant,
        "label": v["label"],
        "a6_enabled": v["a6"],
        "a7_enabled": v["a7"],
        "returncode": result.returncode,
        "elapsed_sec": round(elapsed, 1),
        "output_dir": str(output_dir),
        "log_file": str(log_file),
    }

    if result.returncode != 0:
        print(f"  FAILED (exit code {result.returncode})")
        print(f"  Last 20 lines of stderr:")
        for line in result.stderr.strip().split("\n")[-20:]:
            print(f"    {line}")
    else:
        print(f"  SUCCESS ({elapsed:.1f}s)")

    return status


def main():
    parser = argparse.ArgumentParser(description="Phase 14 Factorial Ablation Runner")
    parser.add_argument("--variants", nargs="+", default=["V00", "V10", "V01", "V11"],
                        help="Variants to run (default: all four)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Pass --dry_run to pipeline (no inference)")
    args = parser.parse_args()

    results = []
    for variant in args.variants:
        if variant not in VARIANTS:
            print(f"Unknown variant: {variant}")
            continue
        status = run_variant(variant, dry_run=args.dry_run)
        results.append(status)

    # Save manifest
    manifest = {
        "phase": 14,
        "variants": results,
        "total_variants": len(results),
        "successful": sum(1 for r in results if r["returncode"] == 0),
    }
    manifest_path = PHASE14_DIR / "Phase14_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nManifest saved: {manifest_path}")

    # Summary
    print(f"\n{'='*60}")
    print("Phase 14 Summary")
    print(f"{'='*60}")
    for r in results:
        status_str = "PASS" if r["returncode"] == 0 else "FAIL"
        print(f"  {r['variant']} ({r['label']}): {status_str} ({r['elapsed_sec']}s)")


if __name__ == "__main__":
    main()
