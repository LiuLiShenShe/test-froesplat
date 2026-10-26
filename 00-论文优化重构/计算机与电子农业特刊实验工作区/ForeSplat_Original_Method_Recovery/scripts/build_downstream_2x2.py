#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Assemble the downstream 2x2 (mask source x M2M3-GO) table from run records.

Reads, for each tag that exists:
    runs/downstream/KongQueZhuYu_<tag>/run_record.json      (status, timing, command)
    runs/downstream/KongQueZhuYu_<tag>/eval/foreground_test.json  (foreground metrics)
    runs/downstream/KongQueZhuYu_<tag>/mask_provenance.json  (camera/mask hashes)

and writes `metrics/downstream_2x2.csv` + `.json`.  A missing tag is reported as
MISSING rather than silently dropped, so the table never looks more complete
than it is.

Usage:  python scripts/build_downstream_2x2.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path("/data/fj/F2DMAS")
RECOVERY = ROOT / "00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_Original_Method_Recovery"
DOWNSTREAM = RECOVERY / "runs/downstream"
METRICS = RECOVERY / "metrics"

TAGS = ["C_R1_baseline", "RAP_FULL", "RAP_FULL_M2M3GO", "C_R1_M2M3GO"]
MASK_SOURCE = {"C_R1_baseline": "frozen C_R1", "RAP_FULL": "full RAP (210)",
               "RAP_FULL_M2M3GO": "full RAP (210)", "C_R1_M2M3GO": "frozen C_R1"}

METRIC_KEYS = [
    ("psnr_fg", "psnr_fg"),
    ("ssim_fg", "ssim_fg"),
    ("lpips_fg", "lpips_fg"),
    ("outside_mask_non_black_ratio", "outside_leak"),
    ("leakage_energy_ratio", "leakage_energy"),
    ("foreground_fraction", "fg_fraction"),
]


def read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def ply_vertex_count(model_dir: Path, iteration: int = 30000) -> str:
    """Gaussian count straight from the PLY header (no assumptions)."""
    ply = model_dir / f"point_cloud/iteration_{iteration}/point_cloud.ply"
    if not ply.exists():
        return ""
    with ply.open("rb") as handle:
        head = handle.read(2048)
    for line in head.split(b"\n"):
        if line.startswith(b"element vertex"):
            return line.split()[-1].decode()
    return ""


def row_for(tag: str) -> dict:
    run_dir = DOWNSTREAM / f"KongQueZhuYu_{tag}"
    record = read_json(run_dir / "run_record.json") or {}
    eval_json = read_json(run_dir / "eval/foreground_test.json") or {}
    prov = read_json(run_dir / "mask_provenance.json") or {}
    summary = eval_json.get("summary", {})

    row = {
        "tag": tag,
        "mask_source": MASK_SOURCE[tag],
        "m2m3_go": bool(record.get("m2m3_go", "M2M3GO" in tag)),
        "status": record.get("status", "MISSING"),
        "train_rc": record.get("train_returncode", ""),
        "train_wall_sec": record.get("train_wall_sec", ""),
        "eval_rc": record.get("eval_returncode", ""),
        "evaluated_views": summary.get("evaluated_views", ""),
        "mask_cameras": prov.get("camera_count", ""),
        "mask_missing": prov.get("missing_count", ""),
        "gaussians": ply_vertex_count(run_dir),
    }
    for src, dst in METRIC_KEYS:
        value = summary.get(src)
        row[dst] = round(float(value["mean"]), 6) if isinstance(value, dict) else ""
    return row


def main() -> int:
    METRICS.mkdir(exist_ok=True)
    rows = [row_for(tag) for tag in TAGS]
    out_csv = METRICS / "downstream_2x2.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (METRICS / "downstream_2x2.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"{out_csv}")
    for row in rows:
        print(f"  {row['tag']:18s} {row['status']:10s} m2m3={row['m2m3_go']!s:5s} "
              f"psnr_fg={row['psnr_fg']} outside={row['outside_leak']}")
    incomplete = [r["tag"] for r in rows if r["status"] != "COMPLETE"]
    if incomplete:
        print(f"[warn] not COMPLETE yet: {incomplete}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
