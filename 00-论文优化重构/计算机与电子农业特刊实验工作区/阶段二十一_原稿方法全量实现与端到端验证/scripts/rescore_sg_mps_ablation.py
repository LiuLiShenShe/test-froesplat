#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Re-score the SG-MPS ablation from the masks that are already on disk.

The per-configuration runs write their final masks under
``runs/sg_mps_ablation/<config>/最终掩膜/``.  A later single-config invocation of
``run_sg_mps_ablation.py`` overwrote the aggregate CSVs with a one-row summary,
so this script recomputes F1/IoU for *every* configuration directly from those
saved masks and the frozen GT, restoring one complete, hash-traceable table.

It never re-runs the production script and never invents a number: a missing
mask or a shape mismatch is reported and excluded, not imputed.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_sg_mps_ablation import CONFIGS, GT, PHASE, sha256  # noqa: E402

RUN_ROOT = PHASE / "runs/sg_mps_ablation"


def f1_iou(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    tp = float(np.logical_and(pred, gt).sum())
    fp = float(np.logical_and(pred, ~gt).sum())
    fn = float(np.logical_and(~pred, gt).sum())
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
    return f1, iou


def main() -> int:
    rows = []
    missing = []
    for name, _args in CONFIGS:
        out = RUN_ROOT / name / "最终掩膜"
        for gt_path in sorted(GT.glob("mask_*.png")):
            stem = gt_path.stem.replace("mask_", "")
            pred_path = out / f"mask_{stem}.png"
            if not pred_path.exists():
                missing.append(f"{name}:{stem}")
                continue
            pred = np.asarray(Image.open(pred_path).convert("L")) > 0
            gt = np.asarray(Image.open(gt_path).convert("L")) > 0
            if pred.shape != gt.shape:
                missing.append(f"{name}:{stem}:shape")
                continue
            f1, iou = f1_iou(pred, gt)
            rows.append({"config": name, "frame": stem, "f1": f1, "iou": iou,
                         "pred_area": float(pred.mean()), "gt_area": float(gt.mean()),
                         "pred_sha256": sha256(pred_path), "gt_sha256": sha256(gt_path)})

    summary = []
    for name, _ in CONFIGS:
        sub = [r for r in rows if r["config"] == name]
        if not sub:
            continue
        summary.append({"config": name, "frames": len(sub),
                        "f1_mean": float(np.mean([r["f1"] for r in sub])),
                        "f1_min": float(np.min([r["f1"] for r in sub])),
                        "iou_mean": float(np.mean([r["iou"] for r in sub]))})

    metrics = PHASE / "metrics"
    metrics.mkdir(exist_ok=True)
    with (metrics / "sg_mps_ablation_per_frame.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["config", "frame", "f1", "iou", "pred_area",
                                                    "gt_area", "pred_sha256", "gt_sha256"])
        writer.writeheader()
        writer.writerows(rows)
    with (metrics / "sg_mps_ablation_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["config", "frames", "f1_mean", "f1_min", "iou_mean"])
        writer.writeheader()
        writer.writerows(summary)
    (metrics / "sg_mps_ablation_rescore.json").write_text(json.dumps(
        {"source": "runs/sg_mps_ablation/*/最终掩膜",
         "gt_dir": str(GT), "rows": len(rows), "missing": missing,
         "summary": summary}, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    if missing:
        print("MISSING:", missing, file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
