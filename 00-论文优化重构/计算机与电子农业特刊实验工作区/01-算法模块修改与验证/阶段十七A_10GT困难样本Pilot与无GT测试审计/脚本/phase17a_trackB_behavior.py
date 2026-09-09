#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 17A — Step 7: Track B behavioral audit (32 locked TEST frames, NO GT).

ABSOLUTE CONSTRAINT (user protocol + test T8): Track B is NOT an accuracy
evaluation. There is NO GT for TEST frames; no F1/IoU/Precision/Recall vs GT
columns are ever computed. Audit dimensions:

  1. Execution / coverage — does every variant produce outputs for all 32
     frames? Empty-mask count, failure rows (失败汇总.json), per-run timing.
  2. Behavioral divergence vs U00 (Control) — per U-variant per frame:
       mask_change_pixels / mask_change_ratio / prediction IoU(control, var)
       bitwise_noop (identical to control)
  3. Sample-level exposure — per TEST sample (8×4), how many frames ACTED
     (change_ratio > threshold) per variant.
  4. Stability — distribution of change_ratio (min/median/max), frames with
     ratio > large-flip threshold flagged (behavior observation only).

Outputs (05_behavior/):
  trackB_execution.csv        — per variant: frames, empty, failures, wall-time
  trackB_behavior.csv         — per frame×variant behavioral deltas
  trackB_exposure.csv         — per sample × variant acted/noop counts
  trackB_stability.csv        — change_ratio distribution + large-flip flags
  trackB_behavior_summary.json— metadata (no_gt=true)
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np

PHASE17A = Path(__file__).resolve().parent.parent
VARIANT_DIRS = PHASE17A / "03_trackB_variants"
INPUT_DIR = PHASE17A / "01_input_trackB"
OUT_DIR = PHASE17A / "05_behavior"
LOGS = PHASE17A / "logs"
RUNS_MANIFEST = PHASE17A / "Phase17A_runs_manifest.json"

VARIANTS = ["U00", "U10", "U01", "U11"]
CONTROL = "U00"
VARIANT_DIRS_MAP = {"U00": "U00_Control", "U10": "U10_A6", "U01": "U01_A7", "U11": "U11_A6A7"}

ACTED_RATIO = 1e-4      # change_ratio above this → the mechanism "acted" on this frame
LARGE_FLIP_RATIO = 0.5  # change_ratio above this → "large flip" (aggressive change)


def load_mask(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    return img > 127


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(a, b).sum() / union)


def read_execution() -> dict[str, dict]:
    """Per-variant execution info from runs manifest + 失败汇总.json + mask counts."""
    runs = {}
    if RUNS_MANIFEST.exists():
        manifest = json.loads(RUNS_MANIFEST.read_text(encoding="utf-8"))
        for r in manifest.get("runs", []):
            runs[r["variant"]] = r
    out = {}
    for v in VARIANTS:
        od = VARIANT_DIRS / VARIANT_DIRS_MAP[v]
        final_dir = od / "最终掩膜"
        n_final = len(list(final_dir.glob("mask_*.png"))) if final_dir.exists() else 0
        fail = {}
        fp = od / "失败汇总.json"
        if fp.exists():
            fail = json.loads(fp.read_text(encoding="utf-8"))
        out[v] = {
            "final_masks": n_final,
            "empty_candidate_count": fail.get("empty_candidate_count", ""),
            "zero_final_mask_count": fail.get("zero_final_mask_count", ""),
            "spnp_rejected_count": fail.get("spnp_rejected_count", ""),
            "zero_final_masks_list": fail.get("zero_final_masks", []),
            "status": runs.get(v, {}).get("status", "n/a"),
            "elapsed_sec": runs.get(v, {}).get("elapsed_sec", ""),
        }
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # frame inventory from input dir (exactly the 32 TEST frames)
    frames = sorted(p.stem for p in INPUT_DIR.glob("*.jpg"))
    assert len(frames) == 32, f"expected 32 TEST frames, found {len(frames)}"

    # ---- per-frame behavioral deltas vs U00 ----
    behavior_rows = []
    for f in frames:
        ctrl = load_mask(VARIANT_DIRS / VARIANT_DIRS_MAP[CONTROL] / "最终掩膜" / f"mask_{f}.png")
        sample = f.rsplit("_", 1)[0]
        for v in VARIANTS:
            r = {"sample": sample, "frame": f, "variant": v}
            if v == CONTROL:
                # control-self reference
                if ctrl is not None:
                    h, w = ctrl.shape
                    r["mask_change_pixels"] = 0
                    r["mask_change_ratio"] = 0.0
                    r["prediction_iou_vs_u00"] = 1.0
                    r["bitwise_noop"] = 1
                    r["mask_empty"] = int(not ctrl.any())
                else:
                    r["mask_change_pixels"] = ""; r["mask_change_ratio"] = ""
                    r["prediction_iou_vs_u00"] = ""; r["bitwise_noop"] = ""
                    r["mask_empty"] = ""
            else:
                var_mask = load_mask(VARIANT_DIRS / VARIANT_DIRS_MAP[v] / "最终掩膜" / f"mask_{f}.png")
                if ctrl is None or var_mask is None:
                    r["mask_change_pixels"] = ""; r["mask_change_ratio"] = ""
                    r["prediction_iou_vs_u00"] = ""; r["bitwise_noop"] = ""
                    r["mask_empty"] = ("NA(missing)" if var_mask is None else "")
                else:
                    h, w = ctrl.shape
                    px = int(np.logical_xor(ctrl, var_mask).sum())
                    r["mask_change_pixels"] = px
                    r["mask_change_ratio"] = round(px / (h * w), 8)
                    r["prediction_iou_vs_u00"] = round(mask_iou(ctrl, var_mask), 6)
                    r["bitwise_noop"] = int(px == 0)
                    r["mask_empty"] = int((not ctrl.any()) and (not var_mask.any()))
            behavior_rows.append(r)

    cols = ["sample", "frame", "variant", "mask_change_pixels", "mask_change_ratio",
            "prediction_iou_vs_u00", "bitwise_noop", "mask_empty"]
    with (OUT_DIR / "trackB_behavior.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(behavior_rows)

    # ---- sample-level exposure ----
    exposure_rows = []
    for sample in sorted({r["sample"] for r in behavior_rows}):
        for v in VARIANTS:
            vr = [r for r in behavior_rows if r["sample"] == sample and r["variant"] == v]
            acted = sum(True for r in vr if isinstance(r["mask_change_ratio"], float) and r["mask_change_ratio"] > ACTED_RATIO and r["variant"] != CONTROL)
            noop = sum(True for r in vr if r["variant"] != CONTROL and r["bitwise_noop"] == 1)
            large = sum(True for r in vr if isinstance(r["mask_change_ratio"], float) and r["mask_change_ratio"] > LARGE_FLIP_RATIO and r["variant"] != CONTROL)
            exposure_rows.append({
                "sample": sample, "variant": v, "n_frames": len(vr),
                "acted": acted, "noop": noop, "large_flip": large,
            })
    with (OUT_DIR / "trackB_exposure.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["sample", "variant", "n_frames", "acted", "noop", "large_flip"])
        w.writeheader()
        w.writerows(exposure_rows)

    # ---- stability ----
    stability_rows = []
    for v in VARIANTS:
        vr = [r for r in behavior_rows if r["variant"] == v and isinstance(r["mask_change_ratio"], (int, float))]
        ratios = [r["mask_change_ratio"] for r in vr]
        large = [r["frame"] for r in vr if r["mask_change_ratio"] > LARGE_FLIP_RATIO and v != CONTROL]
        n_empty = sum(1 for r in vr if r.get("mask_empty") == 1)
        if ratios:
            stability_rows.append({
                "variant": v, "n": len(vr),
                "change_ratio_min": round(float(min(ratios)), 8),
                "change_ratio_median": round(float(np.median(ratios)), 8),
                "change_ratio_mean": round(float(np.mean(ratios)), 8),
                "change_ratio_max": round(float(max(ratios)), 8),
                "large_flip_count": len(large), "large_flip_frames": ";".join(large),
                "empty_mask_count": n_empty,
            })
    with (OUT_DIR / "trackB_stability.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[k for k in stability_rows[0].keys() if k != "large_flip_frames"] + ["large_flip_frames"])
        w.writeheader()
        w.writerows(stability_rows)

    # ---- execution ----
    exec_info = read_execution()
    exec_rows = [{"variant": v, **exec_info[v]} for v in VARIANTS]
    with (OUT_DIR / "trackB_execution.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["variant"] + list(exec_rows[0].keys())[1:] if exec_rows else ["variant"])
        w.writeheader()
        w.writerows(exec_rows)

    # ---- summary json ----
    acted_by_variant = {}
    for v in VARIANTS:
        vr = [r for r in behavior_rows if r["variant"] == v and v != CONTROL]
        acted = sum(1 for r in vr if isinstance(r["mask_change_ratio"], float) and r["mask_change_ratio"] > ACTED_RATIO)
        noop = sum(1 for r in vr if r["bitwise_noop"] == 1)
        empty = sum(1 for r in vr if r.get("mask_empty") == 1)
        acted_by_variant[v] = {"n": len(vr), "acted": acted, "noop": noop, "empty_masks": empty}

    summary = {
        "no_gt": True,
        "no_gt_note": ("Track B = 32 locked TEST frames with NO human GT. This file "
                       "contains ONLY behavioral/execution statistics vs the U00 control. "
                       "No accuracy metric is or can be reported."),
        "n_frames": len(frames),
        "n_samples": len({r["sample"] for r in behavior_rows}),
        "acted_ratio_threshold": ACTED_RATIO,
        "large_flip_ratio_threshold": LARGE_FLIP_RATIO,
        "per_variant": acted_by_variant,
        "stability": {r["variant"]: {
            "median_change_ratio": r["change_ratio_median"],
            "max_change_ratio": r["change_ratio_max"],
            "large_flip_count": r["large_flip_count"],
            "empty_mask_count": r["empty_mask_count"],
        } for r in stability_rows},
        "execution": {r["variant"]: {
            "status": r["status"], "elapsed_sec": r["elapsed_sec"],
            "final_masks": r["final_masks"], "zero_final_masks": r["zero_final_mask_count"],
        } for r in exec_rows},
    }
    (OUT_DIR / "trackB_behavior_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # console
    print("=== Track B behavioral audit (NO GT) ===")
    print(f"  frames: {len(frames)} | samples: {summary['n_samples']} | no_gt=True")
    for v, e in acted_by_variant.items():
        print(f"  {v}: acted={e['acted']}/{e['n']} noop={e['noop']} empty={e['empty_masks']}")
    print(f"\nSaved → 05_behavior/")


if __name__ == "__main__":
    main()