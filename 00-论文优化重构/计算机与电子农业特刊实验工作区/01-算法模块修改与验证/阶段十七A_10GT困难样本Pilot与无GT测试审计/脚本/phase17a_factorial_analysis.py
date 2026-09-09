#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 17A — Step 6: Track A factorial + mechanism analysis (GT10 only).

Descriptive 2×2 factorial effects (N=10, 2 sequences — descriptive only,
NO significance claim; within-sequence dependence acknowledged):
    E_A6 = (M10 + M11 − M00 − M01) / 2
    E_A7 = (M01 + M11 − M00 − M10) / 2
    I_AB = M11 − M10 − M01 + M00
M = F1 per frame.

Mechanism (does A6/A7 actually act, and does acting align with GT outcomes?):
  - A6 per-frame accept/fallback from 共识投票.csv (P10/P11)
  - A7 per-sample seed + propagation from 日志/记忆传播.json (P01/P11)
  - mask-level delta vs P00: change_pixels / change_ratio / prediction IoU
  - A6 "bitwise no-op" count (accepted but identical to Control) — Phase 15 pattern
  - correlation of "acted" frames with GT outcomes (ΔF1>0 / <0), descriptive

Outputs (06_mechanism/):
  trackA_factorial.csv      — per-frame E_A6/E_A7/I_AB + per-variant F1
  trackA_mechanism.csv      — per-frame mechanism columns (A6/A7 action vs GT delta)
  trackA_factorial_summary.json
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np

PHASE17A = Path(__file__).resolve().parent.parent
VARIANTS_DIR = PHASE17A / "02_trackA_variants"
METRICS = PHASE17A / "04_metrics"
OUT_DIR = PHASE17A / "06_mechanism"

VARIANTS = ["P00", "P10", "P01", "P11"]
CONTROL = "P00"
VARIANT_DIRS = {"P00": "P00_Control", "P10": "P10_A6", "P01": "P01_A7", "P11": "P11_A6A7"}
A6_VARIANTS = ["P10", "P11"]
A7_VARIANTS = ["P01", "P11"]

ENGAGEMENT_RATIO = 1e-4  # mask_change_ratio above this → "acted"


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


def read_per_frame_metrics() -> dict[tuple[str, str, str], dict]:
    """{(sample, frame, variant): row} from trackA_per_frame.csv."""
    out = {}
    with (METRICS / "trackA_per_frame.csv").open(encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            out[(r["sample"], r["frame"], r["variant"])] = r
    return out


def read_a6_consensus(variant: str) -> dict[tuple[str, str], dict]:
    """Per-frame A6 info from 共识投票.csv (image column like 'Sample_0000.png')."""
    out = {}
    p = VARIANTS_DIR / VARIANT_DIRS[variant] / "共识投票.csv"
    if not p.exists():
        return out
    with p.open(encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            img = r.get("图像", "")
            stem = img.rsplit(".", 1)[0]
            parts = stem.rsplit("_", 1)
            if len(parts) != 2:
                continue
            sample, frame = parts
            out[(sample, frame)] = {
                "a6_enabled": r.get("共识启用", ""),
                "a6_accept": r.get("共识接受", ""),
                "a6_fallback_iou": r.get("回退IoU", ""),
                "a6_del_px_ratio": r.get("删除像素比例", ""),
                "a6_add_px_ratio": r.get("补回像素比例", ""),
            }
    return out


def read_a7_memory(variant: str) -> dict[str, dict]:
    """Per-sample A7 info from 日志/记忆传播.json 汇总.samples."""
    out = {}
    p = VARIANTS_DIR / VARIANT_DIRS[variant] / "日志/记忆传播.json"
    if not p.exists():
        return out
    data = json.loads(p.read_text(encoding="utf-8"))
    samples = data.get("汇总", {}).get("samples", {})
    for sname, sinfo in samples.items():
        out[sname] = {
            "a7_seed_stem": sinfo.get("seed_stem", ""),
            "a7_frames": sinfo.get("帧数", ""),
            "a7_propagated": sinfo.get("传播掩膜数", ""),
            "a7_status": sinfo.get("状态", ""),
        }
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    mets = read_per_frame_metrics()
    a6_p10 = read_a6_consensus("P10")
    a6_p11 = read_a6_consensus("P11")
    a7_p01 = read_a7_memory("P01")
    a7_p11 = read_a7_memory("P11")

    # GT10 frame list (union of per-frame metric keys)
    frames = sorted({(s, f) for (s, f, v) in mets})

    # ---- factorial effects per frame ----
    fact_rows = []
    for s, f in frames:
        m = {v: float(mets.get((s, f, v), {}).get("f1", float("nan"))) for v in VARIANTS}
        if any(np.isnan(m[v]) for v in VARIANTS):
            continue
        e_a6 = (m["P10"] + m["P11"] - m["P00"] - m["P01"]) / 2
        e_a7 = (m["P01"] + m["P11"] - m["P00"] - m["P10"]) / 2
        i_ab = m["P11"] - m["P10"] - m["P01"] + m["P00"]
        fact_rows.append({
            "sample": s, "frame": f,
            "f1_P00": round(m["P00"], 6), "f1_P10": round(m["P10"], 6),
            "f1_P01": round(m["P01"], 6), "f1_P11": round(m["P11"], 6),
            "E_A6": round(e_a6, 6), "E_A7": round(e_a7, 6), "I_AB": round(i_ab, 6),
        })

    with (OUT_DIR / "trackA_factorial.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fact_rows[0].keys()))
        w.writeheader()
        w.writerows(fact_rows)

    # descriptive summary (frame level + per-sample level)
    def _agg(key: str):
        vals = [r[key] for r in fact_rows]
        return {"mean": round(float(np.mean(vals)), 6),
                "median": round(float(np.median(vals)), 6),
                "min": round(float(min(vals)), 6),
                "max": round(float(max(vals)), 6),
                "n": len(vals)}

    per_sample = {}
    for s in sorted({r["sample"] for r in fact_rows}):
        sr = [r for r in fact_rows if r["sample"] == s]
        per_sample[s] = {k: round(float(np.mean([r[k] for r in sr])), 6)
                         for k in ("E_A6", "E_A7", "I_AB")}

    # ---- mechanism per frame ----
    mech_rows = []
    for s, f in frames:
        ctrl = load_mask(VARIANTS_DIR / VARIANT_DIRS[CONTROL] / "最终掩膜" / f"mask_{s}_{f}.png")
        row = {"sample": s, "frame": f}
        for v in A6_VARIANTS:
            info = (a6_p10 if v == "P10" else a6_p11).get((s, f), {})
            row[f"{v}_a6_accept"] = info.get("a6_accept", "na")
            row[f"{v}_a6_fallback_iou"] = info.get("a6_fallback_iou", "")
        for v in A7_VARIANTS:
            sinfo = (a7_p01 if v == "P01" else a7_p11).get(s, {})
            row[f"{v}_a7_seed"] = sinfo.get("a7_seed_stem", "")
            row[f"{v}_a7_propagated"] = sinfo.get("a7_propagated", "")
        for v in A6_VARIANTS + A7_VARIANTS:
            var_mask = load_mask(VARIANTS_DIR / VARIANT_DIRS[v] / "最终掩膜" / f"mask_{s}_{f}.png")
            if ctrl is not None and var_mask is not None:
                change_px = int(np.logical_xor(ctrl, var_mask).sum())
                h, w = ctrl.shape
                ratio = change_px / (h * w)
                noop = bool(change_px == 0)
                row[f"{v}_change_px"] = change_px
                row[f"{v}_change_ratio"] = round(ratio, 6)
                row[f"{v}_noop"] = int(noop)
                row[f"{v}_pred_iou_vs_p00"] = round(mask_iou(ctrl, var_mask), 6)
                # GT outcome for this variant
                mt = mets.get((s, f, v), {})
                c = mets.get((s, f, CONTROL), {})
                try:
                    d = float(mt.get("f1", "nan")) - float(c.get("f1", "nan"))
                    row[f"{v}_delta_f1"] = round(d, 6)
                except (ValueError, TypeError):
                    row[f"{v}_delta_f1"] = ""
        mech_rows.append(row)

    with (OUT_DIR / "trackA_mechanism.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(mech_rows[0].keys()), extrasaction="ignore")
        w.writeheader()
        w.writerows(mech_rows)

    # ---- engagement × GT-outcome counts (descriptive, A6/A7) ----
    engaged_outcome = {}
    for v in A6_VARIANTS + A7_VARIANTS:
        acted_up = acted_down = acted_neutral = 0
        not_acted_up = not_acted_down = not_acted_neutral = 0
        acted_noop = 0
        for r in mech_rows:
            delta = r.get(f"{v}_delta_f1")
            if delta == "":
                continue
            ratio = r.get(f"{v}_change_ratio", 0)
            noop = r.get(f"{v}_noop", 0)
            acted = (noop == 0) and (ratio is not None and ratio > ENGAGEMENT_RATIO)
            if acted and noop:
                acted_noop += 1
            if delta > 0.005:
                if acted: acted_up += 1
                else: not_acted_up += 1
            elif delta < -0.005:
                if acted: acted_down += 1
                else: not_acted_down += 1
            else:
                if acted: acted_neutral += 1
                else: not_acted_neutral += 1
        engaged_outcome[v] = {
            "acted": {"f1_up": acted_up, "f1_down": acted_down, "neutral": acted_neutral},
            "not_acted": {"f1_up": not_acted_up, "f1_down": not_acted_down, "neutral": not_acted_neutral},
            "noop_count": sum(1 for r in mech_rows if r.get(f"{v}_noop") == 1),
            "acted_count": sum(1 for r in mech_rows if (r.get(f"{v}_noop") == 0) and (r.get(f"{v}_change_ratio") is not None and r.get(f"{v}_change_ratio") > ENGAGEMENT_RATIO)),
        }

    summary = {
        "n_frames": len(fact_rows),
        "factorial_frame_level": {k: _agg(k) for k in ("E_A6", "E_A7", "I_AB")},
        "factorial_per_sample_mean": per_sample,
        "statistical_caveat": ("N=10, 2 sequences, adjacent frames within each sequence — "
                               "dependent observations. Descriptive only; NO significance claim."),
        "engagement_outcome": engaged_outcome,
        "engagement_ratio_threshold": ENGAGEMENT_RATIO,
        "bitwise_noop_note": "accepted-but-identical-to-control frames are counted as noop.",
    }
    (OUT_DIR / "trackA_factorial_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # console
    print("=== Track A factorial + mechanism ===")
    print(f"  frames: {summary['n_frames']}")
    for k in ("E_A6", "E_A7", "I_AB"):
        a = summary["factorial_frame_level"][k]
        print(f"  {k}: mean={a['mean']} median={a['median']} min={a['min']} max={a['max']}")
    print(f"  per-sample means: {json.dumps(per_sample, ensure_ascii=False)}")
    for v in A6_VARIANTS + A7_VARIANTS:
        e = engaged_outcome[v]
        print(f"  {v}: acted={e['acted']} noop={e['noop_count']} not_acted={e['not_acted']}")
    print(f"\nSaved → 06_mechanism/")


if __name__ == "__main__":
    main()