#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 15 — Zero-compute per-stratum factorial analysis.

Reuses Phase 14/14.1 artifacts (no new inference):
  - V00_Control, V10_A6 (Phase 14), V01_A7_rerun, V11_A6A7_rerun (Phase 14.1)
  - GT_potted_clean (Phase 12)

Computes per-frame binary F1 for each variant, merges with the difficulty
stratification CSV (21易帧难度分层.csv), then reports per-stratum:
  - F1 mean/median/min/max per variant
  - E_A6, E_A7, I_AB (factorial, mean F1)
  - Paired ΔF1 (V10-V00, V01-V00, V11-V00)
  - Trigger-conditioned effects (consensus_accepted from 共识投票.csv,
    记忆候选采用 from 提示词选择.csv)
  - Difficulty correlation (Spearman: difficulty score vs ΔF1)

Outputs: 挑战集列表/phase15_per_stratum_metrics.csv and prints a summary.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
PHASE15 = SCRIPT_DIR.parent
PHASE14 = Path("/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区"
               "/01-算法模块修改与验证/阶段十四_A6A7真实数据析因消融")
PHASE14_1 = Path("/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区"
                 "/01-算法模块修改与验证/阶段十四点一_A7有效性封口")
GT_BASE = Path("/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区"
               "/01-算法模块修改与验证/阶段十二_GT_v2_QA与P6正式验收/GT_potted_clean")

VARIANTS = {
    "V00": PHASE14 / "V00_Control" / "选择后掩膜",
    "V10": PHASE14 / "V10_A6" / "选择后掩膜",
    "V01": PHASE14_1 / "V01_A7_rerun" / "选择后掩膜",
    "V11": PHASE14_1 / "V11_A6A7_rerun" / "选择后掩膜",
}


def compute_f1(gt_arr: np.ndarray, pred_arr: np.ndarray) -> float:
    tp = int((gt_arr & pred_arr).sum())
    fp = int((~gt_arr & pred_arr).sum())
    fn = int((gt_arr & ~pred_arr).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0


def load_mask(path: Path, shape: tuple[int, int]) -> np.ndarray:
    arr = np.array(Image.open(path).convert("L"))
    if arr.shape != shape:
        arr = np.array(Image.open(path).convert("L").resize((shape[1], shape[0])))
    return arr > 128


def read_trigger(variant_dir: Path, col: str, key: str = "共识接受") -> dict[str, str]:
    """Read a trigger column from 提示词选择.csv / 共识投票.csv keyed by stem."""
    out: dict[str, str] = {}
    for name, stem_k in [("提示词选择.csv", "图像"), ("共识投票.csv", "图像")]:
        p = variant_dir / name
        if not p.exists():
            continue
        with open(p, encoding="utf-8-sig") as f:
            rd = csv.DictReader(f)
            if col not in rd.fieldnames:
                continue
            for row in rd:
                img = row.get(stem_k, "")
                stem = img.replace(".jpg", "").replace(".png", "")
                out[stem] = row.get(col, "")
    return out


def spearman(x: list[float], y: list[float]) -> tuple[float, float]:
    """Return (rho, p_value) — crude two-sided Spearman."""
    n = len(x)
    if n < 4:
        return (0.0, 1.0)
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    d = rx - ry
    rho = 1.0 - 6.0 * (d ** 2).sum() / (n * (n * n - 1))
    if abs(rho) >= 1.0:
        return (rho, 0.0)
    t = rho * np.sqrt((n - 2) / (1 - rho * rho))
    from scipy import stats
    p = 2 * (1 - stats.t.cdf(abs(t), n - 2))
    return (float(rho), float(p))


def main() -> None:
    # --- load stratification ---
    strat_path = PHASE15 / "挑战集列表" / "21易帧难度分层.csv"
    strata: dict[str, str] = {}
    diff_score: dict[str, float] = {}
    is_h1: dict[str, bool] = {}
    with open(strat_path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            strata[row["stem"]] = row["stratum"]
            diff_score[row["stem"]] = float(row["diff"])
            is_h1[row["stem"]] = row["is_h1_failure"] == "True"

    # --- per-frame F1 ---
    f1: dict[str, dict[str, float]] = {v: {} for v in VARIANTS}
    gt_shape: tuple[int, int] | None = None
    frame_meta: list[dict] = []

    # iterate via V00 mask filenames
    for pred_file in sorted((VARIANTS["V00"]).glob("mask_*.png")):
        parts = pred_file.stem.replace("mask_", "").rsplit("_", 1)
        if len(parts) != 2:
            continue
        sample, frame = parts
        stem = f"{sample}_{frame}"
        gt_file = GT_BASE / sample / f"mask_potted_clean_{frame}.png"
        if not gt_file.exists() or stem not in strata:
            continue
        gt = np.array(Image.open(gt_file).convert("L")) > 128
        gt_shape = gt.shape
        row = {"stem": stem, "sample": sample, "frame": frame,
               "stratum": strata[stem], "diff": diff_score[stem],
               "is_h1": is_h1[stem]}
        for v, vdir in VARIANTS.items():
            pf = vdir / pred_file.name
            if not pf.exists():
                row[v] = ""
                continue
            pred = load_mask(pf, gt.shape)
            f1[v][stem] = compute_f1(gt, pred)
            row[v] = f"{f1[v][stem]:.6f}"
        frame_meta.append(row)

    n = len(frame_meta)
    print(f"Frames analyzed: {n}\n")

    # --- trigger columns (CSVs live in variant ROOT, not 选择后掩膜/) ---
    root_v10 = VARIANTS["V10"].parent
    root_v01 = VARIANTS["V01"].parent
    root_v11 = VARIANTS["V11"].parent
    consensus = read_trigger(root_v10, "共识接受", "共识接受")   # 0/1
    mem_seed_v01 = read_trigger(root_v01, "记忆种子帧")
    mem_sel_v01 = read_trigger(root_v01, "记忆候选采用")       # 0/1
    mem_sel_v11 = read_trigger(root_v11, "记忆候选采用")

    for row in frame_meta:
        s = row["stem"]
        row["consensus_accepted"] = consensus.get(s, "")
        row["mem_seed_v01"] = mem_seed_v01.get(s, "")
        row["mem_sel_v01"] = mem_sel_v01.get(s, "")
        row["mem_sel_v11"] = mem_sel_v11.get(s, "")

    # --- write merged CSV ---
    out_fields = ["stem", "sample", "frame", "stratum", "diff", "is_h1",
                  "V00", "V10", "V01", "V11",
                  "d_V10", "d_V01", "d_V11",
                  "consensus_accepted", "mem_seed_v01", "mem_sel_v01", "mem_sel_v11"]
    out_csv = PHASE15 / "挑战集列表" / "phase15_per_frame_f1.csv"
    with out_csv.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=out_fields, extrasaction="ignore")
        w.writeheader()
        for row in sorted(frame_meta, key=lambda r: -r["diff"]):
            s = row["stem"]
            if s in f1["V00"] and s in f1["V10"]:
                row["d_V10"] = f"{f1['V10'][s] - f1['V00'][s]:+.6f}"
            if s in f1["V00"] and s in f1["V01"]:
                row["d_V01"] = f"{f1['V01'][s] - f1['V00'][s]:+.6f}"
            if s in f1["V00"] and s in f1["V11"]:
                row["d_V11"] = f"{f1['V11'][s] - f1['V00'][s]:+.6f}"
            w.writerow(row)
    print(f"Saved → {out_csv}\n")

    # --- per-stratum summary ---
    print("=== Per-stratum F1 (mean) ===")
    summary_rows = []
    for st in ["hard", "mid", "easy"]:
        stems = [r["stem"] for r in frame_meta if r["stratum"] == st]
        if not stems:
            continue
        line = {"stratum": st, "n": len(stems)}
        vals = {}
        for v in VARIANTS:
            vv = [f1[v][s] for s in stems if s in f1[v]]
            if vv:
                vals[v] = float(np.mean(vv))
                line[v] = f"{vals[v]:.4f}"
                line[f"{v}_min"] = f"{min(vv):.4f}"
                line[f"{v}_max"] = f"{max(vv):.4f}"
        print(f"  {st} (n={len(stems)}): " + "  ".join(f"{v}={line[v]}" for v in VARIANTS if v in line))
        # deltas
        for v in ["V10", "V01", "V11"]:
            if v in vals and "V00" in vals:
                line[f"d_{v}"] = f"{vals[v] - vals['V00']:+.4f}"
        # factorial (only if all 4 present)
        if all(v in vals for v in ["V00", "V10", "V01", "V11"]):
            e_a6 = (vals["V10"] + vals["V11"] - vals["V00"] - vals["V01"]) / 2
            e_a7 = (vals["V01"] + vals["V11"] - vals["V00"] - vals["V10"]) / 2
            i_ab = vals["V11"] - vals["V10"] - vals["V01"] + vals["V00"]
            line["E_A6"] = f"{e_a6:+.4f}"
            line["E_A7"] = f"{e_a7:+.4f}"
            line["I_AB"] = f"{i_ab:+.4f}"
            print(f"      E_A6={line['E_A6']}  E_A7={line['E_A7']}  I_AB={line['I_AB']}")
        summary_rows.append(line)

    # overall
    line = {"stratum": "all", "n": n}
    vals = {}
    for v in VARIANTS:
        vv = [f1[v][s] for s in strata if s in f1[v]]
        if vv:
            vals[v] = float(np.mean(vv))
            line[v] = f"{vals[v]:.4f}"
    for v in ["V10", "V01", "V11"]:
        if v in vals and "V00" in vals:
            line[f"d_{v}"] = f"{vals[v] - vals['V00']:+.4f}"
    if all(v in vals for v in ["V00", "V10", "V01", "V11"]):
        line["E_A6"] = f"{(vals['V10'] + vals['V11'] - vals['V00'] - vals['V01']) / 2:+.4f}"
        line["E_A7"] = f"{(vals['V01'] + vals['V11'] - vals['V00'] - vals['V10']) / 2:+.4f}"
        line["I_AB"] = f"{vals['V11'] - vals['V10'] - vals['V01'] + vals['V00']:+.4f}"
    summary_rows.append(line)
    print(f"  all (n={n}): " + "  ".join(f"{v}={line[v]}" for v in VARIANTS if v in line))
    if "E_A6" in line:
        print(f"      E_A6={line['E_A6']}  E_A7={line['E_A7']}  I_AB={line['I_AB']}")

    sum_csv = PHASE15 / "挑战集列表" / "phase15_per_stratum_metrics.csv"
    with sum_csv.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["stratum", "n", "V00", "V10", "V01", "V11",
                                          "V00_min", "V10_min", "V01_min", "V11_min",
                                          "V00_max", "V10_max", "V01_max", "V11_max",
                                          "d_V10", "d_V01", "d_V11", "E_A6", "E_A7", "I_AB"],
                           extrasaction="ignore")
        w.writeheader()
        for r in summary_rows:
            w.writerow(r)
    print(f"Saved → {sum_csv}\n")

    # --- material regression / improvement counts per stratum ---
    print("=== Material changes (|ΔF1| > 0.005) per stratum ===")
    for v, label in [("V10", "A6"), ("V01", "A7"), ("V11", "A6+A7")]:
        for st in ["hard", "mid", "easy"]:
            stems = [r["stem"] for r in frame_meta if r["stratum"] == st]
            reg = [(s, f1["V00"][s], f1[v][s]) for s in stems
                   if s in f1["V00"] and s in f1[v] and f1[v][s] - f1["V00"][s] < -0.005]
            imp = [(s, f1["V00"][s], f1[v][s]) for s in stems
                   if s in f1["V00"] and s in f1[v] and f1[v][s] - f1["V00"][s] > 0.005]
            print(f"  {label} {st}: reg={len(reg)} imp={len(imp)}"
                  + (f"  REG {reg[0][0]} {reg[0][1]:.4f}->{reg[0][2]:.4f}" if reg else ""))

    # --- trigger-conditioned ---
    print("\n=== Trigger-conditioned mean ΔF1 (V10 = A6) ===")
    for st in ["hard", "mid", "easy"]:
        stems = [r["stem"] for r in frame_meta if r["stratum"] == st]
        on = [f1["V10"][s] - f1["V00"][s] for s in stems
              if consensus.get(s) == "1" and s in f1["V10"] and s in f1["V00"]]
        off = [f1["V10"][s] - f1["V00"][s] for s in stems
               if consensus.get(s) == "0" and s in f1["V10"] and s in f1["V00"]]
        n_on = sum(1 for s in stems if consensus.get(s) == "1")
        print(f"  {st}: A6-consensus accepted n={n_on}, "
              f"mean ΔF1 on={np.mean(on) if on else float('nan'):+.4f}, "
              f"off={np.mean(off) if off else float('nan'):+.4f}")

    print("\n=== Trigger-conditioned mean ΔF1 (V01 = A7) ===")
    for st in ["hard", "mid", "easy"]:
        stems = [r["stem"] for r in frame_meta if r["stratum"] == st]
        on = [f1["V01"][s] - f1["V00"][s] for s in stems
              if mem_sel_v01.get(s) == "1" and s in f1["V01"] and s in f1["V00"]]
        off = [f1["V01"][s] - f1["V00"][s] for s in stems
               if mem_sel_v01.get(s) == "0" and s in f1["V01"] and s in f1["V00"]]
        n_on = sum(1 for s in stems if mem_sel_v01.get(s) == "1")
        print(f"  {st}: A7-selected n={n_on}, "
              f"mean ΔF1 on={np.mean(on) if on else float('nan'):+.4f}, "
              f"off={np.mean(off) if off else float('nan'):+.4f}")

    # --- difficulty correlation ---
    print("\n=== Difficulty correlation with ΔF1 (all 21) ===")
    for v, label in [("V10", "A6"), ("V01", "A7"), ("V11", "A6+A7")]:
        xs, ys = [], []
        for s in strata:
            if s in f1["V00"] and s in f1[v]:
                xs.append(diff_score[s])
                ys.append(f1[v][s] - f1["V00"][s])
        rho, p = spearman(xs, ys)
        print(f"  {label}: Spearman(diff, ΔF1) = {rho:+.3f} (p={p:.3f}, n={len(xs)})")

    # --- H1 reference ---
    print("\n=== H1 historical failure frames (reference) ===")
    for s in strata:
        if is_h1[s] and s in f1["V00"]:
            print(f"  {s}: V00={f1['V00'][s]:.4f} V10={f1['V10'][s]:.4f} "
                  f"V01={f1['V01'][s]:.4f} V11={f1['V11'][s]:.4f}")


if __name__ == "__main__":
    main()