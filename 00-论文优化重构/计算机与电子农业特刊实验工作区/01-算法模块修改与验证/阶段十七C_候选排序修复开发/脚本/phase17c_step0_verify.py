#!/usr/bin/env python3
"""
Phase 17C — Step 0-1: Starting audit, manifest, score reconstruction, data-role freeze.
Zero-inference: reads only frozen candidate artifacts from P00_Control + Phase14 V00_Control.

Run:
  python 脚本/phase17c_step0_verify.py
"""
from __future__ import annotations

import csv
import json
import hashlib
import subprocess
from pathlib import Path

import numpy as np

WORKSPACE = Path("/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/01-算法模块修改与验证")
P00 = WORKSPACE / "阶段十七A_10GT困难样本Pilot与无GT测试审计/02_trackA_variants/P00_Control"
V00 = WORKSPACE / "阶段十四_A6A7真实数据析因消融/V00_Control"
P12 = WORKSPACE / "阶段十二_GT_v2_QA与P6正式验收/P6_raw_baseline"
OUT = WORKSPACE / "阶段十七C_候选排序修复开发"

PHASE17A_SHA = "d656851efd0e687e87238086857bd4d594f60cdd"
PHASE17B_SHA = "26da29e6197ac4d009102a8ee43346c34dc58c55"

# Frozen weights (must match score_weights in P00 参数.json)
WEIGHTS = {"area": 1.0, "comp": 1.0, "edge": 1.0, "temp": 1.0, "contrast": 1.0}
WEIGHTS["sam"] = 0.5  # parsed separately but included in weights dict by pipeline
DENOM = 5.5
Q_COLS = ["面积得分", "连通域得分", "边界得分", "时序得分", "前背景对比得分", "下方泄漏得分", "侧边泄漏得分"]
Q_KEYS = ["area", "comp", "edge", "temp", "contrast", "leak", "side"]


def read_score_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def read_instance_details(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def join_sam_scores(score_rows: list[dict], source_dir: Path) -> dict[str, dict[float, float]]:
    """Return {image_stem: {area_ratio: sam_score}} mapping from 候选评分明细 (per-instance).

    提示词评分.csv's SAM3原始分数 column is the FRAME-LEVEL SAM list (identical for all
    candidate rows of a frame), NOT per-candidate. The authoritative per-instance value is
    候选评分明细.SAM3分数 (S20 L2606: round(item.sam_score,6)), joined positionally by
    area_ratio (Phase 17B pattern).
    """
    out: dict[str, dict[float, float]] = {}
    for csvp in sorted((source_dir / "候选掩膜").glob("候选评分明细_*.csv")):
        stem = csvp.name.replace("候选评分明细_", "").replace(".csv", "")
        m = {}
        with csvp.open(encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                m[float(r["面积比例"])] = float(r["SAM3分数"])
        out[stem] = m
    return out


def reconstruction_check(score_rows: list[dict], source_dir: Path, source_label: str) -> dict:
    """Reconstruct total from 得分 columns + per-instance SAM3分数; compare to recorded 总分.

    base_total = (Σ w_i q_i + w_sam * q_sam) / denom
    q_sam = sigmoid(6*(sam-0.5)), denom = 5.5
    总分 = base_total - penalty  (penalty ∈ {0, 0.25, 0.40} = vertical-coverage + track)
    """
    area_to_sam = join_sam_scores(score_rows, source_dir)

    total = 0
    match = 0
    penalty_count = 0
    anomaly_count = 0
    anomalies = []
    penalty_dist = {}

    for row in score_rows:
        total += 1
        q_values = {k: float(row[c]) for k, c in zip(Q_KEYS, Q_COLS)}
        # per-instance SAM score from 候选评分明细 (join by area_ratio)
        area_ratio = float(row["面积比例"])
        sam_score = area_to_sam.get(row["图像"].rsplit(".", 1)[0], {}).get(area_ratio, 0.0)
        q_sam = 1.0 / (1.0 + np.exp(-6.0 * (sam_score - 0.5)))
        S_full = sum(WEIGHTS.get(k, 0.0) * q_values[k] for k in q_values)
        S_full += WEIGHTS.get("sam", 0.0) * q_sam
        base_total_recon = S_full / DENOM
        recorded_total = float(row["总分"])
        penalty_est = base_total_recon - recorded_total

        # penalty_est = recon − recorded = +0.25 (vertical coverage) / +0.40 (vertical + track)
        is_valid = any(abs(penalty_est - vp) < 1e-6 for vp in [0.0, 0.25, 0.40])
        penalty_dist[f"{penalty_est:+.4f}"] = penalty_dist.get(f"{penalty_est:+.4f}", 0) + 1
        if is_valid:
            match += 1
            if abs(penalty_est) > 0.001:
                penalty_count += 1
        else:
            anomaly_count += 1
            if len(anomalies) < 8:
                anomalies.append(
                    f"{row['图像'].rsplit('.',1)[0]} area={area_ratio:.5f} sam={sam_score:.4f}: "
                    f"recon={base_total_recon:.8f} rec={recorded_total:.8f} pen={penalty_est:.6f}"
                )

    return {
        "source": source_label,
        "n": total,
        "n_match": match,
        "n_penalty": penalty_count,
        "n_anomaly": anomaly_count,
        "penalty_dist": penalty_dist,
        "anomalies": anomalies,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print("=== Phase 17C Step 0-1: Starting Audit + Reconstruction Verification ===\n")

    # --- Git audit ---
    print("--- Git audit ---")
    for cmd, label in [
        (["git", "rev-parse", "HEAD"], "HEAD"),
        (["git", "status", "--short"], "status"),
    ]:
        out = subprocess.check_output(cmd, cwd=WORKSPACE, text=True).strip()
        print(f"  {label}: {out}")
    assert "26da29e6" in subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=WORKSPACE, text=True).strip().lower(), \
        "HEAD != 26da29e6"
    status_out = subprocess.check_output(["git", "status", "--short"], cwd=WORKSPACE, text=True).strip()
    # Only the new (untracked) Phase17C dir is allowed; no modification to tracked files.
    allowed_prefixes = "?? " + str(OUT.stem).split(" ")[0]  # "阶段十七C_候选排序修复开发"
    dirty = [l for l in status_out.splitlines() if l and not l.startswith("?? 阶段十七C_候选排序修复开发")]
    assert not dirty, f"Tracked files modified before Phase17C start: {dirty}"
    print("  HEAD = 26da29e6 ✓, tree clean ✓")

    # --- Manifest ---
    print("\n--- Manifest ---")
    manifest = {
        "phase": "17C",
        "description": "Candidate ranking rescue development (zero-inference counterfactual)",
        "starting_sha": "26da29e6197ac4d009102a8ee43346c34dc58c55",
        "phase17b_sha": "26da29e6197ac4d009102a8ee43346c34dc58c55",
        "phase17a_sha": "d656851efd0e687e87238086857bd4d594f60cdd",
        "phase16_test_manifest_sha256_ref": "7504fcbbc2a495502534a6fbbd27e9a57fb6b24ccefb573291851478d08c0157",
        "no_a6a7": True,
        "no_tuning": True,
        "no_sam3_inference": True,
        "zero_inference": True,
        "denom": DENOM,
        "frozen_weights": "area=1,comp=1,edge=1,temp=1,contrast=1,sam=0.5",
        "dev10_source": str(P00),
        "easy21_source": str(V00),
    }
    (OUT / "Phase17C_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), "utf-8")
    print("  Phase17C_manifest.json written")

    # --- Cross-check artifact counts ---
    print("\n--- Artifact count cross-check ---")
    # DEV10 (P00): 30 frames × varying candidates = 157 total
    p00_score_rows = read_score_rows(P00 / "提示词评分.csv")
    p00_details_all = []
    for csvp in sorted((P00 / "候选掩膜").glob("候选评分明细_*.csv")):
        p00_details_all.extend(read_instance_details(csvp))
    p00_raw_masks = list((P00 / "候选掩膜" / "raw_instance_P6").glob("mask_*.png"))
    print(f"  P00 提示词评分.csv: {len(p00_score_rows)} rows")
    print(f"  P00 候选评分明细 total: {len(p00_details_all)} rows (across 30 files)")
    print(f"  P00 raw_instance_P6: {len(p00_raw_masks)} masks")
    assert len(p00_score_rows) == 157, f"P00 score rows {len(p00_score_rows)} != 157"

    # Easy21 (V00): 21 frames, 25 candidates
    v00_score_rows = read_score_rows(V00 / "提示词评分.csv")
    v00_details_all = []
    for csvp in sorted((V00 / "候选掩膜").glob("候选评分明细_*.csv")):
        v00_details_all.extend(read_instance_details(csvp))
    v00_raw_masks = list((V00 / "候选掩膜" / "raw_instance_P6").glob("mask_*.png"))
    print(f"  V00 提示词评分.csv: {len(v00_score_rows)} rows")
    print(f"  V00 候选评分明细 total: {len(v00_details_all)} rows (across 21 files)")
    print(f"  V00 raw_instance_P6: {len(v00_raw_masks)} masks")
    assert len(v00_score_rows) == 25, f"V00 score rows {len(v00_score_rows)} != 25"
    assert len(v00_details_all) == 25, f"V00 detail rows {len(v00_details_all)} != 25"
    assert len(v00_raw_masks) == 25, f"V00 raw masks {len(v00_raw_masks)} != 25"

    # Phase12 cross-check
    p12_details = []
    for csvp in sorted((P12 / "候选掩膜").glob("候选评分明细_*.csv")):
        p12_details.extend(read_instance_details(csvp))
    p12_raw = list((P12 / "候选掩膜" / "raw_instance_P6").glob("mask_*.png"))
    print(f"  Phase12 P6_raw_baseline 候选评分明细: {len(p12_details)} rows")
    print(f"  Phase12 P6_raw_baseline raw masks: {len(p12_raw)}")
    assert len(p12_details) == 25, f"P12 details {len(p12_details)} != 25"
    assert len(p12_raw) == 25, f"P12 raw masks {len(p12_raw)} != 25"
    print("  Cross-check: Phase14 V00 == Phase12 P6 (25/25) ✓")

    # --- Score reconstruction exactness ---
    print("\n--- Score reconstruction exactness ---")
    p00_check = reconstruction_check(p00_score_rows, P00, "P00_DEV10")
    v00_check = reconstruction_check(v00_score_rows, V00, "V00_Easy21")
    for check in [p00_check, v00_check]:
        print(f"  {check['source']}: {check['n']} rows, "
              f"{check['n_match']}/{check['n']} exact match, "
              f"{check['n_penalty']} with penalty (≥0.25), "
              f"{check['n_anomaly']} anomalies")
        print(f"    penalty distribution: {check['penalty_dist']}")
        if check["anomalies"]:
            for a in check["anomalies"]:
                print(f"    ANOMALY: {a}")
    p00_anomalies = p00_check["n_anomaly"] + v00_check["n_anomaly"]
    print(f"  Total anomalies: {p00_anomalies} (0 expected)")
    assert p00_anomalies == 0, f"Reconstruction anomalies detected: {p00_anomalies}"

    # --- Total candidates ---
    total_candidates = len(p00_score_rows) + len(v00_score_rows)
    print(f"\n--- Total benchmark candidates: {total_candidates} (expect 157+25=182) ---")
    assert total_candidates == 182, f"Total {total_candidates} != 182"

    print("\n=== Step 0-1 PASS ===")


if __name__ == "__main__":
    main()
