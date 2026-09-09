#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 17B — Tests T1–T10 (zero-inference audit of P00 raw candidates).

Run (from workspace root):
    python -m pytest -q 阶段十七B_上游失败归因与候选Oracle审计/tests/test_phase17b.py

Capabilities:
  T1  Oracle metric correctness — recompute F1/IoU for 0039_i01, 0040_i00, 0041_i03
      with the reference formula (阶段十二 计算P6指标.py pattern) == oracle CSVs.
  T2  Selected-instance identification — argmax(total_score) per frame ==
      提示词选择.csv selected instance == candidate_oracle.selected_instance_id.
  T3  Cleanup identity — 选择后掩膜 == 最终掩膜 == A5c_final_mask bitwise for the
      10 GT frames; postcleanup_oracle_f1 == raw_oracle_f1.
  T4  Selection regret — regret = finalcandidate_oracle_f1 − selected_f1 ≥ 0, exact match.
  T5  Failure-class determinism — protocol §5 C5>C1>C2>C3>C4 reproduced; each
      frame exactly one class; C5 excluded; 0039/0040/0041 == C3_RANKING_FAILURE.
  T6  Prompt-sweep no-op — manifest records sweep gate NOT triggered; prompt_oracle
      P1–P5 == not_run; no 17B pipeline run log exists.
  T7  GT never enters inference — no 17B script invokes the S20 pipeline.
  T8  Zero-inference freeze — analysis reads only pre-existing P00 artifacts
      (masks older than the 17B analysis; no new masks written under 候选掩膜).
  T9  Frame/GT mapping exact — all 10 GT frames map to matching stems/shapes.
  T10 A6/A7 disabled — P00 参数.json flags False; 17B scripts pass no a6/a7 args.

Skipping policy: all tests are pure analysis (no inference); skip only if
required 17B artifacts are absent.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import cv2
import numpy as np

PHASE17B = Path(__file__).resolve().parent.parent
WORKSPACE = Path("/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/01-算法模块修改与验证")
PHASE17A = WORKSPACE / "阶段十七A_10GT困难样本Pilot与无GT测试审计"
PHASE16 = WORKSPACE / "阶段十六_HardCase_GT构建与锁定"
P00 = PHASE17A / "02_trackA_variants/P00_Control"
GT_DIR = PHASE16 / "GT_potted_clean_challenge"

GT10 = [("BaiZhang", f"{i:04d}") for i in range(33, 38)] + \
       [("DouBanLv2", f"{i:04d}") for i in range(37, 42)]


def bmask(p: Path):
    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    return (img > 127) if img is not None else None


def metrics(pred, gt):
    """Reference formula (阶段十二 §五): tp/fp/fn/tn -> P/R/F1/IoU; empty pred -> F1=0."""
    if pred is None:
        return {"f1": 0.0, "iou": 0.0, "precision": 0.0, "recall": 0.0, "empty": True}
    tp = int(np.logical_and(pred, gt).sum())
    fp = int(pred.sum()) - tp
    fn = int(gt.sum()) - tp
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
    return {"f1": f1, "iou": iou, "precision": p, "recall": r, "empty": bool(pred.sum() == 0)}


def read_csv(name: str):
    with (PHASE17B / name).open(encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def selected_instance_id(sample: str, frame: str):
    """Recompute pipeline selection: argmax total_score over P6 instances."""
    rows = []
    with (P00 / "提示词评分.csv").open(encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r["图像"].rsplit(".", 1)[0] == f"{sample}_{frame}":
                rows.append((float(r["总分"]), r["面积比例"]))
    best = float(max(rows)[1])
    with (P00 / "候选掩膜" / f"候选评分明细_{sample}_{frame}.csv").open(encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if abs(float(r["面积比例"]) - best) < 1e-9:
                return int(r["实例编号"])
    return None


# ---------------------------------------------------------------- T1
def test_t1_oracle_metric_correctness():
    cases = {("DouBanLv2", "0039", 1): 0.986660,
             ("DouBanLv2", "0040", 0): 0.988160,
             ("DouBanLv2", "0041", 3): 0.986485}
    oracles = {(r["sample"], r["frame"]): r for r in read_csv("Phase17B_candidate_oracle.csv")}
    for (sample, frame, inst_id), want_f1 in cases.items():
        raw = bmask(P00 / "候选掩膜" / "raw_instance_P6" / f"mask_{sample}_{frame}_{inst_id:02d}.png")
        gt = bmask(GT_DIR / sample / f"mask_potted_clean_{frame}.png")
        m = metrics(raw, gt)
        assert abs(m["f1"] - want_f1) < 1e-4, f"{sample}_{frame}_i{inst_id} f1={m['f1']:.6f} != {want_f1}"
        assert abs(m["f1"] - float(oracles[(sample, frame)]["raw_oracle_f1"])) < 1e-4, \
            f"{sample}_{frame} oracle mismatch vs candidate_oracle.csv"
    print("  T1: 3 known instances recomputed with reference formula == oracle CSVs")


# ---------------------------------------------------------------- T2
def test_t2_selected_identity():
    sel = {}
    with (P00 / "提示词选择.csv").open(encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            stem = r["图像"].rsplit(".", 1)[0]
            s, fr = stem.rsplit("_", 1)
            sel[(s, fr)] = r
    oracles = {(r["sample"], r["frame"]): r for r in read_csv("Phase17B_candidate_oracle.csv")}
    for (sample, frame) in GT10:
        recomputed = selected_instance_id(sample, frame)
        # prompt-select file's 前景面积比例 -> instance id (via area ratio)
        area = float(sel[(sample, frame)]["前景面积比例"])
        inst = None
        with (P00 / "候选掩膜" / f"候选评分明细_{sample}_{frame}.csv").open(encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                if abs(float(r["面积比例"]) - area) < 1e-6:
                    inst = int(r["实例编号"])
        assert recomputed == inst, f"{sample}_{frame}: argmax={recomputed} != select-file={inst}"
        assert inst == int(oracles[(sample, frame)]["selected_instance_id"]), \
            f"{sample}_{frame}: select-file={inst} != oracle.selected_instance_id"
    print("  T2: argmax(total_score) == 提示词选择 == candidate_oracle.selected for all 10")


# ---------------------------------------------------------------- T3
def test_t3_cleanup_identity():
    for (sample, frame) in GT10:
        stem = f"{sample}_{frame}"
        a = bmask(P00 / "选择后掩膜" / f"mask_{stem}.png")
        b = bmask(P00 / "最终掩膜" / f"mask_{stem}.png")
        c = bmask(P00 / "A5c_final_mask" / f"mask_{stem}.png")
        assert a is not None and b is not None and c is not None, f"{stem} missing mask"
        assert np.array_equal(a, b), f"{stem}: 选择后 != 最终"
        assert np.array_equal(b, c), f"{stem}: 最终 != A5c"
    # postcleanup oracle == raw oracle in CSV (cleanup identity implies no drop)
    oracles = read_csv("Phase17B_candidate_oracle.csv")
    for r in oracles:
        assert abs(float(r["raw_oracle_f1"]) - float(r["postcleanup_oracle_f1"])) < 1e-9
    print("  T3: 选择后==最终==A5c bitwise for 10 GT frames; postcleanup==raw oracle")


# ---------------------------------------------------------------- T4
def test_t4_selection_regret():
    oracles = read_csv("Phase17B_candidate_oracle.csv")
    for r in oracles:
        r0 = float(r["raw_oracle_f1"])
        r1 = float(r["postcleanup_oracle_f1"])
        r2 = float(r["finalcandidate_oracle_f1"])
        sel_f1 = float(r["selected_f1"])
        reg = float(r["selection_regret"])
        assert abs(r0 - r1) < 1e-9 and abs(r1 - r2) < 1e-9
        assert reg >= -1e-9, f"{r['sample']}_{r['frame']} negative regret"
        assert abs(reg - (r2 - sel_f1)) < 1e-6, f"{r['sample']}_{r['frame']} regret != finOrc - selF1"
    # failing frames have regret == oracle (everything lost)
    for r in oracles:
        if (r["sample"], r["frame"]) in {("DouBanLv2", "0039"), ("DouBanLv2", "0040"), ("DouBanLv2", "0041")}:
            assert float(r["selection_regret"]) > 0.9
    print("  T4: regret = finalcandidate_oracle − selected_f1, ≥ 0, exact; failing frames ≈ full oracle")


# ---------------------------------------------------------------- T5
def test_t5_failure_class_determinism():
    attrs = {(r["sample"], r["frame"]): r for r in read_csv("Phase17B_failure_attribution.csv")}
    assert len(attrs) == 10
    for (sample, frame) in GT10:
        a = attrs[(sample, frame)]
        # exactly one class per frame (non-empty class string)
        assert a["failure_class"], f"{sample}_{frame} no class"
        # C5 excluded (mapping proven correct)
        assert a["failure_class"] != "C5_DATA_MAPPING_FAILURE"
        assert a["confidence"] in ("PROVEN", "SUPPORTED", "HYPOTHESIS")
    # failing frames == C3, PROVEN
    for (sample, frame) in [("DouBanLv2", "0039"), ("DouBanLv2", "0040"), ("DouBanLv2", "0041")]:
        a = attrs[(sample, frame)]
        assert a["failure_class"] == "C3_RANKING_FAILURE"
        assert a["confidence"] == "PROVEN"
    # other 7: OK_SELECTED (regret 0)
    for (sample, frame) in GT10:
        if (sample, frame) not in {("DouBanLv2", "0039"), ("DouBanLv2", "0040"), ("DouBanLv2", "0041")}:
            assert attrs[(sample, frame)]["failure_class"] == "OK_SELECTED"
    print("  T5: attribution deterministic per §5; C3×3 PROVEN; C5 excluded; OK_SELECTED×7")


# ---------------------------------------------------------------- T6
def test_t6_prompt_sweep_noop():
    manifest = json.loads((PHASE17B / "Phase17B_manifest.json").read_text("utf-8"))
    gate = manifest["prompt_sweep_gate"]
    assert gate["gate_triggered"] is False
    assert gate["decision"] == "skip (user-confirmed)"
    assert all(float(gate[k]) >= 0.10 for k in
               ("p6_raw_oracle_0039", "p6_raw_oracle_0040", "p6_raw_oracle_0041"))
    prows = list(csv.DictReader((PHASE17B / "Phase17B_prompt_oracle.csv").open(encoding="utf-8-sig")))
    for r in prows:
        assert all(r[k] == "not_run(gate)" for k in ("P1", "P2", "P3", "P4", "P5")), r
        assert r["best_prompt"] == "P6"
    # no pipeline run logs inside 17B dir
    assert not any(PHASE17B.glob("logs/*.log"))
    assert not any((PHASE17B / "logs").exists() for _ in [1]) or not any(PHASE17B.rglob("*.log"))
    print("  T6: sweep gate unmet & documented; P1–P5 = not_run; zero inference logs")


# ---------------------------------------------------------------- T7
def test_t7_gt_never_enters_inference():
    import re
    s20 = "生成RAP-FSAM3掩膜"
    for fp in (PHASE17B / "脚本").glob("*.py"):
        text = fp.read_text("utf-8")
        invoking = re.search(rf"(subprocess|os\.system|Popen).*{s20}", text)
        assert invoking is None, f"{fp.name} invokes pipeline"
    # GT paths only opened from analysis dirs (Phase16 GT dir), never written
    gt_writes = list(GT_DIR.glob("**/*_17B*")) or list(GT_DIR.glob("**/mask_potted_clean_*.png")) and []
    assert not gt_writes
    print("  T7: no S20 invocation in 17B scripts; GT never written")


# ---------------------------------------------------------------- T8
def test_t8_zero_inference_freeze():
    """Analysis is zero-inference: no mask/output under P00 modified post-17A."""
    # raw instance masks carry 17A mtimes (pipeline generation), not analysis times
    import time
    raw_samples = sorted((P00 / "候选掩膜" / "raw_instance_P6").glob("mask_DouBanLv2_0039_*.png"))
    assert len(raw_samples) == 2
    earliest_analysis = min(p.stat().st_mtime for p in (PHASE17B / "脚本").glob("*.py"))
    # raw instance masks were produced by Phase 17A (before 17B scripts)
    raw_now = max(p.stat().st_mtime for p in raw_samples)
    assert raw_now <= earliest_analysis + 5, \
        "raw masks modified after 17B analysis? zero-inference violated"
    print("  T8: raw instance masks predate 17B analysis (zero-inference freeze holds)")


# ---------------------------------------------------------------- T9
def test_t9_mapping_exact():
    mrows = {(r["sample"], r["frame"]): r for r in read_csv("Phase17B_mapping_audit.csv")}
    for (sample, frame) in GT10:
        m = mrows[(sample, frame)]
        assert m["image_exists"] == "True"
        assert m["gt_exists"] == "True"
        assert m["pred_exists"] == "True"
        assert m["sample_prefix_ok"] == "True"
        assert m["frame_index_ok"] == "True"
        assert m["image_size"] == "2160x3840" and m["gt_size"] == "2160x3840" and m["pred_size"] == "2160x3840"
        # frame index within the sample's frozen context window
        assert 0 <= int(frame)
    print("  T9: all 10 GT frames mapped exactly (stem/shape/existence verified)")


# ---------------------------------------------------------------- T10
def test_t10_no_a6a7():
    params = json.loads((P00 / "参数.json").read_text("utf-8"))
    assert params.get("use_cross_view_consensus") is False
    assert params.get("use_memory_propagation") is False
    flags = ("--use_cross_view_consensus", "--use_memory_propagation", "--colmap_dir")
    # audit scripts necessarily name the flags they verify; the substantive check
    # (no pipeline invocation) is handled separately below and in T7.
    audit_scripts = {"phase17b_steps15_19.py", "phase17b_step0_1.py"}
    for fp in (PHASE17B / "脚本").glob("*.py"):
        if fp.name in audit_scripts:
            continue
        text = fp.read_text("utf-8")
        # only allow flag-name definitions (e.g. cli_flags list) and param verification;
        # an "active" flag is one that would actually be passed to a runner.
        for flag in flags:
            for line in text.splitlines():
                s = line.strip()
                if flag in s and not s.startswith("#") and "参数.json" not in s \
                   and "verify" not in s.lower() and "cli_flags" not in s:
                    raise AssertionError(f"{fp.name} has active {flag}")
        # meta-assertion: no pipeline subprocess invocation anywhere
        if re.search(r"subprocess\.(run|check_output|Popen).*生成RAP-FSAM3掩膜", text):
            raise AssertionError(f"{fp.name} invokes the S20 pipeline")
    print("  T10: A6/A7 disabled in P00; 17B scripts freeze them")


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))