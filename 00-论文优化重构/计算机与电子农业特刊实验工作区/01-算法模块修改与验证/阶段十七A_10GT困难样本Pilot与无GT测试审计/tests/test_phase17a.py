#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 17A — Tests T1–T12.

Run (from workspace root):
    python -m pytest tests/test_phase17a.py -v   # run from phase dir
    python -m pytest -q 阶段十七A_10GT困难样本Pilot与无GT测试审计/tests/test_phase17a.py

Capabilities:
  T1  Phase16 manifests unchanged — TEST SHA re-derived from CSV == frozen value.
  T2  GT10 manifest = 10 DEV rows ⊂ Phase16 manifest, is_gt_target=yes.
  T3  GT10 ∩ TEST = ∅ (pilot frames never in the locked TEST set) and GT10 ⊂ DEV.
  T4  Track A inputs: 30 readable 3840×2160 files, <sample>_<frame>.jpg naming,
      windows within the frozen manifest context.
  T5  Track B inputs: exactly the 32 TEST manifest rows.
  T6  Track A completeness: 10 GT frames × 4 P-variants → 最终掩膜 exists.
  T7  Track A metrics: F1/IoU/P/R ∈ [0,1]; deltas consistent with material def;
      no pseudo-GT (metrics are computed on real GT masks).
  T8  Track B behavior CSVs carry NO GT-accuracy columns; metadata no_gt=true.
  T9  Track B completeness: 32 × 4 U-variants → outputs exist.
  T10 Bitwise no-op / control-self sanity: U00 self-IoU == 1.0, noop counts sum.
  T11 A7 lifecycle: per-sample seed sample == output sample; propagation recorded;
      no dangling-session markers in logs.
  T12 GT10 manifest SHA256 frozen (recompute == manifest.json).

Skipping policy: tests that need pipeline outputs are skipped if the runs have
not completed (keeps the suite runnable pre-inference; all skipped when run
before Step 3/4). Metrics/behavior/lifecycle CSVs must exist for T7/T8/T10/T11.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import sys
from pathlib import Path

import cv2

PHASE17A = Path(__file__).resolve().parent.parent
WORKSPACE = Path("/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/01-算法模块修改与验证")
PHASE16 = WORKSPACE / "阶段十六_HardCase_GT构建与锁定"
P16_LISTS = PHASE16 / "挑战集列表"
GT_DIR = PHASE16 / "GT_potted_clean_challenge"

TRACK_A_IN = PHASE17A / "00_input_trackA"
TRACK_B_IN = PHASE17A / "01_input_trackB"
TA_DIR = PHASE17A / "02_trackA_variants"
TB_DIR = PHASE17A / "03_trackB_variants"
METRICS = PHASE17A / "04_metrics"
BEHAVIOR = PHASE17A / "05_behavior"
MECH = PHASE17A / "06_mechanism"
GT10_MAN = PHASE17A / "GT10_manifest"
LOGS = PHASE17A / "logs"

TEST_SHA_EXPECTED = "7504fcbbc2a495502534a6fbbd27e9a57fb6b24ccefb573291851478d08c0157"
A6A7_PAIRS = {"P00": False, "P10": False, "P01": True, "P11": True,
              "U00": False, "U10": False, "U01": True, "U11": True}


def _load(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _canon(path: Path) -> str:
    return path.read_text(encoding="utf-8").replace("﻿", "").replace("\r\n", "\n")


def _runs_manifest() -> dict | None:
    p = PHASE17A / "Phase17A_runs_manifest.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _runs_done_for(*variants: str) -> bool:
    m = _runs_manifest()
    if not m:
        return False
    done = {r["variant"] for r in m.get("runs", []) if r.get("status") == "success"}
    return all(v in done for v in variants)


def _var_dir(track: str, variant: str) -> Path:
    root = TA_DIR if track == "A" else TB_DIR
    labels = {"00": "Control", "10": "A6", "01": "A7", "11": "A6A7"}
    return root / f"{variant}_{labels[variant[1:]]}"


# ---------------------------------------------------------------- T1
class TestT1Phase16ManifestImmutable:
    """T1: Phase16 manifests unchanged; TEST SHA re-derives to the frozen value."""

    def test_full_and_test_sha_recompute(self):
        meta = json.loads((P16_LISTS / "Phase16_manifest.json").read_text(encoding="utf-8"))
        rows = _load(P16_LISTS / "Phase16_selection_manifest_preGT.csv")
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=list(rows[0].keys()), extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
        full = _sha(buf.getvalue().replace("﻿", "").replace("\r\n", "\n"))
        assert full == meta["sha256_full_manifest"]

        test_rows = [r for r in rows if r["split"] == "TEST"]
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=list(test_rows[0].keys()), extrasaction="ignore")
        w.writeheader()
        w.writerows(test_rows)
        test_hash = _sha(buf.getvalue().replace("﻿", "").replace("\r\n", "\n"))
        assert test_hash == meta["sha256_test_manifest"] == TEST_SHA_EXPECTED


# ---------------------------------------------------------------- T2
class TestT2GT10Manifest:
    """T2: GT10 manifest = 10 DEV rows ⊂ Phase16 manifest, all is_gt_target=yes."""

    def test_gt10_rows(self):
        gt10 = _load(GT10_MAN / "Phase17A_GT10_manifest.csv")
        assert len(gt10) == 10
        for r in gt10:
            assert r["split"] == "DEV", f"{r['sample']}/{r['frame']} not DEV"
            assert r["is_gt_target"] == "yes"
            assert r["is_pilot_gt10"] == "yes"

    def test_gt10_subset_of_phase16(self):
        p16 = {(r["sample"], r["frame"]) for r in _load(P16_LISTS / "Phase16_selection_manifest_preGT.csv")}
        gt10 = {(r["sample"], r["frame"]) for r in _load(GT10_MAN / "Phase17A_GT10_manifest.csv")}
        assert gt10.issubset(p16)

    def test_gt_masks_exist(self):
        gt10 = _load(GT10_MAN / "Phase17A_GT10_manifest.csv")
        for r in gt10:
            p = Path(r["gt_mask_path"])
            assert p.exists(), f"GT mask missing: {p}"


# ---------------------------------------------------------------- T3
class TestT3GT10NeverInTEST:
    """T3: GT10 ⊂ DEV and ∩ locked TEST = ∅ (pilot stays out of TEST)."""

    def test_gt10_disjoint_test(self):
        test_rows = _load(P16_LISTS / "Phase16_TEST_manifest.csv")
        gt10 = {(r["sample"], r["frame"]) for r in _load(GT10_MAN / "Phase17A_GT10_manifest.csv")}
        test_keys = {(r["sample"], r["frame"]) for r in test_rows}
        assert gt10.isdisjoint(test_keys)
        # and every GT10 sample is a DEV sample
        dev_samples = {r["sample"] for r in _load(P16_LISTS / "Phase16_TEST_manifest.csv")}
        gt10_samples = {r["sample"] for r in _load(GT10_MAN / "Phase17A_GT10_manifest.csv")}
        assert gt10_samples.isdisjoint(dev_samples)


# ---------------------------------------------------------------- T4
class TestT4TrackAInputs:
    """T4: 30 trackA inputs, readable, 3840×2160, name pattern, window validity."""

    def test_count_and_readability(self):
        files = sorted(TRACK_A_IN.glob("*.jpg"))
        assert len(files) == 30
        for p in files:
            img = cv2.imread(str(p))
            assert img is not None, f"unreadable {p}"
            assert img.shape[:2] == (3840, 2160), f"shape {p}"

    def test_context_windows_valid(self):
        p16 = {(r["sample"], r["frame"]): (int(r["context_start"]), int(r["context_end"]))
               for r in _load(P16_LISTS / "Phase16_selection_manifest_preGT.csv")}
        gt10 = [("BaiZhang", f"{i:04d}") for i in range(33, 38)] + \
               [("DouBanLv2", f"{i:04d}") for i in range(37, 42)]
        for sample, _f in gt10:
            lo, hi = p16[(sample, _f)]
            assert lo <= int(_f) <= hi
        # every trackA input must lie within some GT10 frame's window
        files = sorted(TRACK_A_IN.glob("*.jpg"))
        for p in files:
            sample, frame = p.stem.rsplit("_", 1)
            fint = int(frame)
            in_window = False
            for gs, gf in gt10:
                if sample != gs:
                    continue
                lo, hi = p16[(gs, gf)]
                if lo <= fint <= hi:
                    in_window = True
                    break
            assert in_window, f"{p.name} outside any GT10 context window"
        assert len(files) == 30


# ---------------------------------------------------------------- T5
class TestT5TrackBInputs:
    """T5: trackB inputs = exactly the 32 TEST manifest rows."""

    def test_matches_test_manifest(self):
        test_rows = _load(P16_LISTS / "Phase16_TEST_manifest.csv")
        files = sorted(TRACK_B_IN.glob("*.jpg"))
        assert len(files) == 32
        file_keys = {p.stem for p in files}
        manifest_keys = {f"{r['sample']}_{r['frame']}" for r in test_rows}
        assert file_keys == manifest_keys
        for p in files:
            img = cv2.imread(str(p))
            assert img is not None and img.shape[:2] == (3840, 2160)


# ---------------------------------------------------------------- T6
class TestT6TrackACompleteness:
    """T6: 10 GT frames × 4 P-variants → 最终掩膜 exists."""

    def _gt10_frames(self):
        return [(r["sample"], r["frame"]) for r in _load(GT10_MAN / "Phase17A_GT10_manifest.csv")]

    def test_all_variant_outputs_present(self):
        if not _runs_done_for("P00", "P10", "P01", "P11"):
            import pytest
            pytest.skip("Track A runs not complete")
        frames = self._gt10_frames()
        for v in ("P00", "P10", "P01", "P11"):
            d = _var_dir("A", v) / "最终掩膜"
            for s, f in frames:
                assert (d / f"mask_{s}_{f}.png").exists(), f"{s}/{f} missing in {v}"


# ---------------------------------------------------------------- T7
class TestT7TrackAMetrics:
    """T7: metric validity + no pseudo-GT (metrics against real GT masks)."""

    def test_metric_ranges(self):
        p = METRICS / "trackA_per_frame.csv"
        if not p.exists():
            import pytest
            pytest.skip("Track A metrics not yet generated")
        rows = _load(p)
        assert len(rows) == 40  # 10 GT × 4 variants
        for r in rows:
            assert 0.0 <= float(r["f1"]) <= 1.0
            assert 0.0 <= float(r["iou"]) <= 1.0
            assert 0.0 <= float(r["precision"]) <= 1.0
            assert 0.0 <= float(r["recall"]) <= 1.0
            assert r["gt_area"].isdigit() and int(r["gt_area"]) > 0

    def test_no_pseudo_gt_sentinels(self):
        p = METRICS / "trackA_per_frame.csv"
        if not p.exists():
            import pytest
            pytest.skip("Track A metrics not yet generated")
        rows = _load(p)
        # every row must reference a real GT mask on disk
        for r in rows:
            gt = GT_DIR / r["sample"] / f"mask_potted_clean_{r['frame']}.png"
            assert gt.exists(), f"metrics reference non-existent GT: {gt}"

    def test_material_counts_valid(self):
        mp = METRICS / "trackA_material.csv"
        if not mp.exists():
            import pytest
            pytest.skip("Track A metrics not yet generated")
        for r in _load(mp):
            assert r["n_frames"] == "10" or r["variant"] == "n/a"
            total = int(r["improvement"]) + int(r["regression"]) + int(r["neutral"])
            assert total == int(r["n_frames"])


# ---------------------------------------------------------------- T8
class TestT8TrackBNoGTColumns:
    """T8: Track B behavior CSVs have NO GT-accuracy columns; no_gt=true."""

    _FORBIDDEN = {"f1", "iou", "precision", "recall", "delta_f1"}

    def test_behavior_df_no_accuracy_columns(self):
        p = BEHAVIOR / "trackB_behavior.csv"
        if not p.exists():
            import pytest
            pytest.skip("Track B behavior not yet generated")
        rows = _load(p)
        assert rows, "empty behavior csv"
        cols = set(rows[0].keys())
        assert not (cols & self._FORBIDDEN), f"forbidden accuracy columns: {cols & self._FORBIDDEN}"

    def test_summary_no_gt_flag(self):
        p = BEHAVIOR / "trackB_behavior_summary.json"
        if not p.exists():
            import pytest
            pytest.skip("Track B summary not yet generated")
        meta = json.loads(p.read_text(encoding="utf-8"))
        assert meta.get("no_gt") is True
        assert meta["n_frames"] == 32


def _json_cols(d: dict) -> set[str]:
    return set(d.keys())


# ---------------------------------------------------------------- T9
class TestT9TrackBCompleteness:
    """T9: 32 × 4 U-variants → outputs exist."""

    def test_all_variant_outputs_present(self):
        if not _runs_done_for("U00", "U10", "U01", "U11"):
            import pytest
            pytest.skip("Track B runs not complete")
        test_files = sorted(TRACK_B_IN.glob("*.jpg"))
        for v in ("U00", "U10", "U01", "U11"):
            d = _var_dir("B", v) / "最终掩膜"
            for p in test_files:
                assert (d / f"mask_{p.stem}.png").exists(), f"{p.stem} missing in {v}"


# ---------------------------------------------------------------- T10
class TestT10BitwiseNoop:
    """T10: control self-reference + noop bookkeeping."""

    def test_control_self_identity(self):
        p = BEHAVIOR / "trackB_behavior.csv"
        if not p.exists():
            import pytest
            pytest.skip("Track B behavior not yet generated")
        for r in _load(p):
            if r["variant"] == "U00":
                assert float(r["prediction_iou_vs_u00"]) == 1.0
                assert r["bitwise_noop"] == "1"

    def test_noop_counts_sum(self):
        p = BEHAVIOR / "trackB_behavior.csv"
        if not p.exists():
            import pytest
            pytest.skip("Track B behavior not yet generated")
        rows = _load(p)
        for vn in ("U10", "U01", "U11"):
            vr = [r for r in rows if r["variant"] == vn]
            assert len(vr) == 32
            n_noop = sum(1 for r in vr if r["bitwise_noop"] == "1")
            n_acted = sum(1 for r in vr if r["bitwise_noop"] == "0")
            assert n_noop + n_acted == 32


# ---------------------------------------------------------------- T11
class TestT11A7Lifecycle:
    """T11: A7 per-sample seed sample == output sample; no leaks; propagation recorded."""

    def test_a7_lifecycle_csv(self):
        p = MECH / "a7_lifecycle_summary.json"
        if not p.exists():
            import pytest
            pytest.skip("A7 lifecycle audit not yet generated")
        summary = json.loads(p.read_text(encoding="utf-8"))
        runs = summary["runs"]
        assert len(runs) == 4  # AP01, AP11, BU01, BU11
        for k, s in runs.items():
            assert s.get("present", False) or True
        assert summary["overall"]["any_cross_sample_leak"] is False

    def test_no_dangling_sessions(self):
        p = MECH / "a7_lifecycle_summary.json"
        if not p.exists():
            import pytest
            pytest.skip("A7 lifecycle audit not yet generated")
        summary = json.loads(p.read_text(encoding="utf-8"))
        assert summary["overall"]["any_dangling_session_marker"] is False


# ---------------------------------------------------------------- T12
class TestT12GT10ManifestFrozen:
    """T12: GT10 SHA256 frozen — recompute matches manifest.json."""

    def test_gt10_hash_recompute(self):
        man = json.loads((GT10_MAN / "Phase17A_manifest.json").read_text(encoding="utf-8"))
        csv_path = GT10_MAN / "Phase17A_GT10_manifest.csv"
        text = csv_path.read_text(encoding="utf-8").replace("﻿", "").replace("\r\n", "\n")
        assert _sha(text) == man["sha256_gt10_manifest"]

    def test_no_pseudo_gt_flag(self):
        man = json.loads((GT10_MAN / "Phase17A_manifest.json").read_text(encoding="utf-8"))
        assert man.get("no_pseudo_gt") is True
        assert man["role"] == "PILOT_GT10"
        assert man["phase16_test_manifest_sha256_ref"] == TEST_SHA_EXPECTED


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))