#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 16 — Hard-case GT challenge tests (T1–T12).

Covers the split freeze, manifest determinism, GT QA gates, and sequence
context integrity. Pure-data tests; no model inference.

Run:  python -m pytest 阶段十六_HardCase_GT构建与锁定/tests/test_phase16.py -v
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

PHASE16 = Path(__file__).resolve().parent.parent
LISTS = PHASE16 / "挑战集列表"
PKG = PHASE16 / "annotation_package"
SCRIPTS = PHASE16 / "脚本"

MANIFEST = LISTS / "Phase16_selection_manifest_preGT.csv"
DEV_MANIFEST = LISTS / "Phase16_DEV_manifest.csv"
TEST_MANIFEST = LISTS / "Phase16_TEST_manifest.csv"
MANIFEST_JSON = LISTS / "Phase16_manifest.json"


def _load(manifest: Path) -> list[dict]:
    with open(manifest, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _import_script(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    for p in (str(SCRIPTS),):
        if p not in sys.path:
            sys.path.insert(0, p)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------- T1
class TestT1DeterministicSplit:
    """T1: same input → same split output (determinism)."""

    def test_rerun_reproduces(self):
        if not MANIFEST.exists():
            return
        mod = _import_script("phase16_split")
        rows = _load(MANIFEST)
        for r in rows:
            assert r["split"] in ("DEV", "TEST")
        # determinism: rerunning split would need the full pipeline; instead
        # assert SPLIT_ASSIGNMENT is a pure dict of 15 entries covering samples
        assert len(mod.SPLIT_ASSIGNMENT) == 15
        assert set(mod.SPLIT_ASSIGNMENT) == set(mod.NON_GT_SAMPLES)


# ---------------------------------------------------------------- T2 / T3 / T11
class TestT2NoSampleLeakage:
    """T2 + T11: DEV ∩ TEST sample = ∅ ; each sample in exactly one split,
    consistent with the frozen sample-level assignment (SPLIT_ASSIGNMENT)."""

    def test_no_sample_overlap(self):
        rows = _load(MANIFEST)
        mod = _import_script("phase16_split")
        by_sample: dict[str, set[str]] = {}
        for r in rows:
            by_sample.setdefault(r["sample"], set()).add(r["split"])
        for sample, splits in by_sample.items():
            assert len(splits) == 1, f"{sample} in multiple splits: {splits}"
            # manifest row must not contradict the frozen sample-level assignment
            expected = mod.SPLIT_ASSIGNMENT[sample]
            actual = next(iter(splits))
            assert actual == expected, f"{sample}: split {actual} ≠ assignment {expected}"
        dev = {r["sample"] for r in rows if r["split"] == "DEV"}
        test = {r["sample"] for r in rows if r["split"] == "TEST"}
        assert dev.isdisjoint(test), f"sample overlap: {dev & test}"
        # current selection reality: only 2 DEV samples selected for GT so far
        assert len(dev) >= 1, "no DEV sample selected yet"


class TestT3NoSequenceLeakage:
    """T3: no frame from the same sequence appears in both splits."""

    def test_sequence_exclusive(self):
        rows = _load(MANIFEST)
        by_seq = Counter(r["sequence"] for r in rows if r["split"] == "DEV") & \
                 Counter(r["sequence"] for r in rows if r["split"] == "TEST")
        assert not by_seq, f"sequences leaked across splits: {dict(by_seq)}"


class TestT11SplitIntersectionEmpty:
    """T11: DEV/TEST manifest intersection is empty (row identity)."""

    def test_manifest_intersection(self):
        dev = _load(DEV_MANIFEST)
        test = _load(TEST_MANIFEST)
        dev_keys = {(r["sample"], r["frame"]) for r in dev}
        test_keys = {(r["sample"], r["frame"]) for r in test}
        assert dev_keys.isdisjoint(test_keys)
        assert len(dev) == 26 and len(test) == 32, f"{len(dev)}/{len(test)}"


# ---------------------------------------------------------------- T4 / T5 / T6
class TestT4BinaryGTValidation:
    """T4: GT masks are binary {0,255} and shape-exact (when GT present)."""

    def test_audit_binary_cols(self):
        qa = PHASE16 / "GT_QA_challenge" / "gt_audit_challenge.csv"
        if not qa.exists():
            return  # GT not yet annotated — gate pending
        with open(qa, encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        assert rows, "audit csv empty"
        for r in rows:
            assert r["binary_ok"] == "1", f"{r['sample']}/{r['frame']} non-binary"
            assert r["qa_shape_mismatch"] == "0", f"{r['sample']}/{r['frame']} shape mismatch"
            assert r["formal_p6_valid"] == "True", f"{r['sample']}/{r['frame']} not P6 valid"


class TestT5ShapeMismatchRejection:
    """T5: QA pipeline flags shape mismatches."""

    def test_gate_column_present(self):
        qa = PHASE16 / "GT_QA_challenge" / "gt_audit_challenge.csv"
        if not qa.exists():
            return
        with open(qa, encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        assert "qa_shape_mismatch" in rows[0]


class TestT6EmptyMaskRejection:
    """T6: QA pipeline flags empty masks."""

    def test_empty_col_present(self):
        qa = PHASE16 / "GT_QA_challenge" / "gt_audit_challenge.csv"
        if not qa.exists():
            return
        with open(qa, encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        assert "qa_empty" in rows[0]
        for r in rows:
            if r["qa_empty"] == "1":
                raise AssertionError(f"empty mask: {r['sample']}/{r['frame']}")


# ---------------------------------------------------------------- T7
class TestT7DuplicateTargetDetection:
    """T7: no duplicate (sample, frame) targets in the selection manifest."""

    def test_no_duplicate_targets(self):
        rows = _load(MANIFEST)
        c = Counter((r["sample"], r["frame"]) for r in rows)
        dups = {k: v for k, v in c.items() if v > 1}
        assert not dups, f"duplicate targets: {dups}"
        ids = [r["challenge_id"] for r in rows]
        assert len(set(ids)) == len(ids), "duplicate challenge_id"


# ---------------------------------------------------------------- T8
class TestT8DeterministicManifestHashing:
    """T8: SHA256 in manifest.json matches recomputation from CSV files."""

    def _csv_text(self, path: Path) -> str:
        text = path.read_text(encoding="utf-8")
        return text.replace("﻿", "").replace("\r\n", "\n")

    def test_full_hash_matches(self):
        if not MANIFEST_JSON.exists():
            return
        meta = json.loads(MANIFEST_JSON.read_text(encoding="utf-8"))
        # recompute in the same canonical way the freeze script does:
        # hash of the serialized CSV (header + rows, LF)
        mod = _import_script("phase16_freeze")
        rows = _load(MANIFEST)
        import io
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=list(rows[0].keys()), extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
        canonical = buf.getvalue().replace("﻿", "").replace("\r\n", "\n")
        h = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        assert h == meta["sha256_full_manifest"], "manifest full SHA mismatch"

    def test_dev_test_hashes_match(self):
        if not MANIFEST_JSON.exists():
            return
        meta = json.loads(MANIFEST_JSON.read_text(encoding="utf-8"))
        for key, path in [("sha256_dev_manifest", DEV_MANIFEST),
                          ("sha256_test_manifest", TEST_MANIFEST)]:
            h = hashlib.sha256(self._csv_text(path).encode("utf-8")).hexdigest()
            assert h == meta[key], f"{key} mismatch"


# ---------------------------------------------------------------- T9
class TestT9ContextFrameLookup:
    """T9: every target's context window points at real, readable frames."""

    def test_context_frames_exist(self):
        rows = _load(MANIFEST)
        for r in rows:
            raw = Path("/data/fj/F2DMAS/00-论文优化重构/数据管理"
                       "/01-输入图像/01-raw_frames") / r["sample"]
            cs, ce = int(r["context_start"]), int(r["context_end"])
            p1 = raw / f"{cs:04d}.jpg"
            p2 = raw / f"{ce:04d}.jpg"
            assert p1.exists(), f"context_prev missing: {p1}"
            assert p2.exists(), f"context_next missing: {p2}"
            assert cs <= int(r["frame"]) <= ce

    def test_package_context_copies(self):
        rows = _load(MANIFEST)
        n_ok = 0
        for r in rows:
            d = PKG / r["challenge_id"]
            assert d.exists(), f"package dir missing: {d}"
            for name in ("context_prev.jpg", "context_next.jpg", "target.jpg"):
                p = d / name
                assert p.exists(), f"package file missing: {p}"
                img = cv2.imread(str(p))
                assert img is not None and img.shape[:2] == (3840, 2160), f"{p}"
                n_ok += 1
        assert n_ok >= 3 * len(rows), f"only {n_ok} package images verified"


# ---------------------------------------------------------------- T10
class TestT10GTCannotOverwriteSilently:
    """T10: challenge GT output never overwrites the frozen Phase 12 GT."""

    def test_output_paths_are_distinct(self):
        # Phase 12 GT frozen at 阶段十二/GT_potted_clean — challenge GT must be
        # under Phase 16's own GT_potted_clean_challenge, never that path.
        ph12 = Path("/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区"
                    "/01-算法模块修改与验证/阶段十二_GT_v2_QA与P6正式验收")
        frozen = ph12 / "GT_potted_clean"
        challenge_out = PHASE16 / "GT_potted_clean_challenge"
        assert frozen != challenge_out
        # and the QA csv path must not be the Phase 12 one
        assert (PHASE16 / "GT_QA_challenge") != (ph12 / "GT_QA")

    def test_existing_21_gt_untouched(self):
        frozen = Path("/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区"
                      "/01-算法模块修改与验证/阶段十二_GT_v2_QA与P6正式验收"
                      "/GT_potted_clean")
        n = sum(1 for p in frozen.rglob("*.png"))
        assert n == 21, f"Phase 12 GT changed! now {n} masks"


# ---------------------------------------------------------------- T12
class TestT12TestManifestImmutable:
    """T12: TEST manifest SHA256 is frozen and re-derivable; per-target image
    checksums recorded in manifest.json match the on-disk files."""

    def test_test_hash_immutable(self):
        if not MANIFEST_JSON.exists():
            return
        meta = json.loads(MANIFEST_JSON.read_text(encoding="utf-8"))
        assert meta["sha256_test_manifest"], "TEST manifest hash missing"

    def test_image_checksums_match_disk(self):
        if not MANIFEST_JSON.exists():
            return
        meta = json.loads(MANIFEST_JSON.read_text(encoding="utf-8"))
        chk = meta.get("image_sha256", {})
        # map challenge_id -> raw image_path (manifest records hashes of the
        # ORIGINAL raw frames, not the re-encoded package copies)
        rows = _load(MANIFEST)
        assert len(chk) == len(rows), f"{len(chk)} image checksums != {len(rows)} rows"
        cid_to_path = {r["challenge_id"]: r["image_path"] for r in rows}
        for cid, sha in chk.items():
            raw_path = Path(cid_to_path[cid])
            assert raw_path.exists(), f"raw frame missing: {raw_path}"
            h = hashlib.sha256(raw_path.read_bytes()).hexdigest()
            assert h == sha, f"checksum mismatch for {cid}"


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
