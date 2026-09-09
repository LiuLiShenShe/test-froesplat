#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 17A — Step 2: Freeze GT10 manifest (PILOT_GT10 role, SHA256 locked).

Extracts the 10 annotated challenge GT frames from the frozen Phase16 manifest
and writes:
  Phase17A_GT10_manifest.csv   — 10 rows, all DEV, is_pilot_gt10=yes
  Phase17A_manifest.json       — SHA256 of GT10 CSV + metadata

Canonical hashing follows phase16_freeze.py (LF, no BOM).
Role = PILOT_GT10 (budgeted pilot subset of DEV material, not confirmatory test).
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path

PHASE17A = Path(__file__).resolve().parent.parent
WORKSPACE = Path("/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/01-算法模块修改与验证")
PHASE16 = WORKSPACE / "阶段十六_HardCase_GT构建与锁定"
P16_LISTS = PHASE16 / "挑战集列表"
P16_SELECT = P16_LISTS / "Phase16_selection_manifest_preGT.csv"
GT_CHALLENGE = PHASE16 / "GT_potted_clean_challenge"
OUT_DIR = PHASE17A / "GT10_manifest"

TEST_SHA_EXPECTED = "7504fcbbc2a495502534a6fbbd27e9a57fb6b24ccefb573291851478d08c0157"

GT10_FRAMES = {
    "BaiZhang":  ["0033", "0034", "0035", "0036", "0037"],
    "DouBanLv2": ["0037", "0038", "0039", "0040", "0041"],
}


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load full Phase16 selection manifest
    with P16_SELECT.open(encoding="utf-8-sig") as f:
        all_rows = list(csv.DictReader(f))

    # Filter to GT10 rows
    gt10_pairs = {(s, fr) for s, frs in GT10_FRAMES.items() for fr in frs}
    gt10 = [r for r in all_rows if (r["sample"], r["frame"]) in gt10_pairs]
    assert len(gt10) == 10, f"Expected 10 GT10 rows, found {len(gt10)}"

    # All must be DEV
    for r in gt10:
        assert r["split"] == "DEV", f"{r['sample']}/{r['frame']} is {r['split']}, expected DEV"
        assert r["is_gt_target"] == "yes", f"{r['sample']}/{r['frame']} is_gt_target != yes"

    # Add pilot-specific columns
    for r in gt10:
        r["is_pilot_gt10"] = "yes"
        r["gt_mask_path"] = str(GT_CHALLENGE / r["sample"] / f"mask_potted_clean_{r['frame']}.png")

    # Write GT10 CSV
    csv_fields = list(gt10[0].keys())
    out_csv = OUT_DIR / "Phase17A_GT10_manifest.csv"
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=csv_fields, extrasaction="ignore")
    w.writeheader()
    w.writerows(gt10)
    csv_text = buf.getvalue().replace("﻿", "").replace("\r\n", "\n")
    out_csv.write_text(csv_text, encoding="utf-8")
    gt10_sha = sha256_hex(csv_text)

    # Re-derive the Phase16 TEST SHA to include in metadata
    test_rows = [r for r in all_rows if r["split"] == "TEST"]
    tbuf = io.StringIO()
    tw = csv.DictWriter(tbuf, fieldnames=list(test_rows[0].keys()), extrasaction="ignore")
    tw.writeheader()
    tw.writerows(test_rows)
    test_text = tbuf.getvalue().replace("﻿", "").replace("\r\n", "\n")
    test_sha_actual = sha256_hex(test_text)

    # Verify GT10 GT masks actually exist on disk
    missing_gt = []
    for r in gt10:
        if not Path(r["gt_mask_path"]).exists():
            missing_gt.append(f"{r['sample']}/{r['frame']}")

    # Write manifest.json
    manifest = {
        "schema_version": "1.0",
        "phase": "17A",
        "description": "Phase 17A budgeted pilot GT10 frozen manifest",
        "role": "PILOT_GT10",
        "role_note": ("10 human-labeled frames from DEV material (BaiZhang + DouBanLv2). "
                      "Budgeted pilot subset, NOT a held-out confirmatory test. "
                      "No pseudo-GT created for unlabeled TEST frames."),
        "frozen_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "gt10_targets": len(gt10),
        "gt10_samples": sorted(set(r["sample"] for r in gt10)),
        "gt10_frames": {s: sorted(GT10_FRAMES[s]) for s in sorted(GT10_FRAMES)},
        "gt_masks_exist_all": len(missing_gt) == 0,
        "gt_masks_missing": missing_gt,
        "no_pseudo_gt": True,
        "sha256_gt10_manifest": gt10_sha,
        "phase16_test_manifest_sha256_ref": test_sha_actual,
        "phase16_test_manifest_note": "Immutable for Phase 17A (re-derived from CSV)",
        "source_manifest": "Phase16_selection_manifest_preGT.csv",
    }
    out_json = OUT_DIR / "Phase17A_manifest.json"
    out_json.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"GT10 manifest: {len(gt10)} rows")
    print(f"  samples: {manifest['gt10_samples']}")
    print(f"  gt_masks exist: {manifest['gt_masks_exist_all']}")
    if missing_gt:
        print(f"  ⚠ missing GT masks: {missing_gt}")
    print(f"  sha256_gt10_manifest: {gt10_sha}")
    print(f"  phase16_test_sha256_ref: {test_sha_actual}")
    print(f"  no_pseudo_gt: {manifest['no_pseudo_gt']}")
    print(f"Saved → {out_csv.name}, {out_json.name}")


if __name__ == "__main__":
    main()