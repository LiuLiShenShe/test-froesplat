#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 16 — Freeze selection manifests with SHA256 hashing.

Reads Phase16_selection_manifest_preGT.csv, splits into
Phase16_DEV_manifest.csv / Phase16_TEST_manifest.csv, computes SHA256 for
the full manifest + per-target image checksums, and writes:
  Phase16_manifest.json   — frozen hashes, split assignment, timestamps, refs
  Phase16_DEV_manifest.csv
  Phase16_TEST_manifest.csv

Check-ref: with the SAME input rows in the SAME order, rerunning this script
reproduces identical SHA256 values (deterministic).

Usage:
    python phase16_freeze.py [--manifest .../Phase16_selection_manifest_preGT.csv]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

PHASE16 = Path(__file__).resolve().parent.parent
LISTS = PHASE16 / "挑战集列表"
DEFAULT_MANIFEST = LISTS / "Phase16_selection_manifest_preGT.csv"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=str, default=str(DEFAULT_MANIFEST))
    args = ap.parse_args()
    man_path = Path(args.manifest)

    with open(man_path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    # deterministic canonical serialization for the hash: full CSV as-is,
    # with normalized newlines so the hash is platform-independent
    import io
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(rows[0].keys()), extrasaction="ignore")
    w.writeheader()
    w.writerows(rows)
    canonical_full_csv = buf.getvalue().replace("\r\n", "\n").replace("﻿", "")

    # split
    dev_rows = [r for r in rows if r["split"] == "DEV"]
    test_rows = [r for r in rows if r["split"] == "TEST"]

    def write_manifest(rows_: list[dict], out: Path) -> str:
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=list(rows[0].keys()), extrasaction="ignore")
        w.writeheader()
        w.writerows(rows_)
        text = buf.getvalue().replace("\r\n", "\n").replace("﻿", "")
        out.write_text(text, encoding="utf-8")
        return sha256_hex(text)

    dev_hash = write_manifest(dev_rows, LISTS / "Phase16_DEV_manifest.csv")
    test_hash = write_manifest(test_rows, LISTS / "Phase16_TEST_manifest.csv")

    # per-target image checksums
    checksums: dict[str, str] = {}
    missing_img: list[str] = []
    for r in rows:
        p = Path(r["image_path"])
        if not p.exists():
            missing_img.append(r["challenge_id"])
            continue
        checksums[r["challenge_id"]] = sha256_file(p)

    utcnow = datetime.now(timezone.utc)
    manifest = {
        "schema_version": "1.0",
        "phase": "16",
        "description": "Phase 16 hard-case GT challenge frozen selection manifest",
        "frozen_at_utc": utcnow.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_csv": man_path.name,
        "total_targets": len(rows),
        "dev_targets": len(dev_rows),
        "test_targets": len(test_rows),
        "dev_samples": sorted({r["sample"] for r in dev_rows}),
        "test_samples": sorted({r["sample"] for r in test_rows}),
        "sha256_full_manifest": sha256_hex(canonical_full_csv),
        "sha256_dev_manifest": dev_hash,
        "sha256_test_manifest": test_hash,
        "image_sha256": checksums,
        "image_checksum_missing": missing_img,
        "note": "Manifest determinism: identical input CSV rows in identical "
                "order reproduce identical SHA256. TEST manifest is immutable "
                "for Phase 17 evaluation.",
    }

    out_json = LISTS / "Phase16_manifest.json"
    out_json.write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                        encoding="utf-8")

    print(f"Total targets: {len(rows)}  DEV: {len(dev_rows)}  TEST: {len(test_rows)}")
    print(f"sha256 full : {manifest['sha256_full_manifest']}")
    print(f"sha256 DEV  : {manifest['sha256_dev_manifest']}")
    print(f"sha256 TEST : {manifest['sha256_test_manifest']}")
    print(f"dev samples : {', '.join(manifest['dev_samples'])}")
    print(f"test samples: {', '.join(manifest['test_samples'])}")
    print(f"image checksums: {len(checksums)}/{len(rows)}  missing: {len(missing_img)}")
    print(f"Saved → {out_json}")


if __name__ == "__main__":
    main()