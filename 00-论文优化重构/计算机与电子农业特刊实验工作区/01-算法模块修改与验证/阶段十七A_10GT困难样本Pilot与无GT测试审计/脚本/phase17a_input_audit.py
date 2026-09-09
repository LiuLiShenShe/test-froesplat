#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 17A — Step 0: Starting-State Audit (read-only, no push).

Verifies the pre-execution state of the repository and all Phase 17A inputs:
  - git branch / HEAD / last 5 commits / working-tree status
  - Phase 16 frozen manifests re-derivable (full / DEV / TEST SHA256)
  - 10 challenge GT masks present + QA csv formal_p6_valid=True
  - 32 locked TEST frames present on disk
  - GPU state snapshot
  - pipeline script + SAM3 checkpoint + COLMAP presence for Track A samples

Output: Phase17A_starting_audit.md   (phase root)
No repository files are modified (audit only).
"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PHASE17A = Path(__file__).resolve().parent.parent
WORKSPACE = Path("/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/01-算法模块修改与验证")
PHASE16 = WORKSPACE / "阶段十六_HardCase_GT构建与锁定"
P16_LISTS = PHASE16 / "挑战集列表"
GT_CHALLENGE = PHASE16 / "GT_potted_clean_challenge"
QA_CSV = PHASE16 / "GT_QA_challenge" / "gt_audit_challenge.csv"

RAW_FRAMES = Path("/data/fj/F2DMAS/00-论文优化重构/数据管理/01-输入图像/01-raw_frames")
COLMAP_LOCKED = Path("/data/fj/F2DMAS/00-论文优化重构/数据管理/02-位姿COLMAP/03-final_locked")
PIPELINE = Path("/data/fj/F2DMAS/00-论文优化重构/数据管理/07-运行脚本与超参"
                "/S20-RAP-FSAM3掩膜生成与验证/脚本/生成RAP-FSAM3掩膜.py")
SAM3_PT = Path("/data/fj/F2DMAS/sam3/sam3.pt")

TRACK_A_SAMPLES = ["BaiZhang", "DouBanLv2"]
TEST_SHA_EXPECTED = "7504fcbbc2a495502534a6fbbd27e9a57fb6b24ccefb573291851478d08c0157"


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def csv_canonical_text(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    return text.replace("﻿", "").replace("\r\n", "\n")


def git(*args: str) -> tuple[int, str]:
    r = subprocess.run(["git", *args], capture_output=True, text=True, cwd=str(WORKSPACE))
    return r.returncode, (r.stdout + r.stderr).strip()


def main() -> None:
    lines: list[str] = []
    def out(s: str = "") -> None:
        print(s)
        lines.append(s)

    out("# Phase 17A — Starting-State Audit")
    out("")
    out(f"**Generated at (UTC)**: {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}")
    out(f"**Phase root**: `{PHASE17A.name}`")
    out("**Purpose**: read-only verification of pre-execution state for the budgeted 10-GT pilot + unlabeled TEST behavioral audit.")
    out("")

    # ---- git state ----
    out("## 1. Repository State")
    out("")
    rc, branch = git("branch", "--show-current")
    out(f"| Item | Value |")
    out(f"|---|---|")
    out(f"| branch | `{branch}` |")
    rc, head = git("rev-parse", "HEAD")
    out(f"| HEAD | `{head}` |")
    rc, status = git("status", "--short")
    out(f"| working tree | `{'clean' if not status else status}` |")
    rc, log = git("log", "-5", "--oneline")
    out("")
    out("```")
    out(log)
    out("```")
    out("")

    # ---- Phase 16 manifest re-derivation ----
    out("## 2. Phase 16 Manifest Integrity (re-derivation)")
    out("")
    manifest_json = P16_LISTS / "Phase16_manifest.json"
    meta = json.loads(manifest_json.read_text(encoding="utf-8"))
    full_rows = list(csv.DictReader((P16_LISTS / "Phase16_selection_manifest_preGT.csv").open(encoding="utf-8-sig")))
    dev_rows = [r for r in full_rows if r["split"] == "DEV"]
    test_rows = [r for r in full_rows if r["split"] == "TEST"]

    import io
    def hash_rows(rows: list[dict]) -> str:
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=list(rows[0].keys()), extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
        return sha256_hex(buf.getvalue().replace("﻿", "").replace("\r\n", "\n"))

    h_full = hash_rows(full_rows)
    h_dev = hash_rows(dev_rows)
    h_test = hash_rows(test_rows)

    out("| Manifest | frozen SHA256 | re-derived SHA256 | match |")
    out("|---|---|---|---|")
    out(f"| full ({len(full_rows)} rows) | `{meta['sha256_full_manifest']}` | `{h_full}` | {'✅' if h_full == meta['sha256_full_manifest'] else '❌ MISMATCH'} |")
    out(f"| DEV ({len(dev_rows)} rows) | `{meta['sha256_dev_manifest']}` | `{h_dev}` | {'✅' if h_dev == meta['sha256_dev_manifest'] else '❌ MISMATCH'} |")
    out(f"| TEST ({len(test_rows)} rows) | `{meta['sha256_test_manifest']}` | `{h_test}` | {'✅' if h_test == meta['sha256_test_manifest'] else '❌ MISMATCH'} |")
    out("")
    test_sha = meta["sha256_test_manifest"]
    out(f"**TEST SHA256 (immutable for Phase 17A)**: `{test_sha}`")
    out(f"**Matches plan expectation**: {'✅' if test_sha == TEST_SHA_EXPECTED else '⚠️ differs — see report note'}")
    out("")

    # ---- GT10 masks ----
    out("## 3. Challenge GT Masks (Track A targets)")
    out("")
    gt_frames: list[tuple[str, str]] = []
    for sdir in sorted(GT_CHALLENGE.iterdir()):
        if not sdir.is_dir():
            continue
        for p in sorted(sdir.glob("mask_potted_clean_*.png")):
            frame = p.stem.replace("mask_potted_clean_", "")
            gt_frames.append((sdir.name, frame))
    out(f"| GT mask | exists | size |")
    out(f"|---|---|---|")
    for s, f in sorted(gt_frames):
        p = GT_CHALLENGE / s / f"mask_potted_clean_{f}.png"
        sz = p.stat().st_size if p.exists() else 0
        out(f"| {s}/{f} | {'✅' if p.exists() else '❌'} | {sz} B |")
    out("")
    out(f"**Total GT masks**: {len(gt_frames)} (expected 10)")
    qa_p6 = {}
    if QA_CSV.exists():
        for r in csv.DictReader(QA_CSV.open(encoding="utf-8-sig")):
            qa_p6[(r["sample"], r["frame"])] = r["formal_p6_valid"]
    p6_all = all(qa_p6.get((s, f)) == "True" for s, f in gt_frames)
    out(f"**QA formal_p6_valid=True for all 10**: {'✅' if p6_all else '❌ CHECK'}")
    out("")

    # ---- TEST frames ----
    out("## 4. Locked TEST Frames (Track B inputs)")
    out("")
    missing = []
    for r in test_rows:
        p = Path(r["image_path"])
        if not p.exists():
            missing.append(r["challenge_id"])
    out(f"| Check | Value |")
    out(f"|---|---|")
    out(f"| TEST rows in frozen manifest | {len(test_rows)} |")
    out(f"| on-disk frames present | {len(test_rows) - len(missing)}/{len(test_rows)} |")
    out(f"| missing | {', '.join(missing) if missing else 'none'} |")
    out("")

    # ---- COLMAP + raw frames for Track A ----
    out("## 5. COLMAP Availability (Track A samples)")
    out("")
    for s in TRACK_A_SAMPLES:
        sp = COLMAP_LOCKED / s / "sparse" / "0"
        ok = (sp / "cameras.bin").exists() and (sp / "images.bin").exists() and (sp / "points3D.bin").exists()
        n_raw = len(list((RAW_FRAMES / s).glob("*.jpg"))) if (RAW_FRAMES / s).exists() else 0
        out(f"| {s} | COLMAP sparse/0 {'✅' if ok else '❌'} | raw frames: {n_raw} |")
    out("")

    # ---- pipeline + SAM3 ----
    out("## 6. Pipeline & Weights")
    out("")
    out(f"| Resource | path | exists |")
    out(f"|---|---|---|")
    out(f"| pipeline script | `{PIPELINE}` | {'✅' if PIPELINE.exists() else '❌'} |")
    out(f"| SAM3 checkpoint | `{SAM3_PT}` | {'✅' if SAM3_PT.exists() else '❌'} |")
    out(f"| python | `{sys.executable}` | ✅ |")
    out("")

    # ---- GPU snapshot ----
    out("## 7. GPU Snapshot")
    out("")
    try:
        g = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,utilization.gpu", "--format=csv"],
                           capture_output=True, text=True, timeout=30)
        out("```")
        out(g.stdout.strip())
        out("```")
    except Exception as exc:  # noqa: BLE001
        out(f"nvidia-smi unavailable: {exc}")
    out("")

    # ---- summary ----
    out("## 8. Audit Conclusion")
    out("")
    ok = (h_full == meta["sha256_full_manifest"] and h_dev == meta["sha256_dev_manifest"]
          and h_test == meta["sha256_test_manifest"] and p6_all and not missing)
    out(f"**Starting-state gate**: {'PASS — proceed' if ok else '⚠️ anomalies above must be resolved before runs'}")
    out("")
    out("---")
    out("")
    out("Relevant constraints carried into Phase 17A:")
    out("- The 10 GT frames are **PILOT_GT10** (DEV material), not a held-out confirmatory test.")
    out("- The 32 locked TEST frames have **no GT**; Track B audits execution/behavior only (no accuracy).")
    out("- No pseudo-GT is created; no TEST frame is scored against GT.")
    out("- **No automatic git push.**")

    out_md = PHASE17A / "Phase17A_starting_audit.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nSaved → {out_md}")


if __name__ == "__main__":
    main()
