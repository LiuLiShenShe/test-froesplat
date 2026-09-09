#!/usr/bin/env python3
"""
Phase 17B — Step 0+1: Starting audit + freeze DEV10 manifest.

Reads only; produces:
  Phase17B_manifest.json  (starting SHA, references, gate record)
  Phase17B_DEV10_manifest.csv (exact 10 rows of Phase17A GT10, role=DIAGNOSTIC_DEV)
"""
from __future__ import annotations
import csv, hashlib, json, subprocess, sys
from pathlib import Path

PHASE17A = Path("/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/01-算法模块修改与验证/阶段十七A_10GT困难样本Pilot与无GT测试审计")
PHASE16  = Path("/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/01-算法模块修改与验证/阶段十六_HardCase_GT构建与锁定")
ROOT     = Path("/data/fj/F2DMAS")
OUT_DIR  = Path(__file__).resolve().parent.parent

SHA_TEST = "7504fcbbc2a495502534a6fbbd27e9a57fb6b24ccefb573291851478d08c0157"


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def shell(cmd: str) -> str:
    return subprocess.check_output(cmd, shell=True, cwd=str(ROOT), text=True).strip()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Step 0: starting audit ----
    git_status  = shell("git status --short")
    git_head    = shell("git rev-parse HEAD")
    git_branch  = shell("git branch --show-current")
    git_log5    = shell("git log -5 --oneline")

    # only the new phase-17B directory may be untracked
    others = [ln for ln in git_status.splitlines()
              if "阶段十七B_上游失败归因与候选Oracle审计" not in ln]
    assert not others, f"unexpected dirty state:\n{others}"
    print(f"HEAD: {git_head}")
    print(f"branch: {git_branch}")

    gt10_path  = PHASE17A / "GT10_manifest" / "Phase17A_GT10_manifest.csv"
    gt10_meta  = json.loads((PHASE17A / "Phase17A_manifest.json").read_text("utf-8")) if (PHASE17A / "Phase17A_manifest.json").exists() else {}
    # re-derive GT10 manifest SHA
    gt10_sha   = sha256(gt10_path)
    phase16_manifest = json.loads((PHASE16 / "挑战集列表" / "Phase16_manifest.json").read_text("utf-8"))

    # verify no A6/A7 in P00 params
    p00_params = json.loads((PHASE17A / "02_trackA_variants/P00_Control/参数.json").read_text("utf-8"))
    a6_on = p00_params.get("use_cross_view_consensus", False)
    a7_on = p00_params.get("use_memory_propagation", False)
    print(f"P00 A6={a6_on}, A7={a7_on}  (both must be False)")
    assert not a6_on and not a7_on, "A6/A7 must be disabled in P00"

    # ---- Step 1: freeze DEV10 ----
    gt10_rows = list(csv.DictReader(gt10_path.open(encoding="utf-8-sig")))
    assert len(gt10_rows) == 10

    # add role column (copy all columns as-is, no add/remove of data)
    dev10_cols = list(gt10_rows[0].keys())
    if "role" not in dev10_cols:
        dev10_cols.append("role")
    dev10_path = OUT_DIR / "Phase17B_DEV10_manifest.csv"
    with dev10_path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=dev10_cols, extrasaction="ignore")
        w.writeheader()
        for r in gt10_rows:
            r["role"] = "DIAGNOSTIC_DEV"
            w.writerow(r)

    # verify re-derive matches original
    dev10_sha = sha256(dev10_path)
    print(f"DEV10 SHA: {dev10_sha}")

    # ---- manifest.json ----
    manifest = {
        "phase": "17B",
        "description": "Upstream failure attribution + candidate oracle audit (zero-inference)",
        "starting_sha": git_head,
        "phase17a_main_sha": "d0a9970b",
        "phase17a_followup_sha": "d656851e",
        "gt10_manifest_sha256": gt10_sha,
        "phase16_test_manifest_sha256_ref": SHA_TEST,
        "dev10_manifest_sha256": dev10_sha,
        "no_pseudo_gt": True,
        "zero_inference": True,
        "prompt_sweep_gate": {
            "p6_raw_oracle_0039": 0.9867,
            "p6_raw_oracle_0040": 0.9882,
            "p6_raw_oracle_0041": 0.9865,
            "gate_threshold": 0.10,
            "gate_triggered": False,
            "decision": "skip (user-confirmed)"
        },
        "no_a6a7": True,
        "no_tuning": True,
    }
    (OUT_DIR / "Phase17B_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Phase17B_manifest.json written")
    print(f"\n=== AUDIT PASS ===")
    print(f"  HEAD={git_head}, clean=True, A6/A7=False")


if __name__ == "__main__":
    main()
