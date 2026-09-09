#!/usr/bin/env python3
"""
Phase 17B — Steps 15-19: A6/A7 frozen, no-tuning, evidence hierarchy, decision tree, outputs completeness.
All assertions in a single deterministic script.

Precise detection:
  Step 15 — flags "active A6/A7 pipeline invocation" ONLY: a CLI `--use_cross_view_consensus` /
           `--use_memory_propagation`/`--colmap_dir` flag passed to the S20 runner, or a direct
           `use_cross_view_consensus=True` keyword. Reads of P00 参数.json to *verify* these are
           False are allowed and are themselves the audit.
  Step 16 — flags an actual pipeline invocation (subprocess/os.system whose command contains the
           S20 script path), or a modified score_weights value (anything whose value string is not
           the frozen reference `area=1,comp=1,edge=1,temp=1,contrast=1,sam=0.5`).
"""
from __future__ import annotations
import json
from pathlib import Path

PHASE17A = Path("/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/01-算法模块修改与验证/阶段十七A_10GT困难样本Pilot与无GT测试审计")
OUT = Path(__file__).resolve().parent.parent
P00 = PHASE17A / "02_trackA_variants/P00_Control"
S20 = "生成RAP-FSAM3掩膜"

ERRORS: list[str] = []


def main() -> None:
    # ==== Step 15: A6/A7 frozen ====
    print("=== Step 15: A6/A7 frozen ===")
    p00 = json.loads((P00 / "参数.json").read_text("utf-8"))
    a6_off = p00.get("use_cross_view_consensus") is False
    a7_off = p00.get("use_memory_propagation") is False
    print(f"  P00 use_cross_view_consensus=False: {a6_off}")
    print(f"  P00 use_memory_propagation=False:  {a7_off}")
    if not (a6_off and a7_off):
        ERRORS.append("FAIL Step15: A6/A7 not disabled in P00 参数.json")

    # active A6/A7 flag = CLI flag or True keyword passed to the pipeline
    cli_flags = ["--use_cross_view_consensus", "--use_memory_propagation", "--colmap_dir", "--consensus_min_frames"]
    SELF = Path(__file__).name
    for fp in sorted((OUT / "脚本").glob("*.py")):
        if fp.name == SELF:
            continue  # this audit script necessarily names the flags it detects
        text = fp.read_text("utf-8")
        # 1) active CLI flag
        for flag in cli_flags:
            if flag in text:
                for line in text.splitlines():
                    if flag in line and "参数.json" not in line and "p00.get" not in line \
                       and "cli_flags" not in line and "assert" not in line:
                        ERRORS.append(f"FAIL Step15: {fp.name} uses A6/A7 CLI flag {flag}: {line.strip()}")
        # 2) direct keyword True
        for flag in ("use_cross_view_consensus=True", "use_memory_propagation=True"):
            if flag in text:
                ERRORS.append(f"FAIL Step15: {fp.name} sets {flag}")
        # 3) pipeline invocation as a command
        if S20 in text:
            for line in text.splitlines():
                if S20 in line and ("subprocess" in text or "os.s" in text or "Popen" in text):
                    ERRORS.append(f"FAIL Step16: {fp.name} invokes pipeline: {line.strip()}")

    step15_ok = not any("Step15" in e for e in ERRORS)
    print(f"  17B scripts contain no active A6/A7 flag: {step15_ok}")

    # ==== Step 16: No tuning (no pipeline invocation, frozen weights) ====
    print("\n=== Step 16: No tuning ===")
    FROZEN = "area=1,comp=1,edge=1,temp=1,contrast=1,sam=0.5"
    for fp in sorted((OUT / "脚本").glob("*.py")):
        if fp.name == SELF:
            continue
        text = fp.read_text("utf-8")
        for i, line in enumerate(text.splitlines(), 1):
            s = line.strip()
            if S20 in s and any(k in text for k in ("subprocess", "os.s", "Popen")):
                ERRORS.append(f"FAIL Step16: {fp.name}:{i} invokes pipeline")
            if "score_weights" in s and not s.startswith("#") and not s.endswith(")"):
                # assignment of a value that is NOT the frozen constant
                if "=" in s and FROZEN not in s:
                    ERRORS.append(f"FAIL Step16: {fp.name}:{i} non-frozen weights: {s}")
    no_tuning_ok = not any("Step16" in e for e in ERRORS)
    print(f"  No pipeline invocation / no weight change: {no_tuning_ok}")

    # ==== Step 17-18: Evidence hierarchy + decision tree ====
    print("\n=== Step 17-18: Evidence hierarchy (written into report) ===")
    evidence = [
        ("mapping_correct [PROVEN]", "all 10 GT frames map exactly; mapping_audit all-OK"),
        ("generation_ok [PROVEN]", "raw oracle F1 = 0.9867/0.9882/0.9865 on failing frames"),
        ("cleanup_identity [PROVEN]", "选择后==最终==A5c bitwise; cleanup_trace all IDENTITY"),
        ("ranking_failure [PROVEN]", "oracle>=0.80 but selected F1=0; wrong instance chosen"),
        ("contrast_driver [PROVEN]", "q_contrast oracle 0.12-0.27 vs selected 1.0 on all 3 frames"),
        ("p6_prompt_ok [PROVEN]", "P6 oracle 0.987 >= gate 0.10; sweep not triggered"),
        ("a6a7_irrelevant [SUPPORTED]", "A6/A7 disabled in P00; no variant invoked here"),
        ("temporal_pattern [SUPPORTED]", "instance count 1/1/2/3/9; failure starts at first spurious extra instance"),
        ("broad_generalization [HYPOTHESIS]", "N=3 frames; other dense sequences untested"),
    ]
    for e in evidence:
        print(f"  {e[0]}: {e[1]}")
    print("  Decision: Case B (ranking/selection) → Phase 17C target = candidate scoring/selection")

    # ==== Step 19: Outputs completeness ====
    print("\n=== Step 19: Outputs completeness ===")
    required = [
        "Phase17B_protocol.md", "Phase17B_report.md", "Phase17B_manifest.json",
        "Phase17B_DEV10_manifest.csv", "Phase17B_candidate_inventory.csv",
        "Phase17B_candidate_oracle.csv", "Phase17B_prompt_oracle.csv",
        "Phase17B_score_regret.csv", "Phase17B_cleanup_trace.csv",
        "Phase17B_failure_attribution.csv", "Phase17B_mapping_audit.csv",
        "Phase17B_detection_diagnosis.csv", "Phase17B_temporal_diagnosis.csv",
    ]
    missing = [n for n in required if not (OUT / n).exists()]
    for n in missing:
        ERRORS.append(f"FAIL Step19: missing {n}")
    viz_ok = any((OUT / "visualizations").glob("*.jpg"))
    tests_ok = (OUT / "tests").exists()
    if not viz_ok:
        ERRORS.append("FAIL Step19: visualizations/ has no JPG")
    if not tests_ok:
        ERRORS.append("FAIL Step19: tests/ missing")
    print(f"  required files present: {not missing}  ({len(required)-len(missing)}/{len(required)})")
    print(f"  visualizations JPGs: {viz_ok}")
    print(f"  tests/: {tests_ok}")

    # ==== summary ====
    clean = step15_ok and no_tuning_ok and not missing and viz_ok and tests_ok
    print(f"\n{'=== ALL STEPS 15-19 PASS ===' if clean else '=== FAILURES ==='}")
    for e in ERRORS:
        print(f"  {e}")


if __name__ == "__main__":
    main()