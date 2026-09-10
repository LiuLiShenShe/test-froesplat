#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 18 — Production-Default / Paper-Code Consistency Tests (T1–T6)

Verifies that the paper's "final algorithm definition" (Phase18_final_algorithm_spec.md)
exactly matches the S20 production code defaults + evidence tables, WITHOUT re-running
any SAM3 inference or modifying anything.

Run:
  pytest 阶段十八_论文实验收束与最终审计/tests/test_phase18_paper_code_consistency.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

S20_SCRIPT = Path("/data/fj/F2DMAS/00-论文优化重构/数据管理/07-运行脚本与超参/S20-RAP-FSAM3掩膜生成与验证/脚本")
WORKSPACE = Path("/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/01-算法模块修改与验证")

if str(S20_SCRIPT) not in sys.path:
    sys.path.insert(0, str(S20_SCRIPT))

from ranking_policy import POLICY_NAMES, primary_and_tiebreak, select_best_policy  # noqa: E402


# ---------------------------------------------------------------------------
# helper: extract the real argparse defaults from the production script (regex, no exec)
# ---------------------------------------------------------------------------
def load_s20_text():
    return (S20_SCRIPT / "生成RAP-FSAM3掩膜.py").read_text(encoding="utf-8")


def probe_s20_defaults() -> dict:
    """Extract argparse defaults directly from the S20 source (source of truth).

    No exec — pure regex on the production file, so no torch/SAM3 import occurs.
    store_true flags (A6/A7) have no explicit default ⇒ argparse default False.
    """
    import re

    text = load_s20_text()
    defaults = {}

    for argname in ("ranking_policy", "score_weights"):
        m = re.search(
            rf'add_argument\(\s*"--{argname}"[^)]*?default\s*=\s*("([^"]*)"|([A-Za-z0-9_.]+))',
            text,
            re.S,
        )
        if m:
            defaults[argname] = m.group(2) if m.group(2) is not None else m.group(3)

    for argname in ("use_cross_view_consensus", "use_memory_propagation"):
        m = re.search(rf'add_argument\(\s*"--{argname}"', text)
        # store_true ⇒ store_true present ⇒ default False
        if m and "store_true" in text[m.start():m.start() + 300]:
            defaults[argname] = False
    return defaults


# ---------------------------------------------------------------------------
# T1 — --ranking_policy default == "r1"
# ---------------------------------------------------------------------------
def test_ranking_policy_default_is_r1():
    defaults = probe_s20_defaults()
    assert defaults.get("ranking_policy") == "r1", f"S20 --ranking_policy default = {defaults.get('ranking_policy')!r}"
    assert "r1" in POLICY_NAMES

    src = load_s20_text()
    # ScoreRecord dataclass default must be r1
    assert 'ranking_policy: str = "r1"' in src, "ScoreRecord.ranking_policy default not r1"
    # getattr fallbacks must default to r1 (3 call sites)
    n = src.count('getattr(args, "ranking_policy", "r1")')
    assert n == 3, f"Expected 3 getattr fallbacks of 'r1', found {n}"


# ---------------------------------------------------------------------------
# T2 — A6 default == False
# ---------------------------------------------------------------------------
def test_a6_cross_view_consensus_default_false():
    defaults = probe_s20_defaults()
    assert defaults.get("use_cross_view_consensus") is False
    # paper spec + architecture decision agree with production
    out = WORKSPACE / "阶段十八_论文实验收束与最终审计"
    spec = (out / "Phase18_final_algorithm_spec.md").read_text(encoding="utf-8")
    assert "DEFAULT OFF" in spec or "disabled by default" in spec


# ---------------------------------------------------------------------------
# T3 — A7 default == False
# ---------------------------------------------------------------------------
def test_a7_memory_propagation_default_false():
    defaults = probe_s20_defaults()
    assert defaults.get("use_memory_propagation") is False


# ---------------------------------------------------------------------------
# T4 — --score_weights default == "area=1,comp=1,edge=1,temp=1,contrast=1,sam=0.5"
# ---------------------------------------------------------------------------
def test_score_weights_default():
    defaults = probe_s20_defaults()
    assert defaults.get("score_weights") == "area=1,comp=1,edge=1,temp=1,contrast=1,sam=0.5", \
        defaults.get("score_weights")
    src = load_s20_text()
    assert 'default="area=1,comp=1,edge=1,temp=1,contrast=1,sam=0.5"' in src


# ---------------------------------------------------------------------------
# T5 — ScoreRecord exposes ranking_policy, primary_score, tie_break_score, contrast_effective
# ---------------------------------------------------------------------------
def test_score_record_fields():
    src = load_s20_text()
    for field in ('ranking_policy: str = "r1"', "primary_score: float = 0.0",
                  "tie_break_score: float = 0.0", "contrast_effective: float = 0.0"):
        assert field in src, f"ScoreRecord missing/diverged field: {field}"


# ---------------------------------------------------------------------------
# T6 — 提示词评分.csv output columns include 排名策略/主评分/对比度调制/平局决胜分
# ---------------------------------------------------------------------------
def test_score_csv_has_policy_columns():
    src = load_s20_text()
    for col in ("排名策略", "主评分", "对比度调制", "平局决胜分"):
        assert col in src, f"提示词评分.csv output missing column: {col}"


# ---------------------------------------------------------------------------
# T7 — paper tables carry the exact protocol numbers (no drift vs Phase17C evidence)
# ---------------------------------------------------------------------------
def test_paper_gt10_table_matches_production_replay():
    """Team-forward guard: the paper GT10 table must equal Phase17C numbers."""
    import csv

    out = WORKSPACE / "阶段十八_论文实验收束与最终审计"
    table = out / "Phase18_final_GT10_table.csv"
    assert table.exists(), "Phase18_final_GT10_table.csv missing"
    with open(table, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    r1 = [r for r in rows if r["policy"].startswith("R1")]
    assert r1, "R1 row missing in GT10 table"
    mean = float(r1[0]["mean_F1"])
    # Phase17C verified: R1 DEV10 mean F1 = 0.9728
    assert abs(mean - 0.9728) < 5e-4, f"GT10 R1 mean {mean} drifts from 0.9728"


# ---------------------------------------------------------------------------
# T8 — failure-rescue table: 3/3 frames rescued to ≥0.95 with zero baseline
# ---------------------------------------------------------------------------
def test_failure_rescue_table_3_of_3():
    import csv

    out = WORKSPACE / "阶段十八_论文实验收束与最终审计"
    table = out / "Phase18_failure_rescue_table.csv"
    assert table.exists()
    with open(table, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3
    for r in rows:
        assert float(r["baseline_F1"]) == 0.0
        assert float(r["R1_selected_F1"]) >= 0.95
        assert float(r["R1_selected_F1"]) >= float(r["oracle_F1"]) - 1e-9


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))