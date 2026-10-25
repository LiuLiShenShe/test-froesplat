#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 21 — disabled-path invariant for the new training-time mechanisms.

Phase 20E3 froze an AST equality test asserting that the branch trainer differs
from the baseline trainer only inside the capacity-control block.  Phase 21 adds
gated foreground/M2M3 code to ``training()``, so that equality is *by design* no
longer true.  This module re-states the invariant in the terms that actually
matter here, at the level Phase 21 can enforce:

  T1 `run_m2m3_go_update` early-returns on `mode=off` and outside its window
  T2 every other Phase-21 helper call in `training()` is switch-guarded
     (`if <switch>:` or the `A if <switch> else None` conditional form)
  T3 the mask-provenance write is guarded by an enabled() switch

`run_m2m3_go_update` is called unconditionally inside the iteration loop on
purpose: it owns its own gate (T1), which keeps the loop body readable and makes
the no-op path testable in one place.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

TRAIN_PY = Path("/data/fj/F2DMAS/2d-gaussian-splatting-m2m3-floor40/train.py")
# helpers that must not run at all when their switch is off
PHASE21_HELPERS = {
    "build_view_weights",
    "compute_view_quality_weights",
    "write_mask_provenance",
    "verify_mask_provenance",
    "init_m2m3_go_state",
    "_sample_gt_colors",
}
SELF_GATED = "run_m2m3_go_update"


def parse_tree() -> ast.Module:
    return ast.parse(TRAIN_PY.read_text(encoding="utf-8"))


def find_function(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"function {name} not found in {TRAIN_PY}")


def is_switch(test: ast.AST) -> bool:
    source = ast.unparse(test)
    return any(token in source for token in ("enabled(", "opt.", "getattr("))


def guarded_calls(function: ast.FunctionDef) -> tuple[list[str], list[str]]:
    """Split Phase-21 helper calls into switch-guarded and unguarded ones."""
    guarded: list[str] = []
    unguarded: list[str] = []

    def visit(node: ast.AST, guard_depth: int) -> None:
        if isinstance(node, ast.If):
            inner = guard_depth + 1 if is_switch(node.test) else guard_depth
            for child in node.body:
                visit(child, inner)
            for child in node.orelse:
                visit(child, guard_depth)
            return
        if isinstance(node, ast.IfExp):
            inner = guard_depth + 1 if is_switch(node.test) else guard_depth
            visit(node.body, inner)
            visit(node.orelse, guard_depth)
            return
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in PHASE21_HELPERS:
                (guarded if guard_depth > 0 else unguarded).append(node.func.id)
        for child in ast.iter_child_nodes(node):
            visit(child, guard_depth)

    for statement in function.body:
        visit(statement, 0)
    return guarded, unguarded


def test_t1_mode_off_is_a_guard_clause():
    tree = parse_tree()
    function = find_function(tree, SELF_GATED)
    gates = [node for node in function.body if isinstance(node, ast.If)]
    assert gates, f"{SELF_GATED} must start with early-return gates"
    for node in gates[:3]:
        assert any(isinstance(child, ast.Return) for child in node.body)
    assert "off" in ast.unparse(gates[0].test)
    assert "m2m3_go_start_iter" in ast.unparse(gates[1].test)
    assert "m2m3_go_interval" in ast.unparse(gates[2].test)


def test_t2_phase21_helpers_are_switch_guarded():
    tree = parse_tree()
    guarded, unguarded = guarded_calls(find_function(tree, "training"))
    assert unguarded == [], (
        "these Phase-21 helpers are called outside any switch guard in training(): "
        f"{unguarded}")
    assert set(guarded) >= {"build_view_weights", "init_m2m3_go_state"}


def test_t3_mask_provenance_write_is_guarded():
    tree = parse_tree()
    guarded, unguarded = guarded_calls(find_function(tree, "training"))
    assert "write_mask_provenance" not in unguarded
    assert "write_mask_provenance" in guarded


if __name__ == "__main__":
    sys.exit(__import__("pytest").main([__file__, "-v"]))
