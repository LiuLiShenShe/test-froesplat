#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Targeted Floor40 contracts (beyond the block-frame smoke tests in
``test_recovery_round2_contracts.py``).

The manuscript's local capacity floor is only meaningful if, across the whole
run, (a) block ids keep denoting the same spatial region even as the cloud
densifies or drifts in coordinate range, (b) the per-block reference tracks the
region's peak population, (c) pruning can never cut a block below its floor, and
(d) densification round-trips back into the *same* frozen frame.

These functions are pure-tensor (see ``utils/m2m3_go.py``), so the tests run on
CPU in the RAP venv without the CUDA rasteriser.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path("/data/fj/F2DMAS")
GS = ROOT / "2d-gaussian-splatting-m2m3-floor40"


def _load_m2m3():
    spec = importlib.util.spec_from_file_location("m2m3_floor40_targeted", GS / "utils/m2m3_go.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["m2m3_floor40_targeted"] = mod
    spec.loader.exec_module(mod)
    return mod


# ── (a) coordinate-range change ───────────────────────────────────────────────

def test_frozen_frame_ignores_a_later_coordinate_range_shift():
    """Translating/scaling the cloud must not move the frozen grid.

    `resolve_block_frame` is *supposed* to freeze on first call.  A rebuild that
    recomputes bounds from the current cloud (the pre-fix behaviour) would move
    every primitive to a different block, silently re-associating each block's
    capacity reference with a new region.
    """
    m2m3 = _load_m2m3()
    state = {}
    first = torch.rand(500, 3) * 4.0
    frame = m2m3.resolve_block_frame(first, 6, state)

    # Same physical cloud, but reported in a shifted coordinate range.
    shifted = first + torch.tensor([100.0, -50.0, 7.0])
    frame_shifted = m2m3.resolve_block_frame(shifted, 6, state)
    assert torch.equal(frame[0], frame_shifted[0])
    assert torch.equal(frame[1], frame_shifted[1])

    # Points far outside the frozen frame clamp to the edge blocks, never
    # producing ids outside [0, n^3) and never rescaling the grid.
    far = torch.tensor([[1e3, 0.0, 0.0], [-1e3, 0.0, 0.0], [0.0, 1e3, 0.0]])
    ids = m2m3.assign_local_blocks(far, 6, frame=frame_shifted)
    assert int(ids.max()) < 6 ** 3
    assert int(ids.min()) >= 0
    # Clamping is per-axis: a point beyond +x only pins the x coordinate.
    lo, hi = frame_shifted

    def block_axis(value, axis):
        span = (hi[axis] - lo[axis]).clamp_min(1e-6)
        return int(((value - lo[axis]) / span * 6).floor().clamp(0, 5))

    assert int(ids[0]) == block_axis(1e3, 0) * 36 + block_axis(0.0, 1) * 6 + block_axis(0.0, 2)
    assert block_axis(1e3, 0) == 5 and block_axis(-1e3, 0) == 0


def test_same_region_keeps_its_id_when_the_cloud_densifies():
    """Densification inside the frozen frame must not renumber existing points."""
    m2m3 = _load_m2m3()
    state = {}
    base = torch.rand(300, 3) * 5.0
    frame = m2m3.resolve_block_frame(base, 6, state)
    ids_base = m2m3.assign_local_blocks(base, 6, frame=frame)

    extra = torch.rand(900, 3) * 5.0
    grown = torch.cat([base, extra], dim=0)
    ids_grown = m2m3.assign_local_blocks(grown, 6, frame=m2m3.resolve_block_frame(grown, 6, state))
    assert torch.equal(ids_base, ids_grown[:300])


# ── (b) per-block reference tracks the region population ─────────────────────

def test_block_reference_is_the_running_maximum_per_region():
    """`reference = max(reference, counts)` only raises blocks that grew."""
    m2m3 = _load_m2m3()
    state = {}
    cloud = torch.rand(1200, 3) * 3.0
    frame = m2m3.resolve_block_frame(cloud, 4, state)
    ids = m2m3.assign_local_blocks(cloud, 4, frame=frame)
    n_blocks = 4 ** 3
    counts_round1 = torch.bincount(ids, minlength=n_blocks)

    reference = torch.maximum(torch.zeros(n_blocks, dtype=torch.long), counts_round1)
    state["block_reference"] = reference

    # Round 2: add 400 points that all land in one block.
    hot_block = int(counts_round1.argmax())
    hot_centre = torch.tensor([
        ((hot_block // 16) + 0.5) / 4,
        (((hot_block // 4) % 4) + 0.5) / 4,
        ((hot_block % 4) + 0.5) / 4,
    ]) * (frame[1] - frame[0]) + frame[0]
    added = hot_centre + (torch.rand(400, 3) - 0.5) * 0.1
    round2 = torch.cat([cloud, added], dim=0)
    ids2 = m2m3.assign_local_blocks(round2, 4, frame=frame)
    counts_round2 = torch.bincount(ids2, minlength=n_blocks)
    reference2 = torch.maximum(state["block_reference"], counts_round2)
    assert int(reference2[hot_block]) == int(counts_round2[hot_block]) > int(counts_round1[hot_block])
    # Blocks that did not grow keep their earlier reference.
    same = counts_round1 == counts_round2
    assert bool((reference2[same] == counts_round1[same]).all())
    # block_min is a per-block fraction of the reference.
    block_min = torch.round(reference2.float() * 0.40).long()
    assert int(block_min[hot_block]) == int(round(int(reference2[hot_block]) * 0.40))


# ── (c) pruning can never cross the floor ────────────────────────────────────

def test_pruning_respects_the_per_block_floor():
    """A block already at its floor must reject every removal candidate."""
    m2m3 = _load_m2m3()
    n = 40
    block_ids = torch.zeros(n, dtype=torch.long)          # all one block
    utility = torch.linspace(0.0, 1.0, n)                 # ascending; 0 = worst
    candidates = torch.arange(n, dtype=torch.long)
    block_counts = torch.tensor([n])
    block_min = torch.tensor([n])                         # floor == population
    remove, report = m2m3.local_floor_select(
        candidates, utility, block_ids, block_counts, block_min, max_remove=n)
    assert int(remove.sum()) == 0
    assert report["floor_blocked"] == n
    assert report["removed"] == 0


def test_pruning_stops_exactly_at_the_floor():
    """With floor 30/40 the controller may remove 10 and no more."""
    m2m3 = _load_m2m3()
    n = 40
    block_ids = torch.zeros(n, dtype=torch.long)
    utility = torch.linspace(0.0, 1.0, n)
    candidates = torch.arange(n, dtype=torch.long)
    block_counts = torch.tensor([n])
    block_min = torch.tensor([30])
    remove, report = m2m3.local_floor_select(
        candidates, utility, block_ids, block_counts, block_min, max_remove=n)
    assert int(remove.sum()) == 10
    assert int(block_counts[0] - remove.sum()) == 30
    # The lowest-utility primitives are the ones kept for removal.
    assert bool(remove[:10].all()) and not bool(remove[10:].any())
    assert report["removed"] == 10


def test_pruning_multiblock_floor_is_per_region():
    """A global budget must not starve one block below its own floor."""
    m2m3 = _load_m2m3()
    # Block 0: 10 prims at floor 10 (protected). Block 1: 10 prims at floor 2.
    n = 20
    block_ids = torch.cat([torch.zeros(10, dtype=torch.long),
                           torch.ones(10, dtype=torch.long)])
    utility = torch.cat([torch.linspace(0.0, 0.5, 10),
                         torch.linspace(0.0, 1.0, 10)])
    candidates = torch.arange(n, dtype=torch.long)
    block_counts = torch.tensor([10, 10])
    block_min = torch.tensor([10, 2])
    remove, report = m2m3.local_floor_select(
        candidates, utility, block_ids, block_counts, block_min, max_remove=n)
    # Block 0 contributes nothing; block 1 may lose at most 8.
    assert not bool(remove[:10].any())
    assert int(remove[10:].sum()) == 8
    assert int(block_counts[1] - remove[10:].sum()) == 2


# ── (d) densification selects a budgeted, ranked subset ──────────────────────

def test_densify_rule_selects_the_manuscript_condition():
    m2m3 = _load_m2m3()
    n = 100
    s_m2g = torch.zeros(n)
    s_g2m = torch.zeros(n)
    residual = torch.zeros(n)
    valid = torch.ones(n, dtype=torch.bool)
    # 30 qualifying cases, 70 non-qualifying.
    s_m2g[:30] = 0.9
    s_g2m[:30] = torch.linspace(0.1, 0.9, 30)
    residual[:30] = 0.2
    selected = m2m3.select_densify_primitives(
        s_m2g, s_g2m, residual, valid, tau_m=0.5, tau_h=0.5, tau_r=0.9, budget=10)
    assert selected.numel() == 10
    # Ranked by (s_g2m + residual) descending -> the ten highest s_g2m.
    expected = torch.argsort(s_g2m[:30], descending=True)[:10]
    assert set(selected.tolist()) == set(expected.tolist())
    assert int(selected.max()) < 30


def test_densify_is_empty_without_deficit_evidence():
    m2m3 = _load_m2m3()
    n = 50
    zeros = torch.zeros(n)
    selected = m2m3.select_densify_primitives(
        zeros, zeros, zeros, torch.ones(n, dtype=torch.bool),
        tau_m=0.5, tau_h=0.5, tau_r=0.5, budget=10)
    assert selected.numel() == 0


def test_densify_budget_is_hard_capped():
    m2m3 = _load_m2m3()
    n = 200
    ones = torch.ones(n)
    selected = m2m3.select_densify_primitives(
        ones, ones, ones, torch.ones(n, dtype=torch.bool),
        tau_m=0.5, tau_h=0.5, tau_r=0.5, budget=7)
    assert selected.numel() == 7
