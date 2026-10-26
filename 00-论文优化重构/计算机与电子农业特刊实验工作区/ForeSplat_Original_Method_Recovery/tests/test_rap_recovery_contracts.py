#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Recovery-workspace contracts for the manuscript-faithful RAP-FSAM3 paths.

These tests target the code that the recovery work actually changed:

  T1  SG-MPS softmax normalises over PROMPTS, not over instances
      (manuscript §2.3.3: a_i^(k), k = prompt index)
  T2  SB-FBPR positive prompts are the distance-transform TopK interior points
      (manuscript §2.3.4: P+ = TopK_{p in M*}(d_M*(p), K+))
  T3  SB-FBPR negative prompts all fall OUTSIDE the mask and are spread over
      more than one sub-region (manuscript §2.3.4: R- = ring ∪ table ∪ bg ∪ nbr)
  T4  the point-prompt adapter produces the (N, 1, 2) / (N, 1) shapes that
      `Prompt.append_points` asserts on
  T5  SB-FBPR with `--spnp_prompt_form points` sends points and zero boxes
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path("/data/fj/F2DMAS/00-论文优化重构/数据管理/07-运行脚本与超参/"
               "S20-RAP-FSAM3掩膜生成与验证/脚本")
PIPELINE = SCRIPTS / "生成RAP-FSAM3掩膜.py"
MOD_NAME = "rap_recovery_contracts"


def load_pipeline():
    if MOD_NAME in sys.modules:
        return sys.modules[MOD_NAME]
    spec = importlib.util.spec_from_file_location(MOD_NAME, PIPELINE)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[MOD_NAME] = mod
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mod():
    return load_pipeline()


def make_args(**overrides):
    base = dict(
        component_min_area_ratio=0.0005,
        area_min_ratio=0.01, area_max_ratio=0.80, area_target_ratio=0.0,
        edge_density_min=0.0, edge_density_max=1.0,
        spnp_positive_points=5, spnp_negative_points=8,
        spnp_ring_radius=15, spnp_negative_nms_radius=0,
        spnp_negative_region_mode="manuscript",
        spnp_table_band_start_ratio=0.58, spnp_side_band_ratio=0.18,
        semantic_green_exg_threshold=12.0,
        semantic_phi_weights="0.5,0.3,0.15,0.15,0.0,0.1",
        semantic_psi_weights="0.3,0.3,0.3,0.2",
        semantic_gate_bias=0.5, semantic_gate_use_contrast=False,
        semantic_gate_mode="gated_softmax",
        semantic_gate_temperature=0.25,
        ranking_policy="r1",
        spnp_prompt_form="points",
        spnp_positive_box_padding=0.03,
        spnp_negative_box_radius=12,
        spnp_use_lower_negative=False,
        spnp_lower_band_ratio=0.70,
        spnp_lower_exg_threshold=0.0,
        spnp_min_refined_iou=0.65,
        spnp_max_remove_ratio=0.15,
        confidence_threshold=0.3,
        sam3_mask_threshold=0.5,
        device="cpu",
        use_semantic_gate=True,
        disable_amp=True,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def test_sgmps_softmax_is_per_prompt(mod):
    """Two instances of P2 must receive the SAME prompt probability, and the
    distribution must normalise over the prompts that actually have candidates."""
    def rec(prompt_id, conf):
        return types_ScoreRecord(mod, prompt_id, conf)

    records = [rec("P1", 0.40), rec("P2", 0.60), rec("P2", 0.30), rec("P3", 0.20)]
    mod.apply_semantic_gate_softmax(records, make_args())
    by_prompt = {}
    for r in records:
        by_prompt.setdefault(r.prompt_id, set()).add(round(r.gate_softmax, 12))
    # same prompt -> single shared probability
    for prompt_id, values in by_prompt.items():
        assert len(values) == 1, f"{prompt_id} got different probabilities {values}"
    total = sum(next(iter(v)) for v in by_prompt.values())
    assert total == pytest.approx(1.0, abs=1e-9), total
    # argmax over prompts == argmax over gate confidence
    assert max(by_prompt, key=lambda p: next(iter(by_prompt[p]))) == "P2"


def types_ScoreRecord(mod, prompt_id, conf):
    return mod.ScoreRecord(
        image_name="x.jpg", prompt_id=prompt_id, prompt_text=prompt_id,
        total_score=conf, q_area=0.0, q_comp=0.0, q_edge=0.0, q_temp=0.0,
        q_contrast=0.0, q_leak=0.0, q_side=0.0, area_ratio=0.0,
        component_count=1, boundary_density=0.0, temporal_iou=0.5,
        contrast=0.0, bottom_leak_fraction=0.0, side_leak_fraction=0.0,
        sam_scores="", instance_id=0, empty_flag=False,
        ranking_policy="r1", primary_score=conf, tie_break_score=0.0,
        contrast_effective=0.0, gate_confidence=conf,
    )


def test_positive_points_are_distance_topk(mod):
    """Every positive prompt must sit inside the mask; the deepest point must be
    selected, and the selection must be a TopK of the distance transform."""
    mask = np.zeros((200, 200), dtype=bool)
    mask[50:150, 60:140] = True           # rectangle, interior depth up to 50 px
    args = make_args(spnp_positive_points=5)
    pts = mod.sample_points(mask, 5, True, args.spnp_ring_radius)
    assert len(pts) == 5
    for x, y in pts:
        assert mask[y, x], "positive prompt outside the mask"
    dist = mod.ndimage.distance_transform_edt(mask)
    deepest = float(dist.max())
    chosen = max(float(dist[y, x]) for x, y in pts)
    assert chosen == pytest.approx(deepest, abs=1e-6), (chosen, deepest)


def test_negative_regions_are_outside_mask(mod):
    from PIL import Image
    mask = np.zeros((240, 240), dtype=bool)
    mask[80:160, 90:150] = True
    image = Image.fromarray(np.full((240, 240, 3), 128, dtype=np.uint8))
    args = make_args()
    regions = mod.manuscript_negative_regions(mask, image, args)
    assert set(regions) == {"ring", "table", "bg", "neighbour"}
    for name, region in regions.items():
        assert not (region & mask).any(), f"{name} region overlaps the mask"
    neg = mod.sample_negative_prompts(regions, 8, 0, mask.shape)
    assert len(neg) == 8
    for x, y in neg:
        assert not mask[y, x], "negative prompt inside the mask"
    hit = {name for name, region in regions.items()
           if region.any() and any(region[y, x] for x, y in neg)}
    assert len(hit) >= 2, f"negative prompts collapsed onto {hit}"


def test_point_adapter_shapes(mod):
    import torch
    adapter = importlib.import_module("sam3_prompt_adapter")

    class FakePrompt:
        def __init__(self):
            self.calls = []

        def append_points(self, points, labels, mask=None):
            assert points.ndim == 3 and points.shape[1:] == (1, 2), points.shape
            assert labels.ndim == 2 and labels.shape[1] == 1, labels.shape
            assert points.shape[0] == labels.shape[0]
            self.calls.append((points.detach().cpu().numpy(),
                               labels.detach().cpu().numpy()))

    state = {"geometric_prompt": FakePrompt()}
    added = adapter.append_point_prompts(state, [(0.2, 0.3), (0.4, 0.5)],
                                        [(0.8, 0.9)], "cpu")
    assert added == {"positive": 2, "negative": 1}
    prompt = state["geometric_prompt"]
    assert len(prompt.calls) == 2
    pos_pts, pos_lab = prompt.calls[0]
    neg_pts, neg_lab = prompt.calls[1]
    assert pos_lab.reshape(-1).tolist() == [1, 1]
    assert neg_lab.reshape(-1).tolist() == [0]
    assert np.allclose(pos_pts.reshape(-1, 2), [[0.2, 0.3], [0.4, 0.5]], atol=1e-6)
    assert np.allclose(neg_pts.reshape(-1, 2), [[0.8, 0.9]], atol=1e-6)


def test_prompt_form_selects_points_not_boxes(mod):
    """The refinement must send points and no boxes when prompt_form=points."""
    sent = {"points": 0, "boxes": 0}

    class FakeProcessor:
        def set_image(self, image):
            return {"backbone_out": None, "geometric_prompt": None}

        def reset_all_prompts(self, state):
            pass

        def set_text_prompt(self, state=None, prompt=""):
            state["geometric_prompt"] = FakePrompt()
            return {"masks": None, "scores": None, "boxes": None}

        def add_geometric_prompt(self, state=None, box=None, label=True):
            sent["boxes"] += 1

        def _forward_grounding(self, state):
            prompt = state["geometric_prompt"]
            sent["points"] += sum(len(c[1]) for c in prompt.calls)
            return {"masks": None, "scores": None, "boxes": None}

    class FakePrompt:
        def __init__(self):
            self.calls = []

        def append_points(self, points, labels, mask=None):
            self.calls.append((points, labels))

    # `sam3_box_refinement` needs a real torch module for autocast; the empty
    # mask path lets us stop right after prompt dispatch.
    torch = pytest.importorskip("torch")
    image = __import__("PIL.Image", fromlist=["Image"]).new("RGB", (64, 64), (10, 20, 10))
    mask = np.zeros((64, 64), dtype=bool)
    mask[20:44, 20:44] = True
    processor = FakeProcessor()
    args = make_args(spnp_prompt_form="points", disable_amp=True)
    mod.sam3_box_refinement(image, mask, "entire potted plant",
                            [(32, 32)], [(2, 2)], processor, torch, args)
    assert sent["boxes"] == 0, "box prompt sent while prompt_form=points"
    assert sent["points"] == 2, sent


def test_gate_margin_defers_on_near_tie(mod):
    """A near-tie must not be broken by the gate: `--semantic_gate_margin`
    makes it report "no preference" so the ranking policy decides."""
    def rec(prompt_id, conf):
        return types_ScoreRecord(mod, prompt_id, conf)

    tied = [rec("P1", 0.501), rec("P2", 0.500), rec("P3", 0.499)]
    assert mod.semantic_gate_prompt(tied, make_args()) == "P1"
    assert mod.semantic_gate_prompt(
        tied, make_args(semantic_gate_margin=0.05)) is None

    separated = [rec("P1", 0.40), rec("P2", 0.90)]
    assert mod.semantic_gate_prompt(
        separated, make_args(semantic_gate_margin=0.05)) == "P2"
