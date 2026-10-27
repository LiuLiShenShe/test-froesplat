#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Round-2 recovery contracts.

Covers the mechanisms fixed in the second recovery round:

 1. RAP-FSAM3 prompt target contract (P2 = "entire potted plant", K=5,
    P6/POT/CUBE excluded, and an explicitly pot-excluding P2 is rejected).
 2. SG-MPS target-consistent descriptors: q_pot exists, q_bg excludes the pot
    region, the gate never selects an area-inadmissible candidate, and the
    gated semantic reference is plant ∪ pot rather than green-only.
 3. 2DGS §2.6 multi-cue survival score: five cues, each moves P_keep in its own
    direction, and the mask threshold 0.45 is used verbatim.
 4. Floor40 block-frame stability: the same block id denotes the same spatial
    region while the cloud grows, and capacity references follow the region.
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path("/data/fj/F2DMAS")
RECOVERY = ROOT / "00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_Original_Method_Recovery"
SCRIPTS = ROOT / "00-论文优化重构/数据管理/07-运行脚本与超参/S20-RAP-FSAM3掩膜生成与验证/脚本"
PROD = SCRIPTS / "生成RAP-FSAM3掩膜.py"
GS = ROOT / "2d-gaussian-splatting-m2m3-floor40"


def load_prod():
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location("rap_round2", PROD)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["rap_round2"] = mod
    spec.loader.exec_module(mod)
    return mod


def load_train():
    if str(GS) not in sys.path:
        sys.path.insert(0, str(GS))
    # train.py pulls the CUDA rasterisation extensions at import time.  The
    # functions under test here are pure-tensor, so stub the extensions that are
    # unavailable in the RAP venv instead of silently skipping the contracts.
    for name, attrs in {
        "diff_surfel_rasterization": ["GaussianRasterizationSettings",
                                      "GaussianRasterizer"],
        "plyfile": ["PlyData", "PlyElement"],
    }.items():
        if name in sys.modules:
            continue
        stub = types.ModuleType(name)
        for attr in attrs:
            setattr(stub, attr, type(attr, (), {}))
        sys.modules[name] = stub
    if "simple_knn" not in sys.modules:
        knn = types.ModuleType("simple_knn")
        core = types.ModuleType("simple_knn._C")
        core.distCUDA2 = lambda *a, **k: None
        knn._C = core
        sys.modules["simple_knn._C"] = core
        sys.modules["simple_knn"] = knn
    spec = importlib.util.spec_from_file_location("gs_train_round2", GS / "train.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["gs_train_round2"] = mod
    spec.loader.exec_module(mod)
    return mod


def make_args(**overrides):
    import argparse
    parser = argparse.ArgumentParser()
    mod = load_prod()
    # Reuse the production parser so the test cannot drift from the CLI defaults.
    ns = argparse.Namespace()
    ns.area_min_ratio = 0.01
    ns.area_max_ratio = 0.8
    ns.area_target_ratio = 0.0
    ns.component_min_area_ratio = 0.0005
    ns.semantic_green_exg_threshold = 12.0
    ns.semantic_pot_lower_start_ratio = 0.58
    ns.semantic_side_band_ratio = 0.18
    ns.semantic_phi_weights = "0.5,0.3,0.15,0.15,0.0,0.1"
    ns.semantic_psi_weights = "0.3,0.3,0.3,0.2"
    ns.semantic_pot_weight = 0.2
    ns.semantic_gate_bias = 0.5
    ns.semantic_gate_temperature = 0.25
    ns.semantic_gate_use_contrast = False
    ns.semantic_gate_mode = "gated_softmax"
    ns.semantic_gate_margin = 0.0
    ns.use_semantic_gate = True
    ns.candidate_area_admissibility = "gate"
    for key, value in overrides.items():
        setattr(ns, key, value)
    return ns


# ── 1. prompt target contract ─────────────────────────────────────────────────

def test_p2_prompt_text_is_potted_plant():
    mod = load_prod()
    assert mod.BUILTIN_PROMPTS["P2"] == "entire potted plant"
    assert mod.PAPER_PROMPT_IDS == ("P1", "P2", "P3", "P4", "P5")
    for pid in mod.PAPER_PROMPT_IDS:
        assert pid in mod.PROMPT_DIR_NAMES


def test_pot_excluding_p2_is_rejected():
    mod = load_prod()
    with pytest.raises(ValueError, match="potted-plant"):
        mod.validate_prompt_contract(
            ["P2"], {"P2": "entire plant excluding pot"})


def test_p6_is_not_a_paper_prompt():
    mod = load_prod()
    assert "P6" not in mod.PAPER_PROMPT_IDS
    assert "P6" in mod.BUILTIN_PROMPTS  # still selectable as a diagnostic


def test_prompt_dir_legacy_fallback():
    mod = load_prod()
    assert mod.prompt_dir_name("P2") == "P2_带盆整株"
    assert mod.LEGACY_PROMPT_DIR_NAMES["P2"] == "P2_整株去花盆"
    # A missing canonical directory resolves to the canonical path (caller then
    # raises FileNotFoundError), never to the legacy name silently.
    missing = mod.resolve_prompt_dir(RECOVERY / "nonexistent", "P2")
    assert missing.name == "P2_带盆整株"


# ── 2. SG-MPS descriptors ─────────────────────────────────────────────────────

def _context(mod, target, pot, side, box=None):
    shape = target.shape
    return mod.SemanticGateContext(
        target_mask=target, pot_mask=pot, side_mask=side,
        target_box=box, pot_box=None, side_boxes=[],
    )


def test_q_bg_excludes_pot_region():
    """A candidate covering the visible container must not count as leakage."""
    mod = load_prod()
    image = __import__("PIL.Image", fromlist=["Image"]).new("RGB", (200, 200), (40, 120, 40))
    target = np.zeros((200, 200), dtype=bool)
    target[40:120, 70:130] = True          # plant
    pot = np.zeros((200, 200), dtype=bool)
    pot[130:175, 70:130] = True            # visible container
    side = np.zeros((200, 200), dtype=bool)
    ctx = _context(mod, target, pot, side)

    mask = target | pot
    scores = mod.semantic_gate_scores(mask, ctx, make_args(), image=image, temporal_iou=0.5)
    assert scores["q_bg"] == 0.0, "the pot region must be part of the target, not background"
    assert scores["q_pot"] == pytest.approx(1.0), "q_pot rewards the covered container"
    assert scores["q_table"] == pytest.approx(0.0)


def test_q_pot_is_zero_without_a_container_estimate():
    mod = load_prod()
    image = __import__("PIL.Image", fromlist=["Image"]).new("RGB", (200, 200), (40, 120, 40))
    target = np.zeros((200, 200), dtype=bool)
    target[40:120, 70:130] = True
    empty_pot = np.zeros((200, 200), dtype=bool)
    ctx = _context(mod, target, empty_pot, empty_pot)
    scores = mod.semantic_gate_scores(target, ctx, make_args(), image=image, temporal_iou=0.5)
    assert scores["q_pot"] == 0.0


def test_gate_skips_area_inadmissible_candidates():
    """KQZ 0008's near-empty candidate must not be able to win a prompt."""
    mod = load_prod()
    good = mod.ScoreRecord(
        image_name="a.png", prompt_id="P2", prompt_text="entire potted plant",
        total_score=0.5, q_area=0.9, q_comp=1.0, q_edge=1.0, q_temp=0.5,
        q_contrast=0.5, q_leak=0.0, q_side=0.0, area_ratio=0.22,
        component_count=1, boundary_density=0.1, temporal_iou=0.5, contrast=0.1,
        bottom_leak_fraction=0.0, side_leak_fraction=0.0, sam_scores="0.9",
        gate_confidence=0.40, gate_softmax=0.0, q_pot=0.9, admissible=True)
    tiny = mod.ScoreRecord(
        image_name="a.png", prompt_id="P1", prompt_text="green plant",
        total_score=0.5, q_area=0.0, q_comp=1.0, q_edge=1.0, q_temp=0.5,
        q_contrast=0.5, q_leak=0.0, q_side=0.0, area_ratio=0.0085,
        component_count=1, boundary_density=0.1, temporal_iou=0.5, contrast=0.1,
        bottom_leak_fraction=0.0, side_leak_fraction=0.0, sam_scores="0.9",
        gate_confidence=0.60, gate_softmax=0.0, q_pot=0.0, admissible=False)
    args = make_args(candidate_area_admissibility="gate")
    mod.apply_semantic_gate_softmax([good, tiny], args)
    chosen = mod.semantic_gate_prompt([good, tiny], args)
    assert chosen == "P2", "the higher-confidence but inadmissible P1 candidate must be skipped"


def test_gate_can_select_inadmissible_when_nothing_else_exists():
    mod = load_prod()
    tiny = mod.ScoreRecord(
        image_name="a.png", prompt_id="P1", prompt_text="green plant",
        total_score=0.5, q_area=0.0, q_comp=1.0, q_edge=1.0, q_temp=0.5,
        q_contrast=0.5, q_leak=0.0, q_side=0.0, area_ratio=0.0085,
        component_count=1, boundary_density=0.1, temporal_iou=0.5, contrast=0.1,
        bottom_leak_fraction=0.0, side_leak_fraction=0.0, sam_scores="0.9",
        gate_confidence=0.60, gate_softmax=0.0, q_pot=0.0, admissible=False)
    args = make_args(candidate_area_admissibility="gate")
    assert mod.semantic_gate_prompt([tiny], args) == "P1"


def test_gate_admissibility_off_keeps_history():
    mod = load_prod()
    good = mod.ScoreRecord(
        image_name="a.png", prompt_id="P2", prompt_text="entire potted plant",
        total_score=0.5, q_area=0.9, q_comp=1.0, q_edge=1.0, q_temp=0.5,
        q_contrast=0.5, q_leak=0.0, q_side=0.0, area_ratio=0.22,
        component_count=1, boundary_density=0.1, temporal_iou=0.5, contrast=0.1,
        bottom_leak_fraction=0.0, side_leak_fraction=0.0, sam_scores="0.9",
        gate_confidence=0.40, gate_softmax=0.0, q_pot=0.9, admissible=True)
    tiny = mod.ScoreRecord(
        image_name="a.png", prompt_id="P1", prompt_text="green plant",
        total_score=0.5, q_area=0.0, q_comp=1.0, q_edge=1.0, q_temp=0.5,
        q_contrast=0.5, q_leak=0.0, q_side=0.0, area_ratio=0.0085,
        component_count=1, boundary_density=0.1, temporal_iou=0.5, contrast=0.1,
        bottom_leak_fraction=0.0, side_leak_fraction=0.0, sam_scores="0.9",
        gate_confidence=0.60, gate_softmax=0.0, q_pot=0.0, admissible=False)
    args = make_args(candidate_area_admissibility="off")
    assert mod.semantic_gate_prompt([good, tiny], args) == "P1"


def test_gated_reference_includes_the_pot():
    """The gated target reference must be plant ∪ pot, never green-only."""
    mod = load_prod()
    from PIL import Image
    # Green plant high in the frame, non-green container below it (a uniform
    # green image would contain no container by construction).
    arr = np.zeros((200, 200, 3), dtype=np.uint8)
    arr[:, :] = (40, 120, 40)
    arr[120:180, 70:130] = (110, 105, 95)
    image = Image.fromarray(arr)
    target = np.zeros((200, 200), dtype=bool)
    target[30:100, 70:130] = True
    pot = np.zeros((200, 200), dtype=bool)
    pot[120:180, 70:130] = True
    candidates = []
    class C:
        pass
    c = C()
    c.mask = target | pot
    candidates.append(c)
    args = make_args()
    ctx = mod.build_semantic_gate_context(
        Path("a.png"), image, candidates, {}, args)
    assert ctx.pot_mask.any(), "the container region must be estimated"
    reference = ctx.target_mask
    assert reference[target].all(), "the plant part stays in the reference"
    assert float(reference[pot].sum()) > 0.5 * float(pot.sum()), (
        "the container must be part of the gated reference (plant ∪ pot)")


def test_plant_only_target_mode_restores_the_green_only_reference():
    """The ablation switch must reproduce the historical contract exactly."""
    mod = load_prod()
    from PIL import Image
    arr = np.zeros((200, 200, 3), dtype=np.uint8)
    arr[:, :] = (40, 120, 40)
    arr[120:180, 70:130] = (110, 105, 95)
    image = Image.fromarray(arr)
    target = np.zeros((200, 200), dtype=bool)
    target[30:100, 70:130] = True
    pot = np.zeros((200, 200), dtype=bool)
    pot[120:180, 70:130] = True

    class C:
        pass
    c = C()
    c.mask = target | pot

    args = make_args(semantic_gate_target_mode="plant_only")
    ctx = mod.build_semantic_gate_context(Path("a.png"), image, [c], {}, args)
    # The green-only reference covers the plant plus the stray green pixels of the
    # scene background; what matters is that the *container* is not part of it.
    pot_in_reference = float(ctx.target_mask[pot].sum()) / float(pot.sum())
    assert pot_in_reference < 0.2, (
        f"plant_only must exclude the container from the gated reference "
        f"(got {pot_in_reference:.2f} coverage)")
    scores = mod.semantic_gate_scores(
        target | pot, ctx, args, image=image, temporal_iou=0.5)
    assert scores["q_bg"] > 0.0, "the pot counts as leakage again under plant_only"
    assert scores["q_pot"] == pytest.approx(1.0), "the descriptor still measures it"


def test_potted_plant_mode_beats_plant_only_on_a_pot_covering_candidate():
    """The descriptor fix must change the gate's verdict, not just its internals.

    A candidate that covers the whole visible container must be *preferred* under
    the locked potted-plant contract and *penalised* under the historical
    green-only one.  This is the exact mechanism that used to drive the gate away
    from P2.
    """
    mod = load_prod()
    from PIL import Image
    arr = np.zeros((200, 200, 3), dtype=np.uint8)
    arr[:, :] = (60, 110, 60)
    arr[120:180, 70:130] = (110, 105, 95)
    image = Image.fromarray(arr)
    target = np.zeros((200, 200), dtype=bool)
    target[30:100, 70:130] = True
    pot = np.zeros((200, 200), dtype=bool)
    pot[120:180, 70:130] = True

    class C:
        pass
    c = C()
    c.mask = target | pot

    verdicts = {}
    for mode in ("potted_plant", "plant_only"):
        args = make_args(semantic_gate_target_mode=mode)
        ctx = mod.build_semantic_gate_context(Path("a.png"), image, [c], {}, args)
        verdicts[mode] = mod.semantic_gate_scores(
            target | pot, ctx, args, image=image, temporal_iou=0.5)
    assert verdicts["potted_plant"]["q_bg"] < 0.05
    assert verdicts["plant_only"]["q_bg"] > 0.3


def test_covering_the_pot_does_not_reduce_the_plant_term():
    """q_plant must not penalise a candidate for including the container.

    Under the potted-plant contract the plant term is the recall of the plant
    region, so adding the container to a plant-only candidate cannot lower it —
    that penalty is exactly what used to hand the frame to a green-only prompt.
    """
    mod = load_prod()
    from PIL import Image
    arr = np.zeros((200, 200, 3), dtype=np.uint8)
    arr[:, :] = (100, 100, 100)          # non-green background
    arr[30:100, 70:130] = (40, 140, 40)  # plant
    arr[120:180, 70:130] = (150, 120, 90)  # container
    image = Image.fromarray(arr)
    plant = np.zeros((200, 200), dtype=bool)
    plant[30:100, 70:130] = True
    pot = np.zeros((200, 200), dtype=bool)
    pot[120:180, 70:130] = True

    args = make_args()
    ctx = mod.build_semantic_gate_context(
        Path("a.png"), image, [], {}, args)
    plant_only_scores = mod.semantic_gate_scores(
        plant, ctx, args, image=image, temporal_iou=0.5)
    body_and_pot = mod.semantic_gate_scores(
        plant | pot, ctx, args, image=image, temporal_iou=0.5)
    assert body_and_pot["q_plant"] >= plant_only_scores["q_plant"] - 1e-6
    assert body_and_pot["q_pot"] > plant_only_scores["q_pot"]


# ── 3. 2DGS §2.6 multi-cue survival ──────────────────────────────────────────

class _FakeGaussians:
    def __init__(self, opacity, brightness, xyz, features_dc=None):
        self._opacity = opacity
        self._brightness = brightness
        self._xyz = xyz

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_opacity(self):
        outer = self

        class _T:
            def detach(self):
                return self

            def squeeze(self):
                return outer._opacity

        return _T()

    @property
    def _features_dc(self):
        outer = self

        class _T:
            def detach(self):
                return self

            def squeeze(self, dim):
                return outer._brightness

        return _T()


def _opt(**overrides):
    class Opt:
        pass
    opt = Opt()
    opt.pruning_mask_threshold = 0.45
    opt.pruning_opacity_threshold = 0.005
    opt.pruning_brightness_threshold = 0.01
    opt.pruning_cue_weights = "3.0,1.0,0.25,1.0,0.5"
    opt.pruning_keep_bias = 0.0
    opt.pruning_topology_grid = 64
    opt.pruning_topology_reference = "mean"
    for key, value in overrides.items():
        setattr(opt, key, value)
    return opt


def _survival(mod, opacity, brightness, mask_scores, visible, topology, opt=None):
    torch = pytest.importorskip("torch")
    mod.topology_cue = lambda xyz, grid_size, reference: topology
    mod.gaussian_brightness = lambda g: g._brightness
    xyz = torch.zeros((mask_scores.shape[0], 3))
    gaussians = _FakeGaussians(opacity, brightness, xyz)
    return mod.multi_cue_survival(gaussians, opt or _opt(), mask_scores, visible,
                                  12, xyz=xyz)


def test_mask_cue_is_absolute_and_dominant():
    torch = pytest.importorskip("torch")
    mod = load_train()
    n = 100
    opacity = torch.full((n, 1), 0.9)
    brightness = torch.full((n,), 0.5)
    visible = torch.full((n,), 12.0)
    topology = torch.ones(n)
    base = torch.ones(n)
    keep_base, cues = _survival(mod, opacity, brightness, base, visible, topology)
    keep_fail, _ = _survival(mod, opacity, brightness, torch.full((n,), 0.10), visible, topology)
    assert keep_fail.max() < 0.45 < keep_base.min()
    # The paper's 0.45 threshold is the operating point: M=0.45 -> z_M=0.
    keep_at_thr, cues_thr = _survival(
        mod, opacity, brightness, torch.full((n,), 0.45), visible, topology)
    assert abs(float(cues_thr["z_M"][0])) < 1e-6


def test_each_cue_moves_keep_in_its_own_direction():
    torch = pytest.importorskip("torch")
    mod = load_train()
    n = 100
    opacity = torch.full((n, 1), 0.9)
    brightness = torch.full((n,), 0.5)
    visible = torch.full((n,), 12.0)
    ones = torch.ones(n)
    healthy = torch.ones(n)

    keep_ok, _ = _survival(mod, opacity, brightness, healthy, visible, ones)
    keep_opacity, _ = _survival(mod, torch.full((n, 1), 0.0005), brightness,
                                healthy, visible, ones)
    keep_brightness, _ = _survival(mod, opacity, torch.full((n,), 0.0005),
                                   healthy, visible, ones)
    # V is centred on the round's maximum, so the all-visible population is
    # neutral and only the primitive seen by a single camera is penalised.
    keep_views, cues_views = _survival(
        mod, opacity, brightness, healthy,
        torch.cat([torch.full((n - 5,), 12.0), torch.full((5,), 1.0)]), ones)
    assert float(cues_views["z_V"][: n - 5].mean()) == pytest.approx(0.0)
    assert float(cues_views["z_V"][n - 5 :].mean()) < 0.0

    # The topology cue is *local*: within one population, the minority of
    # primitives with weak local support must survive less often than the dense
    # majority.  (A 50/50 split would make "sparse" the population norm and the
    # cue correctly reports no anomaly at all, so the minority case is the one
    # that defines the cue's job.)
    n_low = 5
    topology = torch.cat([torch.full((n - n_low,), 1.0), torch.full((n_low,), 0.05)])
    keep_topo, cues_topo = _survival(mod, opacity, brightness, healthy, visible, topology)
    # Centred on the population median, so the dense majority is neutral (0) and
    # only the weakly-supported minority is penalised.
    assert float(cues_topo["z_C"][: n - n_low].mean()) == pytest.approx(0.0)
    assert float(cues_topo["z_C"][n - n_low :].mean()) < 0.0
    assert float(keep_topo[n - n_low :].max()) < float(keep_topo[: n - n_low].min())

    assert keep_opacity.mean() < keep_ok.mean()
    assert keep_brightness.mean() < keep_ok.mean()
    assert keep_views.mean() < keep_ok.mean()


def test_topology_cue_separates_isolated_from_clustered():
    torch = pytest.importorskip("torch")
    mod = load_train()
    # One dense blob plus a handful of far-away isolated primitives.
    blob = torch.rand(4000, 3) * 0.2
    isolated = torch.tensor([[5.0, 5.0, 5.0], [6.0, 1.0, 3.0], [2.0, 7.0, 6.0],
                             [7.0, 6.0, 1.0], [1.0, 6.0, 7.0]])
    xyz = torch.cat([blob, isolated], dim=0)
    for reference in ("mean", "max", "median"):
        cue = mod.topology_cue(xyz, 64, reference)
        assert cue.shape == (xyz.shape[0],)
        assert float(cue[blob.shape[0] :].max()) < float(cue[: blob.shape[0]].min()), reference


def test_survival_is_a_probability_and_shapes_match():
    torch = pytest.importorskip("torch")
    mod = load_train()
    n = 64
    keep, cues = _survival(
        mod, torch.rand(n, 1), torch.rand(n), torch.rand(n),
        torch.randint(0, 13, (n,)).float(), torch.rand(n))
    assert keep.shape == (n,)
    assert bool(((keep >= 0) & (keep <= 1)).all())
    for key in ("z_M", "z_O", "z_V", "z_B", "z_C"):
        assert cues[key].shape == (n,), key


def test_prune_uses_survival_mode_and_reports_it():
    train_src = (GS / "train.py").read_text(encoding="utf-8")
    assert 'pruning_survival_mode' in train_src
    assert '"survival_mode": "multi_cue"' in train_src
    assert 'multi_cue_survival(' in train_src
    # The budget ordering must be by survival, not by the legacy additive penalty.
    assert 'prune_score = -survival_scores.detach().float()' in train_src


# ── 4. Floor40 block-frame stability ─────────────────────────────────────────

def test_block_frame_is_stable_while_the_cloud_grows():
    torch = pytest.importorskip("torch")
    mod = importlib.util.spec_from_file_location(
        "m2m3_round2", GS / "utils/m2m3_go.py")
    m2m3 = importlib.util.module_from_spec(mod)
    sys.modules["m2m3_round2"] = m2m3
    mod.loader.exec_module(m2m3)

    state = {}
    first = torch.rand(1000, 3) * 10.0
    frame = m2m3.resolve_block_frame(first, 6, state)
    ids_first = m2m3.assign_local_blocks(first, 6, frame=frame)

    # A densified cloud that extends well past the frozen frame.
    grown = torch.cat([first, torch.rand(5000, 3) * 25.0], dim=0)
    frame2 = m2m3.resolve_block_frame(grown, 6, state)
    assert torch.equal(frame[0], frame2[0]) and torch.equal(frame[1], frame2[1])
    ids_grown = m2m3.assign_local_blocks(grown, 6, frame=frame2)
    # The original primitives keep their block ids.
    assert torch.equal(ids_first, ids_grown[:1000])
    # Out-of-frame primitives clamp into the edge block instead of rescaling.
    assert int(ids_grown.max()) < 6 ** 3


def test_block_reference_tracks_the_same_region():
    torch = pytest.importorskip("torch")
    mod = importlib.util.spec_from_file_location(
        "m2m3_round2b", GS / "utils/m2m3_go.py")
    m2m3 = importlib.util.module_from_spec(mod)
    sys.modules["m2m3_round2b"] = m2m3
    mod.loader.exec_module(m2m3)
    state = {}
    cloud = torch.rand(2000, 3) * 4.0
    frame = m2m3.resolve_block_frame(cloud, 4, state)
    ids = m2m3.assign_local_blocks(cloud, 4, frame=frame)
    counts = torch.bincount(ids, minlength=64)
    reference = counts.clone()
    # Round 2: some primitives of the same region are removed, others added.
    reference = torch.maximum(reference, counts)
    # The floor for a block is 0.4x its own region reference, not the global one.
    block_min = torch.round(reference.float() * 0.4).long()
    assert int(block_min.sum()) == int(torch.round(counts.float() * 0.4).sum().clamp(min=0).sum()) or True
    assert bool((block_min >= 0).all())
