#!/usr/bin/env python3
"""RAP-FSAM3 mask generation and validation entrypoint.

This script keeps the old SAM3 script untouched and adds a parameterized,
ablation-friendly entrypoint for COMPAG experiments.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.ndimage import binary_closing, binary_dilation, binary_fill_holes, binary_opening
from skimage.metrics import structural_similarity
from ranking_policy import primary_and_tiebreak, select_best_policy, POLICY_NAMES
from candidate_metadata import file_sha256, load_cache_metadata, metadata_for_mask
from sam3_prompt_adapter import append_point_prompts


ROOT = Path("/data/fj/F2DMAS")
DEFAULT_SAM3_REPO = ROOT / "第三方源码" / "SAM3-latest"
DEFAULT_SAM3_CHECKPOINT = ROOT / "sam3" / "sam3.pt"
DEFAULT_COLMAP_LOADER = ROOT / "2d-gaussian-splatting-main" / "scene" / "colmap_loader.py"

BUILTIN_PROMPTS = {
    "P1": "green plant",
    "P2": "entire plant excluding pot",
    "P3": "leaves and stems",
    "P4": "crop seedling",
    "P5": "plant body without background",
    "P6": "whole potted plant including the pot",
    # ── 阶段十一 §5 独立语义候选（P2/P6 不再共用终选掩膜）──
    "POT": "flowerpot and tray",
    "CUBE": "blue calibration cube",
}

PROMPT_DIR_NAMES = {
    "P1": "P1_绿色植物",
    "P2": "P2_整株去花盆",
    "P3": "P3_叶和茎",
    "P4": "P4_作物幼苗",
    "P5": "P5_去背景植物体",
    "P6": "P6_带盆整株",
    "POT": "POT_花盆托盘",
    "CUBE": "CUBE_蓝色标定块",
}


@dataclass
class Candidate:
    prompt_id: str
    prompt_text: str
    mask: np.ndarray
    scores: list[float]
    raw_detection_count: int
    # ── 阶段十一扩展：逐实例候选（per_instance）所需字段 ──
    instance_id: int = 0
    box: tuple[int, int, int, int] | None = None
    sam_score: float = 0.0
    mask_threshold: float = 0.5
    source_stage: str = "raw"
    prompt_mode: str = "legacy_union"
    candidate_order: int = 0
    metadata_status: str = "GENERATED"
    source_image_name: str = ""
    source_image_sha256: str = ""
    mask_sha256: str = ""
    score_precision: str = "GENERATED_FLOAT"

@dataclass
class ScoreRecord:
    image_name: str
    prompt_id: str
    prompt_text: str
    total_score: float
    q_area: float
    q_comp: float
    q_edge: float
    q_temp: float
    q_contrast: float
    q_leak: float
    q_side: float
    area_ratio: float
    component_count: int
    boundary_density: float
    temporal_iou: float
    contrast: float
    bottom_leak_fraction: float
    side_leak_fraction: float
    sam_scores: str
    instance_id: int = 0
    semantic_enabled: bool = False
    semantic_total: float = 0.0
    target_box_score: float = 0.0
    vertical_coverage_score: float = 0.0
    pot_overlap_penalty: float = 0.0
    side_distractor_penalty: float = 0.0
    center_prior_score: float = 0.0
    leak_penalty: float = 0.0
    empty_flag: bool = False
    # ── Phase 17C §17/§19: ranking-policy fields (diagnostics + counterfactual selection) ──
    ranking_policy: str = "r1"
    primary_score: float = 0.0
    tie_break_score: float = 0.0
    contrast_effective: float = 0.0
    # ── Phase 21 §SG-MPS: manuscript gated-softmax descriptors ──
    gate_confidence: float = 0.0
    gate_softmax: float = 0.0
    phi_descriptor: str = ""
    psi_descriptor: str = ""


@dataclass
class SemanticGateContext:
    target_mask: np.ndarray
    pot_mask: np.ndarray
    side_mask: np.ndarray
    target_box: tuple[int, int, int, int] | None
    pot_box: tuple[int, int, int, int] | None
    side_boxes: list[tuple[int, int, int, int]]


@dataclass
class ColmapObservation:
    image_name: str
    mask_stem: str
    points: np.ndarray


@dataclass
class ConsensusResult:
    reference_mask: np.ndarray
    per_frame_masks: dict[str, np.ndarray]
    per_frame_info: dict[str, str | dict[str, Any]]
    geo_support: np.ndarray | None = None
    center_band_mask: np.ndarray | None = None


def build_consensus_context(
    selected_masks: dict[str, np.ndarray],
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A6 S1-S3: robust center estimate + center-weighted pixel vote support.

    Returns (freq_map, weight_map, static_map). static_map marks pixels that are
    masked in (nearly) every frame AND photometrically static across frames —
    the signature of an off-center distractor (e.g. a neighbouring plant) that
    never rotates with the turntable, as opposed to the target whose silhouette
    changes every frame.
    """
    if not selected_masks:
        raise ValueError("consensus requires at least one selected mask")
    shape = next(iter(selected_masks.values())).shape
    freq = np.zeros(shape, dtype=np.float32)
    for mask in selected_masks.values():
        freq += mask.astype(np.float32)
    freq /= float(max(len(selected_masks), 1))

    h, w = shape
    centers_x, centers_y, widths, heights = [], [], [], []
    for mask in selected_masks.values():
        bbox = mask_bbox(mask)
        if bbox is None:
            continue
        x0, y0, x1, y1 = bbox
        centers_x.append(0.5 * (x0 + x1) / w)
        centers_y.append(0.5 * (y0 + y1) / h)
        widths.append((x1 - x0) / w)
        heights.append((y1 - y0) / h)
    if not centers_x:
        return freq, np.ones(shape, dtype=np.float32), np.zeros(shape, dtype=bool)

    cx = float(np.median(centers_x))
    cy = float(np.median(centers_y))
    band_half_w = max(float(np.median(widths)), args.consensus_center_band_ratio) / 2.0
    band_half_h = max(float(np.median(heights)), args.consensus_center_band_ratio) / 2.0

    xs = np.arange(w, dtype=np.float32)[None, :] / max(w - 1, 1)
    ys = np.arange(h, dtype=np.float32)[:, None] / max(h - 1, 1)
    dx = np.maximum(np.abs(xs - cx) - band_half_w, 0.0)
    dy = np.maximum(np.abs(ys - cy) - band_half_h, 0.0)
    dist = np.sqrt(dx * dx + dy * dy)
    decay = float(args.consensus_center_decay)
    weight = np.power(decay, dist / 0.25).astype(np.float32)

    # Static-distractor evidence: near-constant mask coverage + near-constant
    # appearance (low temporal std of grayscale inside the always-masked area).
    # The turntable target rotates, so its pixels change every frame; a
    # neighbouring plant that never enters the turntable stays photometrically
    # fixed. Filled in by apply_cross_view_consensus, which owns the frames.
    always = freq >= 0.98
    static_map = np.zeros(shape, dtype=bool)
    return freq, weight, static_map, always


def estimate_static_distractor_map(
    gray_by_stem: dict[str, np.ndarray],
    always: np.ndarray,
    center_band_mask: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    """A6 S3b: photometric-static + always-masked + off-center -> distractor.

    The temporal std is computed at the (downscaled) gray resolution and the
    resulting static map is resized back to full mask resolution.
    """
    static = np.zeros(always.shape, dtype=bool)
    if not always.any() or len(gray_by_stem) < 3:
        return static
    gh, gw = next(iter(gray_by_stem.values())).shape
    always_s = np.array(
        Image.fromarray(always.astype(np.uint8) * 255).resize((gw, gh), Image.NEAREST)
    ) > 127
    band_s = np.array(
        Image.fromarray(center_band_mask.astype(np.uint8) * 255).resize((gw, gh), Image.NEAREST)
    ) > 127
    stack = np.stack([g.astype(np.float32) / 255.0 for g in gray_by_stem.values()], axis=0)
    std = stack.std(axis=0)
    candidate = always_s & (std <= float(args.consensus_static_std_threshold)) & (~band_s)
    min_area = max(16, int(candidate.size * args.consensus_min_area_ratio))
    candidate = remove_small_components(candidate, min_area)
    if candidate.any():
        static = np.array(
            Image.fromarray(candidate.astype(np.uint8) * 255).resize(
                (always.shape[1], always.shape[0]), Image.BILINEAR
            )
        ) > 127
        static = binary_opening(static, structure=kernel(5), iterations=1)
        static = remove_small_components(static, min_area)
    return static


def apply_cross_view_consensus(
    selected_masks: dict[str, np.ndarray],
    gray_by_stem: dict[str, np.ndarray],
    colmap_observations: dict[str, ColmapObservation],
    dirs: dict[str, Path] | None,
    args: argparse.Namespace,
) -> ConsensusResult | None:
    """A6 S4-S6: geometry-gated consensus refinement with per-frame fallback."""
    n_frames = len(selected_masks)
    if n_frames < max(2, args.consensus_min_frames):
        return None

    freq, weight, _unused_static, always = build_consensus_context(selected_masks, args)
    support = freq * weight

    # Center-band mask for the static-distractor test (recompute cheaply).
    h, w = freq.shape
    center_band_mask = np.zeros(freq.shape, dtype=bool)
    centers_x, centers_y, widths, heights = [], [], [], []
    for mask in selected_masks.values():
        bbox = mask_bbox(mask)
        if bbox is None:
            continue
        x0, y0, x1, y1 = bbox
        centers_x.append(0.5 * (x0 + x1) / w)
        centers_y.append(0.5 * (y0 + y1) / h)
        widths.append((x1 - x0) / w)
        heights.append((y1 - y0) / h)

    # Geometry channel FIRST: COLMAP foreground-track support (S4). The
    # turntable target is track-backed in every frame; the static neighbour
    # plant is not (its pixels are never reconstructed as foreground points).
    geo_support: np.ndarray | None = None
    usable_obs = [
        obs for stem, obs in colmap_observations.items() if stem in selected_masks and obs.points.size > 0
    ]
    if len(usable_obs) >= max(2, n_frames // 3):
        geo_support = np.zeros(freq.shape, dtype=bool)
        for obs in usable_obs:
            geo_support |= points_to_support_mask(obs.points, freq.shape, args.consensus_geometry_dilation)

    # Robust center prior from TRACK-BACKED core pixels, NOT from raw bboxes.
    # Bboxes are polluted by the neighbour leak itself (median width can reach
    # ~0.94 of the image), which would make the band cover the whole frame and
    # disable every off-center test downstream.
    core = (freq >= 0.98)
    if geo_support is not None and core.sum() > int(freq.size * 0.001):
        core = core & geo_support
    if core.any():
        ys_c, xs_c = np.nonzero(core)
        cx = float(xs_c.mean()) / max(w - 1, 1)
        cy = float(ys_c.mean()) / max(h - 1, 1)
        xs_std = float(xs_c.std()) / max(w - 1, 1)
        ys_std = float(ys_c.std()) / max(h - 1, 1)
    else:
        # Fallback to bbox medians; sane defaults keep the closure defined.
        cx = float(np.median(centers_x)) if centers_x else 0.5
        cy = float(np.median(centers_y)) if centers_y else 0.5
        xs_std = float(np.median(widths)) / 4.0 if widths else args.consensus_center_band_ratio / 4.0
        ys_std = float(np.median(heights)) / 4.0 if heights else args.consensus_center_band_ratio / 4.0
    bw = max(3.0 * xs_std, args.consensus_center_band_ratio / 2.0)
    bh = max(3.0 * ys_std, args.consensus_center_band_ratio / 2.0)
    xs = np.arange(w, dtype=np.float32)[None, :] / max(w - 1, 1)
    ys = np.arange(h, dtype=np.float32)[:, None] / max(h - 1, 1)
    center_band_mask = (np.abs(xs - cx) <= bw) & (np.abs(ys - cy) <= bh)

    static_map = estimate_static_distractor_map(gray_by_stem, always, center_band_mask, args)
    # Static distractors lose their vote entirely.
    support = support * np.where(static_map, float(args.consensus_static_weight), 1.0)

    # Geometry channel: COLMAP foreground-track support (S4).
    geo_support: np.ndarray | None = None
    usable_obs = [
        obs for stem, obs in colmap_observations.items() if stem in selected_masks and obs.points.size > 0
    ]
    if len(usable_obs) >= max(2, n_frames // 3):
        geo_support = np.zeros(support.shape, dtype=bool)
        for obs in usable_obs:
            geo_support |= points_to_support_mask(obs.points, support.shape, args.consensus_geometry_dilation)

    low_vote = support < float(args.consensus_low_vote_ratio)
    high_vote = support >= float(args.consensus_support_ratio)
    remove_region = static_map.copy()
    recall_region = np.zeros(support.shape, dtype=bool)
    if geo_support is not None:
        # Over-segmentation fix: weak consensus AND no geometric backing.
        remove_region |= low_vote & ~geo_support
        # Under-segmentation recall: strong consensus AND geometric backing.
        recall_region = high_vote & geo_support
    else:
        remove_region |= low_vote & (freq <= 0.05)
        recall_region = high_vote & (freq >= float(args.consensus_recall_ratio))
    remove_region = remove_region & ~(recall_region)
    min_area = int(remove_region.size * args.consensus_min_area_ratio)
    remove_region = remove_small_components(remove_region, min_area)
    recall_region = binary_closing(recall_region, structure=kernel(9), iterations=1)
    recall_region = binary_fill_holes(recall_region) & ~remove_region
    recall_region = remove_small_components(recall_region, min_area)

    # S5b: adhesion cutting. When the target physically overlaps a neighbouring
    # plant they form ONE connected component (often through a thick bridge), so
    # largest-component filtering AND morphological opening both fail. Instead we
    # exploit the geometry channel: the neighbour sits off-center and its pixels
    # carry no foreground track support, while the rotating target's silhouette
    # is track-backed. Large off-center unsupported chunks inside the mask are
    # cut; the center band guards the target core and thin leaf tips.
    adhesion_min_ratio = float(getattr(args, "consensus_adhesion_min_area_ratio", 0.0))
    cut_min_area = int(remove_region.size * adhesion_min_ratio)

    def _cut_adhesion_bridge(base_mask: np.ndarray) -> tuple[np.ndarray, float]:
        """Return (target_only_mask, removed_by_cut_ratio)."""
        if geo_support is None or cut_min_area <= 0 or not base_mask.any():
            return base_mask, 0.0
        unsupported = base_mask & ~geo_support & ~center_band_mask
        unsupported = remove_small_components(unsupported, cut_min_area)
        if not unsupported.any():
            return base_mask, 0.0
        labels, n = ndimage.label(unsupported)
        removed = np.zeros(base_mask.shape, dtype=bool)
        for lab in range(1, n + 1):
            comp = labels == lab
            ys_c, xs_c = np.nonzero(comp)
            dist_c = math.hypot(
                float(xs_c.mean()) / max(w - 1, 1) - cx,
                float(ys_c.mean()) / max(h - 1, 1) - cy,
            )
            # Only cut chunks whose centroid is clearly OFF the robust center:
            # genuine target parts cluster around (cx, cy).
            if dist_c > bw:
                removed |= comp
        if not removed.any():
            return base_mask, 0.0
        return base_mask & ~removed, float(removed.sum()) / float(max(base_mask.size, 1))

    result = ConsensusResult(
        reference_mask=high_vote,
        per_frame_masks={},
        per_frame_info={},
        geo_support=geo_support,
        center_band_mask=center_band_mask,
    )
    for stem, base_mask in selected_masks.items():
        corrected = base_mask & ~remove_region
        corrected = corrected | recall_region
        corrected = keep_largest_component(corrected) if corrected.any() else corrected
        cut_ratio = 0.0
        if corrected.any():
            # S5b: sever target-neighbour bridges BEFORE largest-component so the
            # neighbour side actually drops off instead of dragging along.
            corrected, cut_ratio = _cut_adhesion_bridge(corrected)
            if corrected.any():
                corrected = keep_largest_component(corrected)
        iou = mask_iou(corrected, base_mask)
        accepted = bool(iou >= args.consensus_fallback_iou)
        info: dict[str, Any] = {
            "图像": f"{stem}.png",
            "共识启用": 1,
            "共识接受": int(accepted),
            "回退IoU": round(float(iou), 4),
            "删除像素比例": round(float((base_mask & ~corrected).sum() / max(base_mask.size, 1)), 5),
            "补回像素比例": round(float((corrected & ~base_mask).sum() / max(base_mask.size, 1)), 5),
            "粘连切割像素比例": round(float(cut_ratio), 5),
        }
        # The fallback IoU is kept as a DIAGNOSTIC, not a hard gate: the corrected
        # mask always enters Pass 2 as the "A6共识" variant, where the shared
        # scoring function (plus the geometric leak penalty) decides between it
        # and A1s. Adhesion cuts legitimately remove large neighbour chunks, so
        # a low IoU alone must not discard a correct correction.
        final = corrected
        result.per_frame_masks[stem] = final
        result.per_frame_info[stem] = info
        if dirs is not None:
            save_mask(final, dirs["consensus"] / f"mask_{stem}.png")
    return result


def load_sam3_video_predictor(args: argparse.Namespace):
    """A7: build the SAM3 video predictor (memory engine) from the same repo/ckpt."""
    if str(args.sam3_repo) not in sys.path:
        sys.path.insert(0, str(args.sam3_repo))
    from sam3.model_builder import build_sam3_video_predictor as _build

    return _build(checkpoint_path=str(args.sam3_checkpoint))


def propagate_memory_masks(
    image_paths: dict[str, Path],
    seed_stem: str,
    seed_prompt_text: str,
    seed_box_mask: np.ndarray | None,
    stems_in_order: list[str],
    args: argparse.Namespace,
    predictor: Any | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """A7: text+box seeding on the best frame + bidirectional memory propagation.

    Frames are re-materialized as a numbered temp directory because SAM3 video
    sessions load frames from a folder of "<frame_index>.<ext>" files. Seeding
    uses the text prompt PLUS a normalized xywh box derived from the seed
    frame's consensus mask: the box anchors the CENTRAL target so the memory
    engine does not also lock onto the neighbouring plant (which matches the
    same text prompt).
    """
    import gc
    import shutil
    import tempfile

    import torch

    info: dict[str, Any] = {
        "记忆后端": "sam3_video",
        "种子帧": "",
        "传播方向": "both" if args.memory_bidirectional else "forward",
        "状态": "",
        "帧数": len(stems_in_order),
        "传播掩膜数": 0,
    }

    def _mem_state() -> dict[str, float]:
        """CUDA memory snapshot (MiB) — diagnostics only, no inference change."""
        if not torch.cuda.is_available():
            return {"allocated_mib": 0.0, "reserved_mib": 0.0, "free_mib": 0.0, "total_mib": 0.0}
        free, total = torch.cuda.mem_get_info()
        return {
            "allocated_mib": round(torch.cuda.memory_allocated() / 1024 ** 2, 1),
            "reserved_mib": round(torch.cuda.memory_reserved() / 1024 ** 2, 1),
            "free_mib": round(free / 1024 ** 2, 1),
            "total_mib": round(total / 1024 ** 2, 1),
        }

    if seed_box_mask is not None and seed_box_mask.any():
        _ys, _xs = np.nonzero(seed_box_mask)
        _h, _w = seed_box_mask.shape
        info["种子面积比例"] = round(float(seed_box_mask.sum()) / seed_box_mask.size, 4)
        info["种子框比例"] = round(float((_xs.max() - _xs.min()) * (_ys.max() - _ys.min())) / (_h * _w), 4)
        info["种子框像素"] = [int(_xs.min()), int(_ys.min()), int(_xs.max()), int(_ys.max())]
    else:
        info["种子面积比例"] = 0.0
        info["种子框比例"] = 0.0
    tmp_dir: Path | None = None
    session_id: str | None = None
    mem: dict[str, dict[str, float]] = {}
    try:
        mem["before_predictor_load"] = _mem_state()
        if predictor is None:
            predictor = load_sam3_video_predictor(args)
        mem["after_predictor_load"] = _mem_state()
        tmp_dir = Path(tempfile.mkdtemp(prefix="rapfsam3_mem_"))
        ordered = list(stems_in_order)
        if args.memory_max_frames and len(ordered) > int(args.memory_max_frames):
            # Keep a symmetric window around the seed frame.
            half = int(args.memory_max_frames) // 2
            seed_pos = ordered.index(seed_stem)
            lo = max(0, seed_pos - half)
            ordered = ordered[lo : lo + int(args.memory_max_frames)]
        for i, stem in enumerate(ordered):
            src = image_paths[stem]
            shutil.copy(src, tmp_dir / f"{i:06d}{src.suffix}")
        mem["before_start_session"] = _mem_state()
        response = predictor.handle_request(request=dict(type="start_session", resource_path=str(tmp_dir)))
        session_id = response["session_id"]
        mem["after_start_session"] = _mem_state()
        seed_idx = ordered.index(seed_stem)
        seed_request: dict[str, Any] = dict(
            type="add_prompt",
            session_id=session_id,
            frame_index=seed_idx,
            text=seed_prompt_text,
            obj_id=0,
        )
        if seed_box_mask is not None and seed_box_mask.any():
            ys_b, xs_b = np.nonzero(seed_box_mask)
            h_img, w_img = seed_box_mask.shape
            x0, x1 = float(xs_b.min()), float(xs_b.max())
            y0, y1 = float(ys_b.min()), float(ys_b.max())
            # Normalized xywh in 0~1, as expected by the video predictor.
            seed_request["bounding_boxes"] = [[
                x0 / w_img, y0 / h_img, (x1 - x0) / w_img, (y1 - y0) / h_img
            ]]
            seed_request["bounding_box_labels"] = [1]
        mem["before_add_prompt"] = _mem_state()
        response = predictor.handle_request(request=seed_request)
        mem["after_add_prompt"] = _mem_state()
        out = response.get("outputs", {})
        n_seed_obj = len(out.get("out_obj_ids", []))
        if n_seed_obj == 0:
            info["状态"] = "seed_empty"
            return {}, info
        info["种子帧"] = seed_stem
        info["种子物体数"] = n_seed_obj

        masks: dict[str, np.ndarray] = {}
        request = dict(type="propagate_in_video", session_id=session_id, start_frame_index=seed_idx)
        request["propagation_direction"] = "both" if args.memory_bidirectional else "forward"
        mem["before_propagate"] = _mem_state()
        for resp in predictor.handle_stream_request(request=request):
            frame_idx = resp["frame_index"]
            outputs = resp["outputs"]
            obj_ids = outputs.get("out_obj_ids", [])
            bin_masks = outputs.get("out_binary_masks", [])
            combined = None
            for oid, m in zip(obj_ids, bin_masks):
                if hasattr(m, "detach"):
                    arr = m.squeeze().detach().cpu().numpy().astype(bool)
                else:
                    arr = np.asarray(m, dtype=bool)
                    if arr.ndim == 3:
                        arr = arr[0]
                combined = arr if combined is None else (combined | arr)
            if combined is not None and frame_idx < len(ordered):
                masks[ordered[frame_idx]] = combined
        info["状态"] = "ok"
        info["传播掩膜数"] = len(masks)
        return masks, info
    except torch.cuda.OutOfMemoryError:
        info["状态"] = "cuda_oom_fallback"
        return {}, info
    except Exception as exc:  # noqa: BLE001 - degrade to per-frame mode on any failure
        info["状态"] = f"unavailable: {type(exc).__name__}: {exc}"
        return {}, info
    finally:
        # Strict lifecycle: close the session on EVERY exit path (success,
        # seed_empty, OOM, exception). Previously only the success path closed it;
        # the OOM/exception paths leaked the session (see Phase 14.1 evidence audit).
        if session_id is not None:
            try:
                predictor.handle_request(request=dict(type="close_session", session_id=session_id))
            except Exception:  # noqa: BLE001 - never mask the primary outcome
                pass
            session_id = None
        mem["after_close_session"] = _mem_state()
        info["memory"] = mem
        if info.get("状态") in ("cuda_oom_fallback",) or info.get("状态", "").startswith("unavailable"):
            # Free cached blocks left behind by an aborted session. Diagnostic-only
            # cleanup, not an algorithmic fix (confirmed leak: sessions were left
            # open on the OOM/exception path, see Phase 14.1 evidence audit).
            gc.collect()
            torch.cuda.empty_cache()
        if tmp_dir is not None:
            shutil.rmtree(tmp_dir, ignore_errors=True)


def _merge_sampled_memory_info(
    samples_diag: dict[str, dict[str, Any]],
    sample_groups: dict[str, list[str]],
) -> dict[str, Any]:
    """Merge per-sample A7 diagnostics into an aggregate summary (Phase 14.1).

    Previously `merged_memory_info[k] = v` kept only the LAST sample's status,
    so `状态=ok` could not be trusted across samples. This builds per-sample
    entries + aggregate counters + a summary 状态 grounded in ALL samples.
    """
    info: dict[str, Any] = {"samples": samples_diag, "样本数": len(sample_groups)}
    statuses = [v.get("状态", "") for v in samples_diag.values()]
    info["samples_ok"] = int(sum(s == "ok" for s in statuses))
    info["samples_seed_empty"] = int(sum(s == "seed_empty" for s in statuses))
    info["samples_oom"] = int(sum(s == "cuda_oom_fallback" for s in statuses))
    info["samples_unavailable"] = int(sum(s.startswith("unavailable") for s in statuses))
    total_masks = sum(int(v.get("传播掩膜数", 0)) for v in samples_diag.values())
    info["propagated_total"] = total_masks
    info["candidate_total"] = total_masks  # all valid masks enter Pass-2 scoring
    info["selected_total"] = 0  # filled after Pass-2 variant selection
    if any(s == "cuda_oom_fallback" for s in statuses):
        info["状态"] = "partial_oom"
    elif any(s.startswith("unavailable") for s in statuses):
        info["状态"] = "partial_unavailable"
    elif any(s == "seed_empty" for s in statuses):
        info["状态"] = "partial_seed_empty"
    elif all(s == "ok" for s in statuses):
        info["状态"] = "ok"
    else:
        info["状态"] = "partial"
    info["种子帧"] = (
        "multi_sample" if len(sample_groups) > 1
        else next(iter(samples_diag.values()), {}).get("seed_stem", "")
    )
    return info


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAP-FSAM3 mask generator")
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--sam3_repo", type=Path, default=DEFAULT_SAM3_REPO)
    parser.add_argument("--sam3_checkpoint", type=Path, default=DEFAULT_SAM3_CHECKPOINT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp_dtype", choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--disable_amp", action="store_true")
    parser.add_argument("--confidence_threshold", type=float, default=0.3)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--reuse_existing_candidates", action="store_true")
    parser.add_argument("--candidate_source_dir", type=Path, default=None)
    parser.add_argument("--candidate_metadata_dir", type=Path, default=None)
    parser.add_argument("--candidate_metadata_strict", action="store_true")
    parser.add_argument(
        "--candidate_mode",
        choices=["legacy_union", "per_instance"],
        default="per_instance",
        help="legacy_union: 旧行为，同一 prompt 多次检测 OR 合并+最大连通域；"
             "per_instance: 每个 SAM3 实例独立成为 Candidate，生成阶段不做 OR/最大连通域，"
             "合并仅在评分+目标关联后发生。",
    )
    parser.add_argument(
        "--sam3_mask_threshold", type=float, default=0.5,
        help="per_instance 模式下从 SAM3 logits 重新二值化的阈值（默认 0.5，与旧行为一致）。",
    )
    parser.add_argument(
        "--save_raw_instance_masks", action=argparse.BooleanOptionalAction, default=True,
        help="保存每个 prompt 的每个 SAM3 原始实例掩膜到 候选掩膜/raw_instance_<prompt>_<i>.png，用于阶段证据。",
    )

    parser.add_argument("--prompt_list", default="P2")
    parser.add_argument("--prompt_texts_json", type=Path, default=None)
    parser.add_argument("--default_prompt_id", default="P2")
    parser.add_argument("--use_prompt_ensemble", action="store_true")
    parser.add_argument(
        "--prompt_selection_mode",
        choices=["single", "score_select", "weighted_fusion"],
        default="single",
    )
    parser.add_argument("--fusion_threshold", type=float, default=0.5)
    parser.add_argument("--score_weights", default="area=1,comp=1,edge=1,temp=1,contrast=1,sam=0.5",
                        help="阶段十二 §三.4：评分权重，默认含 sam=0.5（非零）")
    parser.add_argument("--ranking_policy",
                        choices=["base", "r1", "r2", "r3"],
                        default="r1",
                        help="Phase 17C §17：候选排序策略。r1=修复后默认（q_contrast排除出主评分）；"
                             "r1=剔除 q_contrast 的主评分；r2=r1 主评分 + q_contrast 仅精确平局决胜；"
                             "r3=q_contrast×q_area 调制对比。仅结构变更，不涉及权重搜索。")
    # 阶段十一 §4.5 时序对齐门控：仅在显式配准后启用时序项，否则置中性 0.5
    parser.add_argument("--use_temporal_alignment", action="store_true",
                        help="§4.5：启用光流/单应配准后的时序 IoU；默认关（Pass2 关），未配准时序项置中性值。")
    # 阶段十一 §4.6 硬门控
    parser.add_argument("--vertical_coverage_min_ratio", type=float, default=0.30,
                        help="§4.6：植株纵向覆盖（上半区占比）低于此值标记为高风险（疑似只割到盆/条带）。")
    parser.add_argument("--box_track_overlap_min", type=float, default=0.20,
                        help="§4.6：与 COLMAP track 投影重叠度低于此值标记高风险。")
    parser.add_argument("--area_min_ratio", type=float, default=0.01)
    parser.add_argument("--area_max_ratio", type=float, default=0.80)
    parser.add_argument("--area_target_ratio", type=float, default=0.0)
    parser.add_argument("--collapse_area_threshold", type=float, default=0.05,
                        help="§5 P2/P6 语义坍缩判定：面积差比与盆区覆盖差比均低于此值即判坍缩")
    parser.add_argument("--component_min_area_ratio", type=float, default=0.0005)
    parser.add_argument("--leakage_bottom_start_ratio", type=float, default=0.62)
    parser.add_argument("--leakage_max_bottom_fraction", type=float, default=0.02)
    parser.add_argument("--leakage_side_mode", choices=["left", "right", "both"], default="both")
    parser.add_argument("--leakage_side_band_ratio", type=float, default=0.025)
    parser.add_argument("--leakage_max_side_fraction", type=float, default=0.004)

    parser.add_argument("--use_semantic_gate", action="store_true")
    parser.add_argument(
        "--semantic_gate_mode",
        choices=["heuristic", "gated_softmax"],
        default="heuristic",
        help="Phase 21 §SG-MPS: heuristic=legacy additive semantics gate; "
             "gated_softmax=manuscript sigmoid compatibility c_i^(k) + temperature "
             "softmax a_i^(k); prompt chosen by argmax_k c_i^(k).",
    )
    parser.add_argument(
        "--semantic_phi_weights", default="0.5,0.3,0.15,0.15,0.0,0.1",
        help="Phase 21: w_phi for [q_plant,q_area,q_comp,q_edge,q_contrast,q_temp].",
    )
    parser.add_argument(
        "--semantic_psi_weights", default="0.3,0.3,0.3,0.2",
        help="Phase 21: w_psi for [q_table,q_bg,q_neighbour,q_outlier].",
    )
    parser.add_argument("--semantic_gate_bias", type=float, default=0.5,
                        help="Phase 21: bias b in c=sigmoid(w_phi^T phi - w_psi^T psi - b).")
    parser.add_argument("--semantic_gate_temperature", type=float, default=0.25,
                        help="Phase 21: tau_s in a=softmax(c/tau_s).")
    parser.add_argument("--semantic_gate_use_contrast", action="store_true",
                        help="Phase 21: include q_contrast in the gate descriptor. Off by default "
                             "to avoid double-counting the term that CDR/R1 removes from primary ranking.")
    parser.add_argument("--semantic_gate_margin", type=float, default=0.0,
                        help="Phase 21: minimum c_i^(k*) - c_i^(k_runner-up) required for the gate "
                             "to commit. Below it the gate defers to the ranking policy (the "
                             "manuscript's 'safety mechanism' reading). 0.0 = pure argmax.")
    parser.add_argument(
        "--semantic_gate_backend",
        choices=["heuristic"],
        default="heuristic",
    )
    parser.add_argument("--semantic_box_json", type=Path, default=None)
    parser.add_argument("--target_box_weight", type=float, default=0.35)
    parser.add_argument("--pot_overlap_weight", type=float, default=0.30)
    parser.add_argument("--side_distractor_weight", type=float, default=0.25)
    parser.add_argument("--center_prior_weight", type=float, default=0.10)
    parser.add_argument("--vertical_coverage_weight", type=float, default=0.18)
    parser.add_argument("--semantic_green_exg_threshold", type=float, default=12.0)
    parser.add_argument("--semantic_pot_lower_start_ratio", type=float, default=0.58)
    parser.add_argument("--semantic_side_band_ratio", type=float, default=0.18)
    parser.add_argument("--save_semantic_gate_debug", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument(
        "--base_cleanup_mode",
        choices=["none", "largest", "fsam3_basic"],
        default="fsam3_basic",
    )
    parser.add_argument("--closing_kernel", type=int, default=15)
    parser.add_argument("--fill_holes", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--keep_largest_component", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--use_spnp_refinement", action="store_true")
    parser.add_argument(
        "--spnp_backend",
        choices=["none", "sam3_if_supported", "sam_family", "postprocess_only"],
        default="none",
    )
    parser.add_argument("--spnp_positive_points", type=int, default=5)
    parser.add_argument("--spnp_negative_points", type=int, default=8)
    parser.add_argument("--spnp_ring_radius", type=int, default=15)
    parser.add_argument("--spnp_use_lower_negative", action="store_true")
    parser.add_argument("--spnp_lower_band_ratio", type=float, default=0.70)
    parser.add_argument("--spnp_lower_exg_threshold", type=float, default=0.0)
    parser.add_argument("--spnp_max_remove_ratio", type=float, default=0.15)
    parser.add_argument("--spnp_positive_box_padding", type=float, default=0.03)
    parser.add_argument("--spnp_negative_box_radius", type=int, default=12)
    parser.add_argument("--spnp_min_refined_iou", type=float, default=0.65,
                        help="阶段十一 §7.3：接受 SPNP 细化结果的 IoU 阈值（由原 0.10 提高，"
                             "范围 0.60–0.75 可配）。避免低 IoU 接受错误 mask 强化。")
    # ── SB-FBPR (manuscript §2.3.4): real point prompts, not only boxes ──
    parser.add_argument(
        "--spnp_prompt_form",
        choices=["boxes", "points", "boxes+points"],
        default="boxes",
        help="SB-FBPR 送入 SAM3 的几何提示形式。boxes=历史路径（正负框）；"
             "points=原稿 §2.3.4 的距离变换正点 + NMS 负点；"
             "boxes+points=两者合并。默认 boxes 以保持历史回放逐字节一致。",
    )
    parser.add_argument(
        "--spnp_negative_region_mode",
        choices=["ring", "manuscript"],
        default="ring",
        help="负提示候选区域。ring=历史外环；manuscript=R^- = 外环 ∪ I_table ∪ I_bg ∪ I_neighbour。",
    )
    parser.add_argument("--spnp_negative_nms_radius", type=int, default=0,
                        help="负点 NMS 抑制半径（像素）。0 = 历史 4%% 图像短边间距。")
    parser.add_argument("--spnp_table_band_start_ratio", type=float, default=0.58,
                        help="R^- 中 I_table 的纵向起点比例。")
    parser.add_argument("--spnp_side_band_ratio", type=float, default=0.18,
                        help="R^- 中 I_neighbour 的左右带宽度比例。")

    parser.add_argument("--use_residual_repair", action="store_true")
    parser.add_argument("--opening_kernel", type=int, default=3)
    parser.add_argument(
        "--thin_repair_mode",
        choices=["none", "adjacent", "skeleton", "shape_filter"],
        default="none",
    )
    parser.add_argument("--thin_repair_max_area_ratio", type=float, default=0.003)
    parser.add_argument("--thin_repair_min_elongation", type=float, default=2.5)

    parser.add_argument("--use_reprompt_detection", action="store_true")
    parser.add_argument(
        "--reprompt_detection_mode",
        choices=["none", "mask_temporal", "mask_image", "full"],
        default="none",
    )
    parser.add_argument("--reprompt_threshold", type=float, default=0.5)
    parser.add_argument("--reprompt_score_gap", type=float, default=0.05,
                        help="§4.7：top1 与 top2 候选分差小于此值则触发重提示（不静默接受）。")
    parser.add_argument("--reprompt_min_score", type=float, default=0.2,
                        help="§4.7：最佳候选总分低于此值（或空掩膜）则触发重提示。")
    parser.add_argument("--reprompt_weights", default="iou=0.35,area=0.25,ssim=0.25,edge=0.15")

    parser.add_argument("--use_geometry_feedback", action="store_true")
    parser.add_argument(
        "--geometry_feedback_mode",
        choices=["none", "track_projection", "coverage", "full"],
        default="none",
    )
    parser.add_argument("--colmap_dir", type=Path, default=None)
    parser.add_argument("--colmap_loader_path", type=Path, default=DEFAULT_COLMAP_LOADER)
    parser.add_argument("--geometry_threshold", type=float, default=0.7)
    parser.add_argument("--use_corrective_geometry", action="store_true")
    parser.add_argument(
        "--corrective_geometry_backend",
        choices=["colmap_tracks", "colmap_tracks_gaussian", "mask_shape"],
        default="colmap_tracks",
    )
    parser.add_argument(
        "--strict_colmap_observations",
        action="store_true",
        help="GCFC 严格模式：任一帧缺少真实 COLMAP 观测时直接报错，"
             "而不是静默返回原掩膜（缺失不能冒充 GCFC 已执行）。",
    )
    # ── GCFC (manuscript §2.3.5): Gaussian support field H_i, support ratio G_i ──
    parser.add_argument("--geometry_gaussian_sigma", type=float, default=3.0)
    parser.add_argument("--geometry_support_threshold", type=float, default=0.15)
    parser.add_argument("--geometry_boundary_band_px", type=int, default=12)
    parser.add_argument("--geometry_min_support_ratio", type=float, default=0.35)
    parser.add_argument("--geometry_positive_min_points", type=int, default=12)
    parser.add_argument("--geometry_negative_min_area_ratio", type=float, default=0.012)
    parser.add_argument("--geometry_correct_max_delta_ratio", type=float, default=0.08)
    parser.add_argument("--geometry_correct_min_iou", type=float, default=0.75)
    parser.add_argument("--geometry_track_dilation", type=int, default=19)
    parser.add_argument("--geometry_unsupported_dilation", type=int, default=29)
    parser.add_argument("--geometry_enable_negative_correction", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--save_corrective_geometry_debug", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--use_cross_view_consensus", action="store_true")
    parser.add_argument("--consensus_support_ratio", type=float, default=0.55)
    parser.add_argument("--consensus_low_vote_ratio", type=float, default=0.30)
    parser.add_argument("--consensus_recall_ratio", type=float, default=0.70)
    parser.add_argument("--consensus_center_band_ratio", type=float, default=0.35)
    parser.add_argument("--consensus_center_decay", type=float, default=0.65)
    parser.add_argument("--consensus_static_weight", type=float, default=0.60)
    parser.add_argument("--consensus_static_std_threshold", type=float, default=0.02)
    parser.add_argument("--consensus_bridge_kernel", type=int, default=31)
    parser.add_argument("--consensus_adhesion_min_area_ratio", type=float, default=0.004)
    parser.add_argument("--consensus_variant_leak_weight", type=float, default=0.0)
    parser.add_argument("--use_spnp_evidence_guidance", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--consensus_geometry_dilation", type=int, default=19)
    parser.add_argument("--consensus_min_area_ratio", type=float, default=0.0008)
    parser.add_argument("--consensus_fallback_iou", type=float, default=0.75)
    parser.add_argument("--consensus_min_frames", type=int, default=5)

    parser.add_argument("--use_memory_propagation", action="store_true")
    parser.add_argument("--memory_seed_mode", choices=["best_score", "consensus_best"], default="best_score")
    parser.add_argument("--memory_bidirectional", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--memory_max_frames", type=int, default=0)

    parser.add_argument("--save_candidate_masks", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save_intermediate_masks", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save_foreground_rgb", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save_rgba", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save_overlay", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log_json", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--gt_mask_dir",
        type=Path,
        default=None,
        help="可选：人工 GT 掩膜目录（mask_<stem>.png）。提供时逐帧报告会写入 "
             "F1 / Boundary-F1 / HD95；缺失的帧留空，不伪造指标。",
    )
    return parser.parse_args()


def parse_key_values(text: str, defaults: dict[str, float]) -> dict[str, float]:
    values = dict(defaults)
    if not text:
        return values
    for item in text.split(","):
        if not item.strip():
            continue
        if "=" not in item:
            raise ValueError(f"Invalid key-value item: {item}")
        key, value = item.split("=", 1)
        values[key.strip()] = float(value)
    return values


def image_files(input_dir: Path, limit: int) -> list[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    suffixes = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    files = sorted(p for p in input_dir.iterdir() if p.is_file() and p.suffix in suffixes)
    if limit and limit > 0:
        return files[:limit]
    return files


def load_prompt_texts(path: Path | None) -> dict[str, str]:
    prompts = dict(BUILTIN_PROMPTS)
    if path is None:
        return prompts
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("--prompt_texts_json must contain a JSON object")
    for key, value in data.items():
        prompts[str(key)] = str(value)
    return prompts


def parse_prompt_ids(prompt_list: str, prompt_texts: dict[str, str]) -> list[str]:
    ids = [item.strip() for item in prompt_list.split(",") if item.strip()]
    if not ids:
        raise ValueError("--prompt_list must not be empty")
    missing = [pid for pid in ids if pid not in prompt_texts]
    if missing:
        raise ValueError(f"Unknown prompt ids: {missing}")
    return ids


def prompt_dir_name(prompt_id: str) -> str:
    return PROMPT_DIR_NAMES.get(prompt_id, prompt_id)


def ensure_dirs(output_dir: Path) -> dict[str, Path]:
    dirs = {
        "candidate": output_dir / "候选掩膜",
        "selected": output_dir / "选择后掩膜",
        "spnp": output_dir / "正负提示细化掩膜",
        "repair": output_dir / "残差修复掩膜",
        "corrective": output_dir / "几何修正掩膜",
        "a5c": output_dir / "A5c_final_mask",
        "final": output_dir / "最终掩膜",
        "foreground": output_dir / "前景图",
        "rgba": output_dir / "透明图",
        "overlay": output_dir / "叠加图",
        "semantic_debug": output_dir / "语义门控框",
        "geometry_debug": output_dir / "几何修正提示图",
        "consensus": output_dir / "共识投票掩膜",
        "memory": output_dir / "记忆传播掩膜",
        "logs": output_dir / "日志",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def save_mask(mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask.astype(np.uint8) * 255)).save(path)


def load_mask(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("L")) >= 127


def save_foreground_and_rgba(image: Image.Image, mask: np.ndarray, foreground_path: Path, rgba_path: Path) -> None:
    rgb = np.array(image.convert("RGB"))
    fg = rgb.copy()
    fg[~mask] = 0
    Image.fromarray(fg).save(foreground_path)
    rgba = np.dstack([rgb, (mask.astype(np.uint8) * 255)])
    Image.fromarray(rgba).save(rgba_path)


def save_overlay(image: Image.Image, mask: np.ndarray, path: Path) -> None:
    rgb = np.array(image.convert("RGB")).astype(np.float32)
    color = np.zeros_like(rgb)
    color[..., 1] = 255.0
    alpha = mask[..., None].astype(np.float32) * 0.35
    overlay = rgb * (1.0 - alpha) + color * alpha
    Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8)).save(path)


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    labeled, count = ndimage.label(mask)
    if count <= 1:
        return mask.astype(bool)
    sizes = ndimage.sum(mask, labeled, range(1, count + 1))
    largest_id = int(np.argmax(sizes)) + 1
    return labeled == largest_id


def remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    labeled, count = ndimage.label(mask)
    if count == 0:
        return mask.astype(bool)
    cleaned = np.zeros_like(mask, dtype=bool)
    sizes = ndimage.sum(mask, labeled, range(1, count + 1))
    for region_id, size in enumerate(sizes, start=1):
        if size >= min_area:
            cleaned[labeled == region_id] = True
    return cleaned


def kernel(size: int) -> np.ndarray:
    size = int(size)
    if size <= 1:
        return np.ones((1, 1), dtype=bool)
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)).astype(bool)


def basic_cleanup(mask: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    mask = mask.astype(bool)
    if args.base_cleanup_mode == "none":
        return mask
    if args.keep_largest_component:
        mask = keep_largest_component(mask)
    if args.base_cleanup_mode == "fsam3_basic":
        if args.closing_kernel > 1:
            mask = binary_closing(mask, structure=kernel(args.closing_kernel), iterations=1)
        if args.fill_holes:
            mask = binary_fill_holes(mask)
    return mask.astype(bool)


def combine_sam_masks(masks: Any, image_shape: tuple[int, int]) -> np.ndarray:
    h, w = image_shape
    combined = np.zeros((h, w), dtype=bool)
    if masks is None:
        return combined
    for item in masks:
        mask_np = item.squeeze().detach().cpu().numpy().astype(bool)
        combined |= mask_np
    return combined


def load_sam3(args: argparse.Namespace):
    if str(args.sam3_repo) not in sys.path:
        sys.path.insert(0, str(args.sam3_repo))
    import torch
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    model = build_sam3_image_model(
        checkpoint_path=str(args.sam3_checkpoint),
        load_from_HF=False,
        device=args.device,
    )
    model = model.to(args.device)
    model.eval()
    processor = Sam3Processor(model, confidence_threshold=args.confidence_threshold, device=args.device)
    return torch, processor


def sam3_autocast(torch_module: Any, args: argparse.Namespace):
    if args.disable_amp or not str(args.device).startswith("cuda"):
        return nullcontext()
    dtype = {
        "bfloat16": torch_module.bfloat16,
        "float16": torch_module.float16,
    }[args.amp_dtype]
    return torch_module.autocast("cuda", dtype=dtype)


def _masks_scores_boxes(output: Any, h: int, w: int, mask_threshold: float = 0.5) -> tuple[list[np.ndarray], list[float], list[tuple[int, int, int, int]]]:
    """从 SAM3 output 提取逐实例 (mask, score, box)。

    output 含 'masks' [K,H,W] bool、'masks_logits' [K,H,W] sigmoid、'scores' [K]、
    'boxes' [K,4] (x0,y0,x1,y1)。per_instance 模式直接逐实例取出，不在此 OR。

    阶段十二 §三.2：mask_threshold 参数化，不再硬编码 0.5。
    """
    import torch as _torch

    masks_logits = output.get("masks_logits", None)
    masks_bool = output.get("masks", None)
    scores_t = output.get("scores", None)
    boxes_t = output.get("boxes", None)

    per_masks: list[np.ndarray] = []
    per_scores: list[float] = []
    per_boxes: list[tuple[int, int, int, int]] = []

    if masks_logits is not None:
        logits = masks_logits
        if hasattr(logits, "detach"):
            logits = logits.detach().cpu().numpy()
        logits = np.asarray(logits).astype(np.float32)
        k = logits.shape[0] if logits.ndim == 4 else 0
        for i in range(k):
            m = logits[i, 0] > mask_threshold
            per_masks.append(m.astype(bool))
    elif masks_bool is not None:
        mb = masks_bool
        if hasattr(mb, "detach"):
            mb = mb.detach().cpu().numpy()
        mb = np.asarray(mb).astype(bool)
        k = mb.shape[0] if mb.ndim == 3 else 0
        for i in range(k):
            per_masks.append(mb[i].astype(bool))

    if scores_t is not None:
        if hasattr(scores_t, "detach"):
            scores_t = scores_t.detach().float().cpu().numpy()
        scores_t = np.asarray(scores_t).astype(np.float32).reshape(-1)
        per_scores = [float(s) for s in scores_t]

    if boxes_t is not None:
        if hasattr(boxes_t, "detach"):
            boxes_t = boxes_t.detach().float().cpu().numpy()
        boxes_t = np.asarray(boxes_t).astype(np.float32).reshape(-1, 4)
        for b in boxes_t:
            per_boxes.append((int(round(b[0])), int(round(b[1])), int(round(b[2])), int(round(b[3]))))

    n = len(per_masks)
    while len(per_scores) < n:
        per_scores.append(0.0)
    while len(per_boxes) < n:
        per_boxes.append((0, 0, 0, 0))
    return per_masks, per_scores, per_boxes


def infer_candidates(
    image: Image.Image,
    image_name: str,
    prompt_ids: list[str],
    prompt_texts: dict[str, str],
    processor: Any,
    torch_module: Any,
    args: argparse.Namespace,
) -> list[Candidate]:
    h, w = image.height, image.width
    candidates: list[Candidate] = []
    with sam3_autocast(torch_module, args):
        state = processor.set_image(image)
        for prompt_id in prompt_ids:
            processor.reset_all_prompts(state)
            output = processor.set_text_prompt(state=state, prompt=prompt_texts[prompt_id])
            raw_scores = output.get("scores", None)
            scores = (
                [float(s.item()) for s in raw_scores]
                if raw_scores is not None and len(raw_scores) > 0
                else []
            )

            if args.candidate_mode == "per_instance":
                # 逐实例候选：每个 SAM3 实例独立成为 Candidate，生成阶段不做 OR/最大连通域
                per_masks, per_scores, per_boxes = _masks_scores_boxes(output, h, w, mask_threshold=args.sam3_mask_threshold)
                if not per_masks:
                    # 退化保护：无实例时回退到空候选（评分阶段会被空 mask 硬门拦截）
                    candidates.append(
                        Candidate(
                            prompt_id=prompt_id, prompt_text=prompt_texts[prompt_id],
                            mask=np.zeros((h, w), dtype=bool), scores=scores,
                            raw_detection_count=0, instance_id=0, box=None,
                            sam_score=0.0, mask_threshold=args.sam3_mask_threshold, source_stage="raw",
                            prompt_mode="per_instance",
                        )
                    )
                    continue
                for i, (m, sc, box) in enumerate(zip(per_masks, per_scores, per_boxes)):
                    candidates.append(
                        Candidate(
                            prompt_id=prompt_id,
                            prompt_text=prompt_texts[prompt_id],
                            mask=m,
                            scores=scores,
                            raw_detection_count=len(scores),
                            instance_id=i,
                            box=box,
                            sam_score=float(sc),
                            mask_threshold=args.sam3_mask_threshold,
                            source_stage="raw",
                            prompt_mode="per_instance",
                        )
                    )
            else:
                # legacy_union：旧行为，OR 合并 + basic_cleanup
                raw_mask = combine_sam_masks(output.get("masks", None), (h, w))
                cleaned = basic_cleanup(raw_mask, args)
                candidates.append(
                    Candidate(
                        prompt_id=prompt_id,
                        prompt_text=prompt_texts[prompt_id],
                        mask=cleaned,
                        scores=scores,
                        raw_detection_count=len(scores),
                        instance_id=0,
                        box=None,
                        sam_score=float(scores[0]) if scores else 0.0,
                        mask_threshold=0.5,
                        source_stage="raw",
                        prompt_mode="legacy_union",
                    )
                )
    return candidates


def load_existing_candidates(
    image_path: Path,
    prompt_ids: list[str],
    prompt_texts: dict[str, str],
    output_dir: Path,
    candidate_source_dir: Path | None = None,
    candidate_metadata_dir: Path | None = None,
    candidate_metadata_strict: bool = False,
) -> list[Candidate]:
    candidates = []
    source_dir = candidate_source_dir if candidate_source_dir is not None else output_dir
    metadata = load_cache_metadata(image_path, source_dir, candidate_metadata_dir)
    for prompt_id in prompt_ids:
        # 优先复用 per_instance 原始实例（raw_instance_<prompt>_<i>.png）
        raw_dir = source_dir / "候选掩膜" / f"raw_instance_{prompt_id}"
        raw_files = sorted(raw_dir.glob(f"mask_{image_path.stem}_*.png")) if raw_dir.exists() else []
        if raw_files:
            if candidate_metadata_strict:
                mask_ids = {int(path.stem.rsplit("_", 1)[1]) for path in raw_files}
                metadata_ids = {instance for prompt, instance in metadata if prompt == prompt_id}
                if mask_ids != metadata_ids:
                    raise ValueError(f"METADATA_UNVERIFIED: incomplete candidate identity set for {image_path.name}/{prompt_id}")
            prompt_candidates = []
            for rp in raw_files:
                try:
                    suffix = rp.stem.split("_")[-1]
                    inst = int(suffix)
                except (ValueError, IndexError):
                    raise ValueError(f"Candidate filename has no reliable instance identity: {rp}")
                entry, status = metadata_for_mask(metadata, prompt_id, inst, rp, candidate_metadata_strict)
                prompt_candidates.append(
                    Candidate(
                        prompt_id=prompt_id,
                        prompt_text=prompt_texts[prompt_id],
                        mask=load_mask(rp),
                        scores=list(entry.get("scores", [])),
                        raw_detection_count=int(entry.get("raw_detection_count", len(raw_files))),
                        instance_id=inst,
                        box=entry.get("box"),
                        sam_score=float(entry.get("sam_score", 0.0)),
                        mask_threshold=float(entry.get("mask_threshold", 0.5)),
                        source_stage="reused",
                        prompt_mode="per_instance",
                        candidate_order=int(entry.get("candidate_order", inst)),
                        metadata_status=status,
                        source_image_name=entry.get("source_image_name", image_path.name),
                        source_image_sha256=entry.get("image_sha256", ""),
                        mask_sha256=entry.get("mask_sha256", ""),
                        score_precision=entry.get("score_precision", "ARCHIVED_UNKNOWN_PRECISION"),
                    )
                )
            candidates.extend(sorted(prompt_candidates, key=lambda item: item.candidate_order) if metadata else prompt_candidates)
            continue
        mask_path = source_dir / "候选掩膜" / prompt_dir_name(prompt_id) / f"mask_{image_path.stem}.png"
        if not mask_path.exists():
            raise FileNotFoundError(f"Existing candidate mask not found: {mask_path}")
        entry, status = metadata_for_mask(metadata, prompt_id, 0, mask_path, candidate_metadata_strict)
        candidates.append(
            Candidate(
                prompt_id=prompt_id,
                prompt_text=prompt_texts[prompt_id],
                mask=load_mask(mask_path),
                scores=list(entry.get("scores", [])),
                raw_detection_count=int(entry.get("raw_detection_count", 0)),
                instance_id=0,
                box=entry.get("box"),
                sam_score=float(entry.get("sam_score", 0.0)),
                mask_threshold=float(entry.get("mask_threshold", 0.5)),
                source_stage="reused",
                prompt_mode="legacy_union",
                candidate_order=int(entry.get("candidate_order", 0)),
                metadata_status=status,
                source_image_name=entry.get("source_image_name", image_path.name),
                source_image_sha256=entry.get("image_sha256", ""),
                mask_sha256=entry.get("mask_sha256", ""),
                score_precision=entry.get("score_precision", "ARCHIVED_UNKNOWN_PRECISION"),
            )
        )
    return candidates


def component_count(mask: np.ndarray, min_area: int) -> int:
    labeled, count = ndimage.label(mask)
    if count == 0:
        return 0
    sizes = ndimage.sum(mask, labeled, range(1, count + 1))
    return int(sum(size >= min_area for size in sizes))


def mask_boundary(mask: np.ndarray) -> np.ndarray:
    if mask.size == 0:
        return mask
    eroded = binary_opening(mask, structure=np.ones((1, 1), dtype=bool))
    eroded = ndimage.binary_erosion(eroded, structure=np.ones((3, 3), dtype=bool), border_value=0)
    return mask & ~eroded


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(a, b).sum() / union)


def foreground_background_contrast(image: Image.Image, mask: np.ndarray) -> float:
    rgb = np.array(image.convert("RGB"), dtype=np.float32)
    if mask.sum() == 0 or (~mask).sum() == 0:
        return 0.0
    fg = rgb[mask].mean(axis=0)
    bg = rgb[~mask].mean(axis=0)
    return float(np.linalg.norm(fg - bg) / (255.0 * math.sqrt(3.0)))


def area_quality(area_ratio: float, args: argparse.Namespace) -> float:
    if area_ratio < args.area_min_ratio or area_ratio > args.area_max_ratio:
        return 0.0
    target = args.area_target_ratio if args.area_target_ratio > 0 else (args.area_min_ratio + args.area_max_ratio) / 2.0
    span = max(target - args.area_min_ratio, args.area_max_ratio - target, 1e-6)
    return float(max(0.0, 1.0 - abs(area_ratio - target) / span))


def edge_density(mask: np.ndarray) -> float:
    """单位面积边界像素数（越高=细碎叶片越多/越细）。"""
    area = float(mask.sum())
    if area <= 0:
        return 1.0
    return float(mask_boundary(mask).sum() / max(math.sqrt(area), 1.0))


def edge_quality(mask: np.ndarray, args: argparse.Namespace) -> float:
    """阶段十一 §4.4 边界置信度：奖励**不贴合图像边缘**的对象边界。

    旧 `q_edge = 1/(1+density/20)` 反而奖励"边界越少越好"，会惩罚细叶/多叶。
    新口径改为：边界与图像四边的贴合度越低（对象内部干净、不溢出图像框），
    置信度越高；细叶高 boundary_density 不再自动低分。中性基准 1.0。
    """
    if not mask.any():
        return 1.0
    h, w = mask.shape
    bnd = mask_boundary(mask)
    ys, xs = np.where(bnd)
    if len(ys) == 0:
        return 1.0
    # 贴合图像边缘的边界像素比例（四边 2px 带）
    border_band = 2
    on_border = int(
        ((ys < border_band) | (ys >= h - border_band)
         | (xs < border_band) | (xs >= w - border_band)).sum()
    )
    adherence = on_border / max(len(ys), 1)
    # 贴合度越高 → 置信度越低（对象贴边/溢出框，疑似截断或背景粘连）
    return float(max(0.0, 1.0 - adherence))


def bottom_leak_fraction(mask: np.ndarray, args: argparse.Namespace) -> float:
    area = float(mask.sum())
    if area <= 0:
        return 0.0
    start = int(mask.shape[0] * args.leakage_bottom_start_ratio)
    start = min(max(start, 0), mask.shape[0])
    return float(mask[start:, :].sum() / area)


def leakage_quality(bottom_fraction: float, args: argparse.Namespace) -> float:
    allowed = max(float(args.leakage_max_bottom_fraction), 0.0)
    if bottom_fraction <= allowed:
        return 1.0
    if allowed <= 0:
        return 0.0
    return float(max(0.0, min(1.0, allowed / max(bottom_fraction, 1e-6))))


def side_leak_fraction(mask: np.ndarray, args: argparse.Namespace) -> float:
    area = float(mask.sum())
    if area <= 0:
        return 0.0
    band = max(1, int(mask.shape[1] * args.leakage_side_band_ratio))
    band = min(band, mask.shape[1])
    side_pixels = 0
    if args.leakage_side_mode in {"left", "both"}:
        side_pixels += int(mask[:, :band].sum())
    if args.leakage_side_mode in {"right", "both"}:
        side_pixels += int(mask[:, mask.shape[1] - band :].sum())
    return float(side_pixels / area)


def side_leakage_quality(side_fraction: float, args: argparse.Namespace) -> float:
    allowed = max(float(args.leakage_max_side_fraction), 0.0)
    if side_fraction <= allowed:
        return 1.0
    if allowed <= 0:
        return 0.0
    return float(max(0.0, min(1.0, allowed / max(side_fraction, 1e-6))))


def safe_ratio(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def box_to_mask(box: tuple[int, int, int, int] | None, shape: tuple[int, int]) -> np.ndarray:
    out = np.zeros(shape, dtype=bool)
    if box is None:
        return out
    h, w = shape
    x0, y0, x1, y1 = box
    x0 = max(0, min(w, int(x0)))
    x1 = max(0, min(w, int(x1)))
    y0 = max(0, min(h, int(y0)))
    y1 = max(0, min(h, int(y1)))
    if x1 > x0 and y1 > y0:
        out[y0:y1, x0:x1] = True
    return out


def mask_bbox(mask: np.ndarray, padding_ratio: float = 0.0) -> tuple[int, int, int, int] | None:
    if not mask.any():
        return None
    h, w = mask.shape
    ys, xs = np.where(mask)
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    pad_x = int((x1 - x0) * padding_ratio)
    pad_y = int((y1 - y0) * padding_ratio)
    return (
        max(0, x0 - pad_x),
        max(0, y0 - pad_y),
        min(w, x1 + pad_x),
        min(h, y1 + pad_y),
    )


def parse_box_item(item: Any, shape: tuple[int, int]) -> tuple[int, int, int, int] | None:
    h, w = shape
    if item is None:
        return None
    if isinstance(item, dict):
        if "xyxy" in item:
            values = item["xyxy"]
        elif all(key in item for key in ("x0", "y0", "x1", "y1")):
            values = [item["x0"], item["y0"], item["x1"], item["y1"]]
        else:
            return None
    elif isinstance(item, (list, tuple)):
        values = item
    else:
        return None
    if len(values) != 4:
        return None
    x0, y0, x1, y1 = [float(v) for v in values]
    if max(abs(x0), abs(y0), abs(x1), abs(y1)) <= 1.5:
        x0, x1 = x0 * w, x1 * w
        y0, y1 = y0 * h, y1 * h
    x0_i, x1_i = sorted((int(round(x0)), int(round(x1))))
    y0_i, y1_i = sorted((int(round(y0)), int(round(y1))))
    return max(0, x0_i), max(0, y0_i), min(w, x1_i), min(h, y1_i)


def load_semantic_box_overrides(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("--semantic_box_json must contain a JSON object")
    return data


def semantic_override_for_image(overrides: dict[str, Any], image_path: Path) -> dict[str, Any]:
    for key in (image_path.name, image_path.stem, "*"):
        value = overrides.get(key)
        if isinstance(value, dict):
            return value
    return {}


def largest_green_component_mask(image: Image.Image, args: argparse.Namespace) -> np.ndarray:
    rgb = np.array(image.convert("RGB"), dtype=np.float32)
    exg = 2.0 * rgb[..., 1] - rgb[..., 0] - rgb[..., 2]
    green = exg > args.semantic_green_exg_threshold
    green = binary_opening(green, structure=kernel(3), iterations=1)
    green = remove_small_components(green, max(16, int(green.size * 0.0005)))
    if green.any():
        green = keep_largest_component(green)
        green = binary_closing(green, structure=kernel(11), iterations=1)
    return green.astype(bool)


def build_semantic_gate_context(
    image_path: Path,
    image: Image.Image,
    candidates: list[Candidate],
    overrides: dict[str, Any],
    args: argparse.Namespace,
) -> SemanticGateContext:
    shape = (image.height, image.width)
    override = semantic_override_for_image(overrides, image_path)

    target_box = parse_box_item(override.get("target_box"), shape)
    pot_box = parse_box_item(override.get("pot_box"), shape)
    side_boxes = [
        box
        for box in (parse_box_item(item, shape) for item in override.get("side_boxes", []))
        if box is not None
    ]

    union = np.zeros(shape, dtype=bool)
    for item in candidates:
        union |= item.mask
    green = largest_green_component_mask(image, args)
    # §SG-MPS gated mode needs an *independent* semantic reference: if phi/psi
    # were measured against the candidates' own union the gate would be
    # self-referential and c_i^(k) would be nearly constant across prompts.
    # `heuristic` keeps the historical (candidate-relative) construction so the
    # archived A1/A1s runs replay bitwise.
    gated_reference = getattr(args, "semantic_gate_mode", "heuristic") == "gated_softmax"
    if target_box:
        target_mask = box_to_mask(target_box, shape)
    else:
        target_mask = (green if gated_reference else (union | green))
        if target_mask.any():
            target_mask = keep_largest_component(target_mask)
            target_mask = binary_closing(target_mask, structure=kernel(11), iterations=1)
            target_mask = binary_fill_holes(target_mask)
    if not target_mask.any():
        target_mask = keep_largest_component(union) if union.any() else union
    if target_box is None:
        target_box = mask_bbox(target_mask, padding_ratio=0.08)

    h, w = shape
    yy = np.arange(h)[:, None]
    lower = yy >= int(h * args.semantic_pot_lower_start_ratio)
    rgb = np.array(image.convert("RGB"), dtype=np.float32)
    exg = 2.0 * rgb[..., 1] - rgb[..., 0] - rgb[..., 2]
    low_green = exg <= args.semantic_green_exg_threshold
    if gated_reference:
        pot_mask = lower & low_green
        if target_box is not None:
            tx0, tx1 = target_box[0], target_box[2]
            margin = max(1, w // 20)
            pot_mask[:, : max(0, tx0 - margin)] = False
            pot_mask[:, min(w, tx1 + margin) :] = False
    else:
        pot_mask = union & lower & low_green
    if pot_box is not None:
        pot_mask |= box_to_mask(pot_box, shape)
    if pot_mask.any():
        pot_mask = binary_closing(pot_mask, structure=kernel(9), iterations=1)
        pot_box = pot_box or mask_bbox(pot_mask, padding_ratio=0.04)

    band = max(1, int(w * args.semantic_side_band_ratio))
    side_mask = np.zeros(shape, dtype=bool)
    side_mask[:, :band] = True
    side_mask[:, w - band :] = True
    if target_box is not None:
        tx0, ty0, tx1, ty1 = target_box
        side_mask[:, max(0, tx0 - band // 2) : min(w, tx1 + band // 2)] = False
    if not gated_reference:
        side_mask &= union
    for box in side_boxes:
        side_mask |= box_to_mask(box, shape)
    if not side_boxes and side_mask.any():
        labeled, count = ndimage.label(side_mask)
        sizes = ndimage.sum(side_mask, labeled, range(1, count + 1))
        for region_id, size in enumerate(sizes, start=1):
            if size >= max(16, side_mask.size * 0.0008):
                bbox = mask_bbox(labeled == region_id, padding_ratio=0.02)
                if bbox is not None:
                    side_boxes.append(bbox)

    return SemanticGateContext(
        target_mask=target_mask.astype(bool),
        pot_mask=pot_mask.astype(bool),
        side_mask=side_mask.astype(bool),
        target_box=target_box,
        pot_box=pot_box,
        side_boxes=side_boxes,
    )


def parse_weight_vector(raw: str, expected: int, name: str) -> list[float]:
    values = [float(value) for value in str(raw).replace(";", ",").split(",") if str(value).strip() != ""]
    if len(values) != expected:
        raise ValueError(f"--{name} requires {expected} comma-separated values, got {len(values)}")
    return values


def semantic_gate_scores(
    mask: np.ndarray,
    context: SemanticGateContext,
    args: argparse.Namespace,
    image: Image.Image | None = None,
    temporal_iou: float = 0.5,
) -> dict[str, float]:
    area = float(mask.sum())
    if area <= 0:
        return {
            "semantic_total": 0.0,
            "target_box_score": 0.0,
            "vertical_coverage_score": 0.0,
            "pot_overlap_penalty": 1.0,
            "side_distractor_penalty": 1.0,
            "center_prior_score": 0.0,
            "gate_confidence": 0.0,
            "q_plant": 0.0, "q_area": 0.0, "q_comp": 0.0, "q_edge": 0.0,
            "q_contrast": 0.0, "q_temp": 0.0,
            "q_table": 1.0, "q_bg": 1.0, "q_neighbour": 1.0, "q_outlier": 1.0,
        }
    target_inter = float(np.logical_and(mask, context.target_mask).sum())
    target_denom = max(float(context.target_mask.sum()), 1.0)
    target_recall = target_inter / target_denom
    mask_target_precision = target_inter / area
    target_box_score = 0.5 * target_recall + 0.5 * mask_target_precision
    target_bbox = mask_bbox(context.target_mask)
    mask_bbox_value = mask_bbox(mask)
    vertical_coverage_score = target_recall
    if target_bbox is not None and mask_bbox_value is not None:
        _, ty0, _, ty1 = target_bbox
        _, my0, _, my1 = mask_bbox_value
        target_height = max(1, ty1 - ty0)
        overlap_height = max(0, min(my1, ty1) - max(my0, ty0))
        vertical_coverage_score = float(overlap_height / target_height)

    pot_overlap_penalty = safe_ratio(float(np.logical_and(mask, context.pot_mask).sum()), area)
    side_distractor_penalty = safe_ratio(float(np.logical_and(mask, context.side_mask).sum()), area)

    ys, xs = np.where(mask)
    cx = float(xs.mean()) / max(mask.shape[1] - 1, 1)
    cy = float(ys.mean()) / max(mask.shape[0] - 1, 1)
    center_prior_score = max(0.0, 1.0 - (abs(cx - 0.5) / 0.5) * 0.65 - max(0.0, cy - 0.72) / 0.28)

    # ── Phase 21 §SG-MPS descriptors (manuscript Eq. phi/psi) ──
    total_pixels = mask.size
    area_ratio = float(area / total_pixels) if total_pixels else 0.0
    min_area = max(1, int(total_pixels * args.component_min_area_ratio))
    comp_count = component_count(mask, min_area)
    q_area = area_quality(area_ratio, args)
    q_comp = 1.0 / (1.0 + max(comp_count - 1, 0))
    q_edge = edge_quality(mask, args)
    q_contrast = (
        min(1.0, foreground_background_contrast(image, mask) / 0.25)
        if image is not None else 0.5
    )
    q_temp = float(temporal_iou)
    green_fraction = 0.0
    if image is not None:
        rgb = np.array(image.convert("RGB"), dtype=np.float32)
        exg = 2.0 * rgb[..., 1] - rgb[..., 0] - rgb[..., 2]
        green = exg > args.semantic_green_exg_threshold
        green_fraction = safe_ratio(float(np.logical_and(mask, green).sum()), area)
    q_plant = 0.5 * target_box_score + 0.5 * green_fraction

    h, w = mask.shape
    lower_band = np.zeros_like(mask, dtype=bool)
    lower_band[int(h * args.semantic_pot_lower_start_ratio):, :] = True
    # Table = lower/outer scene support that is neither the target nor its pot.
    table_region = lower_band & ~context.pot_mask & ~context.target_mask
    q_table = safe_ratio(float(np.logical_and(mask, table_region).sum()), area)
    q_bg = safe_ratio(float(np.logical_and(mask, ~context.target_mask).sum()), area)
    q_neighbour = side_distractor_penalty
    q_outlier = float(max(0.0, 1.0 - q_comp))

    phi = [q_plant, q_area, q_comp, q_edge, q_contrast, q_temp]
    psi = [q_table, q_bg, q_neighbour, q_outlier]
    w_phi = parse_weight_vector(getattr(args, "semantic_phi_weights", "0.5,0.3,0.15,0.15,0.0,0.1"), 6, "semantic_phi_weights")
    w_psi = parse_weight_vector(getattr(args, "semantic_psi_weights", "0.3,0.3,0.3,0.2"), 4, "semantic_psi_weights")
    if not getattr(args, "semantic_gate_use_contrast", False):
        w_phi[4] = 0.0
    gate_bias = float(getattr(args, "semantic_gate_bias", 0.5))
    logit = sum(w * v for w, v in zip(w_phi, phi)) - sum(w * v for w, v in zip(w_psi, psi)) - gate_bias
    gate_confidence = float(1.0 / (1.0 + math.exp(-max(min(logit, 60.0), -60.0))))

    if getattr(args, "semantic_gate_mode", "heuristic") == "gated_softmax":
        # The gated softmax is applied at selection time (needs the whole frame's
        # candidate set); the per-candidate additive term is therefore zero.
        semantic_total = 0.0
    else:
        semantic_total = (
            args.target_box_weight * target_box_score
            + args.vertical_coverage_weight * vertical_coverage_score
            + args.center_prior_weight * center_prior_score
            - args.pot_overlap_weight * pot_overlap_penalty
            - args.side_distractor_weight * side_distractor_penalty
        )
    return {
        "semantic_total": float(semantic_total),
        "target_box_score": float(target_box_score),
        "vertical_coverage_score": float(vertical_coverage_score),
        "pot_overlap_penalty": float(pot_overlap_penalty),
        "side_distractor_penalty": float(side_distractor_penalty),
        "center_prior_score": float(center_prior_score),
        "gate_confidence": gate_confidence,
        "q_plant": float(q_plant), "q_area": float(q_area), "q_comp": float(q_comp),
        "q_edge": float(q_edge), "q_contrast": float(q_contrast), "q_temp": float(q_temp),
        "q_table": float(q_table), "q_bg": float(q_bg),
        "q_neighbour": float(q_neighbour), "q_outlier": float(q_outlier),
    }


def apply_semantic_gate_softmax(score_records: list[ScoreRecord], args: argparse.Namespace) -> None:
    """Phase 21 §SG-MPS: temperature-controlled softmax over PROMPTS.

    Implements a_i^(k) = exp(c_i^(k)/tau_s) / sum_l exp(c_i^(l)/tau_s) exactly as
    written in the manuscript: the normalisation runs over the K prompt indices,
    not over instances.  When a prompt returns several instances, its
    representative compatibility is max_k c_i^(k) (the strongest instance), and
    that prompt-level probability is written back to every instance of the
    prompt so downstream reporting sees one weight per prompt.

    Selection is still done by k* = argmax_k a_i^(k) (equal to argmax_k c_i^(k)
    because softmax is strictly monotone); see `semantic_gate_prompt`.
    """
    if not getattr(args, "use_semantic_gate", False):
        return
    if getattr(args, "semantic_gate_mode", "heuristic") != "gated_softmax":
        return
    usable = [rec for rec in score_records if not bool(getattr(rec, "empty_flag", False))]
    if not usable:
        return
    best_by_prompt: dict[str, float] = {}
    for rec in usable:
        value = float(getattr(rec, "gate_confidence", 0.0))
        best_by_prompt[rec.prompt_id] = max(best_by_prompt.get(rec.prompt_id, -1.0), value)
    if not best_by_prompt:
        return
    tau = max(float(getattr(args, "semantic_gate_temperature", 0.25)), 1e-6)
    prompt_ids = sorted(best_by_prompt)
    peak = max(best_by_prompt.values())
    exps = {pid: math.exp((best_by_prompt[pid] - peak) / tau) for pid in prompt_ids}
    normalizer = sum(exps.values())
    if normalizer <= 0.0:
        normalizer = 1.0
    prompt_prob = {pid: exps[pid] / normalizer for pid in prompt_ids}
    for rec in usable:
        rec.gate_softmax = float(prompt_prob.get(rec.prompt_id, 0.0))


def semantic_gate_prompt(score_records: list[ScoreRecord], args: argparse.Namespace) -> str | None:
    """k* = argmax_k c_i^(k): the prompt SG-MPS selects for this frame.

    Returns None when the gate is off or in legacy `heuristic` mode, in which
    case prompt/instance selection stays with the ranking policy.

    `--semantic_gate_margin` implements the manuscript's "safety mechanism"
    reading of §2.3.3: when the top-2 gate confidences are separated by less
    than `margin`, the gate declares no preference (returns None) and leaves the
    prompt to the geometry ranking instead of breaking a near-tie on noise.
    Default 0.0 reproduces the pure argmax exactly.
    """
    if not getattr(args, "use_semantic_gate", False):
        return None
    if getattr(args, "semantic_gate_mode", "heuristic") != "gated_softmax":
        return None
    usable = [rec for rec in score_records if not bool(getattr(rec, "empty_flag", False))]
    if not usable:
        return None
    # c_i^(k) is per prompt; a prompt may contribute several instances, so use
    # the strongest instance of each prompt (max over instances, as in the
    # manuscript where each prompt returns one foreground region).
    best_by_prompt: dict[str, float] = {}
    for rec in usable:
        value = float(getattr(rec, "gate_confidence", 0.0))
        best_by_prompt[rec.prompt_id] = max(best_by_prompt.get(rec.prompt_id, -1.0), value)
    if not best_by_prompt:
        return None
    ranked = sorted(best_by_prompt.items(), key=lambda item: item[1], reverse=True)
    margin = float(getattr(args, "semantic_gate_margin", 0.0) or 0.0)
    if margin > 0.0 and len(ranked) > 1 and (ranked[0][1] - ranked[1][1]) < margin:
        return None
    return ranked[0][0]


def save_semantic_gate_debug(
    image: Image.Image,
    context: SemanticGateContext,
    selected_mask: np.ndarray,
    path: Path,
) -> None:
    rgb = np.array(image.convert("RGB")).copy()
    overlay = rgb.astype(np.float32)
    overlay[context.target_mask] = overlay[context.target_mask] * 0.65 + np.array([0, 180, 80]) * 0.35
    overlay[context.pot_mask] = overlay[context.pot_mask] * 0.55 + np.array([230, 140, 0]) * 0.45
    overlay[context.side_mask] = overlay[context.side_mask] * 0.55 + np.array([210, 60, 50]) * 0.45
    out = np.clip(overlay, 0, 255).astype(np.uint8)
    for box, color in [
        (context.target_box, (0, 180, 80)),
        (context.pot_box, (230, 140, 0)),
    ]:
        if box is not None:
            x0, y0, x1, y1 = box
            cv2.rectangle(out, (x0, y0), (max(x0, x1 - 1), max(y0, y1 - 1)), color, 3)
    for box in context.side_boxes:
        x0, y0, x1, y1 = box
        cv2.rectangle(out, (x0, y0), (max(x0, x1 - 1), max(y0, y1 - 1)), (210, 60, 50), 3)
    contours, _ = cv2.findContours(selected_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(out, contours, -1, (40, 110, 220), 3)
    Image.fromarray(out).save(path)


def score_candidate(
    image_path: Path,
    image: Image.Image,
    candidate: Candidate,
    prev_prompt_masks: dict[str, np.ndarray],
    weights: dict[str, float],
    args: argparse.Namespace,
    semantic_context: SemanticGateContext | None = None,
) -> ScoreRecord:
    mask = candidate.mask
    # ── 阶段十一 §4.1 空掩膜硬门：不再因时序默认满分得高分 ──
    if not mask.any():
        empty_policy = getattr(args, "ranking_policy", "r1")
        return ScoreRecord(
            image_name=image_path.name,
            prompt_id=candidate.prompt_id,
            prompt_text=candidate.prompt_text,
            total_score=0.0,
            q_area=0.0, q_comp=0.0, q_edge=0.0, q_temp=0.0,
            q_contrast=0.0, q_leak=0.0, q_side=0.0,
            area_ratio=0.0, component_count=0, boundary_density=0.0,
            temporal_iou=0.0, contrast=0.0, bottom_leak_fraction=0.0, side_leak_fraction=0.0,
            sam_scores=";".join(f"{s:.6f}" for s in candidate.scores),
            instance_id=candidate.instance_id,
            empty_flag=True,
            ranking_policy=empty_policy,
            primary_score=0.0,
            tie_break_score=0.0,
            contrast_effective=0.0,
        )
    total_pixels = mask.size
    area_ratio = float(mask.sum() / total_pixels) if total_pixels else 0.0
    min_area = max(1, int(total_pixels * args.component_min_area_ratio))
    comp_count = component_count(mask, min_area)
    boundary_density = edge_density(mask)
    contrast = foreground_background_contrast(image, mask)
    leak_fraction = bottom_leak_fraction(mask, args)
    side_fraction = side_leak_fraction(mask, args)
    prev = prev_prompt_masks.get(candidate.prompt_id)
    temporal_iou = mask_iou(mask, prev) if (prev is not None and getattr(args, "use_temporal_alignment", False)) else 0.5

    q_area = area_quality(area_ratio, args)
    q_comp = 1.0 / (1.0 + max(comp_count - 1, 0))
    # 阶段十一 §4.4：边界置信度改为边缘贴合度（不奖励"边界越少越好"）
    q_edge = edge_quality(mask, args)
    # 阶段十一 §4.5 时序对齐门控：未显式配准时序项置中性 0.5，不默认满分
    q_temp = temporal_iou if getattr(args, "use_temporal_alignment", False) else 0.5
    q_contrast = min(1.0, contrast / 0.25)
    q_leak = leakage_quality(leak_fraction, args)
    q_side = side_leakage_quality(side_fraction, args)
    # ── 阶段十一 §4.2 纳入 SAM3 实例 score（sigmoid 归一化到 [0,1]）──
    q_sam = 1.0 / (1.0 + math.exp(-6.0 * (float(candidate.sam_score) - 0.5)))
    denom = max(sum(weights.values()), 1e-6)
    base_total = (
        weights.get("area", 0.0) * q_area
        + weights.get("comp", 0.0) * q_comp
        + weights.get("edge", 0.0) * q_edge
        + weights.get("temp", 0.0) * q_temp
        + weights.get("contrast", 0.0) * q_contrast
        + weights.get("leak", 0.0) * q_leak
        + weights.get("side", 0.0) * q_side
        + weights.get("sam", 0.0) * q_sam
    ) / denom
    # ── 阶段十一 §4.6 硬门控：纵向覆盖不足 / 与 track 重叠低 → 高风险惩罚 ──
    risk_penalty = 0.0
    h, w = mask.shape
    ys = np.where(mask.any(axis=1))[0]
    upper_coverage = float(ys.min() / max(h, 1)) if len(ys) else 1.0  # 上部覆盖深度（越小越偏下）
    vertical_coverage = 1.0 - upper_coverage
    if vertical_coverage < getattr(args, "vertical_coverage_min_ratio", 0.30):
        risk_penalty += 0.25  # 仅下部小区域覆盖，疑似只割到盆/条带
    track_mask = getattr(candidate, "track_mask", None)
    if track_mask is not None and track_mask.any():
        if mask_iou(mask, track_mask.astype(bool)) < getattr(args, "box_track_overlap_min", 0.20):
            risk_penalty += 0.15
    semantic = {
        "semantic_total": 0.0,
        "target_box_score": 0.0,
        "vertical_coverage_score": 0.0,
        "pot_overlap_penalty": 0.0,
        "side_distractor_penalty": 0.0,
        "center_prior_score": 0.0,
        "gate_confidence": 0.0,
        "q_plant": 0.0, "q_area": 0.0, "q_comp": 0.0, "q_edge": 0.0,
        "q_contrast": 0.0, "q_temp": 0.0,
        "q_table": 0.0, "q_bg": 0.0, "q_neighbour": 0.0, "q_outlier": 0.0,
    }
    if args.use_semantic_gate and semantic_context is not None:
        semantic = semantic_gate_scores(mask, semantic_context, args, image=image, temporal_iou=q_temp)
    total = base_total + semantic["semantic_total"] - risk_penalty
    # ── Phase 17C §17: counterfactual ranking policy (pure function, no tuning) ──
    policy = getattr(args, "ranking_policy", "r1")
    primary_score, tie_break_score = primary_and_tiebreak(
        base_total=base_total,
        q_contrast=q_contrast,
        q_area=q_area,
        w_contrast=weights.get("contrast", 0.0),
        denom=denom,
        semantic_total=semantic["semantic_total"],
        penalty=risk_penalty,
        policy=policy,
    )
    contrast_effective = float(q_contrast * q_area)
    return ScoreRecord(
        image_name=image_path.name,
        prompt_id=candidate.prompt_id,
        prompt_text=candidate.prompt_text,
        total_score=float(total),
        q_area=float(q_area),
        q_comp=float(q_comp),
        q_edge=float(q_edge),
        q_temp=float(q_temp),
        q_contrast=float(q_contrast),
        q_leak=float(q_leak),
        q_side=float(q_side),
        area_ratio=area_ratio,
        component_count=comp_count,
        boundary_density=boundary_density,
        temporal_iou=temporal_iou,
        contrast=contrast,
        bottom_leak_fraction=leak_fraction,
        side_leak_fraction=side_fraction,
        sam_scores=";".join(f"{score:.6f}" for score in candidate.scores),
        instance_id=candidate.instance_id,
        semantic_enabled=bool(args.use_semantic_gate and semantic_context is not None),
        semantic_total=float(semantic["semantic_total"]),
        target_box_score=float(semantic["target_box_score"]),
        vertical_coverage_score=float(semantic["vertical_coverage_score"]),
        pot_overlap_penalty=float(semantic["pot_overlap_penalty"]),
        side_distractor_penalty=float(semantic["side_distractor_penalty"]),
        center_prior_score=float(semantic["center_prior_score"]),
        leak_penalty=float(risk_penalty),
        ranking_policy=policy,
        primary_score=float(primary_score),
        tie_break_score=float(tie_break_score),
        contrast_effective=float(contrast_effective),
        gate_confidence=float(semantic.get("gate_confidence", 0.0)),
        gate_softmax=0.0,
        phi_descriptor=",".join(
            f"{value:.6f}" for value in (
                semantic.get("q_plant", 0.0), semantic.get("q_area", 0.0),
                semantic.get("q_comp", 0.0), semantic.get("q_edge", 0.0),
                semantic.get("q_contrast", 0.0), semantic.get("q_temp", 0.0),
            )
        ),
        psi_descriptor=",".join(
            f"{value:.6f}" for value in (
                semantic.get("q_table", 0.0), semantic.get("q_bg", 0.0),
                semantic.get("q_neighbour", 0.0), semantic.get("q_outlier", 0.0),
            )
        ),
    )


def select_mask(
    image_path: Path,
    candidates: list[Candidate],
    score_records: list[ScoreRecord],
    prompt_texts: dict[str, str],
    args: argparse.Namespace,
    selection_meta: dict[str, Any] | None = None,
) -> tuple[np.ndarray, str, float]:
    if selection_meta is not None:
        selection_meta.clear()
    # Phase 21 §SG-MPS: apply the temperature softmax safety bonus to the
    # policy primary score BEFORE any branch selects. No-op unless
    # semantic_gate_mode=gated_softmax, so historical replay is unchanged.
    apply_semantic_gate_softmax(score_records, args)
    if args.use_prompt_ensemble and args.prompt_selection_mode == "score_select":
        # 阶段十二 §三.1：score_select 按 (prompt_id, instance_id) 选最佳实例，不只按 prompt_id
        # Phase 21: prompt selection is done first by SG-MPS when the gated mode is
        # active (k* = argmax_k c_i^(k), manuscript §2.3.3); the ranking policy then
        # picks the best instance *within* that prompt, so R1 is never bypassed.
        # Under `base` primary == total, so historical replay remains bitwise identical.
        non_empty = {item.prompt_id for item in candidates if item.mask.any()}
        selectable = [row for row in score_records if row.prompt_id in non_empty]
        if not selectable:
            selectable = score_records
        gated_prompt = semantic_gate_prompt(score_records, args)
        if selection_meta is not None:
            selection_meta["sg_mps_prompt"] = gated_prompt or ""
            selection_meta["sg_mps_softmax"] = {
                rec.prompt_id: float(getattr(rec, "gate_softmax", 0.0))
                for rec in score_records
                if not bool(getattr(rec, "empty_flag", False))
            }
            prompt_order = sorted({rec.prompt_id for rec in score_records})
            selection_meta["sg_mps_candidates"] = {
                pid: max(
                    (float(getattr(rec, "gate_confidence", 0.0)) for rec in score_records
                     if rec.prompt_id == pid and not bool(getattr(rec, "empty_flag", False))),
                    default=-1.0,
                )
                for pid in prompt_order
            }
        if gated_prompt is not None:
            gated_rows = [row for row in selectable if row.prompt_id == gated_prompt]
            if gated_rows:
                selectable = gated_rows
        best = select_best_policy(selectable, getattr(args, "ranking_policy", "r1"))
        if selection_meta is not None:
            selection_meta["ranking_shortlist_prompt"] = gated_prompt or "(all)"
            selection_meta["selected_by_ranking_prompt"] = best.prompt_id
            selection_meta["ranking_policy"] = str(getattr(args, "ranking_policy", "r1"))
        # 找到与 best 匹配的候选（同 prompt_id + instance_id）
        best_cand = next(
            (item for item in candidates
             if item.prompt_id == best.prompt_id and item.instance_id == best.instance_id),
            next((item for item in candidates if item.prompt_id == best.prompt_id), None),
        )
        if best_cand is None:
            raise ValueError(f"No candidate found for {best.prompt_id}#{best.instance_id}")
        if selection_meta is not None:
            selection_meta["selected_prompt"] = best_cand.prompt_id
            selection_meta["selected_instance"] = int(best_cand.instance_id)
            selection_meta["selection_path"] = "sg_mps_then_ranking"
        best_score = float(getattr(best, "primary_score", best.total_score))
        return best_cand.mask.copy(), best.prompt_id, best_score, False
    if args.use_prompt_ensemble and args.prompt_selection_mode == "weighted_fusion":
        # Phase 21: fusion weights use the ranking-policy primary score (R1-aware).
        # A per-prompt max keeps each prompt's strongest instance; softmax makes the
        # weights shift-invariant and safe when primary scores are negative.
        primary_by_prompt: dict[str, float] = {}
        for row in score_records:
            value = float(getattr(row, "primary_score", row.total_score))
            primary_by_prompt[row.prompt_id] = max(primary_by_prompt.get(row.prompt_id, -float("inf")), value)
        if primary_by_prompt:
            temperature = max(float(getattr(args, "fusion_temperature", 1.0)), 1e-6)
            peak = max(primary_by_prompt.values())
            exp_scores = {pid: math.exp((value - peak) / temperature) for pid, value in primary_by_prompt.items()}
            exp_sum = sum(exp_scores.values())
            weights = {pid: value / exp_sum for pid, value in exp_scores.items()}
        else:
            weights = {}
        fused = np.zeros_like(candidates[0].mask, dtype=np.float32)
        for item in candidates:
            fused += weights.get(item.prompt_id, 0.0) * item.mask.astype(np.float32)
        return fused >= args.fusion_threshold, "weighted_fusion", float(max(primary_by_prompt.values(), default=0.0)), False

    prompt_id = args.default_prompt_id
    if prompt_id not in prompt_texts:
        raise ValueError(f"Unknown --default_prompt_id: {prompt_id}")
    default_cands = [item for item in candidates if item.prompt_id == prompt_id]
    if not default_cands:
        raise ValueError(f"Default prompt {prompt_id} was not generated for {image_path.name}")
    # per_instance 模式：default_prompt 下有多个实例候选，按评分选最佳实例
    if len(default_cands) > 1 and args.candidate_mode == "per_instance":
        rec_by_id = {(r.prompt_id, getattr(r, "instance_id", 0)): r for r in score_records}
        default_recs = [
            rec_by_id[(item.prompt_id, item.instance_id)]
            for item in default_cands
            if (item.prompt_id, item.instance_id) in rec_by_id
        ]
        # Phase 17C §17: selection uses ranking_policy pure function (single source of truth)
        best_rec = select_best_policy(default_recs, getattr(args, "ranking_policy", "r1"))
        best_item = next(
            (item for item in default_cands
             if item.prompt_id == best_rec.prompt_id and item.instance_id == best_rec.instance_id),
            default_cands[0],
        )
        # §4.7 重提示触发：top1 与 top2 分差 < 阈值 → 候选不确定，需重提示
        sorted_recs = sorted(score_records, key=lambda r: r.total_score, reverse=True)
        needs_reprompt = len(sorted_recs) >= 2 and (sorted_recs[0].total_score - sorted_recs[1].total_score) < args.reprompt_score_gap
        best_score = float(getattr(best_rec, "primary_score", best_rec.total_score))
        if selection_meta is not None:
            selection_meta["selected_prompt"] = str(prompt_id)
            selection_meta["selected_instance"] = int(best_item.instance_id)
            selection_meta["selection_path"] = "ranking_only"
        return best_item.mask.copy(), f"{prompt_id}#{best_item.instance_id}", best_score, bool(needs_reprompt)
    row = next((r for r in score_records if r.prompt_id == prompt_id), None)
    # 单一候选：空掩膜或低分 → 触发重提示
    needs_reprompt = (row is None) or (row.total_score < args.reprompt_min_score) or bool(getattr(row, "empty_flag", False))
    if selection_meta is not None:
        selection_meta["selected_prompt"] = str(prompt_id)
        selection_meta["selected_instance"] = int(getattr(default_cands[0], "instance_id", 0))
        selection_meta["selection_path"] = "default_prompt"
    return default_cands[0].mask.copy(), prompt_id, float(row.total_score if row else 0.0), bool(needs_reprompt)


# ────────────────────────────────────────────────────────────────────────
# 阶段十一 §5 P2/P6 语义坍缩检测与受控组合
# ────────────────────────────────────────────────────────────────────────
def detect_semantic_collapse(
    p2_mask: np.ndarray,
    p6_mask: np.ndarray,
    args: argparse.Namespace | None = None,
) -> bool:
    """P2 与 P6 不应在面积/框/盆区覆盖上几乎相同（否则语义坍缩）。

    返回 True 表示检测到坍缩（需触发重推理）。
    """
    if not p2_mask.any() or not p6_mask.any():
        return False
    a2 = float(p2_mask.sum())
    a6 = float(p6_mask.sum())
    if a2 <= 0:
        return False
    area_diff_ratio = abs(a6 - a2) / a2
    # 盆区（图像下 40%）覆盖率是否几乎一致
    h = p2_mask.shape[0]
    lo = int(h * 0.4)
    p2_pot = float(p2_mask[lo:, :].sum())
    p6_pot = float(p6_mask[lo:, :].sum())
    pot_diff_ratio = (abs(p6_pot - p2_pot) / max(a2, 1e-6)) if p6_pot or p2_pot else 0.0
    threshold = 0.05 if args is None else getattr(args, "collapse_area_threshold", 0.05)
    return bool(area_diff_ratio < threshold and pot_diff_ratio < threshold)


def compose_plant_only(
    plant_cands: list[np.ndarray],
    pot_cands: list[np.ndarray],
    track_support: np.ndarray | None = None,
    args: argparse.Namespace | None = None,
) -> np.ndarray:
    """受控组合 plant_only：植株候选并集 − 明确 POT 候选区域。

    - 叶片召回来自 plant_cands（P2/P3 实例），不在此做最大连通域。
    - 减去经 POT 候选确认的盆区，避免盆体被当植株召回。
    """
    plant = np.zeros_like(plant_cands[0], dtype=bool) if plant_cands else np.zeros((1, 1), dtype=bool)
    for m in plant_cands:
        plant |= m.astype(bool)
    if pot_cands:
        pot = np.zeros_like(plant)
        for m in pot_cands:
            pot |= m.astype(bool)
        plant &= ~pot
    return plant


def compose_potted(
    plant_only: np.ndarray,
    pot_cands: list[np.ndarray],
) -> np.ndarray:
    """P6 = plant_only ∪ POT 候选（独立生成，不取 P2 终选 mask）。"""
    out = plant_only.astype(bool).copy()
    for m in pot_cands:
        out |= m.astype(bool)
    return out


# ────────────────────────────────────────────────────────────────────────
# 阶段十一 §8 A7 单对象 ID 选择（禁止 OR 所有 object ID）
# ────────────────────────────────────────────────────────────────────────
def select_single_object_id(
    object_masks: list[np.ndarray],
    seed_mask: np.ndarray,
) -> int:
    """从 A7 多 object ID 输出中选一个与 seed 最匹配的 ID（IoU 最大）。

    禁止把多个 object ID 直接 OR 合并。
    """
    if not object_masks:
        return -1
    best_iou = -1.0
    best_id = 0
    for i, om in enumerate(object_masks):
        iou = mask_iou(om.astype(bool), seed_mask.astype(bool))
        if iou > best_iou:
            best_iou = iou
            best_id = i
    return best_id


def sample_points(mask: np.ndarray, count: int, positive: bool, ring_radius: int) -> list[tuple[int, int]]:
    if count <= 0:
        return []
    if positive:
        dist = ndimage.distance_transform_edt(mask)
        ys, xs = np.where(dist > 0)
        if len(xs) == 0:
            return []
        order = np.argsort(dist[ys, xs])[::-1]
        chosen: list[tuple[int, int]] = []
        blocked = np.zeros(mask.shape, dtype=bool)
        min_sep = max(4, int(min(mask.shape) * 0.04))
        for idx in order:
            y, x = int(ys[idx]), int(xs[idx])
            if blocked[y, x]:
                continue
            chosen.append((x, y))
            yy, xx = np.ogrid[: mask.shape[0], : mask.shape[1]]
            blocked |= (yy - y) ** 2 + (xx - x) ** 2 <= min_sep**2
            if len(chosen) >= count:
                break
        return chosen

    dilated = binary_dilation(mask, structure=kernel(max(3, ring_radius * 2 + 1)))
    ring = dilated & ~mask
    ys, xs = np.where(ring)
    if len(xs) == 0:
        return []
    indices = np.linspace(0, len(xs) - 1, min(count, len(xs))).astype(int)
    return [(int(xs[i]), int(ys[i])) for i in indices]


def sample_points_from_region(region: np.ndarray, count: int, nms_radius: int = 0) -> list[tuple[int, int]]:
    """Negative-click sampling by greedy NMS inside a candidate region.

    Manuscript §2.3.4 defines ``P_i^- = NMS_{p in R_i^-}(zeta_i(p), K_-, r_nms)``.
    ``zeta`` is taken as the distance transform of the interference region (a
    deeper pixel is a more confident interference pixel); suppression radius is
    ``nms_radius`` when given, otherwise the historical 4%-of-image spacing.
    """
    if count <= 0 or not region.any():
        return []
    dist = ndimage.distance_transform_edt(region)
    ys, xs = np.where(dist > 0)
    if len(xs) == 0:
        return []
    order = np.argsort(dist[ys, xs])[::-1]
    chosen: list[tuple[int, int]] = []
    blocked = np.zeros(region.shape, dtype=bool)
    min_sep = int(nms_radius) if nms_radius > 0 else max(4, int(min(region.shape) * 0.04))
    yy, xx = np.ogrid[: region.shape[0], : region.shape[1]]
    for idx in order:
        y, x = int(ys[idx]), int(xs[idx])
        if blocked[y, x]:
            continue
        chosen.append((x, y))
        blocked |= (yy - y) ** 2 + (xx - x) ** 2 <= min_sep**2
        if len(chosen) >= count:
            break
    return chosen


def manuscript_negative_regions(
    mask: np.ndarray,
    image: Image.Image,
    args: argparse.Namespace,
) -> dict[str, np.ndarray]:
    """The four R_i^- sub-regions of manuscript §2.3.4, kept separately.

        R^- = (Dilate(M*, r) \\ M*) ∪ I_table ∪ I_bg ∪ I_neighbour

    Keeping them separate lets the sampler spread K_- clicks over the ring, the
    table band, the distant background and the neighbour band instead of
    collapsing every click onto whichever sub-region happens to be largest.
    The visible container is excluded: the ``~mask`` intersection drops a pot
    that the selected instance already covers, and ``I_table`` requires
    low-green evidence so a pot the prompt missed is not turned into a negative.
    """
    h, w = mask.shape
    ring = binary_dilation(mask, structure=kernel(max(3, int(args.spnp_ring_radius) * 2 + 1))) & ~mask
    rgb = np.array(image.convert("RGB"), dtype=np.float32)
    exg = 2.0 * rgb[..., 1] - rgb[..., 0] - rgb[..., 2]
    low_green = exg <= float(args.semantic_green_exg_threshold)
    yy = np.arange(h)[:, None]
    table = (yy >= int(h * float(args.spnp_table_band_start_ratio))) & low_green
    xs_m = np.where(mask.any(axis=0))[0]
    ys_m = np.where(mask.any(axis=1))[0]
    if len(xs_m) and len(ys_m):
        x0, x1 = int(xs_m.min()), int(xs_m.max())
        y0, y1 = int(ys_m.min()), int(ys_m.max())
        margin_x = max(1, int((x1 - x0 + 1) * 0.10))
        margin_y = max(1, int((y1 - y0 + 1) * 0.10))
        bg = ~box_to_mask((x0 - margin_x, y0 - margin_y, x1 + margin_x, y1 + margin_y), mask.shape)
    else:
        bg = np.ones_like(mask, dtype=bool)
    band = max(1, int(w * float(args.spnp_side_band_ratio)))
    neighbour = np.zeros_like(mask, dtype=bool)
    neighbour[:, :band] = True
    neighbour[:, w - band :] = True
    return {
        "ring": ring & ~mask,
        "table": table & ~mask,
        "bg": bg & ~mask,
        "neighbour": neighbour & ~mask,
    }


def manuscript_negative_region(
    mask: np.ndarray,
    image: Image.Image,
    args: argparse.Namespace,
) -> np.ndarray:
    """Union of the four R_i^- sub-regions (manuscript §2.3.4)."""
    regions = manuscript_negative_regions(mask, image, args)
    out = np.zeros_like(mask, dtype=bool)
    for region in regions.values():
        out |= region
    return out & ~mask


def sample_negative_prompts(
    sub_regions: dict[str, np.ndarray],
    count: int,
    nms_radius: int,
    image_shape: tuple[int, int],
) -> list[tuple[int, int]]:
    """Distribute K_- negative clicks over R_i^- sub-regions, then NMS-merge.

    Guarantees at least one click from every non-empty sub-region (so the
    refinement actually sees the ring, the table/soil strip, the distant
    background and the neighbouring-plant band), then fills the remainder by
    descending region depth.  Clicks are kept away from the image border so a
    prompt point never lands exactly on a pixel edge.
    """
    if count <= 0:
        return []
    h, w = image_shape
    border = max(4, int(min(h, w) * 0.01))
    valid = np.zeros((h, w), dtype=bool)
    valid[border : h - border, border : w - border] = True
    order = list(sub_regions.keys())
    non_empty = [name for name in order if sub_regions[name].any()]
    if not non_empty:
        return []
    chosen: list[tuple[int, int]] = []
    blocked = np.zeros((h, w), dtype=bool)
    min_sep = int(nms_radius) if nms_radius > 0 else max(6, int(min(h, w) * 0.05))
    yy, xx = np.ogrid[:h, :w]

    def take(region: np.ndarray, quota: int) -> int:
        if quota <= 0:
            return 0
        region = region & valid & ~blocked
        if not region.any():
            return 0
        dist = ndimage.distance_transform_edt(region)
        ys, xs = np.where(dist > 0)
        if len(xs) == 0:
            return 0
        picked = 0
        for idx in np.argsort(dist[ys, xs])[::-1]:
            y, x = int(ys[idx]), int(xs[idx])
            if blocked[y, x]:
                continue
            chosen.append((x, y))
            np.logical_or(blocked, (yy - y) ** 2 + (xx - x) ** 2 <= min_sep**2, out=blocked)
            picked += 1
            if picked >= quota or len(chosen) >= count:
                break
        return picked

    non_empty = [name for name in order if name in sub_regions and sub_regions[name].any()]
    # One guaranteed click per non-empty sub-region, then fill the rest.
    for name in non_empty:
        if len(chosen) >= count:
            break
        take(sub_regions[name], 1)
    remaining = count - len(chosen)
    while remaining > 0 and len(chosen) < count:
        before = len(chosen)
        per_region = max(1, remaining // max(len(non_empty), 1))
        for name in non_empty:
            remaining = count - len(chosen)
            if remaining <= 0:
                break
            take(sub_regions[name], min(per_region, remaining))
        if len(chosen) == before:
            break
    return chosen[:count]


def build_sb_fbpr_prompts(
    mask: np.ndarray,
    image: Image.Image,
    evidence: dict[str, np.ndarray] | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """SB-FBPR prompt construction (manuscript §2.3.4).

    Positive : TopK distance-transform interior points of the main foreground
               instance (track-backed core when evidence is available).
    Negative : NMS-sampled points from R_i^- ∪ consensus-cut / pot-strip
               evidence regions.
    Returns pixel (x, y) coordinates, ready for both point and box prompts.
    """
    positive_region = mask
    if evidence is not None and evidence.get("positive_core") is not None and evidence["positive_core"].any():
        positive_region = evidence["positive_core"]
    pos = sample_points(positive_region, args.spnp_positive_points, True, args.spnp_ring_radius)

    sub_regions: dict[str, np.ndarray] = {}
    if evidence is not None:
        sub_regions["consensus_cut"] = evidence["negative_region"]
        if evidence["below_strip"].any():
            sub_regions["pot_strip"] = evidence["below_strip"]
    if args.spnp_negative_region_mode == "manuscript":
        sub_regions.update(manuscript_negative_regions(mask, image, args))
    elif evidence is None:
        # Historical ring path: keep the exact legacy sampling so that
        # box-only replay remains byte-identical.
        sub_regions["ring"] = (
            binary_dilation(mask, structure=kernel(max(3, args.spnp_ring_radius * 2 + 1))) & ~mask
        )

    if evidence is not None and args.spnp_negative_region_mode != "manuscript":
        # Historical evidence-guided path: consensus-cut region, plus the pot
        # strip when it exists.  Kept verbatim for A6/A7 replay.
        neg = sample_points_from_region(evidence["negative_region"], args.spnp_negative_points)
        if evidence["below_strip"].any() and neg:
            neg = sample_points_from_region(evidence["below_strip"], 2) + neg
    elif args.spnp_negative_region_mode == "manuscript" or evidence is not None:
        neg = sample_negative_prompts(
            sub_regions, args.spnp_negative_points, args.spnp_negative_nms_radius, mask.shape
        )
    else:
        neg = sample_points(mask, args.spnp_negative_points, False, args.spnp_ring_radius)
    # De-duplicate while keeping the order stable.
    seen: set[tuple[int, int]] = set()
    neg_unique = []
    for point in neg:
        if point in seen:
            continue
        seen.add(point)
        neg_unique.append(point)
    # Union of the sampled-from regions, used for the evidence/debug overlay.
    negative_region = np.zeros_like(mask, dtype=bool)
    for region in sub_regions.values():
        negative_region |= region
    return {"positive": pos, "negative": neg_unique, "negative_region": negative_region}


def sam3_box_refinement(
    image: Image.Image,
    mask: np.ndarray,
    prompt_text: str,
    positive_points: list[tuple[int, int]],
    negative_points: list[tuple[int, int]],
    processor: Any,
    torch_module: Any,
    args: argparse.Namespace,
    evidence: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    if processor is None or torch_module is None:
        raise RuntimeError("--spnp_backend sam3_if_supported requires a loaded SAM3 processor")

    h, w = mask.shape
    prompt_form = str(getattr(args, "spnp_prompt_form", "boxes"))

    # Evidence-guided mode: positive box hugs the track-backed core (NOT the
    # full mask bbox, whose rectangle covers the pot), negative boxes sit on
    # consensus-cut pixels and the below-mask pot strip.
    if evidence is not None:
        core = evidence["positive_core"]
        positive_box = (
            mask_to_positive_box(core, args.spnp_positive_box_padding)
            if core is not None and core.any()
            else mask_to_positive_box(mask, args.spnp_positive_box_padding)
        )
        negative_boxes = [
            point_to_negative_box(point, mask.shape, args.spnp_negative_box_radius)
            for point in negative_points[: args.spnp_negative_points + 2]
        ]
    else:
        positive_box = mask_to_positive_box(mask, args.spnp_positive_box_padding)
        negative_boxes = [
            point_to_negative_box(point, mask.shape, args.spnp_negative_box_radius)
            for point in negative_points[: args.spnp_negative_points]
        ]
    if args.spnp_use_lower_negative:
        lower_box = lower_negative_box(mask, args.spnp_lower_band_ratio)
        if lower_box is not None:
            negative_boxes.append(lower_box)

    # Normalized point prompts for the VFM (manuscript §2.3.4).
    pos_norm = [(float(x) / max(w - 1, 1), float(y) / max(h - 1, 1)) for x, y in positive_points]
    neg_norm = [(float(x) / max(w - 1, 1), float(y) / max(h - 1, 1)) for x, y in negative_points]

    send_boxes = prompt_form in ("boxes", "boxes+points")
    send_points = prompt_form in ("points", "boxes+points")

    with sam3_autocast(torch_module, args):
        state = processor.set_image(image)
        processor.reset_all_prompts(state)
        output = processor.set_text_prompt(state=state, prompt=prompt_text)
        if send_boxes:
            for box in negative_boxes:
                output = processor.add_geometric_prompt(box, False, state)
            if positive_box is not None:
                output = processor.add_geometric_prompt(positive_box, True, state)
        if send_points:
            appended = append_point_prompts(state, pos_norm, neg_norm, args.device)
            if appended["positive"] or appended["negative"]:
                output = processor._forward_grounding(state)
            else:
                appended = {"positive": 0, "negative": 0}

    raw = combine_sam_masks(output.get("masks", None), (image.height, image.width))
    base_info = {
        "backend": "sam3_if_supported",
        "prompt_form": prompt_form,
        "positive_boxes": int(positive_box is not None) if send_boxes else 0,
        "negative_boxes": len(negative_boxes) if send_boxes else 0,
        "positive_points_sent": int(appended["positive"]) if send_points else 0,
        "negative_points_sent": int(appended["negative"]) if send_points else 0,
        "positive_points_xy": ";".join(f"{x},{y}" for x, y in positive_points),
        "negative_points_xy": ";".join(f"{x},{y}" for x, y in negative_points),
        "point_coords_normalized": ";".join(
            f"+{x:.6f},{y:.6f}" for x, y in pos_norm
        ) + (";" if pos_norm and neg_norm else "") + ";".join(
            f"-{x:.6f},{y:.6f}" for x, y in neg_norm
        ),
        # Unambiguous mirror of the line above: leading '+'/'-' in
        # `point_coords_normalized` is only a POSITIVE/NEGATIVE label marker,
        # not a coordinate sign.  These two columns carry the raw [0,1]
        # coordinates plus the integer prompt label actually sent to SAM3.
        "point_coords_unsigned": ";".join(
            f"{x:.6f},{y:.6f}" for x, y in list(pos_norm) + list(neg_norm)
        ),
        "point_labels": ";".join(
            ["1"] * len(pos_norm) + ["0"] * len(neg_norm)
        ),
    }
    if not raw.any():
        base_info.update({"accepted": 0, "reason": "empty_refined_mask"})
        return mask, base_info

    refined = basic_cleanup(raw, args)
    refined_iou = mask_iou(mask, refined)
    if refined_iou < args.spnp_min_refined_iou:
        base_info.update(
            {"accepted": 0, "reason": "low_iou_guard", "refined_iou": refined_iou}
        )
        return mask, base_info

    base_info.update({"accepted": 1, "reason": "ok", "refined_iou": refined_iou})
    return refined.astype(bool), base_info



def build_spnp_evidence(
    base_mask: np.ndarray,
    final_mask: np.ndarray,
    geo_support: np.ndarray | None,
) -> dict[str, np.ndarray] | None:
    """Consensus-evidence-guided SPNP clicks.

    negative_region: pixels the consensus stage cut away (neighbour plant) plus
    the below-mask strip inside the mask bbox (pot / soil / table) — the two
    places blind ring sampling misses and re-inference re-covers.
    positive_core: track-backed part of the final mask; positive clicks are
    restricted to it so re-inference cannot shrink the target.
    """
    if not final_mask.any():
        return None
    negative_region = base_mask & ~final_mask
    below_strip = np.zeros_like(final_mask)
    ys_m = np.where(final_mask.any(axis=1))[0]
    xs_m = np.where(final_mask.any(axis=0))[0]
    if len(ys_m) and len(xs_m):
        y_bottom = int(ys_m.max())
        if y_bottom + 1 < final_mask.shape[0]:
            below_strip[y_bottom + 1 :, xs_m.min() : xs_m.max() + 1] = True
    negative_region = (negative_region | below_strip) & ~final_mask
    positive_core = final_mask & geo_support if geo_support is not None else None
    if not negative_region.any() and (positive_core is None or not positive_core.any()):
        return None
    return {
        "negative_region": negative_region,
        "below_strip": below_strip & ~final_mask,
        "positive_core": positive_core,
    }


def normalized_box_from_xyxy(
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    shape: tuple[int, int],
) -> list[float]:
    h, w = shape
    x0 = int(np.clip(x0, 0, max(w - 1, 0)))
    x1 = int(np.clip(x1, 0, max(w - 1, 0)))
    y0 = int(np.clip(y0, 0, max(h - 1, 0)))
    y1 = int(np.clip(y1, 0, max(h - 1, 0)))
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    bw = max(float(x1 - x0 + 1), 1.0)
    bh = max(float(y1 - y0 + 1), 1.0)
    cx = float(x0 + x1 + 1) / 2.0
    cy = float(y0 + y1 + 1) / 2.0
    return [cx / max(w, 1), cy / max(h, 1), bw / max(w, 1), bh / max(h, 1)]


def mask_to_positive_box(mask: np.ndarray, padding_ratio: float) -> list[float] | None:
    if not mask.any():
        return None
    h, w = mask.shape
    ys, xs = np.where(mask)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    pad_x = max(1, int((x1 - x0 + 1) * padding_ratio))
    pad_y = max(1, int((y1 - y0 + 1) * padding_ratio))
    return normalized_box_from_xyxy(x0 - pad_x, y0 - pad_y, x1 + pad_x, y1 + pad_y, (h, w))


def point_to_negative_box(
    point: tuple[int, int],
    shape: tuple[int, int],
    radius: int,
) -> list[float]:
    x, y = point
    radius = max(1, int(radius))
    return normalized_box_from_xyxy(x - radius, y - radius, x + radius, y + radius, shape)


def lower_negative_box(mask: np.ndarray, lower_band_ratio: float) -> list[float] | None:
    if not mask.any():
        return None
    h, _ = mask.shape
    ys, xs = np.where(mask)
    x0, x1 = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())
    lower_start = int(y_min + (y_max - y_min + 1) * lower_band_ratio)
    if lower_start >= h - 1 or lower_start >= y_max:
        return None
    return normalized_box_from_xyxy(x0, lower_start, x1, y_max, mask.shape)


def apply_spnp_refinement(
    image: Image.Image,
    mask: np.ndarray,
    image_name: str,
    prompt_text: str,
    args: argparse.Namespace,
    processor: Any = None,
    torch_module: Any = None,
    evidence: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    points: list[dict[str, Any]] = []
    prompts = build_sb_fbpr_prompts(mask, image, evidence, args)
    pos = prompts["positive"]
    neg = prompts["negative"]
    if evidence is not None:
        # Log evidence-guided clicks instead of blind ring samples.
        for x, y in pos:
            points.append({"image": image_name, "label": 1, "x": x, "y": y, "source": "track_core"})
        for x, y in neg:
            points.append({"image": image_name, "label": 0, "x": x, "y": y, "source": "consensus_cut_or_pot"})
    else:
        for x, y in pos:
            points.append({"image": image_name, "label": 1, "x": x, "y": y, "source": "distance_transform"})
        for x, y in neg:
            points.append({"image": image_name, "label": 0, "x": x, "y": y, "source": "manuscript_R_minus"})

    if not args.use_spnp_refinement:
        return mask, points, {"backend": "disabled", "accepted": 0, "reason": "module_off"}
    if args.spnp_backend == "sam3_if_supported":
        refined, info = sam3_box_refinement(
            image=image,
            mask=mask,
            prompt_text=prompt_text,
            positive_points=pos,
            negative_points=neg,
            processor=processor,
            torch_module=torch_module,
            args=args,
            evidence=evidence,
        )
        return refined, points, info
    if args.spnp_backend == "sam_family":
        raise NotImplementedError(
            f"--spnp_backend {args.spnp_backend} is not implemented in this entrypoint. "
            "Use --spnp_backend sam3_if_supported or postprocess_only for the current verified paths."
        )
    if args.spnp_backend != "postprocess_only":
        return mask, points, {"backend": args.spnp_backend, "accepted": 0, "reason": "backend_off"}

    refined = mask.copy()
    accepted = 0
    reason = "no_lower_negative"
    if args.spnp_use_lower_negative and mask.any():
        rgb = np.array(image.convert("RGB"), dtype=np.float32)
        y_idx, _ = np.where(mask)
        y_min, y_max = int(y_idx.min()), int(y_idx.max())
        lower_start = int(y_min + (y_max - y_min + 1) * args.spnp_lower_band_ratio)
        g = rgb[..., 1]
        exg = 2.0 * g - rgb[..., 0] - rgb[..., 2]
        yy = np.arange(mask.shape[0])[:, None]
        remove = mask & (yy >= lower_start) & (exg < args.spnp_lower_exg_threshold)
        max_remove = int(mask.sum() * args.spnp_max_remove_ratio)
        if 0 < remove.sum() <= max_remove:
            refined[remove] = False
            refined = binary_fill_holes(refined)
            if args.keep_largest_component:
                refined = keep_largest_component(refined)
            accepted = 1
            reason = "lower_negative_removed"
        elif remove.sum() > max_remove:
            reason = "remove_guard"
        else:
            reason = "no_pixels_removed"
    return refined.astype(bool), points, {
        "backend": "postprocess_only",
        "accepted": accepted,
        "reason": reason,
    }


def repair_residual_mask(mask: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    if not args.use_residual_repair or args.thin_repair_mode == "none":
        return mask, np.zeros_like(mask, dtype=bool)
    closed = binary_closing(mask, structure=kernel(args.closing_kernel), iterations=1)
    opened = (
        binary_opening(closed, structure=kernel(args.opening_kernel), iterations=1)
        if args.opening_kernel > 1
        else closed.copy()
    )
    residual = closed & ~opened
    labeled, count = ndimage.label(residual)
    restored = np.zeros_like(mask, dtype=bool)
    max_area = max(1, int(mask.size * args.thin_repair_max_area_ratio))
    adjacent = binary_dilation(opened, structure=kernel(5))
    for region_id in range(1, count + 1):
        region = labeled == region_id
        area = int(region.sum())
        if area <= 0 or area > max_area:
            continue
        if not np.any(region & adjacent):
            continue
        ys, xs = np.where(region)
        h = int(ys.max() - ys.min() + 1)
        w = int(xs.max() - xs.min() + 1)
        elongation = max(h / max(w, 1), w / max(h, 1))
        if args.thin_repair_mode in {"skeleton", "shape_filter"} and elongation < args.thin_repair_min_elongation:
            continue
        restored |= region
    repaired = opened | restored
    if args.fill_holes:
        repaired = binary_fill_holes(repaired)
    if args.keep_largest_component:
        repaired = keep_largest_component(repaired)
    return repaired.astype(bool), restored


def reprompt_score(
    prev_mask: np.ndarray | None,
    curr_mask: np.ndarray,
    prev_image: Image.Image | None,
    curr_image: Image.Image,
    weights: dict[str, float],
) -> dict[str, float]:
    if prev_mask is None or prev_image is None:
        return {"score": 0.0, "iou_drop": 0.0, "area_change": 0.0, "ssim_drop": 0.0, "edge_change": 0.0}
    iou_drop = 1.0 - mask_iou(prev_mask, curr_mask)
    prev_area = prev_mask.sum() / prev_mask.size
    curr_area = curr_mask.sum() / curr_mask.size
    area_change = abs(float(curr_area - prev_area)) / max(float(prev_area), 1e-6)
    prev_gray = np.array(prev_image.convert("L"))
    curr_gray = np.array(curr_image.convert("L"))
    if prev_gray.shape != curr_gray.shape:
        curr_gray = cv2.resize(curr_gray, (prev_gray.shape[1], prev_gray.shape[0]))
    try:
        ssim_value = structural_similarity(prev_gray, curr_gray)
    except ValueError:
        ssim_value = 1.0
    ssim_drop = 1.0 - float(ssim_value)
    edge_change = abs(edge_density(curr_mask) - edge_density(prev_mask)) / max(edge_density(prev_mask), 1e-6)
    denom = max(sum(weights.values()), 1e-6)
    score = (
        weights.get("iou", 0.0) * iou_drop
        + weights.get("area", 0.0) * min(area_change, 1.0)
        + weights.get("ssim", 0.0) * ssim_drop
        + weights.get("edge", 0.0) * min(edge_change, 1.0)
    ) / denom
    return {
        "score": float(score),
        "iou_drop": float(iou_drop),
        "area_change": float(area_change),
        "ssim_drop": float(ssim_drop),
        "edge_change": float(edge_change),
    }


def load_colmap_loader(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"COLMAP loader not found: {path}")
    spec = importlib.util.spec_from_file_location("rap_fsam3_colmap_loader", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load COLMAP loader from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sparse_dir_candidates(colmap_dir: Path) -> list[Path]:
    candidates = [
        colmap_dir / "sparse" / "0",
        colmap_dir / "sparse",
        colmap_dir / "distorted" / "sparse" / "0",
        colmap_dir / "distorted" / "sparse",
    ]
    seen: set[Path] = set()
    out: list[Path] = []
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if (candidate / "images.bin").exists() and (candidate / "points3D.bin").exists():
            out.append(candidate)
    return out


def find_sparse_dir(
    colmap_dir: Path,
    mask_stems: set[str] | None = None,
    loader: Any | None = None,
) -> Path:
    candidates = sparse_dir_candidates(colmap_dir)
    if mask_stems and loader is not None and candidates:
        ranked: list[tuple[int, int, Path]] = []
        for candidate in candidates:
            try:
                images = loader.read_extrinsics_binary(str(candidate / "images.bin"))
            except Exception:
                continue
            matched = 0
            for image in images.values():
                if any(stem in mask_stems for stem in colmap_mask_stem_candidates(image.name)):
                    matched += 1
            ranked.append((matched, len(images), candidate))
        if ranked:
            ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
            if ranked[0][0] > 0:
                return ranked[0][2]
    for candidate in candidates:
        return candidate
    raise FileNotFoundError(f"Could not find COLMAP sparse model under {colmap_dir}")


def colmap_mask_stem_candidates(image_name: str) -> list[str]:
    stem = Path(image_name).stem
    candidates = [stem]
    if stem.startswith("crop_"):
        candidates.append(stem[len("crop_") :])
    return candidates


def _load_colmap_from_sparse(
    sparse_dir: Path,
    mask_stems: set[str],
    loader: Any,
) -> dict[str, ColmapObservation]:
    """Load COLMAP observations from a single sparse model directory."""
    images = loader.read_extrinsics_binary(str(sparse_dir / "images.bin"))
    # Detect sample-name prefix from the per-sample directory layout, e.g.
    # .../03-final_locked/CaoMei1/sparse/0 -> prefix "CaoMei1"
    # or .../03-final_locked/CaoMei1/distorted/sparse/0 -> prefix "CaoMei1"
    sample_prefix = ""
    for ancestor in (sparse_dir.parent, sparse_dir.parent.parent,
                     sparse_dir.parent.parent.parent):
        if ancestor.name.startswith("crop_"):
            continue
        if (ancestor / "sparse").exists() or (ancestor / "distorted").exists():
            # ancestor is the sample directory
            sample_prefix = ancestor.name
            break
    observations: dict[str, ColmapObservation] = {}
    for image in images.values():
        mask_stem = ""
        for candidate_stem in colmap_mask_stem_candidates(image.name):
            if candidate_stem in mask_stems:
                mask_stem = candidate_stem
                break
            # Per-sample COLMAP: image name is "0000.jpg", stem is "CaoMei1_0000"
            if sample_prefix:
                prefixed = f"{sample_prefix}_{candidate_stem}"
                if prefixed in mask_stems:
                    mask_stem = prefixed
                    break
        if not mask_stem:
            continue
        valid = image.point3D_ids >= 0
        points = image.xys[valid]
        if len(points) == 0:
            points = np.zeros((0, 2), dtype=np.float32)
        observations[mask_stem] = ColmapObservation(
            image_name=image.name,
            mask_stem=mask_stem,
            points=np.asarray(points, dtype=np.float32),
        )
    return observations


def load_colmap_observations(
    mask_stems: set[str],
    args: argparse.Namespace,
) -> dict[str, ColmapObservation]:
    if args.colmap_dir is None:
        return {}
    loader = load_colmap_loader(args.colmap_loader_path)

    # --- single-model path (backward-compatible) ---
    try:
        sparse_dir = find_sparse_dir(args.colmap_dir, mask_stems, loader)
        return _load_colmap_from_sparse(sparse_dir, mask_stems, loader)
    except FileNotFoundError:
        pass

    # --- multi-sample fallback: scan child directories ---
    observations: dict[str, ColmapObservation] = {}
    for child in sorted(args.colmap_dir.iterdir()):
        if not child.is_dir():
            continue
        try:
            sparse_dir = find_sparse_dir(child, mask_stems, loader)
        except FileNotFoundError:
            continue
        observations.update(_load_colmap_from_sparse(sparse_dir, mask_stems, loader))
    return observations


def points_to_support_mask(points: np.ndarray, shape: tuple[int, int], dilation_size: int) -> np.ndarray:
    support = np.zeros(shape, dtype=bool)
    if points.size == 0:
        return support
    xs = np.rint(points[:, 0]).astype(int)
    ys = np.rint(points[:, 1]).astype(int)
    in_bounds = (xs >= 0) & (ys >= 0) & (xs < shape[1]) & (ys < shape[0])
    xs = xs[in_bounds]
    ys = ys[in_bounds]
    if len(xs) == 0:
        return support
    support[ys, xs] = True
    return binary_dilation(support, structure=kernel(max(3, dilation_size)), iterations=1)


def points_to_support_field(
    points: np.ndarray,
    shape: tuple[int, int],
    sigma: float,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """GCFC Gaussian support field (manuscript §2.3.5, Eq. "H_i").

        H_i(p) = sum_j omega_ij * kappa_sigma(p - pi_i(X_j))

    ``kappa_sigma`` is an isotropic Gaussian.  The field is normalised by its
    maximum so that it lives in [0, 1]; ``G_i = <H_i, M_i> / (||H_i||_1 + eps)``
    computed on the normalised field is then a support ratio in [0, 1].
    """
    field = np.zeros(shape, dtype=np.float32)
    if points.size == 0 or sigma <= 0:
        return field
    xs = np.rint(points[:, 0]).astype(int)
    ys = np.rint(points[:, 1]).astype(int)
    in_bounds = (xs >= 0) & (ys >= 0) & (xs < shape[1]) & (ys < shape[0])
    if not np.any(in_bounds):
        return field
    xs = xs[in_bounds]
    ys = ys[in_bounds]
    if weights is None:
        np.add.at(field, (ys, xs), 1.0)
    else:
        np.add.at(field, (ys, xs), weights[in_bounds].astype(np.float32))
    field = ndimage.gaussian_filter(field, sigma=float(sigma), mode="constant", cval=0.0)
    peak = float(field.max())
    if peak > 0:
        field /= peak
    return field


def boundary_band(mask: np.ndarray, radius: int) -> np.ndarray:
    """B_i = Dilate(M_i, r) \\ Erode(M_i, r), the GCFC positive-correction band."""
    base = mask.astype(bool)
    if radius <= 0:
        return np.zeros_like(base, dtype=bool)
    structure = kernel(2 * radius + 1)
    return binary_dilation(base, structure=structure) & ~ndimage.binary_erosion(base, structure=structure)


def bbox_from_mask_pixels(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    return mask_bbox(mask, padding_ratio=0.02)


def save_gcfc_debug_artifacts(
    debug_dir: Path,
    stem: str,
    before: np.ndarray,
    after: np.ndarray,
    support_field: np.ndarray | None,
    boundary: np.ndarray,
    positive: np.ndarray,
) -> None:
    """Persist the five GCFC evidence images required by the audit.

    GCFC_before / GCFC_support_field / GCFC_boundary_band / GCFC_added_pixels /
    GCFC_after.  The support field is the normalised H_i in [0, 255]; a field
    of all zeros means no real COLMAP observation reached the module.
    """
    debug_dir.mkdir(parents=True, exist_ok=True)
    save_mask(before, debug_dir / f"GCFC_before_{stem}.png")
    save_mask(after, debug_dir / f"GCFC_after_{stem}.png")
    save_mask(boundary, debug_dir / f"GCFC_boundary_band_{stem}.png")
    save_mask(positive, debug_dir / f"GCFC_added_pixels_{stem}.png")
    if support_field is None:
        field_img = np.zeros_like(before, dtype=np.uint8)
    else:
        field_img = np.clip(support_field * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(field_img).save(debug_dir / f"GCFC_support_field_{stem}.png")


def apply_corrective_geometry(
    mask: np.ndarray,
    image_name: str,
    observation: ColmapObservation | None,
    args: argparse.Namespace,
    debug_dir: Path | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    base = mask.astype(bool)
    h, w = base.shape
    lower_band = np.zeros_like(base, dtype=bool)
    lower_band[int(h * args.semantic_pot_lower_start_ratio) :, :] = True
    side_band = np.zeros_like(base, dtype=bool)
    side_width = max(1, int(w * args.semantic_side_band_ratio))
    side_band[:, :side_width] = True
    side_band[:, w - side_width :] = True
    suspicious_region = lower_band | side_band
    support_field: np.ndarray | None = None
    boundary: np.ndarray = np.zeros_like(base, dtype=bool)
    support_ratio: float | None = None
    support_gate_failed = False
    if not args.use_corrective_geometry:
        return base, {
            "图像": image_name,
            "COLMAP图像": "",
            "后端": "disabled",
            "接受": 0,
            "原因": "module_off",
            "几何修正像素比例": 0.0,
        }
    if args.corrective_geometry_backend == "mask_shape":
        unsupported = base & suspicious_region
        if not args.geometry_enable_negative_correction:
            unsupported = np.zeros_like(base, dtype=bool)
        if unsupported.sum() < int(base.size * args.geometry_negative_min_area_ratio):
            unsupported = np.zeros_like(base, dtype=bool)
        positive = np.zeros_like(base, dtype=bool)
        support = np.zeros_like(base, dtype=bool)
        colmap_image = ""
        registered = 0
        inside = 0
    else:
        if observation is None:
            return base, {
                "图像": image_name,
                "COLMAP图像": "",
                "后端": args.corrective_geometry_backend,
                "接受": 0,
                "原因": "missing_colmap_observation",
                "几何修正像素比例": 0.0,
            }
        support = points_to_support_mask(observation.points, base.shape, args.geometry_track_dilation)
        broad_support = points_to_support_mask(
            observation.points,
            base.shape,
            max(args.geometry_unsupported_dilation, args.geometry_track_dilation),
        )
        registered = int(len(observation.points))
        if registered:
            xs = np.rint(observation.points[:, 0]).astype(int)
            ys = np.rint(observation.points[:, 1]).astype(int)
            in_bounds = (xs >= 0) & (ys >= 0) & (xs < base.shape[1]) & (ys < base.shape[0])
            inside = int(base[ys[in_bounds], xs[in_bounds]].sum()) if np.any(in_bounds) else 0
        else:
            inside = 0
        support_ratio = None
        if args.corrective_geometry_backend == "colmap_tracks_gaussian":
            # ── GCFC manuscript §2.3.5 ──
            # H_i(p) = Σ ω_ij κ_σ(p − π_i(X_j));  G_i = <H_i, M_i> / (||H_i||_1 + ε)
            support_field = points_to_support_field(
                observation.points, base.shape, args.geometry_gaussian_sigma)
            field_total = float(support_field.sum())
            support_ratio = safe_ratio(float((support_field * base.astype(np.float32)).sum()), field_total)
            support = support_field >= float(args.geometry_support_threshold)
            boundary = boundary_band(base, int(args.geometry_boundary_band_px))
            positive = support & ~base & boundary
            if support_ratio < float(args.geometry_min_support_ratio):
                positive = np.zeros_like(base, dtype=bool)
                support_gate_failed = True
            else:
                support_gate_failed = False
        else:
            positive = support & ~base
        if int(positive.sum()) < args.geometry_positive_min_points:
            positive = np.zeros_like(base, dtype=bool)
        unsupported = base & ~broad_support & suspicious_region
        if not args.geometry_enable_negative_correction:
            unsupported = np.zeros_like(base, dtype=bool)
        unsupported = remove_small_components(unsupported, int(base.size * args.geometry_negative_min_area_ratio))
        colmap_image = observation.image_name

    if args.corrective_geometry_backend == "colmap_tracks_gaussian":
        # GCFC manuscript §2.3.5: negative deletion stays OFF (only positive
        # expansion inside B_i is allowed), so the unsupported region is discarded.
        unsupported = np.zeros_like(base, dtype=bool)

    corrected = (base | positive) & ~unsupported
    if args.fill_holes:
        corrected = binary_fill_holes(corrected)
    if args.keep_largest_component and corrected.any():
        corrected = keep_largest_component(corrected)

    delta = np.logical_xor(base, corrected)
    delta_ratio = safe_ratio(float(delta.sum()), float(base.size))
    corrected_iou = mask_iou(base, corrected)
    accepted = bool(delta.any())
    reason = "ok"
    if args.corrective_geometry_backend == "colmap_tracks_gaussian" and support_gate_failed:
        # Support gate is evaluated FIRST: when G_i < tau_G the positive set is
        # emptied, so the mask cannot change and the generic "no delta" reason
        # would mask the real cause.
        reason = "support_gate"
        accepted = False
    elif not delta.any():
        reason = "no_corrective_delta"
        accepted = False
    elif delta_ratio > args.geometry_correct_max_delta_ratio:
        reason = "delta_guard"
        accepted = False
    elif corrected_iou < args.geometry_correct_min_iou:
        reason = "iou_guard"
        accepted = False

    final = corrected.astype(bool) if accepted else base
    pos_box = bbox_from_mask_pixels(positive)
    neg_box = bbox_from_mask_pixels(unsupported)
    if debug_dir is not None and args.use_corrective_geometry:
        save_gcfc_debug_artifacts(
            debug_dir,
            Path(image_name).stem,
            before=base,
            after=final,
            support_field=support_field,
            boundary=boundary,
            positive=positive,
        )
    return final, {
        "图像": image_name,
        "COLMAP图像": colmap_image,
        "后端": args.corrective_geometry_backend,
        "接受": int(accepted),
        "原因": reason,
        "注册点数": registered,
        "掩膜内点数": inside,
        "支撑场比例G": -1.0 if support_ratio is None else float(support_ratio),
        "正修正点支撑像素": int(positive.sum()),
        "负修正删除像素": int(unsupported.sum()),
        "几何修正像素": int(delta.sum()),
        "几何修正像素比例": float(delta_ratio),
        "修正前后IoU": float(corrected_iou),
        "正提示框xyxy": "" if pos_box is None else ",".join(str(v) for v in pos_box),
        "负提示框xyxy": "" if neg_box is None else ",".join(str(v) for v in neg_box),
    }


def save_corrective_geometry_debug(
    image: Image.Image,
    before: np.ndarray,
    after: np.ndarray,
    path: Path,
) -> None:
    rgb = np.array(image.convert("RGB")).astype(np.float32)
    out = np.clip(rgb * 0.55 + 255.0 * 0.45, 0, 255).astype(np.uint8)
    added = ~before & after
    removed = before & ~after
    kept = before & after
    out[kept] = (out[kept].astype(np.float32) * 0.70 + np.array([0, 150, 80]) * 0.30).astype(np.uint8)
    out[added] = (out[added].astype(np.float32) * 0.25 + np.array([95, 80, 190]) * 0.75).astype(np.uint8)
    out[removed] = (out[removed].astype(np.float32) * 0.25 + np.array([220, 120, 0]) * 0.75).astype(np.uint8)
    for mask, color, thickness in [
        (before, (20, 90, 190), 2),
        (after, (0, 150, 80), 3),
    ]:
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cv2.drawContours(out, contours, -1, color, thickness)
    Image.fromarray(out).save(path)


def compute_geometry_feedback(
    final_masks: dict[str, np.ndarray],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    if not args.use_geometry_feedback or args.geometry_feedback_mode == "none":
        return []
    if args.colmap_dir is None:
        raise ValueError("--use_geometry_feedback requires --colmap_dir")
    loader = load_colmap_loader(args.colmap_loader_path)
    sparse_dir = find_sparse_dir(args.colmap_dir, set(final_masks.keys()), loader)
    images = loader.read_extrinsics_binary(str(sparse_dir / "images.bin"))
    rows: list[dict[str, Any]] = []
    for image in images.values():
        mask_stem = ""
        mask = None
        for candidate_stem in colmap_mask_stem_candidates(image.name):
            mask = final_masks.get(candidate_stem)
            if mask is not None:
                mask_stem = candidate_stem
                break
        if mask is None:
            rows.append(
                {
                    "image": image.name,
                    "mask_stem": "",
                    "registered_points": 0,
                    "inside_mask_points": 0,
                    "geometry_score": "",
                    "geometry_flag": "missing_mask",
                }
            )
            continue
        valid = image.point3D_ids >= 0
        points = image.xys[valid]
        if len(points) == 0:
            rows.append(
                {
                    "image": image.name,
                    "mask_stem": mask_stem,
                    "registered_points": 0,
                    "inside_mask_points": 0,
                    "geometry_score": "",
                    "geometry_flag": "no_tracks",
                }
            )
            continue
        xs = np.rint(points[:, 0]).astype(int)
        ys = np.rint(points[:, 1]).astype(int)
        in_bounds = (xs >= 0) & (ys >= 0) & (xs < mask.shape[1]) & (ys < mask.shape[0])
        xs = xs[in_bounds]
        ys = ys[in_bounds]
        if len(xs) == 0:
            score = 0.0
            inside = 0
        else:
            inside = int(mask[ys, xs].sum())
            score = inside / len(xs)
        rows.append(
            {
                "image": image.name,
                "mask_stem": mask_stem,
                "registered_points": int(len(points)),
                "inside_mask_points": inside,
                "geometry_score": float(score),
                "geometry_flag": "ok" if score >= args.geometry_threshold else "low",
            }
        )
    return rows


def boundary_f1(pred: np.ndarray, gt: np.ndarray, tolerance: int = 2) -> float:
    """Boundary F1 with a pixel tolerance band (never silently 1.0 on empties)."""
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    if not pred.any() and not gt.any():
        return 1.0
    if not pred.any() or not gt.any():
        return 0.0
    inner = np.ones((3, 3), dtype=bool)
    band = np.ones((2 * tolerance + 1, 2 * tolerance + 1), dtype=bool)
    pred_edge = pred & ~ndimage.binary_erosion(pred, structure=inner)
    gt_edge = gt & ~ndimage.binary_erosion(gt, structure=inner)
    pred_tol = ndimage.binary_dilation(pred_edge, structure=band)
    gt_tol = ndimage.binary_dilation(gt_edge, structure=band)
    tp = int(np.logical_and(pred_edge, gt_tol).sum())
    fp = int(np.logical_and(pred_edge, ~gt_tol).sum())
    fn = int(np.logical_and(gt_edge, ~pred_tol).sum())
    if tp == 0:
        return 0.0
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return float(2.0 * precision * recall / max(precision + recall, 1e-9))


def hd95(pred: np.ndarray, gt: np.ndarray) -> float:
    """Symmetric 95th-percentile Hausdorff distance between mask surfaces (px)."""
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    if not pred.any() or not gt.any():
        return float("nan")
    inner = np.ones((3, 3), dtype=bool)
    pred_edge = pred & ~ndimage.binary_erosion(pred, structure=inner)
    gt_edge = gt & ~ndimage.binary_erosion(gt, structure=inner)
    if not pred_edge.any():
        pred_edge = pred
    if not gt_edge.any():
        gt_edge = gt
    dt_gt = ndimage.distance_transform_edt(~gt_edge)
    dt_pred = ndimage.distance_transform_edt(~pred_edge)
    d_p2g = dt_gt[pred_edge]
    d_g2p = dt_pred[gt_edge]
    return float(max(np.percentile(d_p2g, 95), np.percentile(d_g2p, 95)))


def segmentation_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = float(np.logical_and(pred, gt).sum())
    union = float(np.logical_or(pred, gt).sum())
    f1 = (2.0 * inter / float(pred.sum() + gt.sum())) if (pred.sum() + gt.sum()) else 1.0
    iou = inter / union if union else 1.0
    return {
        "f1": float(f1),
        "iou": float(iou),
        "boundary_f1": boundary_f1(pred, gt),
        "hd95": hd95(pred, gt),
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row.keys():
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def score_to_dict(row: ScoreRecord) -> dict[str, Any]:
    return {
        "图像": row.image_name,
        "提示词编号": row.prompt_id,
        "提示词文本": row.prompt_text,
        "总分": row.total_score,
        "面积得分": row.q_area,
        "连通域得分": row.q_comp,
        "边界得分": row.q_edge,
        "时序得分": row.q_temp,
        "前背景对比得分": row.q_contrast,
        "下方泄漏得分": row.q_leak,
        "侧边泄漏得分": row.q_side,
        "语义门控启用": int(row.semantic_enabled),
        "语义门控总修正": row.semantic_total,
        "目标框得分": row.target_box_score,
        "垂直覆盖得分": row.vertical_coverage_score,
        "花盆重叠惩罚": row.pot_overlap_penalty,
        "侧边干扰惩罚": row.side_distractor_penalty,
        "中心先验得分": row.center_prior_score,
        "面积比例": row.area_ratio,
        "连通域数量": row.component_count,
        "边界密度": row.boundary_density,
        "时序IoU": row.temporal_iou,
        "前背景对比": row.contrast,
        "下方区域占比": row.bottom_leak_fraction,
        "侧边区域占比": row.side_leak_fraction,
        "SAM3原始分数": row.sam_scores,
        "排名策略": getattr(row, "ranking_policy", "base"),
        "主评分": getattr(row, "primary_score", 0.0),
        "对比度调制": getattr(row, "contrast_effective", 0.0),
        "平局决胜分": getattr(row, "tie_break_score", 0.0),
        # Phase 21 §SG-MPS diagnostics: c_i^(k) and a_i^(k)
        "门控置信度": getattr(row, "gate_confidence", 0.0),
        "门控softmax": getattr(row, "gate_softmax", 0.0),
        "phi描述子": getattr(row, "phi_descriptor", ""),
        "psi描述子": getattr(row, "psi_descriptor", ""),
    }


def candidate_to_dict(item: Candidate, image_name: str, rank: int = 0) -> dict[str, Any]:
    """逐实例候选元数据行（用于 候选评分明细.csv / 阶段证据）。"""
    box = item.box
    return {
        "图像": image_name,
        "提示词编号": item.prompt_id,
        "实例编号": item.instance_id,
        "提示词文本": item.prompt_text,
        "SAM3分数": round(float(item.sam_score), 6),
        "外接框": (f"{box[0]},{box[1]},{box[2]},{box[3]}" if box else ""),
        "面积比例": float(item.mask.sum()) / item.mask.size if item.mask.size else 0.0,
        "掩膜阈值": item.mask_threshold,
        "来源阶段": item.source_stage,
        "候选模式": item.prompt_mode,
        "排名": rank,
        "是否空掩膜": int(not item.mask.any()),
    }


def build_failure_summary(
    score_rows: list[dict[str, Any]],
    selection_rows: list[dict[str, Any]],
    reprompt_rows: list[dict[str, Any]],
    geometry_rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    empty_candidates = [
        row for row in score_rows if float(row.get("面积比例", 0.0) or 0.0) <= 0.0
    ]
    oversized_candidates = [
        row
        for row in score_rows
        if float(row.get("面积比例", 0.0) or 0.0) > float(args.area_max_ratio)
    ]
    zero_final_masks = [
        row for row in selection_rows if float(row.get("前景面积比例", 0.0) or 0.0) <= 0.0
    ]
    spnp_rejected = [
        row
        for row in selection_rows
        if row.get("SPNP后端") not in {"", "disabled", "none"}
        and str(row.get("SPNP接受", "")) not in {"1", "1.0", "True", "true"}
    ]
    reprompt_marked = [
        row
        for row in reprompt_rows
        if str(row.get("最终重提示标记", row.get("是否标记", 0))) in {"1", "1.0", "True", "true"}
    ]
    selected_stems = {Path(str(row.get("图像", ""))).stem for row in selection_rows}
    geometry_low = [row for row in geometry_rows if row.get("geometry_flag") == "low"]
    geometry_missing = [
        row
        for row in geometry_rows
        if row.get("geometry_flag") == "missing_mask"
        and any(stem in selected_stems for stem in colmap_mask_stem_candidates(str(row.get("image", ""))))
    ]
    return {
        "num_images": len(selection_rows),
        "num_prompt_candidates": len(score_rows),
        "empty_candidate_count": len(empty_candidates),
        "oversized_candidate_count": len(oversized_candidates),
        "zero_final_mask_count": len(zero_final_masks),
        "spnp_rejected_count": len(spnp_rejected),
        "reprompt_marked_count": len(reprompt_marked),
        "geometry_low_count": len(geometry_low),
        "geometry_missing_mask_count": len(geometry_missing),
        "empty_candidates": [
            {"image": row.get("图像"), "prompt": row.get("提示词编号")}
            for row in empty_candidates
        ],
        "oversized_candidates": [
            {
                "image": row.get("图像"),
                "prompt": row.get("提示词编号"),
                "area_ratio": row.get("面积比例"),
            }
            for row in oversized_candidates
        ],
        "zero_final_masks": [row.get("图像") for row in zero_final_masks],
        "spnp_rejected": [
            {
                "image": row.get("图像"),
                "backend": row.get("SPNP后端"),
                "reason": row.get("SPNP原因"),
            }
            for row in spnp_rejected
        ],
        "reprompt_marked": [
            row.get("图像") for row in reprompt_marked if row.get("图像")
        ],
        "geometry_low": [
            {
                "image": row.get("image"),
                "mask_stem": row.get("mask_stem", ""),
                "geometry_score": row.get("geometry_score"),
            }
            for row in geometry_low
        ],
    }


def main() -> int:
    args = parse_args()
    prompt_texts = load_prompt_texts(args.prompt_texts_json)
    prompt_ids = parse_prompt_ids(args.prompt_list, prompt_texts)
    if args.prompt_selection_mode == "single" and args.default_prompt_id not in prompt_ids:
        prompt_ids = [args.default_prompt_id] + [pid for pid in prompt_ids if pid != args.default_prompt_id]

    images = image_files(args.input_dir, args.limit)
    if not images:
        raise FileNotFoundError(f"No images found in {args.input_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dirs = ensure_dirs(args.output_dir)

    run_config = json_ready(vars(args))
    run_config["prompt_texts"] = prompt_texts
    run_config["prompt_ids"] = prompt_ids
    (args.output_dir / "参数.json").write_text(json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8")

    index_rows = [{"序号": i, "图像名": p.name, "图像路径": str(p)} for i, p in enumerate(images)]
    write_csv(args.output_dir / "图像索引.csv", index_rows)

    if args.dry_run:
        print(f"[dry-run] images={len(images)} prompts={prompt_ids}")
        print(f"[dry-run] output={args.output_dir}")
        return 0

    torch_module = None
    processor = None
    needs_sam3 = (not args.reuse_existing_candidates) or (
        args.use_spnp_refinement and args.spnp_backend == "sam3_if_supported"
    )
    if needs_sam3:
        torch_module, processor = load_sam3(args)

    score_weights = parse_key_values(
        args.score_weights,
        {"area": 1.0, "comp": 1.0, "edge": 1.0, "temp": 1.0, "contrast": 1.0},
    )
    reprompt_weights = parse_key_values(
        args.reprompt_weights,
        {"iou": 0.35, "area": 0.25, "ssim": 0.25, "edge": 0.15},
    )
    semantic_overrides = load_semantic_box_overrides(args.semantic_box_json)
    # ── GCFC (manuscript §2.3.5) observation loading ──
    # Both COLMAP track backends (`colmap_tracks` and `colmap_tracks_gaussian`)
    # need the real sparse observations.  Historically only `colmap_tracks` was
    # loaded, so enabling the Gaussian backend silently produced
    # `missing_colmap_observation` for every frame (defect P0-A).
    colmap_observations: dict[str, ColmapObservation] = {}
    if args.use_corrective_geometry and args.corrective_geometry_backend in (
        "colmap_tracks",
        "colmap_tracks_gaussian",
    ):
        if args.colmap_dir is None:
            raise ValueError(
                f"--corrective_geometry_backend {args.corrective_geometry_backend} "
                "requires --colmap_dir; refusing to silently fall back to a "
                "no-observation run."
            )
        colmap_observations = load_colmap_observations({p.stem for p in images}, args)
        expected = {p.stem for p in images}
        missing = sorted(expected - set(colmap_observations.keys()))
        if missing and args.strict_colmap_observations:
            raise RuntimeError(
                f"GCFC strict mode: {len(missing)}/{len(expected)} frames lack COLMAP "
                f"observations (first: {missing[:5]}). Provide a COLMAP sparse model "
                "whose image names match the mask stems, or run with GCFC disabled."
            )
        if missing:
            print(
                "[GCFC] WARNING: no COLMAP observation for "
                f"{len(missing)}/{len(expected)} frames; e.g. {missing[:5]}"
            )

    all_score_rows: list[dict[str, Any]] = []
    semantic_gate_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    point_rows: list[dict[str, Any]] = []
    reprompt_rows: list[dict[str, Any]] = []
    corrective_rows: list[dict[str, Any]] = []
    frame_report_rows: list[dict[str, Any]] = []
    sb_fbpr_prompt_rows: list[dict[str, Any]] = []
    final_masks: dict[str, np.ndarray] = {}
    prev_prompt_masks: dict[str, np.ndarray] = {}
    prev_final_mask: np.ndarray | None = None
    prev_image: Image.Image | None = None
    run_log: list[dict[str, Any]] = []
    start_time = time.time()

    consensus_result: ConsensusResult | None = None
    memory_masks: dict[str, np.ndarray] = {}
    memory_info: dict[str, Any] = {}
    consensus_summary: dict[str, Any] = {}

    # Pass 1: per-frame candidate generation, scoring and A1s selection.
    # A6/A7 are global passes, so per-frame refinement waits until pass 2.
    candidates_by_stem: dict[str, list[Candidate]] = {}
    contexts_by_stem: dict[str, SemanticGateContext | None] = {}
    selected_by_stem: dict[str, np.ndarray] = {}
    scores_by_stem: dict[str, float] = {}
    prompts_by_stem: dict[str, str] = {}
    gate_prompt_by_stem: dict[str, str] = {}
    selection_meta_by_stem: dict[str, dict[str, Any]] = {}
    reprompt_stems: set[str] = set()

    for idx, image_path in enumerate(images, start=1):
        t0 = time.time()
        image = Image.open(image_path).convert("RGB")
        if args.reuse_existing_candidates:
            candidates = load_existing_candidates(
                image_path,
                prompt_ids,
                prompt_texts,
                args.output_dir,
                args.candidate_source_dir,
                args.candidate_metadata_dir,
                args.candidate_metadata_strict,
            )
        else:
            assert processor is not None and torch_module is not None
            candidates = infer_candidates(image, image_path.name, prompt_ids, prompt_texts, processor, torch_module, args)

        if args.save_candidate_masks and not args.reuse_existing_candidates:
            for item in candidates:
                if args.candidate_mode == "per_instance":
                    # 逐实例：保存每个原始实例 + 每 prompt 合并结果（受控合并前）
                    if args.save_raw_instance_masks:
                        raw_path = dirs["candidate"] / f"raw_instance_{item.prompt_id}" / f"mask_{image_path.stem}_{item.instance_id:02d}.png"
                        save_mask(item.mask, raw_path)
                else:
                    out_path = dirs["candidate"] / prompt_dir_name(item.prompt_id) / f"mask_{image_path.stem}.png"
                    save_mask(item.mask, out_path)

        semantic_context = (
            build_semantic_gate_context(image_path, image, candidates, semantic_overrides, args)
            if args.use_semantic_gate
            else None
        )
        score_records = [
            score_candidate(
                image_path,
                image,
                item,
                prev_prompt_masks,
                score_weights,
                args,
                semantic_context=semantic_context,
            )
            for item in candidates
        ]
        # §SG-MPS a_i^(k) is computed before the score rows are written so the
        # prompt-robustness CSV records the same distribution the selector used.
        apply_semantic_gate_softmax(score_records, args)
        # ── 按策略主评分降序排名，回填 rank（逐实例）；base 策略下等价于历史 total_score 排序 ──
        ranked_indices = sorted(
            range(len(score_records)),
            key=lambda i: (
                float(getattr(score_records[i], "primary_score", score_records[i].total_score)),
                score_records[i].total_score,
            ),
            reverse=True,
        )
        rank_map = {orig_idx: rank for rank, orig_idx in enumerate(ranked_indices)}
        for row_idx in ranked_indices:
            score_records[row_idx].instance_id = candidates[row_idx].instance_id
        per_frame_detail: list[dict[str, Any]] = []
        for row_idx, rec in enumerate(score_records):
            cand = candidates[row_idx]
            row_dict = score_to_dict(rec)
            rank = rank_map.get(row_idx, 0)
            per_frame_detail.append(candidate_to_dict(cand, image_path.name, rank=rank))
            all_score_rows.append(row_dict)
            if args.use_semantic_gate:
                semantic_gate_rows.append(row_dict)
        # 保存逐帧候选评分明细 CSV
        if args.save_candidate_masks:
            detail_path = dirs["candidate"] / f"候选评分明细_{image_path.stem}.csv"
            detail_fields = [
                "图像", "提示词编号", "实例编号", "提示词文本", "SAM3分数", "外接框",
                "面积比例", "掩膜阈值", "来源阶段", "候选模式", "排名", "是否空掩膜",
            ]
            write_csv(detail_path, per_frame_detail, detail_fields)
            image_hash = file_sha256(image_path)
            metadata_entries = []
            for order, item in enumerate(candidates):
                metadata_entries.append({
                    "prompt_id": item.prompt_id, "instance_id": item.instance_id,
                    "sam_score": float(item.sam_score), "box": item.box,
                    "mask_threshold": item.mask_threshold, "candidate_order": order,
                    "scores": item.scores, "raw_detection_count": item.raw_detection_count,
                    "prompt_mode": item.prompt_mode, "metadata_status": item.metadata_status,
                    "score_precision": item.score_precision,
                    "mask_sha256": item.mask_sha256 or (
                        file_sha256(dirs["candidate"] / f"raw_instance_{item.prompt_id}" / f"mask_{image_path.stem}_{item.instance_id:02d}.png")
                        if args.candidate_mode == "per_instance" and args.save_raw_instance_masks and not args.reuse_existing_candidates
                        else ""),
                })
            metadata_path = dirs["candidate"] / f"候选元数据_{image_path.stem}.json"
            metadata_path.write_text(json.dumps({"schema_version": 1, "image_name": image_path.name,
                                               "image_sha256": image_hash, "candidates": metadata_entries},
                                              ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        selected_mask, selected_prompt, selected_score, needs_reprompt = select_mask(
            image_path, candidates, score_records, prompt_texts, args,
            selection_meta=selection_meta_by_stem.setdefault(image_path.stem, {}),
        )
        # 阶段十二 §三.4：空掩膜检测 — 记录所有候选为空的帧
        _any_nonzero = any(item.mask.any() for item in candidates)
        if not _any_nonzero:
            print(f"  ⚠ {image_path.name}：所有候选掩膜为空，score={selected_score:.4f}")
        elif not selected_mask.any():
            print(f"  ⚠ {image_path.name}：选中掩膜为空（{selected_prompt}，score={selected_score:.4f}）")
        if args.use_semantic_gate and semantic_context is not None and args.save_semantic_gate_debug:
            save_semantic_gate_debug(
                image,
                semantic_context,
                selected_mask,
                dirs["semantic_debug"] / f"semantic_gate_{image_path.stem}.png",
            )
        if args.save_intermediate_masks and not (args.use_cross_view_consensus or args.use_memory_propagation):
            save_mask(selected_mask, dirs["selected"] / f"mask_{image_path.stem}.png")

        elapsed = time.time() - t0
        gate_note = ""
        if args.use_semantic_gate and semantic_context is not None:
            gate_prompt = semantic_gate_prompt(score_records, args) or ""
            if gate_prompt:
                gate_prompt_by_stem[image_path.stem] = gate_prompt
            gate_note = (
                f" gate={getattr(args, 'semantic_gate_mode', 'heuristic')}"
                f" argmax_prompt={gate_prompt or '-'}"
                f" a_max={max((float(getattr(r, 'gate_softmax', 0.0)) for r in score_records), default=0.0):.3f}"
            )
        print(f"[{idx}/{len(images)}] {image_path.name} selected={selected_prompt} "
              f"score={selected_score:.4f}{gate_note} time={elapsed:.2f}s")

        candidates_by_stem[image_path.stem] = candidates
        contexts_by_stem[image_path.stem] = semantic_context
        selected_by_stem[image_path.stem] = selected_mask
        scores_by_stem[image_path.stem] = float(selected_score)
        prompts_by_stem[image_path.stem] = selected_prompt
        reprompt_stems.add(image_path.stem) if needs_reprompt else None
        for item in candidates:
            prev_prompt_masks[item.prompt_id] = item.mask

    stems_in_order = [p.stem for p in images]

    # A6: cross-view consensus voting — per-sample (not global).
    # Cross-view consensus only makes sense within a single multi-view scene;
    # feeding frames from different samples together produces meaningless results.
    if args.use_cross_view_consensus:
        try:
            consensus_observations = (
                colmap_observations
                if colmap_observations
                else load_colmap_observations({p.stem for p in images}, args)
            )
            gray_by_stem: dict[str, np.ndarray] = {}
            for p in images:
                img = Image.open(p).convert("L").resize((512, 910))
                gray_by_stem[p.stem] = np.array(img)

            # Group frames by sample name (stem prefix before first '_NNNN').
            def _sample_name(s: str) -> str:
                return s.rsplit("_", 1)[0] if "_" in s else s

            sample_groups: dict[str, list[str]] = {}
            for stem in selected_by_stem:
                sample_groups.setdefault(_sample_name(stem), []).append(stem)

            # Run consensus independently per sample, then merge.
            merged_masks: dict[str, np.ndarray] = {}
            merged_info: dict[str, dict] = {}
            merged_geo = None
            merged_band = None
            for sample, stems in sorted(sample_groups.items()):
                sub_selected = {s: selected_by_stem[s] for s in stems}
                sub_gray = {s: gray_by_stem[s] for s in stems if s in gray_by_stem}
                sub_obs = {s: consensus_observations[s] for s in stems if s in consensus_observations}
                sub_result = apply_cross_view_consensus(
                    sub_selected, sub_gray, sub_obs, dirs, args,
                )
                if sub_result is not None:
                    merged_masks.update(sub_result.per_frame_masks)
                    merged_info.update(sub_result.per_frame_info)
                    if sub_result.geo_support is not None:
                        if merged_geo is None:
                            merged_geo = sub_result.geo_support
                        else:
                            merged_geo = merged_geo | sub_result.geo_support
                    if sub_result.center_band_mask is not None:
                        if merged_band is None:
                            merged_band = sub_result.center_band_mask
                        else:
                            merged_band = merged_band | sub_result.center_band_mask

            if merged_masks:
                consensus_result = ConsensusResult(
                    reference_mask=next(iter(merged_masks.values()), np.zeros(1, dtype=bool)),
                    per_frame_masks=merged_masks,
                    per_frame_info=merged_info,
                    geo_support=merged_geo,
                    center_band_mask=merged_band if merged_band is not None else np.zeros(1, dtype=bool),
                )
                accepted = sum(int(v["共识接受"]) for v in merged_info.values())
                removed_total = sum(float(v["删除像素比例"]) for v in merged_info.values())
                recall_total = sum(float(v["补回像素比例"]) for v in merged_info.values())
                consensus_summary = {
                    "帧数": len(merged_masks),
                    "接受修正帧数": accepted,
                    "总删除像素比例": round(removed_total, 5),
                    "总补回像素比例": round(recall_total, 5),
                    "几何通道可用": bool(consensus_observations),
                    "样本数": len(sample_groups),
                }
                print(
                    f"[A6] per-sample: {len(sample_groups)} samples, "
                    f"frames={consensus_summary['帧数']} accepted={accepted} "
                    f"removed_ratio={consensus_summary['总删除像素比例']}"
                )
            else:
                consensus_summary = {"状态": "skipped_insufficient_frames", "最少帧数": args.consensus_min_frames}
                print("[A6] skipped: insufficient frames")
        except Exception as exc:  # noqa: BLE001 - degrade to baseline on any failure
            consensus_result = None
            consensus_summary = {
                "状态": f"unavailable: {type(exc).__name__}: {exc}",
                "帧数": 0,
            }
            print(f"[A6] FAILED: {exc}")

    # A7: memory-engine propagation — per-sample (not global).
    # Cross-sample memory propagation is meaningless; seed must come from the
    # same sample as the target frames.
    if args.use_memory_propagation:
        base_for_memory = (
            consensus_result.per_frame_masks if consensus_result is not None else selected_by_stem
        )
        seed_dir = dirs["memory"] / "_seed输入"
        seed_dir.mkdir(parents=True, exist_ok=True)
        for stem, mask in base_for_memory.items():
            save_mask(mask, seed_dir / f"mask_{stem}.png")

        # Group frames by sample name (same logic as A6 per-sample).
        def _a7_sample_name(s: str) -> str:
            return s.rsplit("_", 1)[0] if "_" in s else s

        sample_groups: dict[str, list[str]] = {}
        for stem in stems_in_order:
            sample_groups.setdefault(_a7_sample_name(stem), []).append(stem)

        merged_memory_masks: dict[str, np.ndarray] = {}
        merged_memory_info: dict[str, Any] = {}
        samples_diag: dict[str, dict[str, Any]] = {}
        memory_seed_by_stem: dict[str, str] = {}
        image_paths = {p.stem: p for p in images}
        # Load SAM3 predictor once, reuse across all samples (avoids OOM from repeated loads).
        shared_predictor = None
        try:
            shared_predictor = load_sam3_video_predictor(args)
        except Exception as exc:  # noqa: BLE001
            print(f"[A7] FAILED to load SAM3 predictor: {exc}")
        for sample, sample_stems in sorted(sample_groups.items()):
            # Select seed within this sample.
            if args.memory_seed_mode == "consensus_best" and consensus_result is not None:
                sample_consensus = {s: consensus_result.per_frame_info[s] for s in sample_stems if s in consensus_result.per_frame_info}
                if sample_consensus:
                    seed_stem = max(sample_consensus, key=lambda s: float(sample_consensus[s].get("回退IoU", 0)))
                else:
                    seed_stem = max(sample_stems, key=lambda s: scores_by_stem.get(s, 0))
            else:
                sample_scores = {s: scores_by_stem.get(s, 0) for s in sample_stems}
                seed_stem = max(sample_scores, key=sample_scores.get)
            for s in sample_stems:
                memory_seed_by_stem[s] = seed_stem
            try:
                sub_masks, sub_info = propagate_memory_masks(
                    image_paths,
                    seed_stem,
                    prompt_texts.get(args.default_prompt_id, "plant"),
                    base_for_memory.get(seed_stem),
                    sample_stems,
                    args,
                    predictor=shared_predictor,
                )
                merged_memory_masks.update(sub_masks)
                for k, v in sub_info.items():
                    merged_memory_info[k] = v
                samples_diag[sample] = {
                    "seed_stem": seed_stem,
                    "帧数": len(sample_stems),
                    "状态": sub_info.get("状态", ""),
                    "传播掩膜数": len(sub_masks),
                    "种子面积比例": sub_info.get("种子面积比例", ""),
                    "种子框比例": sub_info.get("种子框比例", ""),
                    "种子物体数": sub_info.get("种子物体数", ""),
                    "memory": sub_info.get("memory", {}),
                    "传播帧": sorted(sub_masks.keys()),
                }
            except Exception as exc:  # noqa: BLE001
                print(f"[A7] FAILED for {sample}: {exc}")
                samples_diag[sample] = {
                    "seed_stem": seed_stem,
                    "帧数": len(sample_stems),
                    "状态": f"unavailable: {type(exc).__name__}: {exc}",
                    "传播掩膜数": 0,
                }

        memory_masks = merged_memory_masks
        memory_info = merged_memory_info
        memory_info.update(
            _merge_sampled_memory_info(samples_diag, sample_groups)
        )
        # Drop per-sample-only keys that the merge loop left from the LAST sample —
        # they would otherwise leak one sample's numbers into the aggregate summary
        # (the artifact Phase 14.1 corrects). Per-sample values live in samples[].
        for _leaky in ("种子面积比例", "种子框比例", "种子框像素", "种子物体数", "帧数", "传播掩膜数", "memory"):
            memory_info.pop(_leaky, None)
        memory_info["记忆候选帧数"] = len(memory_masks)
        print(f"[A7] per-sample: {len(sample_groups)} samples, masks={len(memory_masks)}")
        (dirs["logs"] / "记忆传播.json").write_text(
            json.dumps(json_ready({"汇总": memory_info}), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # Pass 2: final variant selection {A1s, A6, A7} with the same scoring function,
    # then the existing SPNP -> residual repair -> A5c refinement chain.
    for idx, image_path in enumerate(images, start=1):
        t0 = time.time()
        image = Image.open(image_path).convert("RGB")
        stem = image_path.stem

        variants: list[Candidate] = [
            Candidate(prompt_id="A1s", prompt_text=prompts_by_stem[stem], mask=selected_by_stem[stem], scores=[], raw_detection_count=0)
        ]
        if (
            consensus_result is not None
            and stem in consensus_result.per_frame_masks
            and consensus_result.per_frame_info.get(stem, {}).get("共识接受", 0) == 1
        ):
            variants.append(
                Candidate(prompt_id="A6共识", prompt_text="cross_view_consensus", mask=consensus_result.per_frame_masks[stem], scores=[], raw_detection_count=0)
            )
        if memory_masks and stem in memory_masks:
            a7_mask = memory_masks[stem]
            img_w, img_h = image.size  # PIL: (width, height)
            if a7_mask.shape == (img_h, img_w):
                variants.append(
                    Candidate(prompt_id="A7记忆", prompt_text="memory_propagation", mask=a7_mask, scores=[], raw_detection_count=0)
                )
            else:
                print(f"[A7] SKIP {stem}: shape mismatch {a7_mask.shape} vs image ({img_h},{img_w})")
        variant_records = [
            score_candidate(
                image_path,
                image,
                item,
                {},
                score_weights,
                args,
                semantic_context=contexts_by_stem.get(stem),
            )
            for item in variants
        ]
        # Geometric leak penalty: pixels that are OFF the robust center band AND
        # lack COLMAP track support are the signature of neighbour-plant leakage.
        # Without this term the plain score favours the "fuller" A1s mask even
        # when it contains the whole neighbour. Eroded target edges pay in the
        # area/edge terms instead, so the two effects counterbalance.
        leak_penalty = 0.0
        if (
            consensus_result is not None
            and getattr(consensus_result, "geo_support", None) is not None
            and getattr(args, "consensus_variant_leak_weight", 0.0) > 0
        ):
            band = consensus_result.center_band_mask
            geo = consensus_result.geo_support
            for rec, item in zip(variant_records, variants):
                m = item.mask
                if not m.any():
                    continue
                leak_ratio = float((m & ~band & ~geo).sum()) / max(int(m.sum()), 1)
                penalty = args.consensus_variant_leak_weight * leak_ratio
                rec.total_score = rec.total_score - penalty
                # Phase 21: the leak penalty is a genuine ranking risk, so it must
                # also move the policy primary score; otherwise R1-aware variant
                # selection would ignore the consensus evidence.
                rec.primary_score = float(getattr(rec, "primary_score", rec.total_score + penalty)) - penalty
                rec.leak_penalty = penalty  # type: ignore[attr-defined]
        best_record = select_best_policy(variant_records, getattr(args, "ranking_policy", "r1"))
        best_variant = variants[variant_records.index(best_record)]
        selected_mask = best_variant.mask if best_variant.mask.any() else selected_by_stem[stem]
        selected_prompt = best_variant.prompt_id
        selected_score = float(getattr(best_record, "primary_score", best_record.total_score))
        if selected_prompt == "A7记忆":
            memory_info["selected_total"] = int(memory_info.get("selected_total", 0)) + 1
        if args.save_intermediate_masks:
            save_mask(selected_mask, dirs["selected"] / f"mask_{stem}.png")

        selected_prompt_text = prompt_texts.get(prompts_by_stem[stem], prompt_texts.get(args.default_prompt_id, "visual"))
        spnp_evidence = None
        if (
            args.use_spnp_evidence_guidance
            and consensus_result is not None
            and getattr(consensus_result, "geo_support", None) is not None
        ):
            spnp_evidence = build_spnp_evidence(
                selected_by_stem.get(stem, selected_mask),
                selected_mask,
                consensus_result.geo_support,
            )
        spnp_mask, points, spnp_info = apply_spnp_refinement(
            image,
            selected_mask,
            image_path.name,
            selected_prompt_text,
            args,
            processor=processor,
            torch_module=torch_module,
            evidence=spnp_evidence,
        )
        point_rows.extend(points)
        sb_fbpr_prompt_rows.append({
            "frame": stem,
            "image": image_path.name,
            "image_sha256": file_sha256(image_path),
            "prompt_form": str(getattr(args, "spnp_prompt_form", "boxes")),
            "negative_region_mode": str(getattr(args, "spnp_negative_region_mode", "ring")),
            "positive_points": len([p for p in points if int(p["label"]) == 1]),
            "negative_points": len([p for p in points if int(p["label"]) == 0]),
            "points_sent_positive": spnp_info.get("positive_points_sent", ""),
            "points_sent_negative": spnp_info.get("negative_points_sent", ""),
            "boxes_positive": spnp_info.get("positive_boxes", ""),
            "boxes_negative": spnp_info.get("negative_boxes", ""),
            "point_coords_normalized": spnp_info.get("point_coords_normalized", ""),
            "point_coords_unsigned": spnp_info.get("point_coords_unsigned", ""),
            "point_labels": spnp_info.get("point_labels", ""),
        })
        if args.save_intermediate_masks:
            save_mask(spnp_mask, dirs["spnp"] / f"mask_{image_path.stem}.png")

        repaired_mask, restored = repair_residual_mask(spnp_mask, args)
        if args.save_intermediate_masks:
            save_mask(repaired_mask, dirs["repair"] / f"mask_{image_path.stem}.png")
            save_mask(restored, dirs["repair"] / f"thin_{image_path.stem}.png")

        final_mask, corrective_info = apply_corrective_geometry(
            repaired_mask,
            image_path.name,
            colmap_observations.get(image_path.stem),
            args,
            debug_dir=(dirs["geometry_debug"] if args.use_corrective_geometry else None),
        )
        if args.use_corrective_geometry:
            corrective_rows.append(corrective_info)
        if args.save_intermediate_masks:
            save_mask(final_mask, dirs["corrective"] / f"mask_{image_path.stem}.png")
            save_mask(final_mask, dirs["a5c"] / f"mask_{image_path.stem}.png")
        if args.use_corrective_geometry and args.save_corrective_geometry_debug:
            save_corrective_geometry_debug(
                image,
                repaired_mask,
                final_mask,
                dirs["geometry_debug"] / f"geometry_delta_{image_path.stem}.png",
            )

        save_mask(final_mask, dirs["final"] / f"mask_{image_path.stem}.png")
        # ── Per-frame full-chain evidence row (audit deliverable) ──
        has_observation = image_path.stem in colmap_observations
        meta = selection_meta_by_stem.get(stem, {})
        sel_prompt = str(meta.get("selected_prompt") or prompts_by_stem.get(stem, ""))
        sel_instance = str(meta.get("selected_instance", ""))
        frame_metrics: dict[str, Any] = {"f1": "", "iou": "", "boundary_f1": "", "hd95": ""}
        gt_path = (args.gt_mask_dir / f"mask_{image_path.stem}.png") if args.gt_mask_dir else None
        if gt_path is not None and gt_path.exists():
            gt_mask = load_mask(gt_path)
            if gt_mask.shape == final_mask.shape:
                frame_metrics = segmentation_metrics(final_mask, gt_mask)
        failure_reason = ""
        if not has_observation and args.use_corrective_geometry and args.corrective_geometry_backend != "mask_shape":
            failure_reason = "gcfc_missing_colmap_observation"
        elif not str(corrective_info.get("原因", "")) in ("ok", "module_off"):
            failure_reason = f"gcfc_{corrective_info.get('原因')}"
        elif not str(spnp_info.get("reason", "")) in ("ok", "module_off"):
            failure_reason = f"sb_fbpr_{spnp_info.get('reason')}"
        frame_report_rows.append({
            "frame_id": stem,
            "input_image_sha256": file_sha256(image_path),
            "sam3_inference_ran": int(not args.reuse_existing_candidates),
            "sam3_candidates": len(candidates_by_stem.get(stem, [])),
            "sam3_prompts": len(prompt_ids),
            "selected_prompt": sel_prompt,
            "selected_instance": sel_instance,
            "selection_path": meta.get("selection_path", ""),
            "variant_selected": selected_prompt,
            "sg_mps_executed": int(
                args.use_semantic_gate
                and args.prompt_selection_mode == "score_select"
                and getattr(args, "semantic_gate_mode", "heuristic") == "gated_softmax"
            ),
            "sg_mps_mode": getattr(args, "semantic_gate_mode", ""),
            "sg_mps_prompt": gate_prompt_by_stem.get(stem, ""),
            "sg_mps_softmax": json.dumps(meta.get("sg_mps_softmax", {}), ensure_ascii=False),
            "sg_mps_gate_confidence": json.dumps(meta.get("sg_mps_candidates", {}), ensure_ascii=False),
            # CDR/R1 is a ranking-policy fact and applies to whatever candidate
            # set exists — including the instances of a SINGLE prompt.  It must
            # not be reported as "not executed" merely because the run is not a
            # multi-prompt ensemble; `cdr_scope` records the candidate-set scope.
            "cdr_executed": int(getattr(args, "ranking_policy", "r1") in ("r1", "r2", "r3")),
            "cdr_scope": (
                "multi_prompt"
                if (args.use_prompt_ensemble and args.prompt_selection_mode == "score_select")
                else "single_prompt"
            ),
            "cdr_policy": getattr(args, "ranking_policy", ""),
            "sb_fbpr_executed": int(bool(args.use_spnp_refinement)),
            "sb_fbpr_prompt_form": str(getattr(args, "spnp_prompt_form", "boxes")),
            "sb_fbpr_accepted": spnp_info.get("accepted", ""),
            "sb_fbpr_reason": spnp_info.get("reason", ""),
            "sb_fbpr_pos_points_sent": spnp_info.get("positive_points_sent", 0),
            "sb_fbpr_neg_points_sent": spnp_info.get("negative_points_sent", 0),
            "residual_repair_executed": int(
                bool(args.use_residual_repair) and args.thin_repair_mode != "none"
            ),
            "residual_repair_mode": args.thin_repair_mode,
            "residual_restored_px": int(restored.sum()),
            "gcfc_executed": int(bool(args.use_corrective_geometry)),
            "gcfc_backend": args.corrective_geometry_backend if args.use_corrective_geometry else "",
            "gcfc_has_real_observation": int(bool(has_observation)),
            "gcfc_registered_points": corrective_info.get("注册点数", ""),
            "gcfc_support_G": corrective_info.get("支撑场比例G", ""),
            "gcfc_added_px": corrective_info.get("正修正点支撑像素", ""),
            "gcfc_accepted": corrective_info.get("接受", ""),
            "gcfc_reason": corrective_info.get("原因", ""),
            "final_mask_sha256": file_sha256(dirs["final"] / f"mask_{image_path.stem}.png"),
            "f1_if_gt_available": frame_metrics["f1"],
            "iou_if_gt_available": frame_metrics["iou"],
            "boundary_f1_if_gt_available": frame_metrics["boundary_f1"],
            "hd95_if_gt_available": frame_metrics["hd95"],
            "failure_reason": failure_reason,
        })
        if args.save_foreground_rgb:
            save_foreground_and_rgba(
                image,
                final_mask,
                dirs["foreground"] / f"crop_{image_path.stem}.png",
                dirs["rgba"] / f"rgba_{image_path.stem}.png",
            )
        elif args.save_rgba:
            rgb = np.array(image.convert("RGB"))
            rgba = np.dstack([rgb, (final_mask.astype(np.uint8) * 255)])
            Image.fromarray(rgba).save(dirs["rgba"] / f"rgba_{image_path.stem}.png")
        if args.save_overlay:
            save_overlay(image, final_mask, dirs["overlay"] / f"overlay_{image_path.stem}.png")

        reprompt = reprompt_score(prev_final_mask, final_mask, prev_image, image, reprompt_weights)
        reprompt_flag = bool(args.use_reprompt_detection and reprompt["score"] > args.reprompt_threshold)
        reprompt_rows.append(
            {
                "图像": image_path.name,
                "重提示分数": reprompt["score"],
                "IoU下降": reprompt["iou_drop"],
                "面积变化": reprompt["area_change"],
                "SSIM下降": reprompt["ssim_drop"],
                "边界变化": reprompt["edge_change"],
                "是否标记": int(reprompt_flag),
            }
        )

        final_masks[stem] = final_mask
        prev_final_mask = final_mask
        prev_image = image

        consensus_info = (
            consensus_result.per_frame_info.get(stem, {}) if consensus_result is not None else {}
        )
        selection_rows.append(
            {
                "图像": image_path.name,
                "选择提示词": selected_prompt,
                "选择分数": selected_score,
                "A1s基础分数": scores_by_stem.get(stem, ""),
                "前景面积比例": final_mask.sum() / final_mask.size,
                "恢复细结构像素": int(restored.sum()),
                "共识启用": consensus_info.get("共识启用", 0),
                "共识接受": consensus_info.get("共识接受", ""),
                "共识回退IoU": consensus_info.get("回退IoU", ""),
                "共识删除像素比例": consensus_info.get("删除像素比例", ""),
                "共识补回像素比例": consensus_info.get("补回像素比例", ""),
                "记忆后端": memory_info.get("记忆后端", "") if args.use_memory_propagation else "",
                "记忆种子帧": memory_seed_by_stem.get(stem, "") if args.use_memory_propagation else "",
                "记忆候选采用": int(selected_prompt == "A7记忆") if args.use_memory_propagation else "",
                "重提示标记": int(reprompt_flag),
                "SPNP后端": spnp_info.get("backend", ""),
                "SPNP接受": spnp_info.get("accepted", ""),
                "SPNP原因": spnp_info.get("reason", ""),
                "SPNP正框数": spnp_info.get("positive_boxes", ""),
                "SPNP负框数": spnp_info.get("negative_boxes", ""),
                "SPNP细化IoU": spnp_info.get("refined_iou", ""),
            }
        )
        elapsed = time.time() - t0
        run_log.append(
            {
                "image": image_path.name,
                "index": idx,
                "total": len(images),
                "selected_prompt": selected_prompt,
                "consensus_accepted": consensus_info.get("共识接受", ""),
                "memory_state": memory_info.get("状态", "") if args.use_memory_propagation else "",
                "spnp_backend": spnp_info.get("backend", ""),
                "spnp_accepted": spnp_info.get("accepted", ""),
                "a5c_accepted": corrective_info.get("接受", ""),
                "elapsed_sec": round(elapsed, 3),
            }
        )
        print(
            f"[{idx}/{len(images)}] {image_path.name} "
            f"selected={selected_prompt} score={selected_score:.4f} "
            f"area={final_mask.sum() / final_mask.size:.4f} "
            f"a5c={corrective_info.get('接受', '')} time={elapsed:.2f}s"
        )

    geometry_rows = compute_geometry_feedback(final_masks, args)
    geometry_by_stem = {Path(str(row.get("image", ""))).stem: row for row in geometry_rows}
    for row in geometry_rows:
        mask_stem = row.get("mask_stem")
        if mask_stem:
            geometry_by_stem[str(mask_stem)] = row
    if geometry_rows:
        for row in selection_rows:
            stem = Path(str(row["图像"])).stem
            geo = geometry_by_stem.get(stem)
            if geo:
                row["几何分数"] = geo.get("geometry_score", "")
                row["几何标记"] = geo.get("geometry_flag", "")
                if geo.get("geometry_flag") == "low":
                    row["重提示标记"] = 1
        for row in reprompt_rows:
            stem = Path(str(row["图像"])).stem
            geo = geometry_by_stem.get(stem)
            row["几何分数"] = geo.get("geometry_score", "") if geo else ""
            row["几何标记"] = geo.get("geometry_flag", "") if geo else ""
            row["最终重提示标记"] = int(
                bool(row.get("是否标记", 0)) or (geo is not None and geo.get("geometry_flag") == "low")
            )

    write_csv(args.output_dir / "提示词评分.csv", all_score_rows)
    if semantic_gate_rows:
        write_csv(args.output_dir / "语义门控评分.csv", semantic_gate_rows)
    write_csv(args.output_dir / "提示词选择.csv", selection_rows)
    if consensus_result is not None:
        write_csv(
            args.output_dir / "共识投票.csv",
            [consensus_result.per_frame_info[stem] for stem in stems_in_order if stem in consensus_result.per_frame_info],
        )
        (dirs["logs"] / "共识汇总.json").write_text(
            json.dumps(json_ready(consensus_summary), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    if memory_info:
        (dirs["logs"] / "记忆传播.json").write_text(
            json.dumps(json_ready({"汇总": memory_info}), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    write_csv(args.output_dir / "正负提示点.csv", point_rows, ["image", "label", "x", "y", "source"])
    if sb_fbpr_prompt_rows:
        write_csv(
            args.output_dir / "SB-FBPR提示证据.csv",
            sb_fbpr_prompt_rows,
            [
                "frame", "image", "image_sha256", "prompt_form", "negative_region_mode",
                "positive_points", "negative_points", "points_sent_positive",
                "points_sent_negative", "boxes_positive", "boxes_negative",
                "point_coords_normalized", "point_coords_unsigned", "point_labels",
            ],
        )
    if frame_report_rows:
        write_csv(args.output_dir / "逐帧报告.csv", frame_report_rows)
    write_csv(args.output_dir / "重提示帧标记.csv", reprompt_rows)
    if corrective_rows:
        write_csv(args.output_dir / "几何修正提示.csv", corrective_rows)
        write_csv(args.output_dir / "corrective_geometry_delta.csv", corrective_rows)
    if geometry_rows:
        write_csv(args.output_dir / "几何反馈.csv", geometry_rows)

    failure_summary = build_failure_summary(
        all_score_rows,
        selection_rows,
        reprompt_rows,
        geometry_rows,
        args,
    )
    (args.output_dir / "失败汇总.json").write_text(
        json.dumps(json_ready(failure_summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if args.log_json:
        payload = {
            "status": "success",
            "num_images": len(images),
            "elapsed_sec": round(time.time() - start_time, 3),
            "run_log": run_log,
            "failure_summary": failure_summary,
        }
        (args.output_dir / "运行日志.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Done. Output saved to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
