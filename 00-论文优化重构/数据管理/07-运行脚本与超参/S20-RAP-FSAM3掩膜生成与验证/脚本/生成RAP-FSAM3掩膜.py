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
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """A7: text+box seeding on the best frame + bidirectional memory propagation.

    Frames are re-materialized as a numbered temp directory because SAM3 video
    sessions load frames from a folder of "<frame_index>.<ext>" files. Seeding
    uses the text prompt PLUS a normalized xywh box derived from the seed
    frame's consensus mask: the box anchors the CENTRAL target so the memory
    engine does not also lock onto the neighbouring plant (which matches the
    same text prompt).
    """
    import shutil
    import tempfile

    import torch

    info: dict[str, Any] = {
        "记忆后端": "sam3_video",
        "种子帧": "",
        "传播方向": "both" if args.memory_bidirectional else "forward",
        "状态": "",
    }
    tmp_dir: Path | None = None
    try:
        predictor = load_sam3_video_predictor(args)
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
        response = predictor.handle_request(request=dict(type="start_session", resource_path=str(tmp_dir)))
        session_id = response["session_id"]
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
        response = predictor.handle_request(request=seed_request)
        out = response.get("outputs", {})
        n_seed_obj = len(out.get("out_obj_ids", []))
        if n_seed_obj == 0:
            info["状态"] = "seed_empty"
            predictor.handle_request(request=dict(type="close_session", session_id=session_id))
            return {}, info
        info["种子帧"] = seed_stem

        masks: dict[str, np.ndarray] = {}
        request = dict(type="propagate_in_video", session_id=session_id, start_frame_index=seed_idx)
        request["propagation_direction"] = "both" if args.memory_bidirectional else "forward"
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
        predictor.handle_request(request=dict(type="close_session", session_id=session_id))
        info["状态"] = "ok"
        return masks, info
    except torch.cuda.OutOfMemoryError:
        info["状态"] = "cuda_oom_fallback"
        return {}, info
    except Exception as exc:  # noqa: BLE001 - degrade to per-frame mode on any failure
        info["状态"] = f"unavailable: {type(exc).__name__}: {exc}"
        return {}, info
    finally:
        if tmp_dir is not None:
            shutil.rmtree(tmp_dir, ignore_errors=True)


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
        choices=["colmap_tracks", "mask_shape"],
        default="colmap_tracks",
    )
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
                            sam_score=0.0, mask_threshold=0.5, source_stage="raw",
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
                            mask_threshold=0.5,
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
) -> list[Candidate]:
    candidates = []
    source_dir = candidate_source_dir if candidate_source_dir is not None else output_dir
    for prompt_id in prompt_ids:
        # 优先复用 per_instance 原始实例（raw_instance_<prompt>_<i>.png）
        raw_dir = source_dir / "候选掩膜" / f"raw_instance_{prompt_id}"
        raw_files = sorted(raw_dir.glob(f"mask_{image_path.stem}_*.png")) if raw_dir.exists() else []
        if raw_files:
            for i, rp in enumerate(raw_files):
                inst = 0
                try:
                    suffix = rp.stem.split("_")[-1]
                    inst = int(suffix)
                except (ValueError, IndexError):
                    inst = i
                candidates.append(
                    Candidate(
                        prompt_id=prompt_id,
                        prompt_text=prompt_texts[prompt_id],
                        mask=load_mask(rp),
                        scores=[],
                        raw_detection_count=len(raw_files),
                        instance_id=inst,
                        box=None,
                        sam_score=0.0,
                        mask_threshold=0.5,
                        source_stage="reused",
                        prompt_mode="per_instance",
                    )
                )
            continue
        mask_path = source_dir / "候选掩膜" / prompt_dir_name(prompt_id) / f"mask_{image_path.stem}.png"
        if not mask_path.exists():
            raise FileNotFoundError(f"Existing candidate mask not found: {mask_path}")
        candidates.append(
            Candidate(
                prompt_id=prompt_id,
                prompt_text=prompt_texts[prompt_id],
                mask=load_mask(mask_path),
                scores=[],
                raw_detection_count=0,
                instance_id=0,
                box=None,
                sam_score=0.0,
                mask_threshold=0.5,
                source_stage="reused",
                prompt_mode="legacy_union",
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
    if target_box:
        target_mask = box_to_mask(target_box, shape)
    else:
        target_mask = union | green
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


def semantic_gate_scores(mask: np.ndarray, context: SemanticGateContext, args: argparse.Namespace) -> dict[str, float]:
    area = float(mask.sum())
    if area <= 0:
        return {
            "semantic_total": 0.0,
            "target_box_score": 0.0,
            "vertical_coverage_score": 0.0,
            "pot_overlap_penalty": 1.0,
            "side_distractor_penalty": 1.0,
            "center_prior_score": 0.0,
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
    }


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
    }
    if args.use_semantic_gate and semantic_context is not None:
        semantic = semantic_gate_scores(mask, semantic_context, args)
    total = base_total + semantic["semantic_total"] - risk_penalty
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
    )


def select_mask(
    image_path: Path,
    candidates: list[Candidate],
    score_records: list[ScoreRecord],
    prompt_texts: dict[str, str],
    args: argparse.Namespace,
) -> tuple[np.ndarray, str, float]:
    if args.use_prompt_ensemble and args.prompt_selection_mode == "score_select":
        # 阶段十二 §三.1：score_select 按 (prompt_id, instance_id) 选最佳实例，不只按 prompt_id
        non_empty = {item.prompt_id for item in candidates if item.mask.any()}
        selectable = [row for row in score_records if row.prompt_id in non_empty]
        if not selectable:
            selectable = score_records
        best = max(selectable, key=lambda row: row.total_score)
        # 找到与 best 匹配的候选（同 prompt_id + instance_id）
        best_cand = next(
            (item for item in candidates
             if item.prompt_id == best.prompt_id and item.instance_id == best.instance_id),
            next((item for item in candidates if item.prompt_id == best.prompt_id), None),
        )
        if best_cand is None:
            raise ValueError(f"No candidate found for {best.prompt_id}#{best.instance_id}")
        return best_cand.mask.copy(), best.prompt_id, best.total_score, False
    if args.use_prompt_ensemble and args.prompt_selection_mode == "weighted_fusion":
        score_by_prompt = {row.prompt_id: max(row.total_score, 0.0) for row in score_records}
        score_sum = sum(score_by_prompt.values())
        if score_sum <= 0:
            weights = {pid: 1.0 / len(candidates) for pid in score_by_prompt}
        else:
            weights = {pid: score / score_sum for pid, score in score_by_prompt.items()}
        fused = np.zeros_like(candidates[0].mask, dtype=np.float32)
        for item in candidates:
            fused += weights.get(item.prompt_id, 0.0) * item.mask.astype(np.float32)
        return fused >= args.fusion_threshold, "weighted_fusion", float(max(score_by_prompt.values(), default=0.0)), False

    prompt_id = args.default_prompt_id
    if prompt_id not in prompt_texts:
        raise ValueError(f"Unknown --default_prompt_id: {prompt_id}")
    default_cands = [item for item in candidates if item.prompt_id == prompt_id]
    if not default_cands:
        raise ValueError(f"Default prompt {prompt_id} was not generated for {image_path.name}")
    # per_instance 模式：default_prompt 下有多个实例候选，按评分选最佳实例
    if len(default_cands) > 1 and args.candidate_mode == "per_instance":
        rec_by_id = {(r.prompt_id, getattr(r, "instance_id", 0)): r for r in score_records}
        best_item = None
        best_score = -1e9
        for item in default_cands:
            r = rec_by_id.get((item.prompt_id, item.instance_id))
            sc = float(r.total_score) if r is not None else 0.0
            if sc > best_score:
                best_score = sc
                best_item = item
        # §4.7 重提示触发：top1 与 top2 分差 < 阈值 → 候选不确定，需重提示
        sorted_recs = sorted(score_records, key=lambda r: r.total_score, reverse=True)
        needs_reprompt = len(sorted_recs) >= 2 and (sorted_recs[0].total_score - sorted_recs[1].total_score) < args.reprompt_score_gap
        return best_item.mask.copy(), f"{prompt_id}#{best_item.instance_id}", float(best_score), bool(needs_reprompt)
    row = next((r for r in score_records if r.prompt_id == prompt_id), None)
    # 单一候选：空掩膜或低分 → 触发重提示
    needs_reprompt = (row is None) or (row.total_score < args.reprompt_min_score) or bool(getattr(row, "empty_flag", False))
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


def sample_points_from_region(region: np.ndarray, count: int) -> list[tuple[int, int]]:
    """Evidence-guided negative clicks: sample well inside the given region
    (far from its boundary via the distance transform), spread apart."""
    if count <= 0 or not region.any():
        return []
    dist = ndimage.distance_transform_edt(region)
    ys, xs = np.where(dist > 0)
    if len(xs) == 0:
        return []
    order = np.argsort(dist[ys, xs])[::-1]
    chosen: list[tuple[int, int]] = []
    blocked = np.zeros(region.shape, dtype=bool)
    min_sep = max(4, int(min(region.shape) * 0.04))
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


def sam3_box_refinement(
    image: Image.Image,
    mask: np.ndarray,
    prompt_text: str,
    negative_points: list[tuple[int, int]],
    processor: Any,
    torch_module: Any,
    args: argparse.Namespace,
    evidence: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    if processor is None or torch_module is None:
        raise RuntimeError("--spnp_backend sam3_if_supported requires a loaded SAM3 processor")

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
        neg_region = evidence["negative_region"]
        neg_points: list[tuple[int, int]] = sample_points_from_region(neg_region, args.spnp_negative_points)
        if evidence["below_strip"].any() and neg_points:
            # Guarantee at least two clicks on the pot strip when it exists.
            pot_points = sample_points_from_region(evidence["below_strip"], 2)
            neg_points = pot_points + neg_points
        negative_boxes = [
            point_to_negative_box(point, mask.shape, args.spnp_negative_box_radius)
            for point in neg_points[: max(args.spnp_negative_points, 0) + 2]
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

    with sam3_autocast(torch_module, args):
        state = processor.set_image(image)
        processor.reset_all_prompts(state)
        output = processor.set_text_prompt(state=state, prompt=prompt_text)
        if positive_box is not None:
            output = processor.add_geometric_prompt(positive_box, True, state)
        for box in negative_boxes:
            output = processor.add_geometric_prompt(box, False, state)

    raw = combine_sam_masks(output.get("masks", None), (image.height, image.width))
    if not raw.any():
        return mask, {
            "backend": "sam3_if_supported",
            "positive_boxes": int(positive_box is not None),
            "negative_boxes": len(negative_boxes),
            "accepted": 0,
            "reason": "empty_refined_mask",
        }

    refined = basic_cleanup(raw, args)
    refined_iou = mask_iou(mask, refined)
    if refined_iou < args.spnp_min_refined_iou:
        return mask, {
            "backend": "sam3_if_supported",
            "positive_boxes": int(positive_box is not None),
            "negative_boxes": len(negative_boxes),
            "accepted": 0,
            "reason": "low_iou_guard",
            "refined_iou": refined_iou,
        }

    return refined.astype(bool), {
        "backend": "sam3_if_supported",
        "positive_boxes": int(positive_box is not None),
        "negative_boxes": len(negative_boxes),
        "accepted": 1,
        "reason": "ok",
        "refined_iou": refined_iou,
    }


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
    if evidence is not None:
        # Log evidence-guided clicks instead of blind ring samples.
        core = evidence["positive_core"]
        pos = sample_points_from_region(core, args.spnp_positive_points) if core is not None and core.any() else []
        neg = sample_points_from_region(evidence["negative_region"], args.spnp_negative_points)
        for x, y in pos:
            points.append({"image": image_name, "label": 1, "x": x, "y": y, "source": "track_core"})
        for x, y in neg:
            points.append({"image": image_name, "label": 0, "x": x, "y": y, "source": "consensus_cut_or_pot"})
    else:
        pos = sample_points(mask, args.spnp_positive_points, True, args.spnp_ring_radius)
        neg = sample_points(mask, args.spnp_negative_points, False, args.spnp_ring_radius)
        for x, y in pos:
            points.append({"image": image_name, "label": 1, "x": x, "y": y, "source": "distance_transform"})
        for x, y in neg:
            points.append({"image": image_name, "label": 0, "x": x, "y": y, "source": "outer_ring"})

    if not args.use_spnp_refinement:
        return mask, points, {"backend": "disabled", "accepted": 0, "reason": "module_off"}
    if args.spnp_backend == "sam3_if_supported":
        refined, info = sam3_box_refinement(
            image=image,
            mask=mask,
            prompt_text=prompt_text,
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


def load_colmap_observations(
    mask_stems: set[str],
    args: argparse.Namespace,
) -> dict[str, ColmapObservation]:
    if args.colmap_dir is None:
        return {}
    loader = load_colmap_loader(args.colmap_loader_path)
    sparse_dir = find_sparse_dir(args.colmap_dir, mask_stems, loader)
    images = loader.read_extrinsics_binary(str(sparse_dir / "images.bin"))
    observations: dict[str, ColmapObservation] = {}
    for image in images.values():
        mask_stem = ""
        for candidate_stem in colmap_mask_stem_candidates(image.name):
            if candidate_stem in mask_stems:
                mask_stem = candidate_stem
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


def bbox_from_mask_pixels(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    return mask_bbox(mask, padding_ratio=0.02)


def apply_corrective_geometry(
    mask: np.ndarray,
    image_name: str,
    observation: ColmapObservation | None,
    args: argparse.Namespace,
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
        positive = support & ~base
        if int(positive.sum()) < args.geometry_positive_min_points:
            positive = np.zeros_like(base, dtype=bool)
        unsupported = base & ~broad_support & suspicious_region
        if not args.geometry_enable_negative_correction:
            unsupported = np.zeros_like(base, dtype=bool)
        unsupported = remove_small_components(unsupported, int(base.size * args.geometry_negative_min_area_ratio))
        colmap_image = observation.image_name

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
    if not delta.any():
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
    return final, {
        "图像": image_name,
        "COLMAP图像": colmap_image,
        "后端": args.corrective_geometry_backend,
        "接受": int(accepted),
        "原因": reason,
        "注册点数": registered,
        "掩膜内点数": inside,
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
    colmap_observations = (
        load_colmap_observations({p.stem for p in images}, args)
        if args.use_corrective_geometry and args.corrective_geometry_backend == "colmap_tracks"
        else {}
    )

    all_score_rows: list[dict[str, Any]] = []
    semantic_gate_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    point_rows: list[dict[str, Any]] = []
    reprompt_rows: list[dict[str, Any]] = []
    corrective_rows: list[dict[str, Any]] = []
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
        # ── 按 total_score 降序排名，回填 rank（逐实例） ──
        ranked_indices = sorted(range(len(score_records)), key=lambda i: score_records[i].total_score, reverse=True)
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
        selected_mask, selected_prompt, selected_score, needs_reprompt = select_mask(
            image_path, candidates, score_records, prompt_texts, args
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
        print(f"[{idx}/{len(images)}] {image_path.name} selected={selected_prompt} score={selected_score:.4f} time={elapsed:.2f}s")

        candidates_by_stem[image_path.stem] = candidates
        contexts_by_stem[image_path.stem] = semantic_context
        selected_by_stem[image_path.stem] = selected_mask
        scores_by_stem[image_path.stem] = float(selected_score)
        prompts_by_stem[image_path.stem] = selected_prompt
        reprompt_stems.add(image_path.stem) if needs_reprompt else None
        for item in candidates:
            prev_prompt_masks[item.prompt_id] = item.mask

    stems_in_order = [p.stem for p in images]

    # A6: cross-view consensus voting (global).
    if args.use_cross_view_consensus:
        try:
            consensus_observations = (
                colmap_observations
                if args.use_corrective_geometry and args.corrective_geometry_backend == "colmap_tracks"
                else load_colmap_observations({p.stem for p in images}, args)
            )
            gray_by_stem: dict[str, np.ndarray] = {}
            for p in images:
                img = Image.open(p).convert("L").resize((512, 910))
                gray_by_stem[p.stem] = np.array(img)
            # Upsample the 512-wide grayscale back to mask resolution inside the estimator.
            consensus_result = apply_cross_view_consensus(
                selected_by_stem,
                gray_by_stem,
                consensus_observations,
                dirs,
                args,
            )
            if consensus_result is not None:
                accepted = sum(int(v["共识接受"]) for v in consensus_result.per_frame_info.values())
                removed_total = sum(float(v["删除像素比例"]) for v in consensus_result.per_frame_info.values())
                recall_total = sum(float(v["补回像素比例"]) for v in consensus_result.per_frame_info.values())
                consensus_summary = {
                    "帧数": len(consensus_result.per_frame_masks),
                    "接受修正帧数": accepted,
                    "总删除像素比例": round(removed_total, 5),
                    "总补回像素比例": round(recall_total, 5),
                    "几何通道可用": bool(consensus_observations),
                }
                print(
                    f"[A6] frames={consensus_summary['帧数']} accepted={accepted} "
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

    # A7: memory-engine propagation seeded from the safest frame (never the raw first frame).
    if args.use_memory_propagation:
        base_for_memory = (
            consensus_result.per_frame_masks if consensus_result is not None else selected_by_stem
        )
        seed_dir = dirs["memory"] / "_seed输入"
        seed_dir.mkdir(parents=True, exist_ok=True)
        for stem, mask in base_for_memory.items():
            save_mask(mask, seed_dir / f"mask_{stem}.png")
        if args.memory_seed_mode == "consensus_best" and consensus_result is not None:
            seed_stem = max(
                consensus_result.per_frame_info,
                key=lambda s: float(consensus_result.per_frame_info[s]["回退IoU"]),
            )
        else:
            seed_stem = max(scores_by_stem, key=scores_by_stem.get)
        image_paths = {p.stem: p for p in images}
        try:
            memory_masks, memory_info = propagate_memory_masks(
                image_paths,
                seed_stem,
                prompt_texts.get(args.default_prompt_id, "plant"),
                base_for_memory.get(seed_stem),
                stems_in_order,
                args,
            )
        except Exception as exc:  # noqa: BLE001 - degrade to per-frame mode on any failure
            memory_masks, memory_info = {}, {
                "记忆后端": "sam3_video",
                "种子帧": seed_stem,
                "状态": f"unavailable: {type(exc).__name__}: {exc}",
            }
        memory_info.setdefault("种子帧", seed_stem)
        memory_info["记忆候选帧数"] = len(memory_masks)
        print(f"[A7] backend={memory_info.get('记忆后端')} state={memory_info.get('状态')} masks={len(memory_masks)}")
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
        if consensus_result is not None and stem in consensus_result.per_frame_masks:
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
                rec.leak_penalty = penalty  # type: ignore[attr-defined]
        best_record = max(variant_records, key=lambda r: r.total_score)
        best_variant = variants[variant_records.index(best_record)]
        selected_mask = best_variant.mask if best_variant.mask.any() else selected_by_stem[stem]
        selected_prompt = best_variant.prompt_id
        selected_score = best_record.total_score
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
                "记忆种子帧": memory_info.get("种子帧", "") if args.use_memory_propagation else "",
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
