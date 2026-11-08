"""Full-sequence per-plant processing: VGGT-Omega inference -> fusion -> GGGS dataset.

Design rules enforced here:

* **No frame sampling**: every valid paired frame of a plant goes into *one*
  joint VGGT-Omega forward pass.  There is no striding, no windowing, no
  splitting a plant into sub-scenes, and no mixing plants.
* **OOM ladder, never frame dropping**: when a forward pass runs out of VRAM
  the config is reduced ``balanced 512 -> max_size 512 -> balanced 448 ->
  max_size 448 -> balanced 384 -> max_size 384`` and the inference is retried
  on the *same* full sequence.
* **Per-view geometry is preserved**: depth / confidence / model RGB / model
  mask are written as individual ``.npy`` files per frame, cameras are written
  with the alignment matrix, so nothing is aggregated into one opaque object.
* **Fusion sees all candidates**: the valid foreground pixels of every view
  (confidence-percentile filtered, no per-view downsampling) enter the
  normal-aware, edge-preserving, confidence-weighted fusion.
* Coordinates follow the same conventions as :func:`visual_util.save_colmap`:
  VGGT-Omega world frame, with the community scene-alignment matrix applied
  exactly once to the points and the cameras.
"""

from __future__ import annotations

import gc
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from foresplat_adapter.discovery import PlantScan, scan_plant  # noqa: E402
from foresplat_adapter.export import export_gggs_dataset  # noqa: E402
from foresplat_adapter.fusion import (  # noqa: E402
    FusionConfig,
    estimate_view_normals,
    fuse_views,
)
from foresplat_adapter.inference import load_model, run_inference  # noqa: E402
from foresplat_adapter.ply import write_point_cloud_ply  # noqa: E402
from visual_util import get_scene_alignment_transform  # noqa: E402

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class LadderAttempt:
    """Result of one VGGT-Omega inference attempt (preprocessing variant)."""

    image_resolution: int
    preprocess_mode: str
    peak_vram_allocated_mib: float | None = None
    peak_vram_reserved_mib: float | None = None
    inference_time_s: float | None = None
    num_frames: int | None = None
    model_input_hw: tuple[int, int] | None = None
    out_of_memory: bool = False
    error: str = ""

    @property
    def key(self) -> str:
        return f"{self.preprocess_mode}@{self.image_resolution}"


OOM_LADDER: list[tuple[int, str]] = [
    (512, "balanced"),
    (512, "max_size"),
    (448, "balanced"),
    (448, "max_size"),
    (384, "balanced"),
    (384, "max_size"),
]


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        return obj.item()
    return obj


def _write_json(path: str, payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(_jsonable(payload), handle, indent=2, ensure_ascii=False)


# --------------------------------------------------------------------------
# preprocessing + inference with OOM ladder
# --------------------------------------------------------------------------

def _preprocess_full_sequence(frames, image_resolution: int, preprocess_mode: str, patch_size: int = 16):
    from vggt_omega.utils.load_fn import load_and_preprocess_images_with_masks

    image_paths = [f.image_path for f in frames]
    mask_paths = [f.mask_path for f in frames]
    return load_and_preprocess_images_with_masks(
        image_paths,
        mask_paths,
        mode=preprocess_mode,
        image_resolution=image_resolution,
        patch_size=patch_size,
    )


def infer_with_ladder(
    model,
    frames,
    attempts: list[LadderAttempt],
    forced: tuple[int, str] | None = None,
    device: str = "cuda",
    background: str = "black",
) -> tuple[dict, torch.Tensor, np.ndarray, list[dict], LadderAttempt]:
    """Run the full sequence once through the OOM ladder.

    Returns ``(predictions, model_masks, transforms, attempt)``.
    """
    ladder = [forced] if forced else OOM_LADDER
    last_error = ""
    for resolution, mode in ladder:
        attempt = LadderAttempt(image_resolution=resolution, preprocess_mode=mode, num_frames=len(frames))
        try:
            images, masks, transforms = _preprocess_full_sequence(frames, resolution, mode)
            model_masks = np.asarray(masks.numpy(), dtype=bool) if isinstance(masks, torch.Tensor) else np.asarray(masks)
            attempt.model_input_hw = (int(images.shape[-2]), int(images.shape[-1]))
            if device == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            predictions = run_inference(model, images, device=device)
            attempt.inference_time_s = round(time.time() - t0, 3)
            if device == "cuda":
                attempt.peak_vram_allocated_mib = round(torch.cuda.max_memory_allocated() / 1024 ** 2, 1)
                attempt.peak_vram_reserved_mib = round(torch.cuda.max_memory_reserved() / 1024 ** 2, 1)
            attempts.append(attempt)
            return predictions, model_masks, transforms, attempt
        except torch.cuda.OutOfMemoryError as exc:  # noqa: PERF203 - ladder by design
            attempt.out_of_memory = True
            attempt.error = f"{type(exc).__name__}: {exc}"
            attempts.append(attempt)
            last_error = attempt.error
            print(f"[infer_with_ladder] OOM at {mode}@{resolution}; moving down the ladder")
            if device == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
            continue
    raise RuntimeError(f"VGGT-Omega OOM on all ladder variants; last error: {last_error}")


# --------------------------------------------------------------------------
# per-view candidate construction (mirrors save_colmap's filter order)
# --------------------------------------------------------------------------

def build_fusion_views(
    predictions: dict,
    masks_model: np.ndarray,
    extrinsic: np.ndarray,
    conf_thres: float = 20.0,
    min_conf: float = 1e-5,
    normal_radius: int = 2,
    edge_rtol: float = 0.03,
) -> list[dict]:
    """Turn per-frame predictions into fusion ``view`` dicts (all candidates)."""
    depth = np.asarray(predictions["depth"])[..., 0] if predictions["depth"].ndim == 4 else np.asarray(predictions["depth"])
    depth_conf = np.asarray(predictions["depth_conf"], dtype=np.float32)
    points_3d = np.asarray(predictions["world_points_from_depth"], dtype=np.float32)
    rgb = np.asarray(predictions["images"])
    if rgb.ndim == 4 and rgb.shape[1] == 3:
        rgb = np.transpose(rgb, (0, 2, 3, 1))

    num_frames, height, width = depth.shape
    masks_model = np.asarray(masks_model, dtype=bool)

    finite_geometry = np.isfinite(points_3d).all(axis=-1) & np.isfinite(depth_conf) & np.isfinite(depth) & (depth > 0)
    valid_fg = masks_model & finite_geometry

    conf_threshold_value = float(min_conf)
    if conf_thres > 0:
        candidates = depth_conf[valid_fg]
        candidates = candidates[np.isfinite(candidates)]
        if candidates.size:
            conf_threshold_value = float(np.percentile(candidates, float(conf_thres)))
            valid_fg &= depth_conf >= conf_threshold_value
    valid_fg &= depth_conf > float(min_conf)

    views = []
    for i in range(num_frames):
        cam_center = -np.asarray(extrinsic[i, :3, :3], dtype=np.float32).T @ np.asarray(extrinsic[i, :3, 3], dtype=np.float32)
        normal_data = estimate_view_normals(
            points_3d[i], valid_fg[i], cam_center, depth=depth[i],
            radius=normal_radius, edge_rtol=edge_rtol,
        )
        views.append(
            {
                "points": points_3d[i],
                "valid": valid_fg[i],
                "colors": rgb[i],
                "conf": depth_conf[i],
                "normals": normal_data["normals"],
                "quality": normal_data["quality"],
                "edge": normal_data["edge"],
                "image_id": i + 1,
                "conf_threshold_value": conf_threshold_value,
            }
        )
    return views, conf_threshold_value


# --------------------------------------------------------------------------
# per-frame artefacts
# --------------------------------------------------------------------------

def save_per_frame_data(
    output_dir: str,
    predictions: dict,
    frames,
    masks_model: np.ndarray,
    transforms: list[dict],
    T_align: np.ndarray,
    attempts: list[LadderAttempt],
) -> dict:
    """Write per-frame depth/conf/model-RGB/model-mask ``.npy`` + camera metadata."""
    per_frame_dir = os.path.join(output_dir, "per_frame")
    os.makedirs(per_frame_dir, exist_ok=True)

    depth = np.asarray(predictions["depth"])
    if depth.ndim == 4:
        depth = depth[..., 0]
    depth_conf = np.asarray(predictions["depth_conf"])
    rgb = np.asarray(predictions["images"])
    if rgb.ndim == 4 and rgb.shape[1] == 3:
        rgb = np.transpose(rgb, (0, 2, 3, 1))
    extrinsic = np.asarray(predictions["extrinsic"], dtype=np.float64)
    intrinsic = np.asarray(predictions["intrinsic"], dtype=np.float64)
    masks_model = np.asarray(masks_model, dtype=bool)

    height, width = depth.shape[1:3]
    cameras = []
    for i, frame in enumerate(frames):
        stem = f"{int(frame.stem):04d}" if frame.stem.isdigit() else frame.stem
        np.save(os.path.join(per_frame_dir, f"depth_{stem}.npy"), depth[i].astype(np.float32))
        np.save(os.path.join(per_frame_dir, f"confidence_{stem}.npy"), depth_conf[i].astype(np.float32))
        np.save(os.path.join(per_frame_dir, f"model_rgb_{stem}.npy"), (rgb[i] * 255).round().astype(np.uint8))
        np.save(os.path.join(per_frame_dir, f"model_mask_{stem}.npy"), masks_model[i].astype(np.uint8))

        R = extrinsic[i, :3, :3]
        t = extrinsic[i, :3, 3]
        center = -R.T @ t
        cameras.append(
            {
                "sequence_index": i,
                "stem": stem,
                "image_name": frame.image_name,
                "intrinsic_model": intrinsic[i].tolist(),
                "extrinsic_model_w2c": extrinsic[i].tolist(),
                "camera_center_model": center.tolist(),
                "camera_center_aligned": (T_align[:3, :3] @ center + T_align[:3, 3]).tolist(),
                "pixel_transform": transforms[i]["pixel_transform"],
                "model_width": int(width),
                "model_height": int(height),
                "foreground_ratio_model": float(masks_model[i].mean()),
            }
        )

    _write_json(
        os.path.join(per_frame_dir, "cameras.json"),
        {
            "description": (
                "Per-view geometry from the full-sequence VGGT-Omega run. Coordinates are the raw "
                "VGGT-Omega world frame; add the scene_alignment_matrix (applied exactly once) to "
                "move into the exported COLMAP/GGGS frame: X' = T_align @ X."
            ),
            "scene_alignment_matrix": T_align.tolist(),
            "num_frames": int(depth.shape[0]),
            "model_height": int(height),
            "model_width": int(width),
            "frames": cameras,
        },
    )
    return {"per_frame_dir": per_frame_dir, "num_frames": int(depth.shape[0])}


# --------------------------------------------------------------------------
# main entry
# --------------------------------------------------------------------------

def process_plant(
    plant_dir: str,
    output_root: str,
    checkpoint: str,
    conf_thres: float = 20.0,
    min_conf: float = 1e-5,
    image_resolution: int = 512,
    preprocess_mode: str = "balanced",
    forced_resolution_mode: tuple[int, str] | None = None,
    filter_depth_edges: bool = False,
    depth_edge_rtol: float = 0.03,
    patch_size: int = 16,
    background: str = "black",
    normal_radius: int = 2,
    edge_rtol: float = 0.03,
    raw_max_points: int = 1_000_000,
    fusion_config: FusionConfig | None = None,
    device: str = "cuda",
    seed: int = 0,
    gggs_repo: str | None = None,
    keep_model: bool = False,
    link_sources: bool = False,
    model=None,
) -> dict:
    """Process one plant end-to-end (full sequence, single joint inference).

    ``model`` may be an already-loaded :class:`VGGTOmega` (shared across plants
    by the batch runner, which calls ``torch.cuda.empty_cache()`` between
    plants); when omitted the checkpoint is loaded here and released after
    inference.
    """
    t_start = time.time()
    plant_id = os.path.basename(os.path.normpath(plant_dir))
    output_dir = os.path.join(output_root, plant_id)
    os.makedirs(output_dir, exist_ok=True)

    scan: PlantScan = scan_plant(plant_id, plant_dir)
    frames = scan.valid_frames
    if len(frames) < 2:
        raise RuntimeError(f"{plant_id}: only {len(frames)} valid frames; cannot run VGGT-Omega")

    from foresplat_adapter.dataset import FramePaths

    frame_objects = [
        FramePaths(index=i, image_path=f.image_path, mask_path=f.mask_path, image_name=f"crop_{f.stem}.png")
        for i, f in enumerate(frames)
    ]

    attempts: list[LadderAttempt] = []
    owns_model = model is None
    if owns_model:
        model = load_model(checkpoint, device=device)
    try:
        predictions, masks_model, transforms, attempt = infer_with_ladder(
            model, frame_objects, attempts, forced=forced_resolution_mode, device=device, background=background,
        )
    finally:
        if owns_model and not keep_model:
            del model
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()

    extrinsic = np.asarray(predictions["extrinsic"], dtype=np.float64)
    T_align = get_scene_alignment_transform(extrinsic)

    save_per_frame_data(
        os.path.join(output_dir, "GGGS_Dataset"),
        predictions, frame_objects, masks_model, transforms, T_align, attempts,
    )

    # ---- fusion -----------------------------------------------------------
    views, conf_threshold_value = build_fusion_views(
        predictions, masks_model, extrinsic, conf_thres=conf_thres, min_conf=min_conf,
        normal_radius=normal_radius, edge_rtol=edge_rtol,
    )
    fusion = fuse_views(views, fusion_config or FusionConfig())
    fused = fusion["fused"]
    observations = fusion["observations"]
    stats = fusion["stats"]

    # Move the fused cloud into the exported (aligned) frame exactly once.
    fused_xyz = fused["xyz"] @ T_align[:3, :3].T + T_align[:3, 3]

    # ---- GGGS dataset -----------------------------------------------------
    dataset_dir = os.path.join(output_dir, "GGGS_Dataset")
    summary = export_gggs_dataset(
        predictions, transforms, masks_model, frame_objects, dataset_dir,
        conf_thres=conf_thres, max_points=int(raw_max_points), filter_depth_edges=filter_depth_edges,
        depth_edge_rtol=depth_edge_rtol, min_conf=min_conf, seed=seed,
        image_resolution=attempt.image_resolution, preprocess_mode=attempt.preprocess_mode,
        checkpoint_path=checkpoint,
        fused_points={
            "xyz": fused_xyz,
            "colors": fused["colors"],
            "normals": fused["normals"],
            "is_edge": fused["is_edge"],
            "observation_count": fused["observation_count"],
            "confidence": fused["confidence"],
        },
        observations=observations,
        link_sources=link_sources,
    )

    # debug PLY of the fused cloud (with quality/observation-count fields)
    debug_dir = os.path.join(output_dir, "GGGS_Dataset", "debug")
    write_point_cloud_ply(
        os.path.join(debug_dir, "fused_pointcloud.ply"),
        fused_xyz, fused["colors"], normals=fused["normals"],
        comments=["ForeSplat full-sequence fused cloud (normal-aware, confidence-weighted, edge-preserving)"],
    )

    _write_json(
        os.path.join(output_dir, "GGGS_Dataset", "metadata", "fusion_metadata.json"),
        {
            "fusion_stats": stats,
            "ladder_attempts": [asdict(a) for a in attempts],
            "attempted": attempt.key,
            "conf_threshold_value": conf_threshold_value,
            "scene_alignment_matrix": T_align.tolist(),
        },
    )
    for name, value in (
        ("peak_vram_allocated_mib", attempt.peak_vram_allocated_mib),
        ("peak_vram_reserved_mib", attempt.peak_vram_reserved_mib),
        ("inference_time_s", attempt.inference_time_s),
    ):
        summary[name] = value
    summary["preprocess_mode"] = attempt.preprocess_mode
    summary["image_resolution"] = attempt.image_resolution
    summary["link_sources"] = bool(link_sources)
    summary["processed_frames"] = len(frame_objects)
    summary["raw_max_points"] = int(raw_max_points)
    summary["model_input_hw"] = list(attempt.model_input_hw) if attempt.model_input_hw else None
    summary["ladder_attempts"] = [asdict(a) for a in attempts]
    summary["mean_observation_confidence"] = float(fused["confidence"].mean()) if fused["confidence"].size else 0.0
    summary["fusion"] = stats
    summary["elapsed_seconds"] = round(time.time() - t_start, 3)
    return summary