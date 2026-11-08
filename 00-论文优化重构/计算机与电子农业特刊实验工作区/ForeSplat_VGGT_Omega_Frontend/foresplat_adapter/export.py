"""Assemble the GGGS-ready dataset from VGGT-Omega predictions.

Output layout (compatible with HKUST-SAIL/Gaussian-Splatting-with-Stochastic-Solids)::

    GGGS_Dataset/
        images/                       RAP-FSAM3 foreground RGB (original resolution)
        masks/                         RAP-FSAM3 binary masks (original resolution)
        sparse/0/
            cameras.bin / images.bin / points3D.bin     (written by save_colmap)
            cameras.txt / images.txt / points3D.txt
            points3D.ply                               (foreground-constrained init cloud)
        metadata/
            preprocess_transforms.json
            image_manifest.json
            reconstruction_metadata.json
        debug/
            model_inputs/              what VGGT-Omega actually saw
            model_masks/               masks at model resolution
            foreground_pointcloud.ply
"""

from __future__ import annotations

import json
import os
import shutil
import time

import numpy as np
import torch
from PIL import Image

import pycolmap

from .dataset import FramePaths, load_mask_binary
from .ply import write_point_cloud_ply

import sys

# Make the community visual_util importable when running from the repo root.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from visual_util import save_colmap  # noqa: E402


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def _write_json(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(_jsonable(payload), handle, indent=2, ensure_ascii=False)


def export_gggs_dataset(
    predictions: dict,
    transforms: list[dict],
    masks_model: torch.Tensor | np.ndarray,
    frames: list[FramePaths],
    output_dir: str,
    conf_thres: float = 20.0,
    max_points: int = 1000000,
    filter_depth_edges: bool = False,
    depth_edge_rtol: float = 0.03,
    apply_alignment: bool = True,
    min_conf: float = 1e-5,
    mask_threshold: int = 127,
    seed: int = 0,
    image_resolution: int = 512,
    preprocess_mode: str = "balanced",
    checkpoint_path: str | None = None,
    run_args: dict | None = None,
    fused_points: dict | None = None,
    observations: dict | None = None,
    link_sources: bool = False,
) -> dict:
    """Run the full ForeSplat export and assemble the GGGS dataset directory."""
    masks_model = np.asarray(masks_model, dtype=bool)

    images_dir = os.path.join(output_dir, "images")
    masks_dir = os.path.join(output_dir, "masks")
    sparse_dir = os.path.join(output_dir, "sparse", "0")
    metadata_dir = os.path.join(output_dir, "metadata")
    debug_dir = os.path.join(output_dir, "debug")
    for directory in (images_dir, masks_dir, metadata_dir, debug_dir):
        os.makedirs(directory, exist_ok=True)
    for sub in ("model_inputs", "model_masks"):
        os.makedirs(os.path.join(debug_dir, sub), exist_ok=True)

    t_start = time.time()

    def _materialise(src: str, dst: str, composite_rgba: bool = True) -> None:
        """Copy ``src`` to ``dst``; with ``link_sources`` use a hard link instead.

        Hard links save the full-resolution image/mask copy on slow disks.  The
        dataset then shares inodes with the source until one side is edited or
        deleted, which the RUN_REPORT records.  RGBA sources are always
        composited over black (matching the RAP-FSAM3 foreground form), never
        linked.
        """
        is_rgba = False
        with Image.open(src) as probe:
            is_rgba = probe.mode == "RGBA"
        if link_sources and not is_rgba:
            if os.path.exists(dst):
                try:
                    if os.path.samefile(dst, src):
                        return
                except OSError:
                    pass
                os.remove(dst)
            try:
                os.link(src, dst)
                return
            except OSError:
                pass  # cross-device or unsupported -> fall back to a copy
        with Image.open(src) as img:
            if composite_rgba and img.mode == "RGBA":
                canvas = Image.new("RGBA", img.size, (0, 0, 0, 255))
                img = Image.alpha_composite(canvas, img)
                img.convert("RGB").save(dst)
                return
            img.save(dst)

    # ---- 1. original-resolution foreground images + masks -----------------
    saved_image_paths = []
    fg_ratios_original = []
    for frame in frames:
        dst_image = os.path.join(images_dir, frame.image_name)
        _materialise(frame.image_path, dst_image)
        dst_mask = os.path.join(masks_dir, os.path.splitext(frame.image_name)[0] + ".png")
        _materialise(frame.mask_path, dst_mask)
        mask_u8 = load_mask_binary(frame.mask_path, mask_threshold=mask_threshold)
        saved_image_paths.append(dst_image)
        fg_ratios_original.append(float((mask_u8 > 0).mean()))

    # ---- 2. COLMAP export (foreground-constrained, transform-correct) ------
    stats = save_colmap(
        predictions,
        output_dir,
        image_names=saved_image_paths,
        conf_thres=conf_thres,
        max_points=max_points,
        foreground_masks=masks_model,
        preprocess_transforms=transforms,
        filter_depth_edges=filter_depth_edges,
        depth_edge_rtol=depth_edge_rtol,
        colmap_subdir="sparse/0",
        apply_alignment=apply_alignment,
        min_conf=min_conf,
        seed=seed,
        return_stats=True,
    )

    # ---- 3. Read back the written model (validates the files) --------------
    reconstruction = pycolmap.Reconstruction(sparse_dir)
    cam_centers = {}
    for image_id, image in reconstruction.images.items():
        R = image.cam_from_world.rotation.matrix()
        t = image.cam_from_world.translation
        cam_centers[image_id] = -R.T @ np.asarray(t, dtype=np.float64)

    # ---- 4. points3D.ply (GGGS initialisation cloud) + debug PLYs ----------
    points3d_xyz = stats["points3d_xyz"]
    points3d_colors = stats["points3d_colors"]
    points3d_quality = stats["points3d_quality"]
    points3d_image_id = stats["points3d_image_id"]

    if len(points3d_xyz) > 0:
        centers = np.stack([cam_centers[int(i)] for i in points3d_image_id], axis=0)
        direction = centers - points3d_xyz
        normals = direction / np.maximum(np.linalg.norm(direction, axis=1, keepdims=True), 1e-8)
    else:
        normals = np.zeros((0, 3), dtype=np.float64)

    raw_comments = [
        "ForeSplat: foreground-mask-constrained depth back-projection points from VGGT-Omega",
        "Each point is observed in exactly one image (single-view depth unprojection, not SfM triangulation)",
    ]
    write_point_cloud_ply(
        os.path.join(debug_dir, "foreground_pointcloud.ply"),
        points3d_xyz,
        points3d_colors,
        normals=normals,
        qualities=points3d_quality,
        comments=raw_comments,
    )
    write_point_cloud_ply(
        os.path.join(debug_dir, "raw_pointcloud.ply"),
        points3d_xyz,
        points3d_colors,
        normals=normals,
        qualities=points3d_quality,
        comments=raw_comments,
    )

    if fused_points is not None:
        # Full-sequence export: points3D.ply is the multi-view FUSED cloud
        # (normal-aware, confidence-weighted, edge-preserving).  The per-view
        # raw cloud stays available as debug/raw_pointcloud.ply and as the
        # single-observation COLMAP points3D.bin.
        fused_xyz = np.asarray(fused_points["xyz"], dtype=np.float64)
        fused_colors = np.asarray(fused_points["colors"], dtype=np.uint8)
        fused_normals = np.asarray(fused_points["normals"], dtype=np.float64)
        fused_comments = [
            "ForeSplat: multi-view fused cloud (voxel hashing + normal-aware clustering,",
            "confidence-weighted, edge-preserving). Fused from single-view depth",
            "back-projection candidates; NOT SfM triangulation. Full provenance in",
            "debug/observations.npz (fused point id -> source image, pixel, weight).",
        ]
        write_point_cloud_ply(
            os.path.join(sparse_dir, "points3D.ply"),
            fused_xyz,
            fused_colors,
            normals=fused_normals,
            qualities=None,
            comments=fused_comments,
        )
        points3d_ply_is_fused = True
        num_fused_points = int(fused_xyz.shape[0])
    else:
        write_point_cloud_ply(
            os.path.join(sparse_dir, "points3D.ply"),
            points3d_xyz,
            points3d_colors,
            normals=normals,
            qualities=None,
            comments=raw_comments,
        )
        points3d_ply_is_fused = False
        num_fused_points = int(len(points3d_xyz))

    if fused_points is not None and observations is not None:
        from .fusion.observation_mapping import save_observations_npz

        save_observations_npz(
            os.path.join(debug_dir, "observations.npz"),
            observations,
            meta={"model_height": np.array([int(predictions["images"].shape[-2])]),
                  "model_width": np.array([int(predictions["images"].shape[-1])])},
        )

    # ---- 5. debug: actual model inputs and model-resolution masks ----------
    model_images = predictions["images"]
    for fidx, frame in enumerate(frames):
        rgb = model_images[fidx]
        if isinstance(rgb, torch.Tensor):
            rgb = rgb.detach().float().cpu().numpy()
        rgb = np.transpose(rgb, (1, 2, 0))
        rgb = (np.clip(rgb, 0, 1) * 255).round().astype(np.uint8)
        stem = os.path.splitext(frame.image_name)[0]
        Image.fromarray(rgb).save(os.path.join(debug_dir, "model_inputs", stem + ".png"))
        Image.fromarray(masks_model[fidx].astype(np.uint8) * 255, mode="L").save(
            os.path.join(debug_dir, "model_masks", stem + ".png")
        )

    # ---- 6. metadata --------------------------------------------------------
    cameras_meta = []
    for image_id in sorted(reconstruction.images):
        image = reconstruction.images[image_id]
        camera = reconstruction.cameras[image.camera_id]
        R = image.cam_from_world.rotation.matrix()
        t = np.asarray(image.cam_from_world.translation, dtype=np.float64)
        cameras_meta.append(
            {
                "image_id": int(image_id),
                "image_name": image.name,
                "camera_id": int(camera.camera_id),
                "camera_model": str(camera.model),
                "width": int(camera.width),
                "height": int(camera.height),
                "params": [float(v) for v in np.asarray(camera.params)],
                "cam_from_world_rotation": R.tolist(),
                "cam_from_world_translation": t.tolist(),
                "camera_center_world": (-R.T @ t).tolist(),
                "num_points2d": len(image.points2D),
            }
        )

    _write_json(
        os.path.join(metadata_dir, "preprocess_transforms.json"),
        {
            "description": (
                "Per-frame VGGT-Omega preprocessing record. pixel_transform is the 3x3 affine A "
                "with p_model = A @ p_original (homogeneous pixel indices, PIL pixel-centre "
                "convention: crop -> bicubic resize -> pad). Map camera intrinsics with "
                "K_original = A^-1 @ K_model and observations with p_original = A^-1 @ p_model."
            ),
            "frames": transforms,
        },
    )

    _write_json(
        os.path.join(metadata_dir, "image_manifest.json"),
        {
            "description": "One entry per exported frame (original resolution inputs and model inputs).",
            "frames": [
                {
                    "index": frame.index,
                    "image_name": frame.image_name,
                    "source_image": frame.image_path,
                    "source_mask": frame.mask_path,
                    "exported_image": os.path.relpath(saved_image_paths[i], output_dir),
                    "exported_mask": os.path.relpath(
                        os.path.join(masks_dir, os.path.splitext(frame.image_name)[0] + ".png"), output_dir
                    ),
                    "original_width": transforms[i]["original_width"],
                    "original_height": transforms[i]["original_height"],
                    "model_width": transforms[i]["model_width"],
                    "model_height": transforms[i]["model_height"],
                    "foreground_ratio_original": fg_ratios_original[i],
                    "foreground_ratio_model": (
                        float(masks_model[i].mean()) if stats.get("foreground_ratios_model") is not None else None
                    ),
                    "exported_point2d_count": int(stats["per_frame_point2d"][i]),
                }
                for i, frame in enumerate(frames)
            ],
        },
    )

    _write_json(
        os.path.join(metadata_dir, "fusion_metadata.json"),
        {
            "description": "Multi-view fusion summary for this dataset (full provenance in debug/observations.npz).",
            "points3d_ply_is_fused": points3d_ply_is_fused,
            "num_fused_points_in_ply": num_fused_points,
            "num_raw_per_view_points": int(stats["num_points3d"]),
        },
    )

    _write_json(
        os.path.join(metadata_dir, "reconstruction_metadata.json"),
        {
            "description": "VGGT-Omega -> COLMAP export summary for the ForeSplat GGGS frontend.",
            "num_frames": stats["num_frames"],
            "model_resolution": stats["model_resolution"],
            "num_points3d": stats["num_points3d"],
            "points3d_ply_is_fused": points3d_ply_is_fused,
            "num_fused_points_in_ply": num_fused_points,
            "confidence_threshold_percentile": float(conf_thres),
            "confidence_threshold_value": stats["conf_threshold_value"],
            "min_conf": float(min_conf),
            "max_points": int(max_points),
            "depth_edge_filtering": {
                "enabled": bool(filter_depth_edges),
                "rtol": float(depth_edge_rtol),
            },
            "scene_alignment_matrix": stats["scene_alignment_matrix"],
            "scene_alignment_note": (
                "T_align = inv(E0) @ opengl_conversion @ rotY(180) - applied ONCE to both the "
                "3D points and the camera poses (identical to the community GLB exporter); "
                "no additional Y/Z flip is performed afterwards."
            ),
            "point_semantics": (
                "Every points3D entry is a single-view depth back-projection from its own "
                "image's VGGT-Omega depth map, filtered by the RAP-FSAM3 foreground mask "
                "(foreground & finite geometry & valid confidence & valid depth). Tracks "
                "contain exactly one 2D observation each; reprojection error is undefined "
                "(zero) and must not be interpreted as bundle-adjustment residuals."
            ),
            "cameras": cameras_meta,
            "per_frame_point2d": stats["per_frame_point2d"],
            "export_config": run_args or {},
            "checkpoint": checkpoint_path,
            "image_resolution": image_resolution,
            "preprocess_mode": preprocess_mode,
            "pycolmap_version": stats["pycolmap_version"],
            "elapsed_seconds": round(time.time() - t_start, 3),
        },
    )

    summary = {
        "output_dir": output_dir,
        "num_frames": stats["num_frames"],
        "num_points3d": stats["num_points3d"],
        "points3d_ply_is_fused": points3d_ply_is_fused,
        "num_fused_points_in_ply": num_fused_points,
        "colmap_dir": sparse_dir,
        "per_frame_point2d": stats["per_frame_point2d"],
        "elapsed_seconds": round(time.time() - t_start, 3),
    }
    print(
        f"[export_gggs_dataset] wrote {summary['num_points3d']} foreground 3D points for "
        f"{summary['num_frames']} frames to {sparse_dir} in {summary['elapsed_seconds']} s"
    )
    return summary
