# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
"""Interactive Gradio demo for dvlt model inference and 3D visualization.

The UI's model dropdown is populated from
:data:`dvlt.util.model_registry.DEFAULT_INFERENCE_MODELS` (every experiment
YAML with a pinned ``trainer.ckpt_dir``). DVLT training configs
(``dvlt-large``, ``dvlt-large-depthconv-stage2``, ...) are deliberately
excluded; use the inference-only ``dvlt.yaml`` instead.

This file is intentionally small: heavy lifting lives in shared modules so
any future batch script can reuse the exact same code path.

- :mod:`dvlt.util.model_registry` — ``ModelEntry`` + curated registry.
- :mod:`dvlt.viz.pointcloud` — depth-unprojection + filtering pipeline.
- :mod:`dvlt.viz.glb` — GLB export with the camera-overlay node.
- :mod:`dvlt.viz.camera_overlay` — trimesh builders for the overlay itself.

Usage::

    # Default launch — dropdown shows all inference models, DVLT preselected.
    python src/dvlt/scripts/gradio_demo.py

    # Pre-select a different model from the curated list:
    python src/dvlt/scripts/gradio_demo.py --default-model "VGGT 1B"

    # Append a custom inference yaml in addition to the curated list.
    # Format: LABEL:CONFIG_NAME  (YAML must pin trainer.ckpt_dir).
    python src/dvlt/scripts/gradio_demo.py \\
        --add-model "DVLT (custom):my_dvlt_eval"

    # Offline / batch mode: skip Gradio and write <cfg_name>.glb / .rrd
    # under demo_outputs/<sequence_name>/. --input may be repeated and
    # accepts a directory of images, a single image, or a video file.
    # --models is a comma-separated list of config names (or 'all').
    python src/dvlt/scripts/gradio_demo.py --offline \\
        --input /path/to/scene_a \\
        --input /path/to/clip.mp4 \\
        --models dvlt,vggt
"""

import gc
import logging
import tempfile
import time
from pathlib import Path
from typing import Optional

import gradio as gr
import numpy as np
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from PIL import Image

from dvlt.common.constants import DataField, PredictionField
from dvlt.common.pose import to4x4
from dvlt.config.cli import cli
from dvlt.config.schema import register_configs
from dvlt.util.model_registry import (
    DEFAULT_INFERENCE_MODELS,
    ModelEntry,
    parse_model_spec,
)
from dvlt.util.preprocess import load_sequence, preprocess_images
from dvlt.viz.depth import overlay_depth_map
from dvlt.viz.glb import pointcloud_to_glb
from dvlt.viz.pointcloud import (
    SPATIAL_PERCENTILE_DEFAULT,
    build_pointcloud,
    zero_depths_on_pad,
)
from dvlt.viz.scene_rerun import visualize_scene


logger = get_logger(__name__)


EXTRA_CLI_ARGS = [
    ("--max-points", {"type": int, "default": 500_000, "help": "Max points to visualize (default: 500000)"}),
    ("--conf-threshold", {"type": float, "default": 50.0, "help": "Confidence percentile threshold (default: 50.0)"}),
    ("--server-port", {"type": int, "default": 7860, "help": "Gradio server port (default: 7860)"}),
    ("--share", {"action": "store_true", "default": False, "help": "Create a public Gradio link"}),
    (
        "--default-model",
        {
            "type": str,
            "default": None,
            "metavar": "LABEL",
            "help": (
                "Label of the dropdown entry to preselect (and pre-load at "
                "startup). Defaults to the first entry in DEFAULT_INFERENCE_MODELS."
            ),
        },
    ),
    (
        "--add-model",
        {
            "action": "append",
            "default": None,
            "dest": "model_specs",
            "metavar": "LABEL:CONFIG_NAME",
            "help": (
                "Append an entry to the model dropdown beyond DEFAULT_INFERENCE_MODELS. "
                "May be repeated. LABEL is shown in the UI, CONFIG_NAME is an "
                "experiment YAML name (without extension) under "
                "src/dvlt/config/experiments. The checkpoint must be pinned "
                "via trainer.ckpt_dir in that YAML."
            ),
        },
    ),
    (
        "--offline",
        {
            "action": "store_true",
            "default": False,
            "help": (
                "Run headlessly: process --input sequences with --models and "
                "write GLB/RRD under --output-root/<sequence_name>/. Skips "
                "Gradio launch entirely."
            ),
        },
    ),
    (
        "--input",
        {
            "action": "append",
            "default": None,
            "dest": "inputs",
            "metavar": "PATH",
            "help": (
                "Offline mode only. Path to a directory of images, a single "
                "image, or a video file. May be repeated; each value becomes "
                "one sequence under --output-root."
            ),
        },
    ),
    (
        "--models",
        {
            "type": str,
            "default": None,
            "metavar": "LIST",
            "help": (
                "Offline mode only. Comma-separated list of config names from "
                "DEFAULT_INFERENCE_MODELS (e.g. dvlt,vggt,pi3). Special value "
                "'all' runs every entry in the registry."
            ),
        },
    ),
    (
        "--output-root",
        {
            "type": str,
            "default": "demo_outputs",
            "metavar": "DIR",
            "help": "Offline mode only. Root directory for outputs (default: demo_outputs).",
        },
    ),
    (
        "--overwrite",
        {
            "action": "store_true",
            "default": False,
            "help": (
                "Offline mode only. Re-run a (sequence, model) pair even if "
                "its .glb and .rrd already exist (default: skip)."
            ),
        },
    ),
    (
        "--frustum-scale",
        {
            "type": float,
            "default": 0.01,
            "metavar": "FRAC",
            "help": (
                "Camera frustum depth as a fraction of the combined point-cloud "
                "+ camera-center scene radius (default: 0.01 = 1%%). Set to 0 "
                "to disable the camera-path overlay in the GLB. Affects offline "
                "GLBs and the initial value of the Gradio slider."
            ),
        },
    ),
]


# ---------------------------------------------------------------------------
# Per-prediction helpers (RRD export + depth gallery)
# ---------------------------------------------------------------------------


def _cameras_to_glb_inputs(predictions: dict, batch: dict) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """Pull ``(c2ws, intrinsics, image_hw)`` numpy/python out of a predictions dict.

    Centralized so the three GLB call sites stay in sync.
    """
    cameras = predictions[PredictionField.CAMERAS][0]
    c2ws = to4x4(cameras.camera_to_worlds).detach().float().cpu().numpy()
    intrinsics = cameras.get_intrinsics_matrices().detach().float().cpu().numpy()
    image_hw = tuple(batch[DataField.IMAGES].shape[-2:])
    return c2ws, intrinsics, image_hw


def pointcloud_to_rrd(
    pts: np.ndarray,
    rgb: np.ndarray,
    predictions: dict,
    batch: dict,
    output_path: Optional[Path] = None,
) -> str:
    """Export predicted point cloud and cameras to an RRD file. Returns path.

    When ``output_path`` is None (Gradio path), writes to ``tempfile.mktemp``.
    Otherwise writes to ``output_path``, creating parent directories.
    """
    cameras_pred = predictions[PredictionField.CAMERAS][0]
    images_np = batch[DataField.IMAGES][0].detach().float().cpu().permute(0, 2, 3, 1).numpy()
    depths_np = predictions[PredictionField.DEPTHS][0].detach().float().cpu().numpy()
    depths_np = zero_depths_on_pad(depths_np, batch)
    blended = [overlay_depth_map(img, d) for img, d in zip(images_np, depths_np)]

    if output_path is None:
        path = tempfile.mktemp(suffix=".rrd")
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        path = str(output_path)
    tag = output_path.stem if output_path is not None else f"gradio_demo_{time.time()}"
    visualize_scene(
        log_path=tag,
        server_address=None,
        cameras=cameras_pred,
        points=pts,
        rgb=rgb,
        images=blended,
        save_path=path,
        view_coordinates="RDF",
        # ``visualize_scene``'s default 200k would re-subsample after
        # build_pointcloud; keep up to 1M for parity with the GLB output.
        max_num_points=min(len(pts), 1_000_000),
    )
    return path


def depth_gallery(predictions: dict, batch: dict) -> list[np.ndarray]:
    """Return list of depth maps alpha-blended with input images for the gallery."""
    depths = predictions[PredictionField.DEPTHS][0].detach().float().cpu().numpy()
    depths = zero_depths_on_pad(depths, batch)
    images = batch[DataField.IMAGES][0].detach().float().cpu().permute(0, 2, 3, 1).numpy()
    return [overlay_depth_map(img, d) for img, d in zip(images, depths)]


# ---------------------------------------------------------------------------
# Offline batch mode
# ---------------------------------------------------------------------------


def _run_offline(
    accelerator: Accelerator,
    registry_by_config: dict[str, ModelEntry],
    model_keys: list[str],
    input_paths: list[Path],
    output_root: Path,
    max_points: int,
    conf_threshold: float,
    overwrite: bool,
    frustum_scale: float = 0.01,
) -> None:
    """Iterate models (outer) x sequences (inner), writing GLB and RRD per pair.

    Loops models on the outside so each checkpoint is loaded once, used for
    every sequence, then unloaded before the next model. This keeps peak GPU
    memory bounded to a single model regardless of ``len(model_keys)``.
    """
    output_root.mkdir(parents=True, exist_ok=True)

    sequences: list[tuple[str, list[Image.Image]]] = []
    for input_path in input_paths:
        seq_name, frames = load_sequence(input_path)
        logger.info(f"[{seq_name}] {len(frames)} frame(s) from {input_path}")
        sequences.append((seq_name, frames))

    for cfg_name in model_keys:
        entry = registry_by_config[cfg_name]
        try:
            entry.ensure_loaded(accelerator)
            for seq_name, frames in sequences:
                seq_dir = output_root / seq_name
                seq_dir.mkdir(parents=True, exist_ok=True)
                glb_path = seq_dir / f"{cfg_name}.glb"
                rrd_path = seq_dir / f"{cfg_name}.rrd"
                if not overwrite and glb_path.exists() and rrd_path.exists():
                    logger.info(f"[{seq_name}/{cfg_name}] skipping (exists). Pass --overwrite to redo.")
                    continue

                logger.info(f"[{seq_name}/{cfg_name}] running inference")
                t0 = time.time()
                batch = preprocess_images(frames, entry.img_size, entry.patch_size, accelerator.device)
                with torch.no_grad(), accelerator.autocast():
                    predictions = entry.model.predict(batch, accelerator)
                pts, rgb = build_pointcloud(
                    predictions, batch, int(max_points), conf_threshold, SPATIAL_PERCENTILE_DEFAULT
                )
                c2ws, intrinsics, image_hw = _cameras_to_glb_inputs(predictions, batch)
                pointcloud_to_glb(
                    pts,
                    rgb,
                    output_path=glb_path,
                    name=f"{seq_name}_{cfg_name}",
                    cameras_c2w=c2ws,
                    intrinsics=intrinsics,
                    image_hw=image_hw,
                    frustum_frac=frustum_scale,
                )
                pointcloud_to_rrd(pts, rgb, predictions, batch, output_path=rrd_path)
                del predictions, batch, pts, rgb
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                logger.info(
                    f"[{seq_name}/{cfg_name}] wrote {glb_path.name} and {rrd_path.name} ({time.time() - t0:.1f}s)"
                )
        finally:
            entry.unload()


# ---------------------------------------------------------------------------
# Gradio app factory
# ---------------------------------------------------------------------------


def _frames_to_gallery(frames: list[Image.Image]) -> list[tuple[np.ndarray, str]]:
    """Format PIL frames as (image, caption) tuples for a gr.Gallery."""
    out: list[tuple[np.ndarray, str]] = []
    for i, img in enumerate(frames):
        caption = f"#{i} (ref)" if i == 0 else f"#{i}"
        out.append((np.array(img.convert("RGB")), caption))
    return out


def _load_files_to_frames(files: list[str]) -> list[Image.Image]:
    """Expand a list of upload paths (images and/or videos) into PIL frames."""
    frames: list[Image.Image] = []
    for f in files:
        _, fs = load_sequence(f)
        frames.extend(fs)
    return frames


def build_app(
    accelerator: Accelerator,
    registry: dict[str, ModelEntry],
    default_label: str,
    default_max_points: int = 500_000,
    default_conf_threshold: float = 50.0,
    default_frustum_scale: float = 0.01,
):
    """Construct and return the Gradio Blocks app."""
    if default_label not in registry:
        raise ValueError(f"default_label {default_label!r} not in registry {list(registry)}")

    state: dict = {"active_label": default_label}

    def _invalidate_inference() -> None:
        for k in ("predictions", "batch", "pts", "rgb"):
            state.pop(k, None)

    def _model_status(label: str) -> str:
        entry = registry[label]
        suffix = "loaded" if entry.is_loaded else "not loaded — will load on first run"
        return f"Active model: **{label}** ({suffix})."

    # ---- model dropdown change handler --------------------------------
    def _on_model_change(label: str):
        if label not in registry:
            raise gr.Error(f"Unknown model: {label}")
        state["active_label"] = label
        _invalidate_inference()
        entry = registry[label]
        if not entry.is_loaded:
            try:
                entry.ensure_loaded(accelerator)
            except Exception as exc:  # noqa: BLE001
                logger.exception(f"Failed to load model {label}: {exc}")
                raise gr.Error(f"Failed to load model '{label}': {exc}") from exc
        return (
            _model_status(label),
            gr.update(value=None),
            gr.update(value=None),
            gr.update(value=None),
        )

    # ---- upload handler: unpack videos -> frames preview -------------
    def _load_files(files):
        _invalidate_inference()
        state["selected_idx"] = None
        if not files:
            state["frames"] = []
            return (
                [],
                "No frames loaded.",
                gr.update(value=None),
                gr.update(value=None),
                gr.update(value=None),
            )

        try:
            frames = _load_files_to_frames(files)
        except ValueError as e:
            raise gr.Error(str(e)) from e
        state["frames"] = frames
        status = f"Loaded {len(frames)} frame(s). Reference is #0."
        return (
            _frames_to_gallery(frames),
            status,
            gr.update(value=None),
            gr.update(value=None),
            gr.update(value=None),
        )

    # ---- gallery selection: remember picked frame --------------------
    def _on_select(evt: gr.SelectData):
        idx = int(evt.index)
        state["selected_idx"] = idx
        return f"Selected frame #{idx}. Click 'Set selected as reference' to move it to position 0."

    # ---- reorder: rotate so picked frame becomes index 0 -------------
    def _set_reference():
        frames = state.get("frames") or []
        if not frames:
            raise gr.Error("Upload images or videos first.")
        idx = state.get("selected_idx")
        if idx is None:
            raise gr.Error("Click a frame in the gallery to select it first.")
        if idx == 0:
            return (
                _frames_to_gallery(frames),
                "Reference is already frame #0.",
                gr.update(),
                gr.update(),
                gr.update(),
            )

        rotated = frames[idx:] + frames[:idx]
        state["frames"] = rotated
        state["selected_idx"] = 0
        _invalidate_inference()
        return (
            _frames_to_gallery(rotated),
            f"Reordered: previous frame #{idx} is now the reference (#0).",
            gr.update(value=None),
            gr.update(value=None),
            gr.update(value=None),
        )

    def _export_glb(pts: np.ndarray, rgb: np.ndarray, frustum_scale_pct: float) -> str:
        """Wrap pointcloud_to_glb so the two handlers stay in sync."""
        c2ws, intrinsics, image_hw = _cameras_to_glb_inputs(state["predictions"], state["batch"])
        return pointcloud_to_glb(
            pts,
            rgb,
            cameras_c2w=c2ws,
            intrinsics=intrinsics,
            image_hw=image_hw,
            frustum_frac=float(frustum_scale_pct) / 100.0,
        )

    # ---- core handler --------------------------------------------------
    def _run(max_pts, conf_thresh, frustum_scale_pct):
        frames = state.get("frames") or []
        if not frames:
            raise gr.Error("Upload at least one image or video.")

        label = state["active_label"]
        entry = registry[label]
        entry.ensure_loaded(accelerator)

        batch = preprocess_images(frames, entry.img_size, entry.patch_size, accelerator.device)
        with torch.no_grad(), accelerator.autocast():
            predictions = entry.model.predict(batch, accelerator)

        state["predictions"] = predictions
        state["batch"] = batch

        pts, rgb = build_pointcloud(predictions, batch, int(max_pts), conf_thresh, SPATIAL_PERCENTILE_DEFAULT)
        state["pts"] = pts
        state["rgb"] = rgb

        glb_path = _export_glb(pts, rgb, frustum_scale_pct)
        rrd_path = pointcloud_to_rrd(pts, rgb, predictions, batch)
        depths_vis = depth_gallery(predictions, batch)

        return glb_path, rrd_path, depths_vis

    # ---- auto-run on model switch when frames are already present ----
    def _run_if_frames_loaded(max_pts, conf_thresh, frustum_scale_pct):
        if not state.get("frames"):
            # Nothing to reconstruct; preserve the cleared outputs from
            # _on_model_change (the .then chain still has to return values).
            return gr.update(), gr.update(), gr.update()
        return _run(max_pts, conf_thresh, frustum_scale_pct)

    # ---- re-visualize with new slider values ---------------------------
    def _update_viz(max_pts, conf_thresh, frustum_scale_pct):
        if "predictions" not in state:
            raise gr.Error("Run inference first.")
        pts, rgb = build_pointcloud(
            state["predictions"], state["batch"], int(max_pts), conf_thresh, SPATIAL_PERCENTILE_DEFAULT
        )
        state["pts"] = pts
        state["rgb"] = rgb

        glb_path = _export_glb(pts, rgb, frustum_scale_pct)
        rrd_path = pointcloud_to_rrd(pts, rgb, state["predictions"], state["batch"])
        return glb_path, rrd_path

    # ---- UI layout -----------------------------------------------------
    with gr.Blocks(title="dvlt 3D Demo") as demo:
        gr.Markdown("# dvlt Interactive 3D Demo")
        gr.Markdown(
            "Upload images or videos (mp4/mov/gif/...), run model inference, and "
            "explore the depth-unprojected 3D point cloud. Videos are sampled at "
            "2 fps (assuming 24 fps when "
            "metadata is missing). Click a frame and use 'Set selected as "
            "reference' to rotate the sequence so it becomes frame #0. Use the "
            "download button on the 3D viewer for GLB, or the link below for "
            "RRD (Rerun)."
        )

        with gr.Row():
            with gr.Column(scale=1):
                model_dropdown = gr.Dropdown(
                    choices=list(registry.keys()),
                    value=default_label,
                    label="Model",
                    interactive=len(registry) > 1,
                )
                model_status = gr.Markdown(_model_status(default_label))
                # Constrain height so many uploads don't blow up the panel; the
                # internal list becomes scrollable.
                file_input = gr.File(
                    label="Upload Images or Videos",
                    file_count="multiple",
                    file_types=["image", "video", ".gif"],
                    type="filepath",
                    height=240,
                )
                max_pts_slider = gr.Slider(
                    minimum=1_000,
                    maximum=2_000_000,
                    value=default_max_points,
                    step=1_000,
                    label="Max Points",
                )
                conf_slider = gr.Slider(
                    minimum=0.0,
                    maximum=100.0,
                    value=default_conf_threshold,
                    step=0.5,
                    label="Confidence Threshold (percentile)",
                )
                frustum_scale_slider = gr.Slider(
                    minimum=0.0,
                    maximum=5.0,
                    value=float(default_frustum_scale) * 100.0,
                    step=0.01,
                    label="Camera Frustum Size (% of scene radius; 0 disables)",
                )
                run_btn = gr.Button("Run Inference", variant="primary")
                update_btn = gr.Button("Update Visualization")
                rrd_download = gr.File(label="Download RRD (Rerun)", interactive=False)

            with gr.Column(scale=3):
                model3d_output = gr.Model3D(
                    label="3D Point Cloud (download GLB via viewer)",
                    height=560,
                )

        with gr.Row():
            with gr.Column(scale=1):
                image_gallery_output = gr.Gallery(
                    label="Input Frames (#0 is reference)",
                    columns=4,
                    height=320,
                    show_label=True,
                    allow_preview=True,
                )
                selection_status = gr.Markdown("No frames loaded.")
                set_ref_btn = gr.Button("Set selected as reference")
            with gr.Column(scale=1):
                depth_gallery_output = gr.Gallery(label="Depth Maps", columns=4, height=320)

        model_dropdown.change(
            fn=_on_model_change,
            inputs=[model_dropdown],
            outputs=[model_status, model3d_output, rrd_download, depth_gallery_output],
        ).then(
            fn=_run_if_frames_loaded,
            inputs=[max_pts_slider, conf_slider, frustum_scale_slider],
            outputs=[model3d_output, rrd_download, depth_gallery_output],
        )
        file_input.change(
            fn=_load_files,
            inputs=[file_input],
            outputs=[
                image_gallery_output,
                selection_status,
                model3d_output,
                rrd_download,
                depth_gallery_output,
            ],
        )
        image_gallery_output.select(
            fn=_on_select,
            inputs=None,
            outputs=[selection_status],
        )
        set_ref_btn.click(
            fn=_set_reference,
            inputs=None,
            outputs=[
                image_gallery_output,
                selection_status,
                model3d_output,
                rrd_download,
                depth_gallery_output,
            ],
        )
        run_btn.click(
            fn=_run,
            inputs=[max_pts_slider, conf_slider, frustum_scale_slider],
            outputs=[model3d_output, rrd_download, depth_gallery_output],
        )
        update_btn.click(
            fn=_update_viz,
            inputs=[max_pts_slider, conf_slider, frustum_scale_slider],
            outputs=[model3d_output, rrd_download],
        )

    return demo


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

register_configs()


# The composed config is consumed only for `accelerator.mixed_precision`; the
# actual models come from DEFAULT_INFERENCE_MODELS (+ --add-model). Defaulting
# to `dvlt` means no --config-name flag is needed in the common case.
@cli(
    config_path="../config/experiments",
    config_name="dvlt",
    version_base=None,
    extra_args=EXTRA_CLI_ARGS,
)
def main(
    config,
    *,
    max_points: int,
    conf_threshold: float,
    server_port: int,
    share: bool,
    default_model: Optional[str] = None,
    model_specs: Optional[list[str]] = None,
    offline: bool = False,
    inputs: Optional[list[str]] = None,
    models: Optional[str] = None,
    output_root: str = "demo_outputs",
    overwrite: bool = False,
    frustum_scale: float = 0.01,
):
    """Launch Gradio demo, or run headless when ``--offline`` is set."""
    accelerator = Accelerator(mixed_precision=config.trainer.mixed_precision)
    logging.basicConfig(
        format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )

    logger.info(f"Using device: {accelerator.device}")
    logger.info(f"Mixed precision: {config.trainer.mixed_precision}")

    # ---- Build registry from the curated inference list ----
    registry: dict[str, ModelEntry] = {}
    for label, cfg_name in DEFAULT_INFERENCE_MODELS:
        registry[label] = ModelEntry(label=label, config_name=cfg_name)

    # ---- Optional extras from --add-model ----
    for spec in model_specs or []:
        label, cfg_name = parse_model_spec(spec)
        if label in registry:
            raise ValueError(f"Duplicate model label in --add-model: {label!r}")
        logger.info(f"Registering extra model entry '{label}' (config={cfg_name})")
        registry[label] = ModelEntry(label=label, config_name=cfg_name)

    # ---- Offline branch ------------------------------------------------
    if offline:
        if not inputs:
            raise ValueError("--offline requires at least one --input PATH.")
        if not models:
            raise ValueError("--offline requires --models LIST (comma-separated config names, or 'all').")

        by_config: dict[str, ModelEntry] = {
            entry.config_name: entry for entry in registry.values() if entry.config_name
        }
        if models.strip().lower() == "all":
            model_keys = list(by_config)
        else:
            requested = [m.strip() for m in models.split(",") if m.strip()]
            unknown = [m for m in requested if m not in by_config]
            if unknown:
                raise ValueError(f"--models contains unknown config name(s) {unknown}. Available: {sorted(by_config)}")
            model_keys = requested

        input_paths = [Path(p).expanduser().resolve() for p in inputs]
        out_root = Path(output_root).expanduser().resolve()
        logger.info(f"Offline mode: {len(input_paths)} sequence(s) x {len(model_keys)} model(s) -> {out_root}")
        _run_offline(
            accelerator=accelerator,
            registry_by_config=by_config,
            model_keys=model_keys,
            input_paths=input_paths,
            output_root=out_root,
            max_points=max_points,
            conf_threshold=conf_threshold,
            overwrite=overwrite,
            frustum_scale=frustum_scale,
        )
        return

    # ---- Resolve which entry is preselected ----
    if default_model is not None:
        if default_model not in registry:
            raise ValueError(
                f"--default-model {default_model!r} not in registry. Available labels: {list(registry.keys())}"
            )
        default_label = default_model
    else:
        default_label = next(iter(registry))

    # Pre-load only the preselected entry so the first run is snappy without
    # paying the cost of every model in the dropdown.
    logger.info(f"Pre-loading default model '{default_label}'")
    registry[default_label].ensure_loaded(accelerator)

    logger.info(
        f"Launching Gradio on port {server_port} (share={share}) with "
        f"{len(registry)} model(s): {list(registry.keys())}"
    )
    demo = build_app(
        accelerator,
        registry=registry,
        default_label=default_label,
        default_max_points=max_points,
        default_conf_threshold=conf_threshold,
        default_frustum_scale=frustum_scale,
    )
    demo.launch(server_port=server_port, share=share)


if __name__ == "__main__":
    main()
