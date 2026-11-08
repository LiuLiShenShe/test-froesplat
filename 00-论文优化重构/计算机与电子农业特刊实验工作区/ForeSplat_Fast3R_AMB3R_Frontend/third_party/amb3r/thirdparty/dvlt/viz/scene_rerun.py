# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path
from typing import List, Optional, Union, Dict

import numpy as np
import rerun as rr
from torch import Tensor

from dvlt.common.io import normalize_depth, normalize_image, read_depth, read_image_cv2
from dvlt.common.numpy.rotation import mat_to_quat
from dvlt.struct.cameras import Cameras
from dvlt.viz.depth import overlay_depth_map
from dvlt.viz.util import calculate_auto_image_plane_distance


def _compute_shared_pointcloud_indices(points_dict, max_num_points):
    """Pre-compute a single random subsample index set shared across all entries
    in ``points_dict`` so paired pointclouds (e.g. ``pred`` and ``gt``) end up
    with corresponding rows after subsampling.

    Returns ``None`` when there's only one entry, or when entries disagree in
    shape (in which case the caller falls back to independent per-entity
    subsampling, preserving legacy behavior), or when no entry exceeds
    ``max_num_points``.
    """
    values = list(points_dict.values())
    if len(values) <= 1:
        return None

    def _length(x):
        return x.shape[0] if hasattr(x, "shape") else len(x)

    n = _length(values[0])
    for v in values[1:]:
        if _length(v) != n:
            return None
    if n <= max_num_points:
        return None
    return np.random.choice(n, size=max_num_points, replace=False)


def visualize_scene(
    log_path: str,
    cameras: Union[Cameras, Dict[str, Cameras]],
    points: Optional[Union[Tensor, np.ndarray, List[np.ndarray], Dict[str, Union[Tensor, np.ndarray, List[np.ndarray]]]]] = None,
    rgb: Optional[Union[Tensor, np.ndarray, List[np.ndarray], Dict[str, Union[Tensor, np.ndarray, List[np.ndarray]]]]] = None,
    images: Optional[Union[List[Union[str, np.ndarray, Tensor]], Dict[str, List[Union[str, np.ndarray, Tensor]]]]] = None,
    depths: Optional[Union[List[Union[str, np.ndarray, Tensor]], Dict[str, List[Union[str, np.ndarray, Tensor]]]]] = None,
    depth_scale_factor: float = 1.0,
    server_address: Optional[str] = None,
    image_plane_distance: Union[float, str] = "auto",
    image_max_size: int = 0,
    save_path: Optional[str] = None,
    view_coordinates: str = "RFU",
    max_num_points: int = 200_000,
    app_id: Optional[str] = None,
):
    """Visualize a 3D scene using Rerun with cameras, pointclouds, images, and 3D bounding boxes.

    This function creates a comprehensive 3D scene visualization that includes camera poses,
    RGB/depth images, 3D pointclouds, instance segmentation masks, and 3D bounding boxes.
    The visualization is displayed using the Rerun framework and can be viewed in real-time
    or saved to a file for later analysis.

    Args:
        log_path (str): The logging path/name for the Rerun visualization session.

        cameras (Union[Cameras, Dict[str, Cameras]]): Camera objects containing intrinsics and poses.
            Can be a single Cameras object or a dictionary mapping split names to Cameras objects
            (e.g., {"train": cameras_train, "val": cameras_val}).

        points (Union[Optional[Union[Tensor, np.Union[ndarray, List[np.ndarray], Dict[str, ...]]], optional): 3D point positions
            for pointcloud visualization. Shape: (N, 3) for static scenes or (T, N, 3) for dynamic scenes.
            Can also be a dict mapping names to arrays, in which case each is logged under its own entity path
            (e.g., ``{"pred": pts_pred, "gt": pts_gt}``). Defaults to None.

        rgb (Union[Optional[Union[Tensor, np.Union[ndarray, List[np.ndarray], Dict[str, ...]]], optional): Colors for the
            pointcloud points. Shape should match points but with 3 color channels. Must be a dict when
            ``points`` is a dict, with matching keys. Defaults to None.

        images (Union[Optional[List[Union[str, np.Union[ndarray, Tensor], Dict[str, List[Union[str, np.Union[ndarray, Tensor]]]], optional):
            RGB images to display in camera views. Can be file paths, numpy arrays, or tensors.
            If cameras is a dict, this should also be a dict with matching keys. Defaults to None.

        depths (Union[Optional[List[Union[str, np.Union[ndarray, Tensor], Dict[str, List[Union[str, np.Union[ndarray, Tensor]]]], optional):
            Depth images corresponding to the RGB images. Can be file paths, numpy arrays, or tensors.
            If cameras is a dict, this should also be a dict with matching keys. Defaults to None.

        depth_scale_factor (float, optional): Scale factor to apply to depth values. Defaults to 1.0.

        server_address (Optional[str], optional): TCP address of Rerun server to connect to. Defaults to None.

        image_plane_distance (Union[float, str], optional): Distance of image planes from camera centers
            for visualization. If "auto", automatically calculated based on scene size. Defaults to "auto".

        image_max_size (int, optional): Maximum size (width or height) for displayed images.
            If > 0, images will be resized if they exceed this size. Defaults to 0 (no resizing).

        save_path (Optional[str], optional): File path to save the Rerun recording.
            If provided, the visualization will be saved to this location. Defaults to None.

        view_coordinates (str, optional): View coordinates to use for the visualization.
            Defaults to "RFU" (Right-Forward-Up). See rerun.ViewCoordinates for available options.

        max_num_points (int, optional): Maximum number of points to visualize. Defaults to 100_000.

        app_id (Optional[str], optional): Rerun ``application_id`` to use when initializing
            the recording. If ``None`` (default), ``log_path`` is used, preserving the historical
            behavior. Set this to a method- or experiment-qualified name (e.g. ``"da3-G/dtu_scan1"``)
            to make multiple recordings of the same sequence distinguishable in the Rerun viewer's
            recordings sidebar.

    Note:
        - The function automatically handles different input formats (file paths vs arrays/tensors)
        - When using dict inputs, all dict parameters must have matching keys

    Example:
        ```python
        # Simple single-camera visualization
        visualize_scene(
            log_path="my_scene",
            cameras=camera_objects,
            points=pointcloud_positions,
            rgb=pointcloud_colors,
            images=["image1.jpg", "image2.jpg"],
            depths=["depth1.png", "depth2.png"]
        )

        # Multi-split visualization
        visualize_scene(
            log_path="training_data",
            cameras={"train": train_cams, "val": val_cams},
            images={"train": train_images, "val": val_images},
            save_path="scene_recording.rrd"
        )
        ```
    """
    rr.init(app_id if app_id is not None else log_path)
    if server_address is not None:
        if "://" in server_address:
            url = server_address
        else:
            url = f"rerun+http://{server_address}/proxy"
        rr.connect_grpc(url)
    if save_path is not None:
        if os.path.dirname(save_path) != "" and not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        rr.save(save_path)
    rr.set_time_seconds("stable_time", 0)
    rr.log(log_path, getattr(rr.ViewCoordinates, view_coordinates), static=True)

    if image_plane_distance == "auto":
        first_points = next(iter(points.values())) if isinstance(points, dict) else points
        image_plane_distance = calculate_auto_image_plane_distance(cameras, first_points)

    def _add_cameras(cams, ims=None, deps=None, name="cameras"):
        for idx, camera in enumerate(cams):
            camera: Cameras
            width, height = int(camera.width), int(camera.height)
            if image_max_size > 0:
                scale_factor = image_max_size / max((width, height))
                if scale_factor < 1.0:
                    width = int(width * scale_factor)
                    height = int(height * scale_factor)
                    camera.rescale_output_resolution(scale_factor)

            intrinsic = camera.get_intrinsics_matrices().detach().cpu().numpy()
            pose = camera.camera_to_worlds.detach().cpu().numpy()
            rr.log(
                f"{log_path}/{name}/{idx}",
                rr.Transform3D(
                    translation=pose[:3, 3],
                    quaternion=mat_to_quat(pose[:3, :3]),
                    from_parent=False,
                ),
            )
            rr.log(
                f"{log_path}/{name}/{idx}",
                rr.Pinhole(
                    image_from_camera=intrinsic,
                    height=height,
                    width=width,
                    camera_xyz=rr.ViewCoordinates.RDF,
                    image_plane_distance=image_plane_distance,
                ),
            )
            if ims is not None:
                # Handle path vs array/tensor
                if isinstance(ims[idx], (str, Path)):
                    img_data = read_image_cv2(ims[idx])
                    if img_data is None:
                        print(f"Skipping image at {ims[idx]} - could not be loaded")
                        continue
                else:
                    img_data = ims[idx]

                processed_img = normalize_image(img_data, height, width)

                if deps is not None:
                    # Handle path vs array/tensor
                    if isinstance(deps[idx], (str, Path)):
                        depth_data = read_depth(deps[idx], height, width, depth_scale_factor)
                    else:
                        depth_data = deps[idx]

                    depth = normalize_depth(depth_data, height, width, depth_scale_factor)
                    processed_img = overlay_depth_map(processed_img, depth)

                rr.log(
                    f"{log_path}/{name}/{idx}/rgb",
                    rr.Image(processed_img),
                )

    if isinstance(cameras, dict):
        if images is not None:
            assert isinstance(images, dict)
        if depths is not None:
            assert isinstance(depths, dict)
        for split, cams in cameras.items():
            ims = images.get(split) if images is not None else None
            deps = depths.get(split) if depths is not None else None
            _add_cameras(cams, ims, deps, name=split)
    else:
        _add_cameras(cameras, images, depths)

    def _log_pointcloud(pts, colors, entity_prefix, shared_indices=None):
        """Log a single pointcloud under entity_prefix.

        If ``shared_indices`` is provided it is used directly as the subsample,
        which lets paired pointclouds (e.g. ``pred`` and ``gt``) keep their
        per-row correspondence.
        """
        positions = pts.cpu().numpy() if isinstance(pts, Tensor) else pts
        col_np = colors.cpu().numpy() if isinstance(colors, Tensor) else colors

        if shared_indices is not None:
            positions = positions[shared_indices]
            col_np = col_np[shared_indices]
        elif len(positions) > max_num_points:
            indices = np.random.choice(len(positions), size=max_num_points, replace=False)
            positions = positions[indices]
            col_np = col_np[indices]

        rr.log(entity_prefix, rr.Points3D(positions=positions, colors=col_np))

    if points is not None:
        assert rgb is not None
        if isinstance(points, dict):
            assert isinstance(rgb, dict)
            shared_indices = _compute_shared_pointcloud_indices(points, max_num_points)
            for name, pts in points.items():
                _log_pointcloud(pts, rgb[name], f"{log_path}/{name}/pointcloud", shared_indices=shared_indices)
        else:
            _log_pointcloud(points, rgb, f"{log_path}/pointcloud")

    rr.disconnect()
