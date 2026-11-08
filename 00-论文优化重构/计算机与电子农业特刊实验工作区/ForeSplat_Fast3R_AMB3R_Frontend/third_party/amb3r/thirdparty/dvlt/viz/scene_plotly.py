# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
import numpy as np
import plotly.graph_objects as go

from dvlt.common.pose import to4x4
from dvlt.struct.cameras import Cameras
from dvlt.viz.util import view_matrix_from_string


def _frustum_lines_plotly(
    K: np.ndarray, extr_c2w: np.ndarray, base_scale: float = 0.2, reference_focal_length: float = 1000.0
) -> tuple[list[float], list[float], list[float]]:
    """Return x, y, z coordinates for frustum lines for Plotly.

    Frustum size is scaled based on focal length to visualize field of view differences.
    Longer focal length = smaller frustum (narrower FOV).
    Shorter focal length = larger frustum (wider FOV).

    Returns:
        tuple: (x_coords, y_coords, z_coords) where each is a list containing
               coordinates for all frustum lines with None separators
    """
    cx, cy = K[0, 2], K[1, 2]
    fx, fy = K[0, 0], K[1, 1]
    img_w, img_h = 2 * cx, 2 * cy
    corners_pix = np.array([[0, 0, 1], [img_w, 0, 1], [img_w, img_h, 1], [0, img_h, 1]]).T  # (3,4)
    Kinv = np.linalg.inv(K)
    dirs = Kinv @ corners_pix  # (3,4)

    # Scale based on focal length - use average focal length and reference of 1000
    avg_focal_length = (fx + fy) / 2
    focal_scale = reference_focal_length / avg_focal_length
    scale = base_scale * focal_scale

    dirs = dirs / np.linalg.norm(dirs, axis=0, keepdims=True) * scale
    # Accept 3x4 [R|t] or 4x4 matrices
    if extr_c2w.shape == (3, 4):
        R = extr_c2w[:, :3]
        c = extr_c2w[:, 3]
    else:  # assume 4x4
        R = extr_c2w[:3, :3]
        c = extr_c2w[:3, 3]
    corners_w = (R @ dirs).T + c  # (4,3)

    # Create line coordinates (center to corners + connecting corners)
    x_coords, y_coords, z_coords = [], [], []

    # Lines from camera center to corners
    for corner in corners_w:
        x_coords.extend([c[0], corner[0], None])
        y_coords.extend([c[1], corner[1], None])
        z_coords.extend([c[2], corner[2], None])

    # Lines connecting corners (forming rectangle)
    for i in range(4):
        corner1 = corners_w[i]
        corner2 = corners_w[(i + 1) % 4]
        x_coords.extend([corner1[0], corner2[0], None])
        y_coords.extend([corner1[1], corner2[1], None])
        z_coords.extend([corner1[2], corner2[2], None])

    return x_coords, y_coords, z_coords


def scene_to_plotly(
    seq_name: str,
    pts_pred: np.ndarray,
    pred_rgb: np.ndarray,
    cameras_pred: Cameras,
    pts_gt: np.ndarray | None = None,
    gt_rgb: np.ndarray | None = None,
    cameras_gt: Cameras | None = None,
    view_coordinates: str = "RFU",
    alpha: float = 0.3,
) -> go.Figure:
    """Create Plotly 3D figure with point clouds + camera frustums.

    Args:
        seq_name: Name of the sequence.
        pts_pred: Predicted point cloud coordinates, shape (N, 3)
        pts_gt: Ground truth point cloud coordinates, shape (N, 3)
        pred_rgb: RGB uint8 colors for predicted points, shape (N, 3)
        gt_rgb: RGB uint8 colors for ground truth points, shape (N, 3)
        cameras_pred: Predicted Cameras object containing multiple cameras
        cameras_gt: Ground truth Cameras object containing multiple cameras
        view_coordinates: Coordinate convention string (e.g. "RFU", "RDF") that
            defines the positive directions of the displayed x, y, z axes.
        alpha: Alpha value for blending red/green with predicted/ground truth points.

    Returns:
        Plotly Figure object with 3D visualization
    """

    # Extract intrinsics (do not depend on coordinate system)
    intrinsics_pred = cameras_pred.get_intrinsics_matrices().detach().cpu().numpy()
    extrinsics_pred = to4x4(cameras_pred.camera_to_worlds).detach().cpu().numpy()

    # Convert to 4x4 matrices using helper function, then to numpy
    if cameras_gt is not None:
        extrinsics_gt = to4x4(cameras_gt.camera_to_worlds).detach().cpu().numpy()
        valid_cameras = extrinsics_gt[:3, :3].any()
        extrinsics_gt = extrinsics_gt if valid_cameras else []
        intrinsics_gt = cameras_gt.get_intrinsics_matrices().detach().cpu().numpy()
        intrinsics_gt = intrinsics_gt if valid_cameras else []
    else:
        extrinsics_gt = []
        intrinsics_gt = []

    # 2) Apply view coordinate transform to points and extrinsics
    M = view_matrix_from_string(view_coordinates)

    pts_pred = (M @ pts_pred.T).T
    if pts_gt is not None:
        pts_gt = (M @ pts_gt.T).T

    for i in range(len(extrinsics_pred)):
        extrinsics_pred[i, :3, :3] = M @ extrinsics_pred[i, :3, :3]
        extrinsics_pred[i, :3, 3] = M @ extrinsics_pred[i, :3, 3]

    for i in range(len(extrinsics_gt)):
        extrinsics_gt[i, :3, :3] = M @ extrinsics_gt[i, :3, :3]
        extrinsics_gt[i, :3, 3] = M @ extrinsics_gt[i, :3, 3]

    if pts_gt is not None:
        # Build coloured point arrays (blend with red / green)
        red = np.array([[255, 0, 0]], dtype=np.float32)
        green = np.array([[0, 255, 0]], dtype=np.float32)
        pred_rgb = (pred_rgb * (1 - alpha) + red * alpha).astype(np.uint8)
        gt_rgb = (gt_rgb * (1 - alpha) + green * alpha).astype(np.uint8)

    # Create figure
    fig = go.Figure()

    # Add predicted point cloud (already transformed)
    fig.add_trace(
        go.Scatter3d(
            x=pts_pred[:, 0],
            y=pts_pred[:, 1],
            z=pts_pred[:, 2],
            mode="markers",
            marker=dict(size=2, color=[f"rgb({int(r)},{int(g)},{int(b)})" for r, g, b in pred_rgb], opacity=0.8),
            name="Predicted Points",
            hovertemplate="<b>Predicted</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<extra></extra>",
        )
    )

    # Add ground truth point cloud
    if pts_gt is not None:
        fig.add_trace(
            go.Scatter3d(
                x=pts_gt[:, 0],
                y=pts_gt[:, 1],
                z=pts_gt[:, 2],
                mode="markers",
                marker=dict(size=2, color=[f"rgb({int(r)},{int(g)},{int(b)})" for r, g, b in gt_rgb], opacity=0.8),
                name="Ground Truth Points",
                hovertemplate="<b>Ground Truth</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<extra></extra>",
            )
        )

    # Add camera frustums
    mean_pred_x = intrinsics_pred[:, 0, 0].mean()
    mean_pred_y = intrinsics_pred[:, 1, 1].mean()
    mean_gt_x = intrinsics_gt[:, 0, 0].mean() if len(intrinsics_gt) > 0 else mean_pred_x
    mean_gt_y = intrinsics_gt[:, 1, 1].mean() if len(intrinsics_gt) > 0 else mean_pred_y
    avg_focal_length_x = (mean_pred_x + mean_gt_x) / 2
    avg_focal_length_y = (mean_pred_y + mean_gt_y) / 2
    avg_focal_length = (avg_focal_length_x + avg_focal_length_y) / 2

    # Predicted camera frustums (red)
    pred_x_all, pred_y_all, pred_z_all = [], [], []
    for K, E in zip(intrinsics_pred, extrinsics_pred):
        x_coords, y_coords, z_coords = _frustum_lines_plotly(K, E, reference_focal_length=avg_focal_length)
        pred_x_all.extend(x_coords)
        pred_y_all.extend(y_coords)
        pred_z_all.extend(z_coords)

    if pred_x_all:  # Only add if there are cameras
        fig.add_trace(
            go.Scatter3d(
                x=pred_x_all,
                y=pred_y_all,
                z=pred_z_all,
                mode="lines",
                line=dict(color="red", width=3),
                name="Predicted Cameras",
                hoverinfo="skip",
            )
        )

    # Ground truth camera frustums (green)
    gt_x_all, gt_y_all, gt_z_all = [], [], []
    for K, E in zip(intrinsics_gt, extrinsics_gt):
        x_coords, y_coords, z_coords = _frustum_lines_plotly(K, E, reference_focal_length=avg_focal_length)
        gt_x_all.extend(x_coords)
        gt_y_all.extend(y_coords)
        gt_z_all.extend(z_coords)

    if gt_x_all:  # Only add if there are cameras
        fig.add_trace(
            go.Scatter3d(
                x=gt_x_all,
                y=gt_y_all,
                z=gt_z_all,
                mode="lines",
                line=dict(color="green", width=3),
                name="Ground Truth Cameras",
                hoverinfo="skip",
            )
        )

    # Configure layout for better 3D visualization
    fig.update_layout(
        title=f"3D Scene: Point Clouds and Camera Poses - {seq_name}",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data",
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5),
                up=dict(x=0, y=0, z=1),
            ),
        ),
        width=800,
        height=600,
        showlegend=True,
        legend=dict(x=0, y=1, bgcolor="rgba(255, 255, 255, 0.8)"),
    )
    return fig
