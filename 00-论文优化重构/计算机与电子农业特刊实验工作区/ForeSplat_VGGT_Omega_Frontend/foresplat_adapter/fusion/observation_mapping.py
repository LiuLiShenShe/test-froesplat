"""Observation provenance for the fused point cloud.

Every fused point is an aggregate of one or more per-view observations.  This
module builds that one-to-many mapping and stores it compactly, so any fused
point can be traced back to the exact source image and source pixel it came
from (required by the ForeSplat full-sequence export).

Stored per observation:

========================  ==========================================
field                     meaning
========================  ==========================================
``fused_point_id``        index into the fused cloud / points3D.ply
``source_image_id``       COLMAP image id (1-based, same as export)
``view_index``            zero-based position in the processed sequence
``source_pixel_index``    flattened pixel index (``y * W + x``, model res)
``source_confidence``     VGGT-Omega depth confidence at that pixel
``source_weight``         fusion weight (confidence x quality x edge boost)
``source_normal``         normal at that pixel (3 x float16)
``source_is_edge``        pixel lies on a depth discontinuity
========================  ==========================================
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "build_observation_table",
    "save_observations_npz",
    "load_observations_npz",
    "pixel_index_to_xy",
    "observations_for_point",
    "export_observations_csv",
]


def build_observation_table(
    fused_point_id: np.ndarray,
    source_image_id: np.ndarray,
    view_index: np.ndarray,
    source_pixel: np.ndarray,
    source_confidence: np.ndarray,
    source_normal: np.ndarray,
    source_edge: np.ndarray,
    source_weight: np.ndarray,
    num_fused_points: int,
) -> dict:
    """Assemble the (observation-level) provenance arrays."""
    fused_point_id = np.asarray(fused_point_id, dtype=np.int64)
    if fused_point_id.size and (fused_point_id.min() < 0 or fused_point_id.max() >= num_fused_points):
        raise ValueError("fused_point_id out of range for the fused cloud")
    return {
        "fused_point_id": fused_point_id,
        "source_image_id": np.asarray(source_image_id, dtype=np.int32),
        "view_index": np.asarray(view_index, dtype=np.int32),
        "source_pixel_index": np.asarray(source_pixel, dtype=np.int64),
        "source_confidence": np.asarray(source_confidence, dtype=np.float32),
        "source_weight": np.asarray(source_weight, dtype=np.float32),
        "source_normal": np.asarray(source_normal, dtype=np.float16),
        "source_is_edge": np.asarray(source_edge, dtype=bool),
    }


def save_observations_npz(path: str, observations: dict, meta: dict | None = None) -> None:
    """Save the provenance table as a compressed ``.npz`` (plus ``meta`` arrays)."""
    payload = dict(observations)
    if meta:
        for key, value in meta.items():
            payload[f"meta__{key}"] = np.asarray(value)
    np.savez_compressed(path, **payload)


def load_observations_npz(path: str) -> tuple[dict, dict]:
    """Load a provenance table, splitting ``meta__*`` entries back out."""
    data = np.load(path, allow_pickle=False)
    observations = {}
    meta = {}
    for key in data.files:
        if key.startswith("meta__"):
            meta[key[len("meta__") :]] = data[key]
        else:
            observations[key] = data[key]
    return observations, meta


def pixel_index_to_xy(pixel_index: np.ndarray, width: int) -> tuple[np.ndarray, np.ndarray]:
    """Flattened pixel index -> (x, y) at the given image width."""
    idx = np.asarray(pixel_index)
    return (idx % width).astype(np.int32), (idx // width).astype(np.int32)


def observations_for_point(observations: dict, point_id: int) -> dict:
    """All observations of one fused point."""
    mask = observations["fused_point_id"] == int(point_id)
    return {key: value[mask] for key, value in observations.items()}


def export_observations_csv(path: str, observations: dict, max_rows: int | None = None) -> int:
    """Write the provenance table to CSV (bounded by ``max_rows``); returns rows."""
    import csv

    n = observations["fused_point_id"].size
    if max_rows is not None:
        n = min(n, int(max_rows))
    columns = [
        "fused_point_id", "source_image_id", "view_index", "source_pixel_index",
        "source_confidence", "source_weight", "source_is_edge",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for i in range(n):
            writer.writerow([int(observations[c][i]) if np.issubdtype(observations[c].dtype, np.integer)
                             else float(observations[c][i]) for c in columns])
    return n