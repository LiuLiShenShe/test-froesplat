"""Guarded geometry primitives for Phase20F.

The functions deliberately require calibrated inputs for physical traits and
explicit organ identities for leaf measurements.
"""

from __future__ import annotations

import numpy as np


class MeasurementBlocked(ValueError):
    """Raised when the evidence required for a measurement is absent."""


def _points(value: np.ndarray, name: str, minimum: int = 1) -> np.ndarray:
    points = np.asarray(value, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < minimum:
        raise MeasurementBlocked(f"{name}_MUST_BE_Nx3_WITH_AT_LEAST_{minimum}_POINTS")
    if not np.isfinite(points).all():
        raise MeasurementBlocked(f"{name}_CONTAINS_NONFINITE_COORDINATES")
    return points


def normalize_axis(axis: np.ndarray, name: str = "axis") -> np.ndarray:
    vector = np.asarray(axis, dtype=np.float64)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise MeasurementBlocked(f"{name.upper()}_MUST_BE_A_FINITE_3_VECTOR")
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise MeasurementBlocked(f"{name.upper()}_MUST_BE_NONZERO")
    return vector / norm


def metric_length(value_units: float, scale_cm_per_unit: float | None,
                  scale_source: str | None) -> float:
    if scale_cm_per_unit is None or not scale_source:
        raise MeasurementBlocked("PHYSICAL_SCALE_NOT_AVAILABLE")
    scale = float(scale_cm_per_unit)
    value = float(value_units)
    if not np.isfinite(scale) or scale <= 0 or not np.isfinite(value) or value < 0:
        raise MeasurementBlocked("INVALID_SCALE_OR_LENGTH")
    return value * scale


def height_along_axis(plant_points: np.ndarray, base_point: np.ndarray,
                      vertical_axis: np.ndarray) -> float:
    points = _points(plant_points, "plant_points")
    base = np.asarray(base_point, dtype=np.float64)
    if base.shape != (3,) or not np.isfinite(base).all():
        raise MeasurementBlocked("BASE_POINT_MUST_BE_A_FINITE_3_VECTOR")
    vertical = normalize_axis(vertical_axis, "vertical_axis")
    return float(np.max((points - base) @ vertical))


def canopy_spans(plant_points: np.ndarray, vertical_axis: np.ndarray,
                 horizontal_axis_a: np.ndarray,
                 horizontal_axis_b: np.ndarray) -> tuple[float, float, float]:
    points = _points(plant_points, "plant_points", minimum=2)
    vertical = normalize_axis(vertical_axis, "vertical_axis")
    axis_a = normalize_axis(horizontal_axis_a, "horizontal_axis_a")
    axis_b = normalize_axis(horizontal_axis_b, "horizontal_axis_b")
    basis = np.stack([vertical, axis_a, axis_b])
    if not np.allclose(basis @ basis.T, np.eye(3), atol=1e-6):
        raise MeasurementBlocked("CANOPY_AXES_MUST_BE_MUTUALLY_ORTHOGONAL")
    projections = points @ basis.T
    spans = np.ptp(projections, axis=0)
    return float(spans[1]), float(spans[2]), float(max(spans[1], spans[2]))


def leaf_length(leaf_id: str | None, base_point: np.ndarray,
                tip_point: np.ndarray, method: str = "BASE_TO_TIP_STRAIGHT") -> float:
    if not leaf_id or leaf_id in {"LEAF_ID_NOT_AVAILABLE", "UNKNOWN", "NONE"}:
        raise MeasurementBlocked("LEAF_ID_NOT_AVAILABLE")
    if method != "BASE_TO_TIP_STRAIGHT":
        raise MeasurementBlocked("UNSUPPORTED_OR_UNDECLARED_LEAF_LENGTH_METHOD")
    base = np.asarray(base_point, dtype=np.float64)
    tip = np.asarray(tip_point, dtype=np.float64)
    if base.shape != (3,) or tip.shape != (3,) or not np.isfinite([base, tip]).all():
        raise MeasurementBlocked("LEAF_BASE_AND_TIP_MUST_BE_FINITE_3_VECTORS")
    length = float(np.linalg.norm(tip - base))
    if length <= 0:
        raise MeasurementBlocked("LEAF_LENGTH_MUST_BE_POSITIVE")
    return length


def leaf_width(leaf_id: str | None, boundary_points: np.ndarray,
               width_axis: np.ndarray) -> float:
    if not leaf_id or leaf_id in {"LEAF_ID_NOT_AVAILABLE", "UNKNOWN", "NONE"}:
        raise MeasurementBlocked("LEAF_ID_NOT_AVAILABLE")
    points = _points(boundary_points, "boundary_points", minimum=2)
    axis = normalize_axis(width_axis, "width_axis")
    return float(np.ptp(points @ axis))


def post_vertices_subset_of_raw(raw_vertices: np.ndarray,
                                post_vertices: np.ndarray) -> bool:
    raw = np.ascontiguousarray(_points(raw_vertices, "raw_vertices"))
    post = np.ascontiguousarray(_points(post_vertices, "post_vertices"))
    if raw.dtype != post.dtype:
        post = post.astype(raw.dtype, copy=False)
    key_dtype = np.dtype((np.void, raw.dtype.itemsize * 3))
    raw_keys = raw.view(key_dtype).reshape(-1)
    post_keys = post.view(key_dtype).reshape(-1)
    return bool(np.isin(post_keys, np.unique(raw_keys), assume_unique=False).all())


def pca_extents(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return center, descending PCA directions, and projected axis ranges."""
    vertices = _points(points, "points", minimum=3)
    center = vertices.mean(axis=0)
    centered = vertices - center
    covariance = centered.T @ centered / len(vertices)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    directions = eigenvectors[:, np.argsort(eigenvalues)[::-1]]
    projected = centered @ directions
    extents = np.ptp(projected, axis=0)
    return center, directions, extents
