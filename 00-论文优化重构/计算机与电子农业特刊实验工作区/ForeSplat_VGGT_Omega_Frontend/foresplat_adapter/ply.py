"""Minimal PLY writers (binary, little endian) used by the ForeSplat adapter."""

from __future__ import annotations

import numpy as np


def _vertex_dtype(with_normals: bool, with_quality: bool):
    fields = [("x", "<f4"), ("y", "<f4"), ("z", "<f4")]
    if with_normals:
        fields += [("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4")]
    fields += [("red", "u1"), ("green", "u1"), ("blue", "u1")]
    if with_quality:
        fields += [("quality", "<f4")]
    return np.dtype(fields)


def _ply_header(num_vertices: int, with_normals: bool, with_quality: bool, comments: list[str]) -> bytes:
    lines = ["ply", "format binary_little_endian 1.0"]
    for comment in comments:
        lines.append(f"comment {comment}")
    lines.append(f"element vertex {num_vertices}")
    lines += ["property float x", "property float y", "property float z"]
    if with_normals:
        lines += ["property float nx", "property float ny", "property float nz"]
    lines += ["property uchar red", "property uchar green", "property uchar blue"]
    if with_quality:
        lines.append("property float quality")
    lines.append("end_header")
    return ("\n".join(lines) + "\n").encode("ascii")


def write_point_cloud_ply(
    path: str,
    vertices: np.ndarray,
    colors: np.ndarray,
    normals: np.ndarray | None = None,
    qualities: np.ndarray | None = None,
    comments: list[str] | None = None,
) -> int:
    """Write a binary PLY point cloud.  ``vertices`` (N,3), ``colors`` uint8 (N,3)."""
    vertices = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    num = vertices.shape[0]
    colors = np.asarray(colors).reshape(-1, 3)
    if colors.dtype != np.uint8:
        colors = np.clip(colors, 0, 255).astype(np.uint8)

    with_normals = normals is not None
    with_quality = qualities is not None
    dtype = _vertex_dtype(with_normals, with_quality)
    data = np.zeros(num, dtype=dtype)
    data["x"] = vertices[:, 0]
    data["y"] = vertices[:, 1]
    data["z"] = vertices[:, 2]
    if with_normals:
        normals = np.asarray(normals, dtype=np.float64).reshape(-1, 3)
        data["nx"] = normals[:, 0]
        data["ny"] = normals[:, 1]
        data["nz"] = normals[:, 2]
    data["red"] = colors[:, 0]
    data["green"] = colors[:, 1]
    data["blue"] = colors[:, 2]
    if with_quality:
        data["quality"] = np.asarray(qualities, dtype=np.float32).reshape(-1)

    with open(path, "wb") as handle:
        handle.write(_ply_header(num, with_normals, with_quality, comments or []))
        handle.write(data.tobytes())
    return num
