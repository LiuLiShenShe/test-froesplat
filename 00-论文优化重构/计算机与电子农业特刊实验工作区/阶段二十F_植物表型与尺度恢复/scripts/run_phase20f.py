#!/usr/bin/env python3
"""Run the Phase20F frozen-M0 geometry and phenotype evidence audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from PIL import Image

from phenotype_core import pca_extents, post_vertices_subset_of_raw


SCENES = {
    "WanNianQing1": {
        "group": "B_R0",
        "image_root": "WanNianQing1/B_R0/images_rgba",
        "aliases": ("万年青", "wanniannqing", "wannianqing"),
        "visual_note": "Leaf canopy and pot are present in source RGB; plant-only segmentation is not supplied.",
    },
    "CaoMei2": {
        "group": "B_R0",
        "image_root": "CaoMei2/B_R0/images_rgba",
        "aliases": ("草莓", "caomei", "strawberry"),
        "visual_note": "Leaves and pot are present in source RGB; the camera-oriented plant appears strongly tilted; plant-only segmentation is not supplied.",
    },
    "HongZhang": {
        "group": "C_R1",
        "image_root": "HongZhang/C_R1/images_rgba",
        "aliases": ("红掌", "hongzhang", "anthurium"),
        "visual_note": "Leaves, inflorescences, and pot are present in source RGB; plant-only segmentation is not supplied.",
    },
}

TRAITS = ("plant_height", "canopy_width", "leaf_length", "leaf_width")


def repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / ".git").exists() and (parent / "植株数据.xlsx").exists():
            return parent
    raise RuntimeError("REPOSITORY_ROOT_NOT_FOUND")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=fields,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def json_cell(value) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"))


def compact_float(value: float) -> str:
    """Keep component audit coordinates readable without repeating long floats."""
    return format(float(value), ".6g")


def vector_bbox(points: np.ndarray) -> tuple[list[float], list[float]]:
    return points.min(axis=0).astype(float).tolist(), points.max(axis=0).astype(float).tolist()


def sample_indices(size: int, maximum: int) -> np.ndarray:
    if size <= maximum:
        return np.arange(size, dtype=np.int64)
    return np.linspace(0, size - 1, maximum, dtype=np.int64)


def read_rgba(image_path: Path, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    with Image.open(image_path) as source:
        rgba = source.convert("RGBA")
        if rgba.size != (width, height):
            rgba = rgba.resize((width, height), Image.Resampling.LANCZOS)
        array = np.asarray(rgba)
    return array[..., :3], array[..., 3]


def project_points(points: np.ndarray, camera: dict) -> tuple[np.ndarray, np.ndarray]:
    transform = np.asarray(camera["world_to_camera"], dtype=np.float64)
    camera_points = points @ transform[:3, :3].T + transform[:3, 3]
    z = camera_points[:, 2]
    fx = camera["width"] / (2.0 * math.tan(float(camera["FoVx"]) / 2.0))
    fy = camera["height"] / (2.0 * math.tan(float(camera["FoVy"]) / 2.0))
    x = fx * camera_points[:, 0] / np.where(z == 0, np.nan, z) + (camera["width"] - 1) / 2.0
    y = fy * camera_points[:, 1] / np.where(z == 0, np.nan, z) + (camera["height"] - 1) / 2.0
    pixels = np.column_stack((x, y))
    visible = (
        (z > 0)
        & np.isfinite(pixels).all(axis=1)
        & (x >= 0)
        & (x < camera["width"])
        & (y >= 0)
        & (y < camera["height"])
    )
    return pixels, visible


def component_analysis(scene: str, mesh: o3d.geometry.TriangleMesh,
                       post_vertex_count: int, raw_sha: str, post_sha: str,
                       camera_records: list[dict], image_root: Path,
                       component_rows: list[dict], projection_rows: list[dict],
                       labels: np.ndarray, face_counts: np.ndarray) -> dict:
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    labels = np.asarray(labels, dtype=np.int32)
    face_counts = np.asarray(face_counts, dtype=np.int64)
    if len(labels) != len(faces) or len(face_counts) == 0:
        raise RuntimeError(f"COMPONENT_ANALYSIS_FAILED:{scene}")
    largest_id = int(np.argmax(face_counts))
    if int(face_counts[largest_id]) < 50:
        raise RuntimeError(f"NO_VALID_RETAINED_COMPONENT:{scene}")

    tri = faces.astype(np.int64, copy=False)
    centroids = vertices[tri].mean(axis=1)
    order = np.argsort(labels, kind="stable")
    ordered_labels = labels[order]
    starts = np.r_[0, np.flatnonzero(np.diff(ordered_labels)) + 1]
    component_ids = ordered_labels[starts]
    component_counts = np.diff(np.r_[starts, len(order)])
    sorted_centroids = centroids[order]
    component_means = np.add.reduceat(sorted_centroids, starts, axis=0) / component_counts[:, None]
    component_mins = np.minimum.reduceat(sorted_centroids, starts, axis=0)
    component_maxs = np.maximum.reduceat(sorted_centroids, starts, axis=0)

    for idx, component_id in enumerate(component_ids):
        if int(component_id) == largest_id:
            continue
        component_rows.append({
            "record_type": "REMOVED_COMPONENT",
            "sample_id": scene,
            "component_id": int(component_id),
            "retained_by_post": False,
            "triangle_count": int(component_counts[idx]),
            "triangle_fraction_of_raw": compact_float(component_counts[idx] / len(faces)),
            "centroid_x": compact_float(component_means[idx][0]),
            "centroid_y": compact_float(component_means[idx][1]),
            "centroid_z": compact_float(component_means[idx][2]),
            "bbox_min_x": compact_float(component_mins[idx][0]),
            "bbox_min_y": compact_float(component_mins[idx][1]),
            "bbox_min_z": compact_float(component_mins[idx][2]),
            "bbox_max_x": compact_float(component_maxs[idx][0]),
            "bbox_max_y": compact_float(component_maxs[idx][1]),
            "bbox_max_z": compact_float(component_maxs[idx][2]),
            "trace_note": "Coordinates summarize triangle centroids; SUMMARY row stores source hashes.",
        })

    deleted_faces = np.flatnonzero(labels != largest_id)
    removed_vertices = max(0, len(vertices) - post_vertex_count)
    summary = {
        "scene": scene,
        "vertices": len(vertices),
        "faces": len(faces),
        "component_count": len(face_counts),
        "largest_component_id": largest_id,
        "largest_component_faces": int(face_counts[largest_id]),
        "removed_component_count": int(len(face_counts) - 1),
        "removed_faces": int(len(deleted_faces)),
        "removed_vertices": removed_vertices,
        "removed_face_fraction": float(len(deleted_faces) / len(faces)),
        "removed_vertex_fraction": float(removed_vertices / len(vertices)),
        "component_labels": labels,
        "triangle_centroids": centroids,
        "deleted_faces": deleted_faces,
    }
    component_rows.append({
        "record_type": "SUMMARY",
        "sample_id": scene,
        "component_id": "SUMMARY",
        "retained_by_post": "largest connected triangle component",
        "triangle_count": int(len(deleted_faces)),
        "triangle_fraction_of_raw": compact_float(summary["removed_face_fraction"]),
        "removed_component_count": summary["removed_component_count"],
        "raw_vertex_count": len(vertices),
        "post_vertex_count": post_vertex_count,
        "removed_vertex_count": removed_vertices,
        "removed_vertex_fraction": compact_float(summary["removed_vertex_fraction"]),
        "raw_face_count": len(faces),
        "post_face_count": int(face_counts[largest_id]),
        "component_count": summary["component_count"],
        "largest_component_id": largest_id,
        "largest_component_faces": int(face_counts[largest_id]),
        "raw_mesh_sha256": raw_sha,
        "post_mesh_sha256": post_sha,
        "trace_note": "All non-largest face-connected components are retained; no size-based noise assumption.",
    })

    if len(deleted_faces):
        selected = deleted_faces[sample_indices(len(deleted_faces), 120_000)]
        points = centroids[selected]
        for camera in camera_records:
            image_path = image_root / f"{camera['camera']}.png"
            if not image_path.is_file():
                projection_rows.append({
                    "sample_id": scene,
                    "camera": camera["camera"],
                    "image_path": str(image_path),
                    "status": "FROZEN_RGB_NOT_FOUND",
                })
                continue
            _, alpha = read_rgba(image_path, camera["width"], camera["height"])
            pixels, visible = project_points(points, camera)
            rounded = np.rint(pixels[visible]).astype(np.int64)
            if len(rounded):
                rounded[:, 0] = np.clip(rounded[:, 0], 0, camera["width"] - 1)
                rounded[:, 1] = np.clip(rounded[:, 1], 0, camera["height"] - 1)
                foreground = alpha[rounded[:, 1], rounded[:, 0]] >= 128
                foreground_fraction = float(foreground.mean())
            else:
                foreground_fraction = None
            projection_rows.append({
                "sample_id": scene,
                "camera": camera["camera"],
                "image_path": str(image_path),
                "removed_face_centroid_sample_n": len(points),
                "projected_inside_image_n": int(visible.sum()),
                "projected_inside_image_fraction": float(visible.mean()),
                "projected_centroids_on_source_alpha_foreground_fraction": foreground_fraction,
                "status": "PROJECTED_USING_FROZEN_CAMERA",
                "camera_world_to_camera": json_cell(camera["world_to_camera"]),
            })
    return summary


def add_component_geometry_figure(scene_data: dict, out_path: Path) -> None:
    fig = plt.figure(figsize=(15, 5.8), constrained_layout=True)
    colors = ["#d1495b", "#edae49", "#00798c", "#6a4c93", "#f07167", "#70a288", "#8d99ae", "#f2cc8f"]
    scene = scene_data["scene"]
    raw_v, raw_f = scene_data["raw_vertices"], scene_data["raw_faces"]
    post_v, post_f = scene_data["post_vertices"], scene_data["post_faces"]
    labels, counts, retained = scene_data["labels"], scene_data["counts"], scene_data["largest_id"]
    combined = np.vstack((raw_v, post_v))
    low, high = vector_bbox(combined)
    center = (np.asarray(low) + np.asarray(high)) / 2.0
    span = np.asarray(high) - np.asarray(low)
    limit = max(float(span.max()) * 0.56, 1e-4)
    raw_ax = fig.add_subplot(1, 3, 1, projection="3d")
    post_ax = fig.add_subplot(1, 3, 2, projection="3d")
    removed_ax = fig.add_subplot(1, 3, 3, projection="3d")
    axes = [raw_ax, post_ax, removed_ax]
    centers_raw = raw_v[raw_f].mean(axis=1)
    top_removed = [int(x) for x in np.argsort(counts)[::-1] if int(x) != retained][:8]
    categories = [(retained, "#2a9d8f")]
    categories += [(cid, colors[i % len(colors)]) for i, cid in enumerate(top_removed)]
    categories.append((-1, "#a8a8a8"))
    raw_indices = sample_indices(len(raw_f), 70_000)
    for cid, color in categories:
        if cid == -1:
            ids = raw_indices[labels[raw_indices] != retained]
            if top_removed:
                ids = ids[~np.isin(labels[ids], top_removed)]
        else:
            ids = raw_indices[labels[raw_indices] == cid]
        if len(ids):
            raw_ax.scatter(*centers_raw[ids].T, s=0.12, c=color, alpha=0.55, depthshade=False)
    post_centers = post_v[post_f].mean(axis=1)
    post_indices = sample_indices(len(post_centers), 70_000)
    post_ax.scatter(*post_centers[post_indices].T, s=0.12, c="#2a9d8f", alpha=0.65, depthshade=False)
    for cid, color in categories[1:]:
        if cid == -1:
            ids = raw_indices[labels[raw_indices] != retained]
            if top_removed:
                ids = ids[~np.isin(labels[ids], top_removed)]
        else:
            ids = raw_indices[labels[raw_indices] == cid]
        if len(ids):
            removed_ax.scatter(*centers_raw[ids].T, s=0.18, c=color, alpha=0.68, depthshade=False)
    for ax, title in zip(axes, ("Raw mesh", "Post mesh", "Removed components")):
        ax.set_title(title)
        ax.set_xlim(center[0] - limit, center[0] + limit)
        ax.set_ylim(center[1] - limit, center[1] + limit)
        ax.set_zlim(center[2] - limit, center[2] + limit)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=20, azim=-60)
        ax.set_xlabel("COLMAP X")
        ax.set_ylabel("COLMAP Y")
        ax.set_zlabel("COLMAP Z")
    fig.suptitle(f"{scene} M0 | fixed equal-axis view | no gravity axis assumed")
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def add_rgb_projection_figure(scene_data: dict, image_root: Path,
                              camera_records: list[dict], out_path: Path) -> None:
    raw_v, raw_f = scene_data["raw_vertices"], scene_data["raw_faces"]
    labels, retained = scene_data["labels"], scene_data["largest_id"]
    centers = raw_v[raw_f].mean(axis=1)
    deleted = np.flatnonzero(labels != retained)
    indices = deleted[sample_indices(len(deleted), 55_000)] if len(deleted) else deleted
    points, labels_sample = centers[indices], labels[indices]
    counts = scene_data["counts"]
    top_removed = [int(x) for x in np.argsort(counts)[::-1] if int(x) != retained][:8]
    palette = ["#d1495b", "#edae49", "#00798c", "#6a4c93", "#f07167", "#70a288", "#8d99ae", "#f2cc8f"]
    fig, axes = plt.subplots(1, len(camera_records), figsize=(5 * len(camera_records), 9), squeeze=False)
    for ax, camera in zip(axes[0], camera_records):
        image_path = image_root / f"{camera['camera']}.png"
        if not image_path.is_file():
            ax.text(0.5, 0.5, f"missing RGB: {camera['camera']}", ha="center", va="center")
            ax.axis("off")
            continue
        rgb, _ = read_rgba(image_path, camera["width"], camera["height"])
        ax.imshow(rgb)
        pixels, visible = project_points(points, camera)
        for index, cid in enumerate(top_removed):
            mask = visible & (labels_sample == cid)
            if np.any(mask):
                ax.scatter(pixels[mask, 0], pixels[mask, 1], s=1.7,
                           c=palette[index % len(palette)], alpha=0.68, linewidths=0,
                           label=f"component {cid} ({counts[cid]} faces)")
        other = visible & ~np.isin(labels_sample, top_removed)
        if np.any(other):
            ax.scatter(pixels[other, 0], pixels[other, 1], s=1.1,
                       c="#ededed", alpha=0.45, linewidths=0, label="other removed components")
        ax.set_title(f"Frozen RGB {camera['camera']}\n{int(visible.sum())}/{len(visible)} projected centroids in frame")
        ax.set_axis_off()
    handles, labels_legend = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels_legend, loc="lower center", ncol=3, fontsize=7)
    fig.suptitle(f"{scene_data['scene']} M0 | removed raw face-centroid samples projected with frozen camera poses")
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def historical_matching(workbook: Path) -> list[dict]:
    ns = {
        "main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
        "rel": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
        "pkg": "http://schemas.openxmlformats.org/package/2006/relationships",
    }
    with zipfile.ZipFile(workbook) as archive:
        workbook_xml = ElementTree.fromstring(archive.read("xl/workbook.xml"))
        sheet = workbook_xml.find("main:sheets/main:sheet", ns)
        if sheet is None:
            raise RuntimeError("XLSX_HAS_NO_WORKSHEET")
        sheet_name = sheet.attrib["name"]
        relation_id = sheet.attrib[f"{{{ns['rel']}}}id"]
        relationships = ElementTree.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        relation = next(item for item in relationships.findall("pkg:Relationship", ns)
                        if item.attrib["Id"] == relation_id)
        target = relation.attrib["Target"].lstrip("/")
        sheet_path = target if target.startswith("xl/") else f"xl/{target}"
        strings = []
        if "xl/sharedStrings.xml" in archive.namelist():
            shared_xml = ElementTree.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in shared_xml.findall("main:si", ns):
                strings.append("".join(node.text or "" for node in item.findall(".//main:t", ns)))
        sheet_xml = ElementTree.fromstring(archive.read(sheet_path))
        rows = []
        for row_element in sheet_xml.findall("main:sheetData/main:row", ns):
            values = []
            for cell in row_element.findall("main:c", ns):
                ref = cell.attrib.get("r", "")
                letters = "".join(char for char in ref if char.isalpha())
                column = 0
                for char in letters:
                    column = column * 26 + ord(char.upper()) - ord("A") + 1
                while len(values) < column:
                    values.append(None)
                kind = cell.attrib.get("t", "")
                value_node = cell.find("main:v", ns)
                if kind == "inlineStr":
                    value = "".join(node.text or "" for node in cell.findall(".//main:t", ns))
                elif value_node is None:
                    value = None
                elif kind == "s":
                    value = strings[int(value_node.text)]
                else:
                    value = value_node.text
                if column:
                    values[column - 1] = value
            rows.append(values)
    headers = [str(value) if value is not None else "" for value in rows[0]]
    data = []
    for excel_row, values in enumerate(rows[1:], start=2):
        variety = "" if not values or values[0] is None else str(values[0]).strip()
        data.append((excel_row, variety, values))

    scene_candidates = {}
    for scene, spec in SCENES.items():
        aliases = tuple(alias.casefold() for alias in spec["aliases"])
        scene_candidates[scene] = [
            (row, variety) for row, variety, _ in data
            if any(alias in variety.casefold() for alias in aliases)
        ]

    def cell(values: list, index: int) -> str:
        if index >= len(values) or values[index] is None:
            return ""
        return str(values[index])

    result = []
    workbook_sha = sha256_file(workbook)
    plant_columns = {
        "plant_height_manual_column": 1,
        "plant_height_virtual_column": 2,
        "canopy_width_manual_column": 3,
        "canopy_width_virtual_column": 4,
    }
    leaf_slots = (
        {"leaf_slot": 1, "leaf_length_manual_column": 5, "leaf_length_virtual_column": 6,
         "leaf_width_manual_column": 7, "leaf_width_virtual_column": 8},
        {"leaf_slot": 2, "leaf_length_manual_column": 9, "leaf_length_virtual_column": 10,
         "leaf_width_manual_column": 11, "leaf_width_virtual_column": 12},
        {"leaf_slot": 3, "leaf_length_manual_column": 13, "leaf_length_virtual_column": 14,
         "leaf_width_manual_column": 15, "leaf_width_virtual_column": 16},
    )
    for excel_row, variety, values in data:
        candidate_scene = next((
            scene for scene, spec in SCENES.items()
            if any(alias.casefold() in variety.casefold() for alias in spec["aliases"])
        ), "")
        candidates = scene_candidates.get(candidate_scene, [])
        status = (
            "AMBIGUOUS" if len(candidates) > 1 else
            "SAMPLE_MATCH_ONLY" if candidates else
            "NO_MATCH"
        )
        candidate_rows = "|".join(str(row) for row, _ in candidates)
        common = {
            "new_mesh_sample_id": candidate_scene,
            "new_mesh_identity": "M0_SEED0_FIXED_BY_E3_MANIFEST" if candidate_scene else "",
            "workbook_path": str(workbook),
            "workbook_sha256": workbook_sha,
            "sheet": sheet_name,
            "historical_record_id": f"{sheet_name}!A{excel_row}",
            "excel_row": excel_row,
            "candidate_variety": variety,
            "candidate_excel_rows_for_scene": candidate_rows,
            "candidate_count": len(candidates),
            "matching_status": status,
            "specimen_id_available": False,
            "measurement_time_available": False,
            "unit_recorded_in_workbook": "NOT_RECORDED_IN_WORKBOOK_HEADERS",
            "virtual_values_are_new_mesh_results": False,
            "reason": "Cultivar text and workbook row only; no specimen ID, capture time, mesh ID, or per-value scale/method link.",
            "leaf_identity_status": "NOT_APPLICABLE_AT_PLANT_LEVEL",
        }
        plant_row = {
            **common,
            "record_level": "PLANT",
            "leaf_slot_in_workbook": "",
            "historical_leaf_id": "NOT_APPLICABLE",
            "trait_columns": json_cell([headers[index] for index in plant_columns.values()]),
            "trait_values": json_cell([cell(values, index) for index in plant_columns.values()]),
        }
        result.append(plant_row)
        for slot in leaf_slots:
            result.append({
                **common,
                "record_level": "LEAF",
                "leaf_slot_in_workbook": slot["leaf_slot"],
                "historical_leaf_id": "LEAF_ID_NOT_AVAILABLE",
                "trait_columns": json_cell([headers[index] for index in (
                    slot["leaf_length_manual_column"], slot["leaf_length_virtual_column"],
                    slot["leaf_width_manual_column"], slot["leaf_width_virtual_column"],
                )]),
                "trait_values": json_cell([cell(values, index) for index in (
                    slot["leaf_length_manual_column"], slot["leaf_length_virtual_column"],
                    slot["leaf_width_manual_column"], slot["leaf_width_virtual_column"],
                )]),
                "leaf_identity_status": "LEAF_ID_NOT_AVAILABLE; ordinal slot is not an organ identity",
            })
    return result


def main() -> int:
    root = repo_root()
    phase = Path(__file__).resolve().parents[1]
    output = phase
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh-root", type=Path,
                        default=Path("/data/fj/F2DMAS_phase20e3_data/1771e53f_v1/meshes"))
    parser.add_argument("--source-root", type=Path,
                        default=Path("/data/fj/F2DMAS_phase20e0_data/f53bf8f3_v1"))
    args = parser.parse_args()
    log_path = output / "logs" / "phase20f_execution.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(log_path, mode="w", encoding="utf-8"),
                                  logging.StreamHandler(sys.stdout)])
    logger = logging.getLogger("phase20f")
    start = time.time()
    started_at = datetime.now(timezone.utc).isoformat()
    e3 = root / "00-论文优化重构/计算机与电子农业特刊实验工作区/阶段二十E3_M2M3真实消融与Mesh验证"
    phase_b = root / "00-论文优化重构/计算机与电子农业特刊实验工作区/阶段二十B_历史实验与证据修复"
    references = {row["sample"]: row for row in read_csv(e3 / "MESH_REFERENCE_MANIFEST.csv")}
    structure = {(row["sample"], row["mesh_stage"]): row for row in read_csv(e3 / "MESH_STRUCTURE_METRICS.csv")
                 if row["method"] == "M0" and row["variant"] == "main"}
    if set(references) != set(SCENES):
        raise RuntimeError("FROZEN_SCENE_REFERENCE_SET_MISMATCH")

    reference_rows, geometry_rows, physical_rows, leaf_rows = [], [], [], []
    scale_rows, frame_rows, component_rows, projection_rows = [], [], [], []
    scene_data, checked_hashes = {}, {}

    for scene, spec in SCENES.items():
        manifest_ref = references[scene]
        if manifest_ref["method"] != "M0" or manifest_ref["actual_group"] != spec["group"] or manifest_ref["seed"] != "0":
            raise RuntimeError(f"FROZEN_MODEL_IDENTITY_MISMATCH:{scene}")
        camera_path = Path(manifest_ref["camera_json_path"])
        gaussian_path = Path(manifest_ref["PLY_path"])
        for asset_name, path, expected_sha in (
            ("camera_json", camera_path, manifest_ref["camera_json_sha256"]),
            ("gaussian_ply", gaussian_path, manifest_ref["PLY_sha256"]),
        ):
            actual_sha = sha256_file(path)
            if actual_sha != expected_sha:
                raise RuntimeError(f"HASH_MISMATCH:{scene}:{asset_name}")
            checked_hashes[str(path)] = actual_sha

        camera_manifest_path = args.mesh_root / scene / "M0/main/FUSION_FLOAT_MANIFEST.json"
        all_camera_records = json.loads(camera_manifest_path.read_text(encoding="utf-8"))
        if not all_camera_records:
            raise RuntimeError(f"NO_FROZEN_CAMERA_RECORDS:{scene}")
        camera_records = [all_camera_records[0], all_camera_records[len(all_camera_records) // 2], all_camera_records[-1]]
        image_root = args.source_root / spec["image_root"]
        mesh_data = {}
        for stage in ("raw", "post"):
            mesh_path = args.mesh_root / scene / f"M0/main/{stage}.ply"
            expected = structure.get((scene, stage))
            if expected is None:
                raise RuntimeError(f"MESH_HASH_INDEX_MISSING:{scene}:{stage}")
            actual_sha = sha256_file(mesh_path)
            if actual_sha != expected["mesh_sha256"]:
                raise RuntimeError(f"HASH_MISMATCH:{scene}:{stage}_mesh")
            checked_hashes[str(mesh_path)] = actual_sha
            mesh = o3d.io.read_triangle_mesh(str(mesh_path), enable_post_processing=False)
            vertices = np.asarray(mesh.vertices)
            faces = np.asarray(mesh.triangles)
            if mesh.is_empty() or not np.isfinite(vertices).all() or len(faces) == 0:
                raise RuntimeError(f"INVALID_MESH:{scene}:{stage}")
            if faces.min() < 0 or faces.max() >= len(vertices):
                raise RuntimeError(f"INVALID_TRIANGLE_INDEX:{scene}:{stage}")
            low, high = vector_bbox(vertices)
            mesh_data[stage] = (mesh, vertices, faces, low, high, mesh_path, actual_sha)
        raw_mesh, raw_vertices, raw_faces, _, _, raw_path, raw_sha = mesh_data["raw"]
        post_mesh, post_vertices, post_faces, _, _, post_path, post_sha = mesh_data["post"]
        subset_ok = post_vertices_subset_of_raw(raw_vertices, post_vertices)
        if not subset_ok:
            raise RuntimeError(f"RAW_POST_COORDINATE_FRAME_MISMATCH:{scene}")
        component_labels, component_counts, _ = raw_mesh.cluster_connected_triangles()
        component_labels = np.asarray(component_labels, dtype=np.int32)
        component_counts = np.asarray(component_counts, dtype=np.int64)
        largest_id = int(np.argmax(component_counts))
        if len(post_faces) != int(component_counts[largest_id]):
            raise RuntimeError(f"POST_FACE_COUNT_NOT_LARGEST_COMPONENT:{scene}")

        pcd = o3d.io.read_point_cloud(str(gaussian_path), remove_nan_points=False,
                                      remove_infinite_points=False)
        gaussian_points = np.asarray(pcd.points)
        if len(gaussian_points) == 0 or not np.isfinite(gaussian_points).all():
            raise RuntimeError(f"INVALID_FROZEN_GAUSSIAN_PLY:{scene}")
        gaussian_low, gaussian_high = vector_bbox(gaussian_points)

        for stage, data in (("raw", mesh_data["raw"]), ("post", mesh_data["post"])):
            _, vertices, faces, low, high, mesh_path, mesh_sha = data
            center, directions, extents = pca_extents(vertices)
            reference_rows.append({
                "sample_id": scene,
                "method": "M0",
                "reference_group": "B_R0" if scene != "HongZhang" else "B_C_SHARED_C_R1",
                "seed": 0,
                "mesh_stage": stage,
                "mesh_path": str(mesh_path),
                "mesh_sha256": mesh_sha,
                "vertices": len(vertices),
                "faces": len(faces),
                "bbox_min_colmap_units": json_cell(low),
                "bbox_max_colmap_units": json_cell(high),
                "pca_center_colmap_units": json_cell(center.tolist()),
                "pca_directions_columns": json_cell(directions.tolist()),
                "pca_extents_colmap_units_descending": json_cell(extents.tolist()),
                "camera_json_path": str(camera_path),
                "camera_json_sha256": checked_hashes[str(camera_path)],
                "gaussian_ply_path": str(gaussian_path),
                "gaussian_ply_sha256": checked_hashes[str(gaussian_path)],
                "gaussian_points": len(gaussian_points),
                "gaussian_bbox_min": json_cell(gaussian_low),
                "gaussian_bbox_max": json_cell(gaussian_high),
                "coordinate_units": "COLMAP_arbitrary_units",
                "coordinate_frame_assessment": "E3 frozen GaussianExtractor world frame; raw/post same frame" if stage == "raw" else "exact coordinate subset of raw; no affine transform",
                "post_vertices_exact_subset_of_raw": subset_ok,
                "plant_visibility_rgb_review": "VISIBLE_IN_FROZEN_RGBA",
                "pot_reconstructed_or_visible": "YES",
                "background_status": "RGBA_MASKED_INPUT; mesh may retain foreground/background artifacts",
                "leaf_completeness": "NOT_VALIDATED_PER_ORGAN",
                "visual_note": spec["visual_note"],
            })
            for trait, value, method in (
                ("plant_height", float(extents[0]), "PCA_MAJOR_AXIS_EXTENT_PROXY"),
                ("canopy_width", float(max(extents[1], extents[2])), "MAX_PCA_MINOR_PLANE_EXTENT_PROXY"),
            ):
                geometry_rows.append({
                    "sample_id": scene,
                    "method": "M0",
                    "mesh_stage": stage,
                    "trait": trait,
                    "value": value,
                    "unit": "COLMAP_arbitrary_unit",
                    "status": "GEOMETRIC_EXTENT_PROXY",
                    "geometry_method": method,
                    "scope": "WHOLE_RECONSTRUCTED_MESH_INCLUDES_POT_OR_ARTIFACTS; NOT_PLANT_ONLY",
                    "physical_trait_claim": False,
                })
            for trait in TRAITS:
                physical_rows.append({
                    "sample_id": scene,
                    "method": "M0",
                    "mesh_stage": stage,
                    "trait": trait,
                    "value_cm": "",
                    "unit": "NOT_EMITTED_WITHOUT_VERIFIED_SCALE",
                    "status": "PHYSICAL_SCALE_NOT_AVAILABLE",
                    "scale_source": "",
                    "same_plant_reference": False,
                    "reason": "No verified real-world length anchor and no specimen-linked trait reference.",
                })
            leaf_rows.append({
                "sample_id": scene,
                "leaf_id": "LEAF_ID_NOT_AVAILABLE",
                "mesh_stage": stage,
                "surface_present_in_mesh": "LEAF_LIKE_SURFACES_VISIBLE; ORGAN_BOUNDARIES_NOT_SEGMENTED",
                "organ_identity_evidence": "NONE_FOUND",
                "base_tip_landmarks": "NOT_AVAILABLE",
                "leaf_boundary_points": "NOT_AVAILABLE",
                "length_method": "BASE_TO_TIP_STRAIGHT_NOT_EXECUTED",
                "width_method": "MAXIMUM_CROSS_SECTION_NOT_EXECUTED",
                "value_length": "",
                "value_width": "",
                "unit": "",
                "validity": "LEAF_ID_NOT_AVAILABLE",
                "secondary_blocker": "LEAF_GEOMETRY_INCOMPLETE_AT_ORGAN_LEVEL",
                "source_evidence": "No leaf instance labels or archived landmarks in repository audit.",
            })

        frame_rows.append({
            "sample_id": scene,
            "coordinate_frame": "COLMAP_WORLD_ARBITRARY_UNITS",
            "vertical_axis": "NOT_AVAILABLE",
            "vertical_source": "NONE_FOUND",
            "plant_base_reference": "NOT_AVAILABLE",
            "pot_or_ground_plane": "VISIBLE_POT_BUT_NO_VERIFIED_GEOMETRIC_REFERENCE_PLANE",
            "formal_height_status": "BLOCKED_NO_VERTICAL_OR_BASE_REFERENCE",
            "formal_canopy_status": "BLOCKED_NO_VERTICAL_REFERENCE_OR_PLANT_ONLY_MASK",
            "exploratory_pca_axis": "COMPUTED_FOR_GEOMETRIC_EXTENT_ONLY",
            "exploratory_pca_axis_is_gravity": False,
            "camera_poses_define_gravity": False,
        })
        scale_rows.append({
            "sample_id": scene,
            "status": "PHYSICAL_SCALE_NOT_AVAILABLE",
            "scale_cm_per_colmap_unit": "",
            "physical_reference_length_cm": "",
            "mesh_correspondence_points": "",
            "scale_source_file": "",
            "uncertainty": "NOT_ESTIMABLE_WITHOUT_CALIBRATION_RECORD",
            "applicable_model": "M0_RAW_AND_POST",
            "evidence_checked": "; ".join([
                str(phase_b / "reports/SCALE_RECOVERY_AUDIT.md"),
                str(phase_b / "reports/PHENOTYPE_PROVENANCE_AUDIT.md"),
                str(phase_b / "tables/PHENOTYPE_DATA_SOURCE_MATRIX.csv"),
                str(root / "植株数据.xlsx"),
                str(e3 / "MESH_REFERENCE_MANIFEST.csv"),
                str(e3 / "MESH_STRUCTURE_METRICS.csv"),
                str(e3 / "RECOVERY_AND_REPRODUCTION.md"),
            ]) + "; no verified physical anchor found",
        })

        component_summary = component_analysis(
            scene, raw_mesh, len(post_vertices), raw_sha, post_sha,
            camera_records, image_root, component_rows, projection_rows,
            component_labels, component_counts,
        )
        scene_data[scene] = {
            "scene": scene,
            "raw_vertices": raw_vertices,
            "raw_faces": raw_faces,
            "post_vertices": post_vertices,
            "post_faces": post_faces,
            "labels": component_summary["component_labels"],
            "counts": component_counts,
            "largest_id": largest_id,
            "camera_records": camera_records,
            "image_root": image_root,
        }
        add_component_geometry_figure(scene_data[scene], output / "figures" / f"{scene}_M0_raw_post_removed.png")
        add_rgb_projection_figure(scene_data[scene], image_root, camera_records,
                                  output / "figures" / f"{scene}_M0_removed_rgb_projection.png")
        logger.info(
            "%s: raw=%dV/%dF, post=%dV/%dF, components=%d, removed_faces=%.3f%%, removed_vertices=%.3f%%",
            scene, len(raw_vertices), len(raw_faces), len(post_vertices), len(post_faces),
            component_summary["component_count"], 100 * component_summary["removed_face_fraction"],
            100 * component_summary["removed_vertex_fraction"],
        )

    workbook = root / "植株数据.xlsx"
    workbook_sha = sha256_file(workbook)
    matrix = read_csv(phase_b / "tables/PHENOTYPE_DATA_SOURCE_MATRIX.csv")
    expected_workbook_sha = next(row["sha256"] for row in matrix
                                 if row["source"] == "植株数据.xlsx" and row["role"] == "manual_and_virtual_trait_values")
    if workbook_sha != expected_workbook_sha:
        raise RuntimeError("HISTORICAL_WORKBOOK_HASH_MISMATCH")
    matching_rows = historical_matching(workbook)

    write_csv(output / "PHENOTYPE_MODEL_REFERENCE.csv", reference_rows, list(reference_rows[0]))
    write_csv(output / "PHENOTYPE_GEOMETRY_RESULTS.csv", geometry_rows, list(geometry_rows[0]))
    write_csv(output / "PHENOTYPE_PHYSICAL_RESULTS.csv", physical_rows, list(physical_rows[0]))
    write_csv(output / "LEAF_MEASURABILITY_REPORT.csv", leaf_rows, list(leaf_rows[0]))
    write_csv(output / "PHYSICAL_SCALE_PROVENANCE.csv", scale_rows, list(scale_rows[0]))
    write_csv(output / "GRAVITY_AND_REFERENCE_FRAME.csv", frame_rows, list(frame_rows[0]))
    component_fields = [
        "record_type", "sample_id", "component_id", "retained_by_post",
        "triangle_count", "triangle_fraction_of_raw",
        "centroid_x", "centroid_y", "centroid_z",
        "bbox_min_x", "bbox_min_y", "bbox_min_z",
        "bbox_max_x", "bbox_max_y", "bbox_max_z",
        "raw_vertex_count", "post_vertex_count", "removed_vertex_count",
        "removed_vertex_fraction", "raw_face_count", "post_face_count",
        "component_count", "removed_component_count", "largest_component_id",
        "largest_component_faces", "raw_mesh_sha256", "post_mesh_sha256",
        "trace_note",
    ]
    write_csv(output / "REMOVED_STRUCTURE_AUDIT.csv", component_rows, component_fields)
    write_csv(output / "HISTORICAL_PHENOTYPE_MATCHING.csv", matching_rows, list(matching_rows[0]))
    write_csv(output / "PHENOTYPE_ERROR_METRICS.csv", [], [
        "trait", "n_valid_pairs", "mae", "rmse", "bias", "mape_percent", "r_squared",
        "bland_altman_bias", "bland_altman_loa_low", "bland_altman_loa_high", "status",
    ])
    projection_fields = list(dict.fromkeys(key for row in projection_rows for key in row.keys()))
    write_csv(output / "logs/REMOVED_COMPONENT_RGB_PROJECTION.csv", projection_rows, projection_fields)

    execution = {
        "phase": "20F",
        "status": "PARTIAL_EVIDENCE_GATES_ENFORCED",
        "started_at_utc": started_at,
        "ended_at_utc": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": round(time.time() - start, 3),
        "repository_head": __import__("subprocess").check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "python": sys.version,
        "numpy": np.__version__,
        "open3d": o3d.__version__,
        "matplotlib": matplotlib.__version__,
        "scenes_analyzed": list(SCENES),
        "checked_input_sha256": checked_hashes,
        "historical_workbook_sha256": workbook_sha,
        "physical_scale": "PHYSICAL_SCALE_NOT_AVAILABLE",
        "vertical_reference": "NOT_AVAILABLE",
        "valid_physical_trait_values": 0,
        "valid_error_metric_rows": 0,
        "raw_post_alignment": "EXACT_POST_VERTEX_SUBSET_AND_LARGEST_COMPONENT_FACE_COUNT_CHECKED",
        "mesh_files_modified": False,
        "gaussian_or_camera_files_modified": False,
        "paper_tex_modified": False,
    }
    (output / "logs/EXECUTION.json").write_text(json.dumps(execution, indent=2) + "\n", encoding="utf-8")
    logger.info("Completed: 3/3 frozen M0 raw/post pairs analyzed; no cm values or phenotype errors emitted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
