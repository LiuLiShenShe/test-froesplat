from __future__ import annotations

import base64
import copy
import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


STAGE = Path(__file__).resolve().parents[2]
PACKAGE = STAGE / "gt_protocol_v2"
sys.path.insert(0, str(STAGE / "scripts"))
from phase20d_common import DIFFICULTIES, ROOT, read_csv, sha256, unique_index
import validate_annotations as gate_v1


PROTOCOL = "FORESPLAT_VISIBLE_GT_V2"
LABELS = {"potted_plant", "plant", "pot", "blue_cube"}
POT_VISIBILITY = {"VISIBLE", "PARTIALLY_VISIBLE", "NOT_VISIBLE", "UNVERIFIED"}
DERIVATION_RULE = "PLANT_EQUALS_HUMAN_POTTED_PLANT_WHEN_CONFIRMED_PLANT_ONLY"
CONFIRM_FIELDS = ("complete_region_boundary", "obvious_open_gap", "closed_region_confirmed",
                  "confirmed_by", "confirmed_is_human", "confirmed_at", "confirmed_source", "confirmation_reason",
                  "confirmation_status")
EPSILON = 1e-9


def verify_original_sources(stage, frames):
    gate_v1.verify_frozen_inputs(stage, frames)
    archive_path = stage / "completion/GT32_SOURCE_ARCHIVE.csv"
    saved = subprocess.check_output(["git", "show", f"6092e2ea:{archive_path.relative_to(ROOT)}"], cwd=ROOT)
    if archive_path.read_bytes() != saved:
        raise ValueError("Original completion source archive changed")
    archive = unique_index(read_csv(archive_path), "frame_id")
    if set(archive) != {frame["frame_id"] for frame in frames}:
        raise ValueError("All32 original human exports must be permanently retained")
    for frame in frames:
        row = archive[frame["frame_id"]]
        source = stage / "annotation_package/annotations" / f"{frame['frame_id']}.json"
        if sha256(source) != row["source_sha256_before"] or sha256(ROOT / row["source_json"]) != row["source_sha256_before"]:
            raise ValueError("Original human GT bytes changed; new versions must not overwrite originals")


def json_digest(document):
    return hashlib.sha256(json.dumps(document, ensure_ascii=False, sort_keys=True, allow_nan=False,
                                     separators=(",", ":")).encode()).hexdigest()


def timestamp_valid(value):
    try:
        parsed = datetime.fromisoformat(value)
        return parsed.tzinfo is not None and parsed.utcoffset() is not None
    except (TypeError, ValueError):
        return False


def human_code(value):
    return bool(isinstance(value, str) and value.strip() and value.strip().casefold() not in gate_v1.MACHINE_NAMES)


def signed_confirmation(row, prefix):
    if not human_code(row.get(prefix + "by", "")) or row.get(prefix + "is_human") != "YES":
        raise ValueError(f"Real human confirmation required: {prefix}")
    if not timestamp_valid(row.get(prefix + "at", "")) or not row.get(prefix + "source", "").strip():
        raise ValueError(f"Actual timestamp and traceable human record required: {prefix}")


def points_array(shape):
    try:
        points = np.asarray(shape.get("points", []), dtype=float)
    except (ValueError, TypeError) as error:
        raise ValueError("Invalid vertex list") from error
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3 or not np.isfinite(points).all():
        raise ValueError("At least3 finite two-dimensional vertices required")
    if len(np.unique(points, axis=0)) < 3:
        raise ValueError("At least3 distinct vertices required")
    return points


def orientation(first, second, third):
    return ((second[0] - first[0]) * (third[1] - first[1])
            - (second[1] - first[1]) * (third[0] - first[0]))


def segment_intersects(first, second, third, fourth):
    if max(first[0], second[0]) + EPSILON < min(third[0], fourth[0]) or max(third[0], fourth[0]) + EPSILON < min(first[0], second[0]):
        return False
    if max(first[1], second[1]) + EPSILON < min(third[1], fourth[1]) or max(third[1], fourth[1]) + EPSILON < min(first[1], second[1]):
        return False
    side_first, side_second = orientation(first, second, third), orientation(first, second, fourth)
    side_third, side_fourth = orientation(third, fourth, first), orientation(third, fourth, second)
    return (min(side_first, side_second) <= EPSILON and max(side_first, side_second) >= -EPSILON
            and min(side_third, side_fourth) <= EPSILON and max(side_third, side_fourth) >= -EPSILON)


def intersection_pairs(points, closed):
    working = points.tolist()
    if closed and working[0] == working[-1]:
        working = working[:-1]
    edges = list(zip(working[:-1], working[1:]))
    if closed:
        edges.append((working[-1], working[0]))
    result = []
    for first_index, (first, second) in enumerate(edges):
        if first == second:
            continue
        for second_index in range(first_index + 1, len(edges)):
            third, fourth = edges[second_index]
            if third == fourth:
                continue
            adjacent = second_index == first_index + 1 or closed and first_index == 0 and second_index == len(edges) - 1
            if adjacent:
                if second_index == first_index + 1:
                    shared, before, after = second, first, fourth
                else:
                    shared, before, after = first, second, third
                overlap = ((before[0] - shared[0]) * (after[0] - shared[0])
                           + (before[1] - shared[1]) * (after[1] - shared[1]))
                if abs(orientation(before, shared, after)) <= EPSILON and overlap > EPSILON:
                    result.append([first_index, second_index])
                continue
            if segment_intersects(first, second, third, fourth):
                result.append([first_index, second_index])
    return result


def shape_geometry(shape, width, height):
    result = {"vertex_count": len(shape.get("points", [])), "endpoint_distance_px": "",
              "endpoint_gap_image_diagonal_ratio": "", "first_point_json": "", "last_point_json": "",
              "median_open_edge_length_px": "", "max_open_edge_length_px": "", "signed_area_px2": "",
              "open_self_intersection_count": "", "closed_self_intersection_count": "",
              "open_intersection_edge_pairs_json": "[]", "closed_intersection_edge_pairs_json": "[]",
              "geometry_status": "INVALID", "geometry_errors_json": "[]"}
    errors = []
    if shape.get("label") not in LABELS:
        errors.append("UNKNOWN_LABEL")
    if shape.get("shape_type", "polygon") not in ("polygon", "linestrip"):
        errors.append("UNSUPPORTED_SHAPE_TYPE")
    try:
        points = points_array(shape)
    except ValueError as error:
        result["geometry_errors_json"] = json.dumps(errors + [str(error)])
        return result
    outside = np.flatnonzero((points[:, 0] < 0) | (points[:, 0] >= width)
                             | (points[:, 1] < 0) | (points[:, 1] >= height)).tolist()
    if outside:
        errors.append("OUT_OF_BOUNDS_POINT_INDICES=" + json.dumps(outside))
    duplicate_edges = np.flatnonzero(np.all(points[1:] == points[:-1], axis=1)).tolist()
    if duplicate_edges:
        errors.append("ZERO_LENGTH_OPEN_EDGE_INDICES=" + json.dumps(duplicate_edges))
    endpoint = float(np.linalg.norm(points[0] - points[-1]))
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    twice_area = float(np.dot(points[:, 0], np.roll(points[:, 1], 1))
                       - np.dot(points[:, 1], np.roll(points[:, 0], 1)))
    if not math.isfinite(twice_area) or twice_area == 0:
        errors.append("ZERO_OR_NONFINITE_CLOSED_AREA")
    open_pairs, closed_pairs = intersection_pairs(points, False), intersection_pairs(points, True)
    if closed_pairs:
        errors.append("CLOSED_OUTLINE_SELF_INTERSECTION")
    result.update(endpoint_distance_px=endpoint, endpoint_gap_image_diagonal_ratio=endpoint / math.hypot(width, height),
                  first_point_json=json.dumps(shape["points"][0]), last_point_json=json.dumps(shape["points"][-1]),
                  median_open_edge_length_px=float(np.median(lengths)), max_open_edge_length_px=float(np.max(lengths)),
                  signed_area_px2=twice_area / 2, open_self_intersection_count=len(open_pairs),
                  closed_self_intersection_count=len(closed_pairs), open_intersection_edge_pairs_json=json.dumps(open_pairs),
                  closed_intersection_edge_pairs_json=json.dumps(closed_pairs), geometry_errors_json=json.dumps(errors),
                  geometry_status="VALID_CLOSED_GEOMETRY_ONLY" if not errors else "INVALID")
    return result


def derive_difficulty(row):
    if any(row.get(code) not in ("YES", "NO") for code in DIFFICULTIES):
        raise ValueError("Preserved human D decisions require explicit YES/NO values")
    return ";".join(code for code in DIFFICULTIES if row[code] == "YES") or "none"


def validate_source(annotation, frame, source_path, source_hash):
    if sha256(source_path) != source_hash:
        raise ValueError("Original annotation SHA256 changed")
    if (annotation.get("imageWidth"), annotation.get("imageHeight")) != (int(frame["width"]), int(frame["height"])):
        raise ValueError("Original RGB/annotation coordinate dimensions differ")
    if Path(annotation.get("imagePath", "")).name not in (f"{frame['frame_id']}.jpg", f"{frame['frame']}.jpg"):
        raise ValueError("Annotation imagePath/frozen frame identity mismatch")
    if annotation.get("imageData"):
        embedded = base64.b64decode(annotation["imageData"], validate=True)
        if hashlib.sha256(embedded).hexdigest() != frame["rgb_sha256"]:
            raise ValueError("Embedded image differs from frozen original RGB")
    if not isinstance(annotation.get("shapes"), list) or not annotation["shapes"]:
        raise ValueError("Nonempty existing human shapes required")


def validate_visibility(row, source_hash):
    if row.get("source_annotation_sha256") != source_hash:
        raise ValueError("Visibility record does not bind this original annotation")
    if row.get("pot_visibility") not in POT_VISIBILITY - {"UNVERIFIED"}:
        raise ValueError("Frame-specific pot visibility is UNVERIFIED")
    if row.get("plant_visible") not in ("YES", "NO") or row.get("blue_cube_present") not in ("YES", "NO"):
        raise ValueError("Actual plant visibility and blue-cube presence decisions required")
    signed_confirmation(row, "visibility_confirmed_")
    if row.get("visibility_status") != "CONFIRMED":
        raise ValueError("Visibility confirmation is not complete")


def validate_shape_confirmation(row, shape, source_hash, index):
    if row.get("source_annotation_sha256") != source_hash or int(row.get("shape_index", -1)) != index or row.get("source_shape_sha256") != json_digest(shape):
        raise ValueError("Human outline confirmation hash/index mismatch")
    if row.get("closed_region_confirmed") != "YES" or row.get("complete_region_boundary") != "YES" or row.get("obvious_open_gap") != "NO":
        raise ValueError("Uncertain/open linestrip: HUMAN_REVIEW_REQUIRED")
    signed_confirmation(row, "confirmed_")
    if row.get("confirmation_status") != "CONFIRMED_CLOSED_REGION" or not row.get("confirmation_reason", "").strip():
        raise ValueError("Specific human conversion reason/status required")


def derive_document(annotation, frame, visibility, shape_records, source_hash):
    validate_visibility(visibility, source_hash)
    shapes = annotation["shapes"]
    expected = set(range(len(shapes)))
    if set(shape_records) != expected:
        raise ValueError("Every source shape must be accounted for; no deletion or subset conversion")
    derived = copy.deepcopy(annotation)
    converted_indices = []
    for shape_index, shape in enumerate(shapes):
        geometry = shape_geometry(shape, int(frame["width"]), int(frame["height"]))
        if geometry["geometry_status"] != "VALID_CLOSED_GEOMETRY_ONLY":
            raise ValueError(f"Shape{shape_index} invalid: {geometry['geometry_errors_json']}")
        if shape.get("shape_type", "polygon") == "linestrip":
            validate_shape_confirmation(shape_records[shape_index], shape, source_hash, shape_index)
            derived["shapes"][shape_index]["shape_type"] = "polygon"
            converted_indices.append(shape_index)
        if derived["shapes"][shape_index]["points"] != shape["points"]:
            raise ValueError("Conversion must not change any source vertex")
    present = {shape["label"] for shape in shapes}
    plant_sources = []
    if "plant" not in present and visibility["plant_visible"] == "YES":
        if visibility["pot_visibility"] != "NOT_VISIBLE" or visibility.get("potted_only_plant_confirmed") != "YES" or visibility.get("plant_equal_potted_derivation_confirmed") != "YES":
            raise ValueError("Missing visible plant cannot be copied unless pot is explicitly NOT_VISIBLE and region is confirmed plant-only")
        signed_confirmation(visibility, "plant_derivation_confirmed_")
        if visibility.get("plant_derivation_rule") != DERIVATION_RULE:
            raise ValueError("Explicit deterministic plant-equals-potted derivation rule required")
        for shape_index, shape in enumerate(derived["shapes"][:len(shapes)]):
            if shape["label"] == "potted_plant":
                plant_shape = copy.deepcopy(shape)
                plant_shape["label"] = "plant"
                derived["shapes"].append(plant_shape)
                plant_sources.append(shape_index)
    derived["gt_protocol_v2"] = {"protocol": PROTOCOL, "source_annotation_sha256": source_hash,
                                "linestrip_to_polygon_indices": converted_indices,
                                "closing_rule": "IMPLICIT_LAST_TO_FIRST_EDGE_NO_VERTEX_CHANGE",
                                "plant_equal_potted_source_indices": plant_sources,
                                "plant_derivation_rule": DERIVATION_RULE if plant_sources else "NONE"}
    validate_visible_masks(derived, frame, visibility)
    return derived


def validate_visible_masks(annotation, frame, visibility):
    width, height = int(frame["width"]), int(frame["height"])
    if (annotation.get("imageWidth"), annotation.get("imageHeight")) != (width, height):
        raise ValueError("Converted polygon dimensions changed; no resampling allowed")
    masks = {label: np.zeros((height, width), dtype=np.uint8) for label in sorted(LABELS)}
    for index, shape in enumerate(annotation["shapes"]):
        if shape.get("shape_type", "polygon") != "polygon":
            raise ValueError("V2 rasterization accepts only checked polygon regions")
        info = shape_geometry(shape, width, height)
        if info["geometry_status"] != "VALID_CLOSED_GEOMETRY_ONLY":
            raise ValueError(f"Converted shape{index} invalid: {info['geometry_errors_json']}")
        cv2.fillPoly(masks[shape["label"]], [np.rint(points_array(shape)).astype(np.int32)], 1)
    if not masks["potted_plant"].any():
        raise ValueError("All frames require nonempty visible potted_plant")
    for label in ("plant", "pot"):
        if (masks[label].astype(bool) & ~masks["potted_plant"].astype(bool)).any():
            raise ValueError(f"Visible {label} extends outside the same target potted_plant")
    pot_expected = visibility["pot_visibility"] in ("VISIBLE", "PARTIALLY_VISIBLE")
    if not pot_expected and visibility["plant_visible"] == "NO":
        raise ValueError("Nonempty target requires an actually visible plant or pot component")
    if bool(masks["pot"].any()) != pot_expected:
        raise ValueError("Pot polygons disagree with actual human pot visibility")
    if bool(masks["plant"].any()) != (visibility["plant_visible"] == "YES"):
        raise ValueError("Plant polygons disagree with actual human plant visibility")
    if bool(masks["blue_cube"].any()) != (visibility["blue_cube_present"] == "YES"):
        raise ValueError("Blue-cube polygons disagree with actual human presence")
    target = masks["potted_plant"].astype(bool) & ~masks["blue_cube"].astype(bool)
    if not target.any():
        raise ValueError("potted_clean cannot be empty")
    return target, masks


def validate_review_v2(review, visibility):
    if review.get("approval") != "APPROVED" or review.get("protocol") != PROTOCOL:
        raise ValueError("Actual V2 human approval missing")
    people = {field: review.get(field, "").strip() for field in ("annotator", "reviewer", "second_reviewer")}
    if any(not human_code(person) for person in people.values()):
        raise ValueError("Real human annotator/reviewers required; no AI signatures")
    if people["second_reviewer"].casefold() in (people["annotator"].casefold(), people["reviewer"].casefold()):
        raise ValueError("Distinct independent second human reviewer required")
    for field in ("annotator_is_human", "reviewer_is_human", "second_reviewer_is_human", "manual_annotation_confirmed",
                  "target_identity_confirmed", "visible_components_confirmed", "difficulty_manually_confirmed"):
        if review.get(field) != "YES":
            raise ValueError(f"Actual human confirmation missing: {field}")
    if review.get("annotation_origin") not in ("MANUAL_VISIBLE_REGIONS", "MANUAL_VISIBLE_REGIONS_WITH_RECORDED_FORMAT_DERIVATION"):
        raise ValueError("Human visible-region source attestation required; predictions are not GT")
    if not review.get("target_identity_description", "").strip():
        raise ValueError("Specific target-instance identity description required")
    if not all(timestamp_valid(review.get(field, "")) for field in ("reviewed_at", "second_reviewed_at")):
        raise ValueError("Both actual timezone-aware human review timestamps required")
    if review.get("blue_cube_present") != visibility["blue_cube_present"]:
        raise ValueError("Human review/visibility cube decision disagrees")
    labels = review.get("human_difficulty_labels", "").split(";")
    if len(set(labels)) != len(labels) or not set(labels) <= set(DIFFICULTIES) | {"none"} or "none" in labels and len(labels) != 1:
        raise ValueError("Exact manual difficulty labels or none required")
    try:
        evidence = json.loads(review.get("difficulty_evidence_json", ""))
    except (ValueError, TypeError) as error:
        raise ValueError("Actual image-specific human difficulty evidence required") from error
    if not isinstance(evidence, dict) or set(evidence) != set(labels) or any(not isinstance(value, str) or not value.strip() for value in evidence.values()):
        raise ValueError("Visual evidence must cover exactly each label, including none")
    approved_people = set(people.values())
    if visibility["visibility_confirmed_by"] not in approved_people:
        raise ValueError("Visibility confirmation must trace to a recorded real annotator/reviewer")
    if visibility.get("plant_equal_potted_derivation_confirmed") == "YES" and visibility.get("plant_derivation_confirmed_by") not in approved_people:
        raise ValueError("Plant derivation consent must trace to a recorded human")
