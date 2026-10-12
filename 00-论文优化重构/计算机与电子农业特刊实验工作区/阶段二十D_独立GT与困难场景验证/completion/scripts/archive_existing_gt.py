from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from pathlib import Path

import numpy as np


STAGE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(STAGE / "scripts"))

from phase20d_common import (DIFFICULTIES, ROOT, original_link, read_csv, relative,
                            sha256, unique_index, write_csv, write_json)
import validate_annotations as original_gate


REQUIRED_LABELS = ("potted_plant", "plant", "pot")
HUMAN_FIELDS = (
    "annotation_origin", "annotator", "reviewer", "second_reviewer",
    "annotator_is_human", "reviewer_is_human", "second_reviewer_is_human",
    "manual_annotation_confirmed", "target_identity_description", "target_identity_confirmed",
    "plant_confirmed", "pot_confirmed", "blue_cube_present", "difficulty_manually_confirmed",
    "difficulty_evidence_json", "reviewed_at", "second_reviewed_at", "approval",
)


def derive_labels(row):
    if any(row.get(code) not in ("YES", "NO") for code in DIFFICULTIES):
        raise ValueError("Every original D column must contain an explicit YES/NO")
    return ";".join(code for code in DIFFICULTIES if row[code] == "YES") or "none"


def preserve_or_fill(row, field, expected):
    if row.get(field, "") and row[field] != expected:
        raise ValueError(f"Existing {field} disagrees with objective source; manual resolution required")
    row[field] = expected


def inspect_shapes(annotation, width, height):
    issues, details = [], []
    shapes = annotation.get("shapes", [])
    if not isinstance(shapes, list):
        return [{"code": "INVALID_SHAPES", "shape_index": None}], []
    present = {shape.get("label") for shape in shapes if isinstance(shape, dict)}
    for label in REQUIRED_LABELS:
        if label not in present:
            issues.append({"code": "MISSING_REQUIRED_LABEL", "label": label, "shape_index": None})
    for shape_index, shape in enumerate(shapes):
        if not isinstance(shape, dict):
            issues.append({"code": "INVALID_SHAPE", "shape_index": shape_index})
            continue
        label = shape.get("label")
        detail = {"shape_index": shape_index, "label": label,
                  "shape_type": shape.get("shape_type", "polygon")}
        if label not in (*REQUIRED_LABELS, "blue_cube"):
            issues.append({"code": "UNKNOWN_LABEL", "shape_index": shape_index, "label": label})
        if detail["shape_type"] != "polygon":
            issues.append({"code": "UNSUPPORTED_SHAPE_TYPE", "shape_index": shape_index,
                           "label": label, "actual": detail["shape_type"], "required": "polygon"})
        try:
            points = np.asarray(shape.get("points", []), dtype=float)
        except (ValueError, TypeError):
            points = np.asarray([])
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3 or not np.isfinite(points).all():
            issues.append({"code": "INVALID_POINTS", "shape_index": shape_index, "label": label})
        else:
            detail["point_count"] = len(points)
            detail["endpoints_equal"] = bool(np.array_equal(points[0], points[-1]))
            outside = np.flatnonzero((points[:, 0] < 0) | (points[:, 0] >= width)
                                     | (points[:, 1] < 0) | (points[:, 1] >= height)).tolist()
            if outside:
                issues.append({"code": "OUT_OF_BOUNDS", "shape_index": shape_index,
                               "label": label, "point_indices": outside})
            twice_area = float(np.dot(points[:, 0], np.roll(points[:, 1], 1))
                               - np.dot(points[:, 1], np.roll(points[:, 0], 1)))
            if len(np.unique(points, axis=0)) < 3 or not np.isfinite(twice_area) or twice_area == 0:
                issues.append({"code": "DEGENERATE_POLYGON", "shape_index": shape_index, "label": label})
        details.append(detail)
    return issues, details


def geometry_record(frame, annotation_path):
    base = {"challenge_id": frame["challenge_id"], "frame_id": frame["frame_id"],
            "sample": frame["sample"], "annotation_json": relative(annotation_path),
            "annotation_sha256": "", "rgb_sha256": frame["rgb_sha256"],
            "expected_width": frame["width"], "expected_height": frame["height"],
            "image_width": "", "image_height": "", "geometry_status": "INVALID",
            "potted_clean_status": "NOT_EVALUATED", "containment_status": "NOT_EVALUATED",
            "blue_cube_shape_count": "", "human_blue_cube_confirmation": "NOT_VERIFIED",
            "source_checked_flag": "", "shape_details_json": "[]", "issues_json": "[]",
            "frozen_geometry_error": ""}
    if not annotation_path.is_file():
        base["issues_json"] = json.dumps([{"code": "ANNOTATION_MISSING"}])
        return base
    base["annotation_sha256"] = sha256(annotation_path)
    try:
        annotation = json.loads(annotation_path.read_text(encoding="utf-8-sig"))
        if not isinstance(annotation, dict):
            raise ValueError("Annotation must be a JSON object")
    except (ValueError, OSError) as error:
        base["frozen_geometry_error"] = str(error)
        base["issues_json"] = json.dumps([{"code": "INVALID_JSON"}])
        return base
    width, height = int(frame["width"]), int(frame["height"])
    issues, details = inspect_shapes(annotation, width, height)
    if (annotation.get("imageWidth"), annotation.get("imageHeight")) != (width, height):
        issues.append({"code": "IMAGE_DIMENSION_MISMATCH"})
    base.update(image_width=annotation.get("imageWidth", ""), image_height=annotation.get("imageHeight", ""),
                source_checked_flag=json.dumps(annotation.get("checked")),
                blue_cube_shape_count=sum(detail["label"] == "blue_cube" for detail in details),
                shape_details_json=json.dumps(details, ensure_ascii=False))
    binding = {"annotation_sha256": base["annotation_sha256"], "rgb_sha256": frame["rgb_sha256"],
               "blue_cube_present": "YES" if base["blue_cube_shape_count"] else "NO"}
    try:
        original_gate.validate_geometry(annotation, frame, binding, annotation_path)
    except (ValueError, KeyError, TypeError, OSError) as error:
        base["frozen_geometry_error"] = str(error)
        issues.append({"code": "FROZEN_GEOMETRY_REJECTED", "message": str(error)})
    if not issues:
        base.update(geometry_status="VALID", potted_clean_status="NONEMPTY", containment_status="PASS")
    base["issues_json"] = json.dumps(issues, ensure_ascii=False)
    return base


def csv_patch(path, old_text, rows):
    fields = list(dict.fromkeys(field for row in rows for field in row))
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    new_text = "\ufeff" + buffer.getvalue()
    old_lines, new_lines = old_text.splitlines(), new_text.splitlines()
    return (f"*** Update File: {relative(path)}\n@@\n"
            + "".join(f"-{line}\n" for line in old_lines)
            + "".join(f"+{line}\n" for line in new_lines))


def missing_human_fields(review):
    return [field for field in HUMAN_FIELDS
            if not review.get(field, "").strip() or field == "approval" and review[field] != "APPROVED"]


def prepare(stage, output, patch_output):
    frames = read_csv(stage / "TEST32_INPUT_MANIFEST.csv")
    if len(frames) != 32 or len(unique_index(frames, "frame_id")) != 32:
        raise ValueError("Exactly 32 frozen frames required")
    original_gate.verify_frozen_inputs(stage, frames)
    review_path = stage / "TEST32_ANNOTATION_REVIEW.csv"
    difficulty_path = stage / "DIFFICULTY_HUMAN_VERIFICATION.csv"
    reviews, difficulty = read_csv(review_path), read_csv(difficulty_path)
    review_map, difficulty_map = unique_index(reviews, "frame_id"), unique_index(difficulty, "frame_id")
    expected = {frame["frame_id"] for frame in frames}
    if len(reviews) != 32 or len(difficulty) != 32 or set(review_map) != expected or set(difficulty_map) != expected:
        raise ValueError("Both human tables must cover exactly the frozen 32 frame IDs")
    old_texts = {path: path.read_text(encoding="utf-8") for path in (review_path, difficulty_path)}
    table_sources = [{"path": relative(path), "sha256_before": sha256(path)} for path in old_texts]
    archive, geometry, supplemental = [], [], []
    for frame in frames:
        stem = frame["frame_id"]
        review, human_difficulty = review_map[stem], difficulty_map[stem]
        for table_row in (review, human_difficulty):
            if table_row["challenge_id"] != frame["challenge_id"] or table_row["rgb_sha256"] != frame["rgb_sha256"]:
                raise ValueError(f"Human record/frozen identity mismatch: {stem}")
        canonical = stage / frame["annotation_json"]
        beside_rgb = stage / "annotation_package/rgb" / f"{stem}.json"
        if canonical.is_file() and beside_rgb.is_file() and sha256(canonical) != sha256(beside_rgb):
            raise ValueError(f"Conflicting existing annotation versions: {stem}; no automatic replacement")
        source = canonical.resolve() if canonical.is_file() else beside_rgb
        before = sha256(source) if source.is_file() else ""
        if not canonical.exists() and source.is_file():
            original_link(source, canonical)
        record = geometry_record(frame, canonical)
        geometry.append(record)
        if before and sha256(source) != before:
            raise ValueError(f"Original human annotation changed: {stem}")
        archive.append({"challenge_id": frame["challenge_id"], "frame_id": stem,
                        "source_json": relative(source.resolve()), "canonical_json": relative(canonical),
                        "bytes": source.stat().st_size if source.is_file() else "",
                        "source_sha256_before": before, "source_sha256_after": before,
                        "canonical_sha256": record["annotation_sha256"],
                        "archive_action": "READ_ONLY_SYMLINK" if canonical.is_symlink() else "EXISTING_OR_MISSING",
                        "approved_final_gt": "NOT_VERIFIED"})
        labels = derive_labels(human_difficulty)
        for table_row in (review, human_difficulty):
            if record["annotation_sha256"]:
                preserve_or_fill(table_row, "annotation_sha256", record["annotation_sha256"])
            preserve_or_fill(table_row, "human_difficulty_labels", labels)
            for field, value in (("image_width", frame["width"]), ("image_height", frame["height"]),
                                 ("geometry_validation_status", record["geometry_status"])):
                preserve_or_fill(table_row, field, value)
        supplemental.append({"challenge_id": frame["challenge_id"], "frame_id": stem,
                             "annotation_sha256": record["annotation_sha256"], "rgb_sha256": frame["rgb_sha256"],
                             **{code: human_difficulty[code] for code in DIFFICULTIES},
                             "human_difficulty_labels": labels,
                             "missing_human_fields": ";".join(missing_human_fields(review)),
                             "geometry_issues_json": record["issues_json"],
                             **{field: "" for field in HUMAN_FIELDS},
                             "supplement_status": "REAL_HUMAN_ENTRY_REQUIRED"})
    for table_row in (*reviews, *difficulty):
        if any(table_row.get(field, "").strip() for field in ("annotator", "reviewer", "second_reviewer")):
            raise ValueError("Private reviewer identities present; use an explicitly approved private patch location")
    patch = "*** Begin Patch\n" + "".join(csv_patch(path, old_texts[path], rows)
                                             for path, rows in ((review_path, reviews), (difficulty_path, difficulty)))
    patch += "*** End Patch\n"
    patch_output.parent.mkdir(parents=True, exist_ok=True)
    with patch_output.open("x", encoding="utf-8") as handle:
        handle.write(patch)
    write_csv(output / "GT32_GEOMETRY_REPORT.csv", geometry)
    write_csv(output / "GT32_SOURCE_ARCHIVE.csv", archive)
    write_csv(output / "HUMAN_REVIEW_FRAME_SUPPLEMENT.csv", supplemental)
    batch = [{"batch_id": "TEST32-existing-review", "scope": "EXACT_FROZEN_32_FRAME_IDS",
              "scope_index": "GT32_SOURCE_ARCHIVE.csv", "annotator_code": "", "reviewer_code": "",
              "second_reviewer_code": "", "reviewed_at": "", "second_reviewed_at": "",
              "actual_review_scope_confirmed": "", "private_identity_map_location": "",
              "pseudonym_publication_confirmed": "", "approval": "",
              "frame_specific_decisions_required": "HUMAN_REVIEW_FRAME_SUPPLEMENT.csv"}]
    write_csv(output / "HUMAN_REVIEW_BATCH_REGISTRATION.csv", batch)
    code_counts = {code: sum(row[code] == "YES" for row in difficulty) for code in DIFFICULTIES}
    summary = {"baseline": "22073ca5", "frozen_frames": 32, "frozen_sequences": 8,
               "verified_original_candidates": 71, "annotations_found": sum(bool(row["canonical_sha256"]) for row in archive),
               "valid_geometry": sum(row["geometry_status"] == "VALID" for row in geometry),
               "manual_decision_counts_not_yet_approved": code_counts,
               "none_decision_count_not_yet_approved": sum(derive_labels(row) == "none" for row in difficulty),
               "human_fields_fabricated": False, "original_annotations_modified": False,
               "predictions_run": False, "accuracy_metrics_computed": False, "table_sources": table_sources,
               "source": "Existing local LabelMe exports and manual D columns at 22073ca5",
               "note": "SHA bindings refer to current exports, not an approval of final GT; links do not convert shape types."}
    write_json(output / "logs/ARCHIVE_PREPARATION.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Archive existing human work; no GT edits, approval, inference or accuracy")
    parser.add_argument("--stage-dir", type=Path, default=STAGE)
    parser.add_argument("--output-dir", type=Path, default=STAGE / "completion")
    parser.add_argument("--patch-output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.stage_dir.resolve(), args.output_dir.resolve(), args.patch_output.resolve())


if __name__ == "__main__":
    main()
