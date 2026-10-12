from __future__ import annotations

import argparse
import json
from pathlib import Path

from protocol_v2 import (DIFFICULTIES, PACKAGE, PROTOCOL, ROOT, STAGE, derive_difficulty, derive_document, gate_v1,
                         read_csv, sha256, shape_geometry, unique_index, validate_review_v2, validate_shape_confirmation,
                         validate_source, validate_visibility, verify_original_sources)
from phase20d_common import write_json, write_text


def batch_review(base, batch, records_dir, identities_hash):
    row = dict(base)
    if not row.get("batch_record_id"):
        return row
    if row["batch_record_id"] != batch.get("batch_record_id") or batch.get("scope_source_identities_sha256") != identities_hash:
        raise ValueError("Batch review scope/hash binding mismatch")
    if batch.get("actual_human_review_completed") != "YES" or batch.get("approval") != "APPROVED" or not batch.get("human_record_source", "").strip():
        raise ValueError("Existing real batch review confirmation must be provided; no automatic batch approval")
    stem = row["frame_id"]
    if stem not in batch.get("scope_frame_ids", []) or batch.get("scope_final_derived_sha256_by_frame", {}).get(stem) != row["annotation_sha256"]:
        raise ValueError("Batch review must bind exact final converted JSON for each frame")
    fields = ("annotator", "reviewer", "second_reviewer", "annotator_is_human", "reviewer_is_human", "second_reviewer_is_human",
              "reviewed_at", "second_reviewed_at", "public_reviewer_codes_confirmed")
    for field in fields:
        if row.get(field) and row[field] != batch.get(field):
            raise ValueError("Existing frame/batch human record conflict")
        row[field] = batch.get(field, "")
    return row


def preflight(stage, records_dir, converted_manifest):
    frames = read_csv(stage / "TEST32_INPUT_MANIFEST.csv")
    if len(frames) != 32 or len(unique_index(frames, "frame_id")) != 32:
        raise ValueError("Frozen32 full coverage required; no subset or replacement")
    verify_original_sources(stage, frames)
    identities = unique_index(read_csv(records_dir / "SOURCE_IDENTITIES.csv"), "frame_id")
    visibility = unique_index(read_csv(records_dir / "GT_VISIBILITY_MANIFEST.csv"), "frame_id")
    reviews = unique_index(read_csv(records_dir / "HUMAN_REVIEW_V2.csv"), "frame_id")
    converted = unique_index(read_csv(converted_manifest), "frame_id")
    expected = {frame["frame_id"] for frame in frames}
    if any(set(indexed) != expected for indexed in (identities, visibility, reviews, converted)):
        raise ValueError("All V2 tables must retain exact frozen32 coverage")
    original_difficulty = unique_index(read_csv(stage / "DIFFICULTY_HUMAN_VERIFICATION.csv"), "frame_id")
    shape_records = {}
    for row in read_csv(records_dir / "LINESTRIP_CONVERSION_AUDIT.csv"):
        key = (row["frame_id"], int(row["shape_index"]))
        if key in shape_records:
            raise ValueError("Duplicate human shape review entry")
        shape_records[key] = row
    if {stem for stem, index in shape_records} != expected:
        raise ValueError("All original shape frame identities must remain represented")
    batch_path = records_dir / "HUMAN_BATCH_RECORD.json"
    batch = json.loads(batch_path.read_text()) if batch_path.is_file() else {}
    rows = []
    for frame in frames:
        stem = frame["frame_id"]
        identity, visible, selected = identities[stem], visibility[stem], converted[stem]
        errors = []
        source = stage / "annotation_package/annotations" / f"{stem}.json"
        source_hash = sha256(source)
        annotation = json.loads(source.read_text(encoding="utf-8-sig"))
        for name, row in (("source", identity), ("visibility", visible), ("review", reviews[stem]), ("converted", selected)):
            if row.get("protocol") != PROTOCOL:
                errors.append(f"{name} protocol version mismatch")
            if row.get("challenge_id") != frame["challenge_id"] or row.get("rgb_sha256") != frame["rgb_sha256"]:
                errors.append(f"{name} frozen frame/RGB identity mismatch")
            if row.get("source_annotation_sha256") != source_hash:
                errors.append(f"{name} source annotation SHA256 mismatch")
        try:
            validate_source(annotation, frame, source, identity["source_annotation_sha256"])
        except (ValueError, KeyError, TypeError) as error:
            errors.append(str(error))
        try:
            validate_visibility(visible, source_hash)
        except ValueError as error:
            errors.append(str(error))
        frame_records = {index: row for (record_stem, index), row in shape_records.items() if record_stem == stem}
        if set(frame_records) != set(range(len(annotation["shapes"]))):
            errors.append("Every original shape must have exactly one review record")
        for index, shape in enumerate(annotation["shapes"]):
            geometry = shape_geometry(shape, int(frame["width"]), int(frame["height"]))
            if geometry["geometry_status"] != "VALID_CLOSED_GEOMETRY_ONLY":
                errors.append(f"shape{index}: source geometry needs human inspection: {geometry['geometry_errors_json']}")
            shape_row = frame_records.get(index, {})
            if (shape_row.get("protocol") != PROTOCOL or shape_row.get("rgb_sha256") != frame["rgb_sha256"]
                    or shape_row.get("challenge_id") != frame["challenge_id"]):
                errors.append(f"shape{index}: version/frozen RGB identity mismatch")
            if shape.get("shape_type", "polygon") == "linestrip":
                try:
                    validate_shape_confirmation(frame_records[index], shape, source_hash, index)
                except (ValueError, KeyError) as error:
                    errors.append(f"shape{index}: {error}")
        try:
            review = batch_review(reviews[stem], batch, records_dir, sha256(records_dir / "SOURCE_IDENTITIES.csv"))
            if review["human_difficulty_labels"] != derive_difficulty(review):
                raise ValueError("V2 labels must derive from the same frame's manual D decisions")
            if any(review[code] != original_difficulty[stem][code] for code in DIFFICULTIES):
                raise ValueError("Historical manual D decisions changed; preserve existing judgments")
            validate_review_v2(review, visible)
            if review["source_annotation_sha256"] != source_hash:
                raise ValueError("Human review source hash mismatch")
            for index, confirmation in frame_records.items():
                if annotation["shapes"][index].get("shape_type", "polygon") == "linestrip" and confirmation.get("confirmed_by") not in {review["annotator"], review["reviewer"], review["second_reviewer"]}:
                    raise ValueError("Outline confirmation must trace to the named real reviewer/annotator")
        except (ValueError, KeyError, TypeError) as error:
            errors.append(str(error))
            review = reviews[stem]
        derived_path = ROOT / selected.get("derived_annotation_json", "")
        if selected.get("conversion_status") != "CONVERTED_PENDING_FINAL_HUMAN_REVIEW" or not selected.get("derived_annotation_json") or not derived_path.is_file():
            errors.append("Human-authorized converted polygon JSON missing")
        else:
            try:
                if sha256(derived_path) != selected["derived_annotation_sha256"] or review.get("annotation_sha256") != sha256(derived_path):
                    raise ValueError("Final human-reviewed derived SHA256 mismatch")
                actual = json.loads(derived_path.read_text(encoding="utf-8-sig"))
                expected_document = derive_document(annotation, frame, visible, frame_records, source_hash)
                if actual != expected_document:
                    raise ValueError("Converted GT does not equal authorized deterministic conversion; no coordinate edits or unrecorded masks")
            except (ValueError, KeyError, TypeError) as error:
                errors.append(str(error))
        rows.append({"frame_id": stem, "pot_visibility": visible.get("pot_visibility", "UNVERIFIED"),
                     "status": "VERIFIED_V2" if not errors else "HUMAN_REVIEW_REQUIRED", "errors": errors})
    allowed = all(row["status"] == "VERIFIED_V2" for row in rows)
    return {"protocol": PROTOCOL, "frames": 32, "original_candidates_verified": 71,
            "verified_frames": sum(row["status"] == "VERIFIED_V2" for row in rows), "evaluation_allowed": allowed,
            "READY_FOR_PHASE20D_EVALUATION": "YES" if allowed else "NO", "per_frame": rows,
            "predictions_executed": False, "accuracy_metrics_computed": False,
            "limitation": "Protocol readiness only; no independent segmentation result or method lock is claimed."}


def main():
    parser = argparse.ArgumentParser(description="Versioned visible-region human-only V2 Gate; never runs independent evaluation")
    parser.add_argument("--stage-dir", type=Path, default=STAGE)
    parser.add_argument("--records-dir", type=Path, default=PACKAGE)
    parser.add_argument("--converted-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    try:
        result = preflight(args.stage_dir.resolve(), args.records_dir.resolve(), args.converted_manifest.resolve())
    except (ValueError, KeyError, OSError, TypeError) as error:
        result = {"protocol": PROTOCOL, "evaluation_allowed": False, "READY_FOR_PHASE20D_EVALUATION": "NO",
                  "predictions_executed": False, "accuracy_metrics_computed": False, "error": str(error)}
    write_json(args.output, result)
    if args.report:
        text = ["# GT32 Gate V2 report", "", f"READY_FOR_PHASE20D_EVALUATION={result['READY_FOR_PHASE20D_EVALUATION']}", "",
                f"Protocol: {PROTOCOL}; verified frames: {result.get('verified_frames', 0)}/32.",
                "Original visible-region annotation is preserved. Missing pot alone is not an annotation error.",
                "No prediction, independent metric, method lock or Phase20E execution.", ""]
        if "error" in result:
            text.append(result["error"])
        for row in result.get("per_frame", []):
            text.append(f"## {row['frame_id']} — {row['status']}")
            text += ["- " + error for error in row["errors"]]
        write_text(args.report, "\n".join(text))
    print(json.dumps({key: value for key, value in result.items() if key != "per_frame"}, ensure_ascii=False, indent=2))
    return 0 if result["evaluation_allowed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
