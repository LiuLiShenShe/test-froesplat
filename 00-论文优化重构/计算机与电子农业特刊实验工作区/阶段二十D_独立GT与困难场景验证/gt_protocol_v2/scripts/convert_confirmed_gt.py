from __future__ import annotations

import argparse
import json
from pathlib import Path

from protocol_v2 import (PACKAGE, PROTOCOL, STAGE, derive_document, json_digest,
                         read_csv, sha256, unique_index, validate_source, verify_original_sources)
from phase20d_common import relative, write_csv, write_json


def convert(stage, records_dir, output):
    frames = read_csv(stage / "TEST32_INPUT_MANIFEST.csv")
    verify_original_sources(stage, frames)
    identities = unique_index(read_csv(records_dir / "SOURCE_IDENTITIES.csv"), "frame_id")
    visibility = unique_index(read_csv(records_dir / "GT_VISIBILITY_MANIFEST.csv"), "frame_id")
    shape_rows = read_csv(records_dir / "LINESTRIP_CONVERSION_AUDIT.csv")
    shape_records = {}
    for row in shape_rows:
        key = (row["frame_id"], int(row["shape_index"]))
        if key in shape_records:
            raise ValueError("Duplicate shape record")
        shape_records[key] = row
    expected = {frame["frame_id"] for frame in frames}
    if len(frames) != 32 or set(identities) != expected or set(visibility) != expected or {stem for stem, index in shape_records} != expected:
        raise ValueError("Frozen32 frame/source/shape record coverage must be preserved")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Refusing to overwrite any conversion run; use a fresh output directory")
    manifest, ledger = [], []
    for frame in frames:
        stem = frame["frame_id"]
        identity = identities[stem]
        if any(row.get("protocol") != PROTOCOL or row.get("challenge_id") != frame["challenge_id"] or row.get("rgb_sha256") != frame["rgb_sha256"]
               for row in (identity, visibility[stem])):
            raise ValueError("Versioned record/frozen identity disagreement")
        original = stage / "annotation_package/annotations" / f"{stem}.json"
        source_hash = sha256(original)
        if identity["source_annotation_sha256"] != source_hash or identity["rgb_sha256"] != frame["rgb_sha256"]:
            raise ValueError("Identity inventory does not match original human JSON/RGB")
        annotation = json.loads(original.read_text(encoding="utf-8-sig"))
        validate_source(annotation, frame, original, source_hash)
        selected_records = {index: row for (record_stem, index), row in shape_records.items() if record_stem == stem}
        row = {"protocol": PROTOCOL, "challenge_id": frame["challenge_id"], "frame_id": stem,
               "source_annotation_json": relative(original.resolve()), "source_annotation_sha256": source_hash,
               "rgb_sha256": frame["rgb_sha256"], "derived_annotation_json": "", "derived_annotation_sha256": "",
               "conversion_status": "HUMAN_REVIEW_REQUIRED", "pot_visibility": visibility[stem]["pot_visibility"],
               "converted_linestrip_count": "", "derived_plant_shape_count": "", "human_final_review_status": "NOT_VERIFIED",
               "conversion_reason": "", "errors_json": "[]"}
        try:
            if any(record.get("protocol") != PROTOCOL or record.get("challenge_id") != frame["challenge_id"]
                   or record.get("rgb_sha256") != frame["rgb_sha256"] for record in selected_records.values()):
                raise ValueError("Shape confirmation/frozen frame/RGB identity disagreement")
            derived = derive_document(annotation, frame, visibility[stem], selected_records, source_hash)
        except (ValueError, KeyError, TypeError) as error:
            row["errors_json"] = json.dumps([str(error)], ensure_ascii=False)
            manifest.append(row)
            continue
        destination = output / "annotations" / f"{stem}.json"
        write_json(destination, derived)
        if sha256(original) != source_hash:
            raise ValueError("Original manual annotation changed during conversion")
        row.update(derived_annotation_json=relative(destination), derived_annotation_sha256=sha256(destination),
                   conversion_status="CONVERTED_PENDING_FINAL_HUMAN_REVIEW", conversion_reason="EXPLICIT_HUMAN_CONFIRMED_VISIBLE_REGION_FORMAT_CONVERSION",
                   converted_linestrip_count=len(derived["gt_protocol_v2"]["linestrip_to_polygon_indices"]),
                   derived_plant_shape_count=len(derived["gt_protocol_v2"]["plant_equal_potted_source_indices"]))
        manifest.append(row)
        for index, shape in enumerate(annotation["shapes"]):
            confirmation = selected_records[index]
            converted_outline = shape.get("shape_type", "polygon") == "linestrip"
            ledger.append({"frame_id": stem, "shape_index": index, "source_shape_index": index, "derived_shape_index": index,
                           "label": shape["label"], "operation": "LINESTRIP_TO_POLYGON" if converted_outline else "EXISTING_POLYGON_RETAINED",
                           "source_hash": source_hash, "derived_hash": sha256(destination),
                           "source_shape_hash": json_digest(shape), "derived_shape_hash": json_digest(derived["shapes"][index]),
                           "source_points_hash": json_digest(shape["points"]),
                           "derived_points_hash": json_digest(derived["shapes"][index]["points"]),
                           "conversion_rule": "ONLY_SHAPE_TYPE_AND_IMPLICIT_CLOSURE_NO_VERTEX_CHANGE" if converted_outline else "NO_CHANGE",
                           "human_confirmation_status": confirmation.get("confirmation_status", "ORIGINAL_POLYGON"),
                           "confirmed_by": confirmation.get("confirmed_by", ""), "confirmed_at": confirmation.get("confirmed_at", ""),
                           "confirmed_source": confirmation.get("confirmed_source", ""), "conversion_reason": confirmation.get("confirmation_reason", "")})
        for offset, source_index in enumerate(derived["gt_protocol_v2"]["plant_equal_potted_source_indices"]):
            derived_index = len(annotation["shapes"]) + offset
            source_shape, derived_shape = annotation["shapes"][source_index], derived["shapes"][derived_index]
            consent = visibility[stem]
            ledger.append({"frame_id": stem, "shape_index": derived_index, "source_shape_index": source_index,
                           "derived_shape_index": derived_index, "label": "plant", "operation": "AUTHORIZED_PLANT_EQUALS_POTTED",
                           "source_hash": source_hash, "derived_hash": sha256(destination),
                           "source_shape_hash": json_digest(source_shape), "derived_shape_hash": json_digest(derived_shape),
                           "source_points_hash": json_digest(source_shape["points"]), "derived_points_hash": json_digest(derived_shape["points"]),
                           "conversion_rule": consent["plant_derivation_rule"], "human_confirmation_status": "EXPLICIT_EQUAL_DERIVATION_CONFIRMED",
                           "confirmed_by": consent["plant_derivation_confirmed_by"], "confirmed_at": consent["plant_derivation_confirmed_at"],
                           "confirmed_source": consent["plant_derivation_confirmed_source"],
                           "conversion_reason": "RECORDED_HUMAN_CONFIRMED_NOT_VISIBLE_POT_AND_PLANT_ONLY"})
    write_csv(output / "CONVERTED_GT_MANIFEST.csv", manifest)
    ledger_fields = ["frame_id", "shape_index", "source_shape_index", "derived_shape_index", "label", "operation", "source_hash", "derived_hash", "source_shape_hash", "derived_shape_hash",
                     "source_points_hash", "derived_points_hash", "conversion_rule", "human_confirmation_status", "confirmed_by", "confirmed_at", "confirmed_source", "conversion_reason"]
    write_csv(output / "SHAPE_CONVERSION_LEDGER.csv", ledger, fields=ledger_fields)
    result = {"protocol": PROTOCOL, "frames": 32,
              "converted_frames": sum(row["conversion_status"] == "CONVERTED_PENDING_FINAL_HUMAN_REVIEW" for row in manifest),
              "evaluation_executed": False, "human_final_approval_generated": False,
              "original_gt_modified": False, "rule": "No confirmed visibility/outline => no derived polygon written"}
    write_json(output / "CONVERSION_RESULT.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Deterministic human-authorized visible-region format conversion; no prediction or accuracy")
    parser.add_argument("--stage-dir", type=Path, default=STAGE)
    parser.add_argument("--records-dir", type=Path, default=PACKAGE)
    parser.add_argument("--output-dir", "--output", type=Path, required=True)
    args = parser.parse_args()
    convert(args.stage_dir.resolve(), args.records_dir.resolve(), args.output_dir.resolve())


if __name__ == "__main__":
    main()
