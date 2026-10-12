from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import convert_confirmed_gt as converter
import protocol_v2 as protocol
import validate_gt_v2 as gate
from test_protocol_v2 import annotation_fixture, human_review_fixture, shape_records, visibility_fixture


def save_table(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def synthetic32(tmp_path, monkeypatch):
    stage = tmp_path / "SYNTHETIC_ONLY_STAGE"
    records = tmp_path / "SYNTHETIC_ONLY_RECORDS"
    frames, identities, visibility, outlines, reviews, difficulties = [], [], [], [], [], []
    for index in range(32):
        stem = f"synthetic_sequence_{index // 4:02d}_{index:04d}"
        frame = {"frame_id": stem, "frame": f"{index:04d}", "challenge_id": f"SYNTHETIC_{index:04d}",
                 "width": "20", "height": "20", "rgb_sha256": "SYNTHETIC_RGB_ONLY"}
        frames.append(frame)
        annotation = annotation_fixture()
        annotation["imagePath"] = stem + ".jpg"
        source = stage / "annotation_package/annotations" / (stem + ".json")
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(json.dumps(annotation))
        source_hash = protocol.sha256(source)
        base = {"protocol": protocol.PROTOCOL, "challenge_id": frame["challenge_id"], "frame_id": stem,
                "rgb_sha256": frame["rgb_sha256"], "source_annotation_sha256": source_hash}
        identities.append(dict(base))
        visible = visibility_fixture()
        visible.update(base)
        visibility.append(visible)
        for shape_record in shape_records(annotation).values():
            shape_record.update(base)
            outlines.append(shape_record)
        review = human_review_fixture()
        review.update(base)
        review.update({code: "NO" for code in protocol.DIFFICULTIES})
        review["annotation_sha256"] = ""
        review["batch_record_id"] = ""
        reviews.append(review)
        difficulties.append({"frame_id": stem, **{code: "NO" for code in protocol.DIFFICULTIES}})
    save_table(stage / "TEST32_INPUT_MANIFEST.csv", frames)
    save_table(stage / "DIFFICULTY_HUMAN_VERIFICATION.csv", difficulties)
    save_table(records / "SOURCE_IDENTITIES.csv", identities)
    save_table(records / "GT_VISIBILITY_MANIFEST.csv", visibility)
    save_table(records / "LINESTRIP_CONVERSION_AUDIT.csv", outlines)
    save_table(records / "HUMAN_REVIEW_V2.csv", reviews)

    def synthetic_input_only(candidate_stage, candidate_frames):
        assert candidate_stage == stage
        assert len(candidate_frames) == 32
        assert all(frame["frame_id"].startswith("synthetic_") for frame in candidate_frames)

    monkeypatch.setattr(converter, "verify_original_sources", synthetic_input_only)
    monkeypatch.setattr(gate, "verify_original_sources", synthetic_input_only)
    return stage, records, tmp_path / "SYNTHETIC_ONLY_CONVERSION"


def bind_final_review(records, output):
    converted = protocol.unique_index(protocol.read_csv(output / "CONVERTED_GT_MANIFEST.csv"), "frame_id")
    reviews = protocol.read_csv(records / "HUMAN_REVIEW_V2.csv")
    for row in reviews:
        row["annotation_sha256"] = converted[row["frame_id"]]["derived_annotation_sha256"]
    save_table(records / "HUMAN_REVIEW_V2.csv", reviews)


def test_full32_conversion_and_gate_pass_for_synthetic_records_only(synthetic32):
    stage, records, output = synthetic32
    originals = {path: path.read_bytes() for path in (stage / "annotation_package/annotations").glob("*.json")}
    converter.convert(stage, records, output)
    bind_final_review(records, output)
    result = gate.preflight(stage, records, output / "CONVERTED_GT_MANIFEST.csv")
    assert result["verified_frames"] == 32
    assert result["evaluation_allowed"] is True
    assert result["READY_FOR_PHASE20D_EVALUATION"] == "YES"
    assert result["accuracy_metrics_computed"] is False
    assert all(path.read_bytes() == contents for path, contents in originals.items())
    assert len(protocol.read_csv(output / "SHAPE_CONVERSION_LEDGER.csv")) == 96


def test_missing_visibility_blocks_conversion_without_filling_gt(synthetic32):
    stage, records, output = synthetic32
    rows = protocol.read_csv(records / "GT_VISIBILITY_MANIFEST.csv")
    for row in rows:
        row["pot_visibility"] = "UNVERIFIED"
    save_table(records / "GT_VISIBILITY_MANIFEST.csv", rows)
    converter.convert(stage, records, output)
    assert not list(output.rglob("annotations/*.json"))
    manifest = protocol.read_csv(output / "CONVERTED_GT_MANIFEST.csv")
    assert len(manifest) == 32
    assert all(row["derived_annotation_sha256"] == "" for row in manifest)
    assert gate.preflight(stage, records, output / "CONVERTED_GT_MANIFEST.csv")["evaluation_allowed"] is False


def test_shape_rgb_identity_disagreement_stops_affected_frame(synthetic32):
    stage, records, output = synthetic32
    rows = protocol.read_csv(records / "LINESTRIP_CONVERSION_AUDIT.csv")
    rows[0]["rgb_sha256"] = "WRONG_RGB"
    save_table(records / "LINESTRIP_CONVERSION_AUDIT.csv", rows)
    converter.convert(stage, records, output)
    manifest = protocol.read_csv(output / "CONVERTED_GT_MANIFEST.csv")
    assert manifest[0]["derived_annotation_sha256"] == ""
    assert "identity disagreement" in manifest[0]["errors_json"]
    assert sum(bool(row["derived_annotation_sha256"]) for row in manifest) == 31


def test_derived_point_change_cannot_pass_even_if_hash_fields_are_updated(synthetic32):
    stage, records, output = synthetic32
    converter.convert(stage, records, output)
    rows = protocol.read_csv(output / "CONVERTED_GT_MANIFEST.csv")
    path = protocol.ROOT / rows[0]["derived_annotation_json"]
    document = json.loads(path.read_text())
    document["shapes"][0]["points"][0][0] += 0.25
    path.write_text(json.dumps(document))
    rows[0]["derived_annotation_sha256"] = protocol.sha256(path)
    save_table(output / "CONVERTED_GT_MANIFEST.csv", rows)
    bind_final_review(records, output)
    result = gate.preflight(stage, records, output / "CONVERTED_GT_MANIFEST.csv")
    assert result["verified_frames"] == 31
    assert result["evaluation_allowed"] is False
    assert any("does not equal authorized" in error for error in result["per_frame"][0]["errors"])


def test_final_human_sha_binding_is_required(synthetic32):
    stage, records, output = synthetic32
    converter.convert(stage, records, output)
    result = gate.preflight(stage, records, output / "CONVERTED_GT_MANIFEST.csv")
    assert result["verified_frames"] == 0
    assert all(any("Final human-reviewed derived SHA256 mismatch" in error for error in row["errors"])
               for row in result["per_frame"])


def test_historical_difficulty_decisions_cannot_be_reclassified(synthetic32):
    stage, records, output = synthetic32
    converter.convert(stage, records, output)
    bind_final_review(records, output)
    reviews = protocol.read_csv(records / "HUMAN_REVIEW_V2.csv")
    reviews[0].update(D1="YES", human_difficulty_labels="D1", difficulty_evidence_json='{"D1":"SYNTHETIC_ONLY"}')
    save_table(records / "HUMAN_REVIEW_V2.csv", reviews)
    result = gate.preflight(stage, records, output / "CONVERTED_GT_MANIFEST.csv")
    assert result["verified_frames"] == 31
    assert any("Historical manual D decisions changed" in error for error in result["per_frame"][0]["errors"])


def test_partial_or_duplicate_frame_coverage_is_rejected(synthetic32):
    stage, records, output = synthetic32
    rows = protocol.read_csv(records / "SOURCE_IDENTITIES.csv")
    save_table(records / "SOURCE_IDENTITIES.csv", rows[:-1] + [rows[0]])
    with pytest.raises(ValueError):
        converter.convert(stage, records, output)


def test_existing_conversion_run_is_never_overwritten(synthetic32):
    stage, records, output = synthetic32
    converter.convert(stage, records, output)
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        converter.convert(stage, records, output)


def test_authorized_plant_derivation_has_separate_human_bound_ledger_entry(synthetic32):
    stage, records, output = synthetic32
    frame = protocol.read_csv(stage / "TEST32_INPUT_MANIFEST.csv")[0]
    stem = frame["frame_id"]
    source = stage / "annotation_package/annotations" / (stem + ".json")
    annotation = json.loads(source.read_text())
    annotation["shapes"] = annotation["shapes"][:1]
    source.write_text(json.dumps(annotation))
    source_hash = protocol.sha256(source)
    for filename in ("SOURCE_IDENTITIES.csv", "GT_VISIBILITY_MANIFEST.csv", "HUMAN_REVIEW_V2.csv"):
        rows = protocol.read_csv(records / filename)
        rows[0]["source_annotation_sha256"] = source_hash
        if filename == "GT_VISIBILITY_MANIFEST.csv":
            rows[0]["pot_visibility"] = "NOT_VISIBLE"
        save_table(records / filename, rows)
    rows = protocol.read_csv(records / "LINESTRIP_CONVERSION_AUDIT.csv")
    rows = [row for row in rows if row["frame_id"] != stem or row["shape_index"] == "0"]
    rows[0]["source_annotation_sha256"] = source_hash
    save_table(records / "LINESTRIP_CONVERSION_AUDIT.csv", rows)
    converter.convert(stage, records, output)
    ledger = [row for row in protocol.read_csv(output / "SHAPE_CONVERSION_LEDGER.csv") if row["frame_id"] == stem]
    assert len(ledger) == 2
    assert ledger[1]["operation"] == "AUTHORIZED_PLANT_EQUALS_POTTED"
    assert ledger[1]["source_shape_index"] == "0" and ledger[1]["derived_shape_index"] == "1"
    assert ledger[1]["source_points_hash"] == ledger[1]["derived_points_hash"]
    assert ledger[1]["conversion_rule"] == protocol.DERIVATION_RULE
    assert ledger[1]["confirmed_source"] == "SYNTHETIC_TEST_ONLY"
    bind_final_review(records, output)
    assert gate.preflight(stage, records, output / "CONVERTED_GT_MANIFEST.csv")["evaluation_allowed"] is True
