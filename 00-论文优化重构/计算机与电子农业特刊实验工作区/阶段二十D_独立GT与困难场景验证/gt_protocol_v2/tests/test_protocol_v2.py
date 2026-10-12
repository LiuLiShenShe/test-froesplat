from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import protocol_v2 as protocol
import validate_gt_v2 as gate


def frame_fixture():
    return {"frame_id": "synthetic_0001", "frame": "0001", "width": "20", "height": "20", "rgb_sha256": "synthetic_rgb"}


def annotation_fixture(components=True):
    shapes = [{"label": "potted_plant", "shape_type": "linestrip", "points": [[2, 2], [17, 2], [17, 17], [2, 17]]}]
    if components:
        shapes += [{"label": "plant", "shape_type": "linestrip", "points": [[3, 3], [16, 3], [16, 10], [3, 10]]},
                   {"label": "pot", "shape_type": "linestrip", "points": [[5, 11], [14, 11], [14, 16], [5, 16]]}]
    return {"imageWidth": 20, "imageHeight": 20, "imagePath": "synthetic_0001.jpg", "shapes": shapes}


def visibility_fixture(pot_visibility="VISIBLE"):
    return {"source_annotation_sha256": "synthetic_hash", "pot_visibility": pot_visibility, "plant_visible": "YES",
            "blue_cube_present": "NO", "visibility_confirmed_by": "Fixture Human A", "visibility_confirmed_is_human": "YES",
            "visibility_confirmed_at": "2026-10-12T10:00:00+08:00", "visibility_confirmed_source": "SYNTHETIC_TEST_ONLY",
            "visibility_status": "CONFIRMED", "potted_only_plant_confirmed": "YES",
            "plant_equal_potted_derivation_confirmed": "YES", "plant_derivation_confirmed_by": "Fixture Human A",
            "plant_derivation_confirmed_is_human": "YES", "plant_derivation_confirmed_at": "2026-10-12T10:00:00+08:00",
            "plant_derivation_confirmed_source": "SYNTHETIC_TEST_ONLY", "plant_derivation_rule": protocol.DERIVATION_RULE}


def shape_records(annotation):
    return {index: {"source_annotation_sha256": "synthetic_hash", "shape_index": str(index),
                    "source_shape_sha256": protocol.json_digest(shape), "closed_region_confirmed": "YES",
                    "complete_region_boundary": "YES", "obvious_open_gap": "NO", "confirmed_by": "Fixture Human A",
                    "confirmed_is_human": "YES", "confirmed_at": "2026-10-12T10:00:00+08:00",
                    "confirmed_source": "SYNTHETIC_TEST_ONLY", "confirmation_reason": "Synthetic square denotes a closed region",
                    "confirmation_status": "CONFIRMED_CLOSED_REGION"}
            for index, shape in enumerate(annotation["shapes"])}


def human_review_fixture():
    return {"approval": "APPROVED", "protocol": protocol.PROTOCOL, "annotator": "Fixture Human A", "reviewer": "Fixture Human A",
            "second_reviewer": "Fixture Human B", "annotator_is_human": "YES", "reviewer_is_human": "YES",
            "second_reviewer_is_human": "YES", "manual_annotation_confirmed": "YES", "target_identity_confirmed": "YES",
            "visible_components_confirmed": "YES", "difficulty_manually_confirmed": "YES",
            "annotation_origin": "MANUAL_VISIBLE_REGIONS_WITH_RECORDED_FORMAT_DERIVATION",
            "target_identity_description": "Synthetic fixture, not an actual TEST32 claim",
            "reviewed_at": "2026-10-12T10:00:00+08:00", "second_reviewed_at": "2026-10-12T11:00:00+08:00",
            "blue_cube_present": "NO", "human_difficulty_labels": "none", "difficulty_evidence_json": '{"none":"Synthetic test evidence"}'}


def test_visible_pot_with_all_regions_converts_without_vertex_change():
    source = annotation_fixture()
    before = copy.deepcopy(source)
    derived = protocol.derive_document(source, frame_fixture(), visibility_fixture(), shape_records(source), "synthetic_hash")
    assert source == before
    assert all(derived["shapes"][index]["points"] == shape["points"] for index, shape in enumerate(source["shapes"]))
    assert all(shape["shape_type"] == "polygon" for shape in derived["shapes"])
    assert derived["gt_protocol_v2"]["plant_equal_potted_source_indices"] == []


def test_confirmed_not_visible_pot_allows_exact_plant_derivation():
    source = annotation_fixture(False)
    derived = protocol.derive_document(source, frame_fixture(), visibility_fixture("NOT_VISIBLE"), shape_records(source), "synthetic_hash")
    target, masks = protocol.validate_visible_masks(derived, frame_fixture(), visibility_fixture("NOT_VISIBLE"))
    assert np.array_equal(masks["plant"], masks["potted_plant"])
    assert not masks["pot"].any()
    assert np.array_equal(target, masks["potted_plant"].astype(bool))
    assert derived["shapes"][0]["points"] == derived["shapes"][1]["points"]
    assert len(source["shapes"]) == 1
    assert derived["gt_protocol_v2"]["plant_derivation_rule"] == protocol.DERIVATION_RULE


@pytest.mark.parametrize("visibility", ["VISIBLE", "PARTIALLY_VISIBLE", "UNVERIFIED"])
def test_missing_pot_never_implies_not_visible(visibility):
    source = annotation_fixture(False)
    with pytest.raises(ValueError):
        protocol.derive_document(source, frame_fixture(), visibility_fixture(visibility), shape_records(source), "synthetic_hash")


def test_partially_visible_pot_requires_real_polygon():
    source = annotation_fixture()
    protocol.derive_document(source, frame_fixture(), visibility_fixture("PARTIALLY_VISIBLE"), shape_records(source), "synthetic_hash")
    source["shapes"] = source["shapes"][:2]
    with pytest.raises(ValueError, match="Pot polygons"):
        protocol.derive_document(source, frame_fixture(), visibility_fixture("PARTIALLY_VISIBLE"), shape_records(source), "synthetic_hash")


@pytest.mark.parametrize("field", ["potted_only_plant_confirmed", "plant_equal_potted_derivation_confirmed",
                                  "plant_derivation_confirmed_by", "plant_derivation_confirmed_is_human",
                                  "plant_derivation_confirmed_at", "plant_derivation_confirmed_source", "plant_derivation_rule"])
def test_plant_derivation_requires_each_real_confirmation(field):
    source = annotation_fixture(False)
    visibility = visibility_fixture("NOT_VISIBLE")
    visibility[field] = ""
    with pytest.raises(ValueError):
        protocol.derive_document(source, frame_fixture(), visibility, shape_records(source), "synthetic_hash")


@pytest.mark.parametrize("field,value", [("closed_region_confirmed", ""), ("obvious_open_gap", "YES"),
                                        ("complete_region_boundary", ""), ("confirmed_by", "AI"),
                                        ("confirmed_is_human", ""), ("confirmed_at", ""), ("confirmed_source", ""),
                                        ("confirmation_reason", ""), ("confirmation_status", "PENDING")])
def test_open_linestrip_is_not_automatically_closed(field, value):
    source = annotation_fixture()
    records = shape_records(source)
    records[0][field] = value
    with pytest.raises(ValueError):
        protocol.derive_document(source, frame_fixture(), visibility_fixture(), records, "synthetic_hash")


def test_shape_hash_binds_points_and_confirmation():
    source = annotation_fixture()
    records = shape_records(source)
    source["shapes"][0]["points"][0] = [1, 2]
    with pytest.raises(ValueError, match="hash/index mismatch"):
        protocol.derive_document(source, frame_fixture(), visibility_fixture(), records, "synthetic_hash")


def test_closed_and_open_self_intersections_report_edge_indices():
    crossing = {"label": "plant", "points": [[1, 1], [9, 9], [1, 9], [9, 1]], "shape_type": "linestrip"}
    record = protocol.shape_geometry(crossing, 20, 20)
    assert record["geometry_status"] == "INVALID"
    assert [0, 2] in json.loads(record["closed_intersection_edge_pairs_json"])
    assert record["open_self_intersection_count"] == 1


def test_only_closure_segment_creates_crossing():
    shape = {"label": "plant", "points": [[1, 1], [8, 1], [2, 5], [8, 8]], "shape_type": "linestrip"}
    record = protocol.shape_geometry(shape, 20, 20)
    assert record["open_self_intersection_count"] == 0
    assert record["closed_self_intersection_count"] > 0


def test_adjacent_edge_backtracking_is_not_a_valid_simple_polygon():
    points = np.asarray([[2, 2], [8, 2], [5, 2], [8, 8], [2, 8]], dtype=float)
    assert [0, 1] in protocol.intersection_pairs(points, False)
    assert [0, 1] in protocol.intersection_pairs(points, True)
    shape = {"label": "potted_plant", "shape_type": "polygon", "points": points.tolist()}
    assert protocol.shape_geometry(shape, 20, 20)["geometry_status"] == "INVALID"


def test_consecutive_identical_points_are_reported_not_removed():
    shape = {"label": "potted_plant", "shape_type": "polygon",
             "points": [[2, 2], [8, 2], [8, 2], [8, 8], [2, 8]]}
    before = copy.deepcopy(shape)
    assert "ZERO_LENGTH_OPEN_EDGE" in protocol.shape_geometry(shape, 20, 20)["geometry_errors_json"]
    assert shape == before


def test_endpoint_distance_not_an_approval_threshold():
    square = annotation_fixture()["shapes"][0]
    info = protocol.shape_geometry(square, 20, 20)
    assert info["endpoint_distance_px"] == 15
    assert info["geometry_status"] == "VALID_CLOSED_GEOMETRY_ONLY"
    records = shape_records(annotation_fixture())
    records[0]["closed_region_confirmed"] = ""
    with pytest.raises(ValueError, match="HUMAN_REVIEW_REQUIRED"):
        protocol.derive_document(annotation_fixture(), frame_fixture(), visibility_fixture(), records, "synthetic_hash")


def test_duplicate_endpoint_is_retained_not_repaired():
    source = annotation_fixture(False)
    source["shapes"][0]["points"].append(source["shapes"][0]["points"][0].copy())
    derived = protocol.derive_document(source, frame_fixture(), visibility_fixture("NOT_VISIBLE"), shape_records(source), "synthetic_hash")
    assert len(derived["shapes"][0]["points"]) == 5
    assert derived["shapes"][0]["points"] == source["shapes"][0]["points"]


@pytest.mark.parametrize("point", [[-1, 2], [20, 2], [2, 20], [float("nan"), 2]])
def test_illegal_points_cannot_be_clipped_or_repaired(point):
    shape = annotation_fixture()["shapes"][0]
    shape["points"][0] = point
    assert protocol.shape_geometry(shape, 20, 20)["geometry_status"] == "INVALID"


def test_no_plant_or_pot_visible_contradicts_nonempty_target():
    source = annotation_fixture(False)
    visibility = visibility_fixture("NOT_VISIBLE")
    visibility["plant_visible"] = "NO"
    with pytest.raises(ValueError, match="actually visible"):
        protocol.derive_document(source, frame_fixture(), visibility, shape_records(source), "synthetic_hash")


def test_blue_cube_still_subtracted_from_primary_gt():
    source = annotation_fixture()
    source["shapes"].append({"label": "blue_cube", "shape_type": "linestrip", "points": [[8, 8], [11, 8], [11, 11], [8, 11]]})
    visibility = visibility_fixture()
    visibility["blue_cube_present"] = "YES"
    derived = protocol.derive_document(source, frame_fixture(), visibility, shape_records(source), "synthetic_hash")
    target, masks = protocol.validate_visible_masks(derived, frame_fixture(), visibility)
    assert np.array_equal(target, masks["potted_plant"].astype(bool) & ~masks["blue_cube"].astype(bool))
    assert target.sum() < masks["potted_plant"].sum()


def test_containment_and_visibility_cannot_be_waived():
    source = annotation_fixture()
    source["shapes"][1]["points"][0] = [1, 1]
    with pytest.raises(ValueError, match="outside"):
        protocol.derive_document(source, frame_fixture(), visibility_fixture(), shape_records(source), "synthetic_hash")
    source = annotation_fixture()
    with pytest.raises(ValueError, match="Pot polygons"):
        protocol.derive_document(source, frame_fixture(), visibility_fixture("NOT_VISIBLE"), shape_records(source), "synthetic_hash")


@pytest.mark.parametrize("field,value", [("approval", "PENDING"), ("protocol", "V1"), ("reviewer", "SAM3"),
                                        ("second_reviewer", "Fixture Human A"), ("visible_components_confirmed", ""),
                                        ("reviewed_at", "2026-10-12"), ("annotation_origin", "SAM3"),
                                        ("difficulty_evidence_json", "{}"), ("difficulty_evidence_json", '{"none":""}')])
def test_v2_retains_actual_independent_human_approval(field, value):
    review = human_review_fixture()
    review[field] = value
    with pytest.raises(ValueError):
        protocol.validate_review_v2(review, visibility_fixture())


def test_real_review_schema_can_pass_for_synthetic_fixture_only():
    protocol.validate_review_v2(human_review_fixture(), visibility_fixture())


def test_batch_record_requires_exact_scope_and_real_completed_record():
    base = {"frame_id": "synthetic_0001", "batch_record_id": "synthetic_batch", "annotation_sha256": "synthetic_derived"}
    batch = {"batch_record_id": "synthetic_batch", "scope_source_identities_sha256": "synthetic_scope",
             "actual_human_review_completed": "YES", "human_record_source": "SYNTHETIC_TEST_ONLY", "approval": "APPROVED",
             "scope_frame_ids": ["synthetic_0001"], "scope_final_derived_sha256_by_frame": {"synthetic_0001": "synthetic_derived"},
             "reviewer": "Fixture Human A"}
    merged = gate.batch_review(base, batch, Path.cwd(), "synthetic_scope")
    assert merged["reviewer"] == "Fixture Human A"
    assert "approval" not in merged
    assert "reviewer" not in base
    batch["scope_final_derived_sha256_by_frame"]["synthetic_0001"] = "unapproved_version"
    with pytest.raises(ValueError, match="exact final"):
        gate.batch_review(base, batch, Path.cwd(), "synthetic_scope")


def test_all_manual_d_decisions_are_explicit_and_frame_specific():
    first = {code: "NO" for code in protocol.DIFFICULTIES}
    second = {**first, "D1": "YES"}
    assert protocol.derive_difficulty(first) == "none"
    assert protocol.derive_difficulty(second) == "D1"
    with pytest.raises(ValueError):
        protocol.derive_difficulty({**first, "D3": ""})
