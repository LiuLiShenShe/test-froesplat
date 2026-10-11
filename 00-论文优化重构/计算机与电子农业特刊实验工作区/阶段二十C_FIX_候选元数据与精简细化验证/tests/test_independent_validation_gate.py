from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location("phase20d_fixed_gate_test", SCRIPTS / "evaluate_independent_test.py")
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_pending_review_cannot_be_accuracy_ground_truth():
    with pytest.raises(ValueError, match="human approval"):
        module.validate_review({"approval": "PENDING", "reviewer": "", "second_reviewer": ""})


def test_two_person_confirmation_is_required():
    row = {"approval": "APPROVED", "reviewer": "a", "second_reviewer": "a",
           "target_identity_confirmed": "YES", "plant_confirmed": "YES", "pot_confirmed": "YES",
           "difficulty_manually_confirmed": "YES", "blue_cube_present": "NO", "human_difficulty_labels": "none"}
    with pytest.raises(ValueError, match="second-review"):
        module.validate_review(row)
    row["second_reviewer"] = "b"
    module.validate_review(row)


def test_heuristic_difficulty_is_not_human_confirmation():
    row = {"approval": "APPROVED", "reviewer": "a", "second_reviewer": "b",
           "target_identity_confirmed": "YES", "plant_confirmed": "YES", "pot_confirmed": "YES",
           "difficulty_manually_confirmed": "", "blue_cube_present": "NO", "human_difficulty_labels": "D1/D11"}
    with pytest.raises(ValueError):
        module.validate_review(row)


def test_label_and_coordinate_gates():
    annotation = {"imageWidth": 20, "imageHeight": 20, "shapes": []}
    with pytest.raises(ValueError, match="polygons required"):
        module.annotation_masks(annotation, 20, 20)
    with pytest.raises(ValueError, match="coordinate mismatch"):
        module.annotation_masks(annotation, 21, 20)
    annotation["shapes"] = [{"label": label, "shape_type": "polygon", "points": [[2, 2], [12, 2], [12, 12], [2, 12]]}
                            for label in ["potted_plant", "plant", "pot"]]
    gt, masks = module.annotation_masks(annotation, 20, 20)
    assert gt.any() and not masks["blue_cube"].any()
