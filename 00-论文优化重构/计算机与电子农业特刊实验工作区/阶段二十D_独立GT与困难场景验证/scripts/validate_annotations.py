from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from phase20d_common import (DIFFICULTIES, FIX, PHASE16, PROD, ROOT, SOURCE, WORK, git_bytes,
                            load_module, read_csv, relative, sha256, unique_index, write_json)


MACHINE_NAMES = {"ai", "sam3", "sam", "r0", "r1", "agnes", "codex", "chatgpt", "gpt", "claude", "assistant"}


def validate_human_review(row):
    if row.get("approval") != "APPROVED":
        raise ValueError("USER_ANNOTATION_REQUIRED: human approval missing")
    people = {field: row.get(field, "").strip() for field in ("annotator", "reviewer", "second_reviewer")}
    if any(not person or person.casefold() in MACHINE_NAMES for person in people.values()):
        raise ValueError("Named real human annotator/reviewers required; AI cannot approve")
    if people["second_reviewer"].casefold() in {people["reviewer"].casefold(), people["annotator"].casefold()}:
        raise ValueError("Distinct independent second reviewer required")
    confirmations = ("annotator_is_human", "reviewer_is_human", "second_reviewer_is_human",
                     "manual_annotation_confirmed", "target_identity_confirmed", "plant_confirmed",
                     "pot_confirmed", "difficulty_manually_confirmed")
    if any(row.get(field) != "YES" for field in confirmations):
        raise ValueError("Explicit manual annotation and real-human confirmations required")
    if row.get("annotation_origin") != "MANUAL_POLYGONS_ONLY":
        raise ValueError("SAM/R1/automatic predictions cannot be GT")
    if not row.get("target_identity_description", "").strip():
        raise ValueError("Manual target-instance identity description required")
    if not row.get("reviewed_at", "").strip() or not row.get("second_reviewed_at", "").strip():
        raise ValueError("Both human review timestamps required")
    if row.get("blue_cube_present") not in ("YES", "NO"):
        raise ValueError("Manual blue-cube presence decision required")
    labels = row.get("human_difficulty_labels", "").split(";")
    if not labels or len(set(labels)) != len(labels) or not set(labels) <= set(DIFFICULTIES) | {"none"}:
        raise ValueError("Explicit D1/D3/D6/D11/D12 labels or none required; no heuristic defaults")
    if "none" in labels and len(labels) != 1:
        raise ValueError("none cannot coexist with difficulty labels")
    try:
        evidence = json.loads(row.get("difficulty_evidence_json", ""))
    except (ValueError, TypeError) as error:
        raise ValueError("Human visual difficulty evidence JSON required") from error
    if not isinstance(evidence, dict) or set(evidence) != set(labels):
        raise ValueError("Evidence must cover exactly the human-confirmed labels")
    if any(not isinstance(description, str) or not description.strip() for description in evidence.values()):
        raise ValueError("Every label needs a human visual description (including none)")


def validate_geometry(annotation, frame, review, annotation_path):
    if sha256(annotation_path) != review.get("annotation_sha256"):
        raise ValueError("Approved annotation SHA256 mismatch; new independent review required")
    if review.get("rgb_sha256") != frame["rgb_sha256"]:
        raise ValueError("Review/frozen RGB SHA256 mismatch")
    expected_names = {f"{frame['frame_id']}.jpg", f"{frame['frame']}.jpg"}
    if Path(annotation.get("imagePath", "")).name not in expected_names:
        raise ValueError("Annotation imagePath/frozen frame identity mismatch")
    if annotation.get("imageData"):
        import hashlib
        embedded = base64.b64decode(annotation["imageData"], validate=True)
        if hashlib.sha256(embedded).hexdigest() != frame["rgb_sha256"]:
            raise ValueError("Embedded annotation image is not the original frozen RGB")
    for shape in annotation.get("shapes", []):
        points = np.asarray(shape.get("points", []), dtype=float)
        if points.ndim != 2 or points.shape[1] != 2 or len(np.unique(points, axis=0)) < 3:
            raise ValueError("Degenerate human polygon")
        twice_area = np.dot(points[:, 0], np.roll(points[:, 1], 1)) - np.dot(points[:, 1], np.roll(points[:, 0], 1))
        if not np.isfinite(twice_area) or twice_area == 0:
            raise ValueError("Degenerate or invalid human polygon area")
    sys.path.insert(0, str(FIX / "scripts"))
    gate = load_module("phase20d_original_polygon_gate", FIX / "scripts/evaluate_independent_test.py")
    target, masks = gate.annotation_masks(annotation, int(frame["width"]), int(frame["height"]))
    if not target.any():
        raise ValueError("potted_clean cannot be empty")
    if bool(masks["blue_cube"].any()) != (review["blue_cube_present"] == "YES"):
        raise ValueError("Blue cube polygons disagree with human review")
    for label in ("plant", "pot"):
        if (masks[label].astype(bool) & ~masks["potted_plant"].astype(bool)).any():
            raise ValueError(f"Target {label} extends outside target potted_plant; human correction required")


def validate_difficulty(row, review, frame):
    if row.get("status") != "APPROVED" or row.get("rgb_sha256") != frame["rgb_sha256"]:
        raise ValueError("Independent human difficulty audit is not approved for this RGB")
    for field in ("annotation_sha256", "reviewer", "second_reviewer", "human_difficulty_labels", "difficulty_evidence_json"):
        if row.get(field) != review.get(field):
            raise ValueError(f"Difficulty audit/review disagreement: {field}")
    labels = set(review["human_difficulty_labels"].split(";"))
    for code in DIFFICULTIES:
        if row.get(code) != ("YES" if code in labels else "NO"):
            raise ValueError(f"Manual {code} decision missing/inconsistent")


def verify_frozen_inputs(stage, frames):
    frozen_path = PHASE16 / "挑战集列表/Phase16_manifest.json"
    test_path = PHASE16 / "挑战集列表/Phase16_TEST_manifest.csv"
    if frozen_path.read_bytes() != git_bytes(frozen_path) or test_path.read_bytes() != git_bytes(test_path):
        raise ValueError("Immutable Phase16 TEST32 source changed")
    frozen = json.loads(frozen_path.read_text())
    authoritative = unique_index(read_csv(test_path), "challenge_id")
    spec = json.loads((stage / "configs/FROZEN_TEST_SPEC.json").read_text())
    if spec["test_manifest_sha256"] != sha256(test_path) or spec["input_manifest_sha256"] != sha256(stage / "TEST32_INPUT_MANIFEST.csv"):
        raise ValueError("Frozen TEST32 input manifest hash mismatch")
    if set(authoritative) != {frame["challenge_id"] for frame in frames}:
        raise ValueError("Frozen TEST32 frame identities changed; no subset or replacement allowed")
    for name in ("ranking_policy.py", "candidate_metadata.py", "生成RAP-FSAM3掩膜.py"):
        if (PROD / name).read_bytes() != git_bytes(PROD / name):
            raise ValueError("Frozen foreground production source changed")
    prior_candidates_path = FIX / "TEST32_CANDIDATE_AUDIT.csv"
    if prior_candidates_path.read_bytes() != git_bytes(prior_candidates_path):
        raise ValueError("Original candidate identity/hash inventory changed")
    expected_masks = unique_index(read_csv(prior_candidates_path), "path")
    if spec["original_candidates"] != 71 or len(expected_masks) != 71:
        raise ValueError("TEST32 must keep all 71 original P6 candidates")
    loader = load_module("phase20d_preflight_frozen_metadata", PROD / "candidate_metadata.py")
    for frame in frames:
        source = authoritative[frame["challenge_id"]]
        stem = f"{source['sample']}_{source['frame']}"
        if any(frame[field] != source[field] for field in ("sample", "sequence", "frame")) or frame["frame_id"] != stem:
            raise ValueError("TEST32 frame/sample/sequence identity mismatch")
        rgb = Path(source["image_path"])
        if ROOT / frame["rgb_path"] != rgb or sha256(rgb) != frozen["image_sha256"][source["challenge_id"]] or frame["rgb_sha256"] != sha256(rgb):
            raise ValueError("Original TEST32 RGB identity mismatch")
        with Image.open(rgb) as handle:
            if handle.size != (int(frame["width"]), int(frame["height"])):
                raise ValueError("Original RGB coordinate grid mismatch")
        metadata_path = ROOT / frame["candidate_metadata_path"]
        if not metadata_path.is_relative_to(stage / "metadata") or sha256(metadata_path) != frame["candidate_metadata_sha256"]:
            raise ValueError("Strict metadata snapshot hash/path mismatch")
        raw_paths = list((SOURCE / "候选掩膜/raw_instance_P6").glob(f"mask_{stem}_*.png"))
        mask_ids = {int(path.stem.rsplit("_", 1)[1]) for path in raw_paths}
        authoritative_paths = {path for path, original in expected_masks.items() if original["challenge_id"] == frame["challenge_id"]}
        if {relative(path) for path in raw_paths} != authoritative_paths:
            raise ValueError("Original candidate set incomplete; do not replace or skip affected frame")
        indexed = loader.load_cache_metadata(stage / f"inputs/TEST32/{stem}.jpg", SOURCE, metadata_path.parent)
        if len(raw_paths) != int(frame["candidate_count_verified"]) or mask_ids != {instance for prompt, instance in indexed if prompt == "P6"}:
            raise ValueError("Candidate identity set changed; affected frame stopped")
        detail_path = SOURCE / f"候选掩膜/候选评分明细_{stem}.csv"
        details = {int(row["实例编号"]): row for row in read_csv(detail_path)}
        for raw in raw_paths:
            original = expected_masks[relative(raw)]
            if sha256(raw) != original["sha256"] or sha256(detail_path) != original["detail_sha256"]:
                raise ValueError("Original candidate mask/detail identity changed")
            instance = int(raw.stem.rsplit("_", 1)[1])
            entry, status = loader.metadata_for_mask(indexed, "P6", instance, raw, strict=True)
            detail = details[instance]
            if entry["sam_score"] != float(detail["SAM3分数"]) or entry["box"] != tuple(int(value) for value in detail["外接框"].split(",")) or entry["candidate_order"] != instance or entry["mask_threshold"] != float(detail["掩膜阈值"]):
                raise ValueError("Archived score/box/order/threshold changed")


def preflight(stage=WORK):
    stage = Path(stage).resolve()
    frames = read_csv(stage / "TEST32_INPUT_MANIFEST.csv")
    if len(frames) != 32 or len(unique_index(frames, "challenge_id")) != 32:
        raise ValueError("Immutable TEST32 must contain exactly 32 unique frames")
    verify_frozen_inputs(stage, frames)
    reviews = read_csv(stage / "TEST32_ANNOTATION_REVIEW.csv")
    difficulty = read_csv(stage / "DIFFICULTY_HUMAN_VERIFICATION.csv")
    review_map = unique_index(reviews, "challenge_id")
    difficulty_map = unique_index(difficulty, "challenge_id")
    expected = {frame["challenge_id"] for frame in frames}
    if set(review_map) != expected or set(difficulty_map) != expected:
        raise ValueError("Full TEST32 review/difficulty coverage required; no subset evaluation")
    rows = []
    for frame in frames:
        review = review_map[frame["challenge_id"]]
        annotation_path = stage / "annotation_package/annotations" / f"{frame['frame_id']}.json"
        errors = []
        if frame.get("input_status") != "VERIFIED":
            errors.append("Original RGB/candidate identity not verified; affected frame stopped")
        if review.get("annotation_json") != frame["annotation_json"] or review.get("frame_id") != frame["frame_id"]:
            errors.append("Human review frame/annotation path identity mismatch")
        if not annotation_path.is_file():
            errors.append("Manual annotation JSON missing")
        try:
            validate_human_review(review)
        except ValueError as error:
            errors.append(str(error))
        if annotation_path.is_file():
            try:
                annotation = json.loads(annotation_path.read_text(encoding="utf-8-sig"))
                validate_geometry(annotation, frame, review, annotation_path)
            except (ValueError, KeyError, TypeError, OSError) as error:
                errors.append(str(error))
        try:
            validate_difficulty(difficulty_map[frame["challenge_id"]], review, frame)
        except ValueError as error:
            errors.append(str(error))
        rows.append({"frame_id": frame["frame_id"], "annotation_exists": annotation_path.is_file(),
                     "status": "VERIFIED" if not errors else "USER_ANNOTATION_REQUIRED", "errors": errors})
    allowed = all(row["status"] == "VERIFIED" for row in rows)
    return {"human_gt": "VERIFIED" if allowed else "USER_ANNOTATION_REQUIRED",
            "evaluation_allowed": allowed, "frames": 32,
            "annotation_files_found": sum(row["annotation_exists"] for row in rows),
            "verified_frames": sum(row["status"] == "VERIFIED" for row in rows),
            "accuracy_metrics_computed": False, "predictions_accessed": False, "per_frame": rows,
            "limitation": "Human attestations and visual evidence must be provided by actual people, not inferred by code or AI."}


def main():
    parser = argparse.ArgumentParser(description="Human-only TEST32 preflight; never computes segmentation accuracy")
    parser.add_argument("--stage-dir", type=Path, default=WORK)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = preflight(args.stage_dir)
    except (ValueError, FileNotFoundError) as error:
        result = {"human_gt": "USER_ANNOTATION_REQUIRED", "evaluation_allowed": False,
                  "accuracy_metrics_computed": False, "error": str(error)}
    write_json(args.output, result)
    print(json.dumps({key: value for key, value in result.items() if key != "per_frame"}, ensure_ascii=False))
    return 0 if result["evaluation_allowed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
