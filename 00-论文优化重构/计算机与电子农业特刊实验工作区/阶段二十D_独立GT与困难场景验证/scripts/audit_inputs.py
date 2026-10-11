from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from phase20d_common import (ALG, BASE, BASELINE, DIFFICULTIES, FIX, PHASE16, PHASE17A,
                            PHASE17C, PROD, ROOT, SOURCE, WORK, git_bytes, load_module,
                            original_link, read_csv, relative, sha256, unique_index,
                            write_csv, write_json)


def rgb_info(path):
    with Image.open(path) as image:
        image.load()
        width, height = image.size
        original_mode = image.mode
        pixels = image.convert("RGB").tobytes()
    pixel_hash = hashlib.sha256(f"{width}x{height}:RGB:".encode() + pixels).hexdigest()
    return {"width": width, "height": height, "mode": original_mode, "rgb_sha256": sha256(path),
            "decoded_rgb_sha256": pixel_hash}


def verify_candidate(detail, historical_rows, mask, width, height, candidate_count):
    if mask.ndim != 2 or mask.shape != (height, width) or mask.dtype != np.uint8:
        raise ValueError("Candidate/RGB grid or uint8 format mismatch; resampling forbidden")
    if set(np.unique(mask)) - {0, 255}:
        raise ValueError("Original candidate is not binary {0,255}")
    area = float(np.count_nonzero(mask) / mask.size)
    if abs(area - float(detail["面积比例"])) > 1e-14:
        raise ValueError("Explicit-ID candidate/detail mask-area mismatch")
    matches = [row for row in historical_rows if row["图像"] == detail["图像"] and
               row["提示词编号"] == detail["提示词编号"] and abs(float(row["面积比例"]) - area) < 1e-14]
    if len(matches) != 1:
        raise ValueError("Historical score/detail identity join is missing or ambiguous")
    score_row = matches[0]
    scores = [float(value) for value in score_row["SAM3原始分数"].split(";") if value]
    instance = int(detail["实例编号"])
    confidence = float(detail["SAM3分数"])
    if len(scores) != candidate_count or instance >= len(scores) or instance < 0 or scores[instance] != confidence:
        raise ValueError("Original instance enumeration/confidence-list mismatch")
    if not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError("Invalid archived SAM confidence")
    box = [int(value) for value in detail["外接框"].split(",")]
    if len(box) != 4 or box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError("Invalid original predicted box")
    if detail["提示词编号"] != "P6" or detail["候选模式"] != "per_instance" or float(detail["掩膜阈值"]) != 0.5:
        raise ValueError("Candidate is not frozen P6/per_instance/0.5")
    return {"instance_id": instance, "box": box, "sam_score": confidence, "scores": scores,
            "area_ratio": area, "empty_mask": int(not mask.any()), "historical_score": score_row}


def annotation_locations(frame, output):
    sample, number = frame["sample"], frame["frame"]
    stem = f"{sample}_{number}"
    return [output / f"annotation_package/annotations/{stem}.json",
            FIX / f"annotation_package/annotations/{stem}.json",
            ROOT / f"03-GT-区分_challenge/{sample}/{number}.json",
            ROOT / f"03-GT-区分/{sample}/{number}.json",
            ROOT / f"03-GT/{sample}/{number}.json",
            PHASE16 / f"annotation_package/{frame['challenge_id']}/target.json",
            PHASE16 / f"annotation_package/annotations/{stem}.json"]


def main():
    parser = argparse.ArgumentParser(description="Read-only TEST32 identity audit; no scoring, inference or GT synthesis")
    parser.add_argument("--output-dir", type=Path, default=WORK)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if (output / "TEST32_INPUT_MANIFEST.csv").exists() or (output / "TEST32_ANNOTATION_REVIEW.csv").exists():
        raise FileExistsError("Existing audit/reviews preserved; use a new --output-dir for re-audit")
    output.mkdir(parents=True, exist_ok=True)
    (output / "annotation_package/annotations").mkdir(parents=True, exist_ok=True)
    source_hashes = {}

    def record(path, kind, expected=None):
        path = Path(path)
        digest = sha256(path)
        if expected is not None and digest != expected:
            raise ValueError(f"Frozen source hash mismatch: {path}")
        source_hashes[relative(path)] = {"path": relative(path), "kind": kind,
                                       "bytes": path.stat().st_size, "sha256": digest}
        return digest

    frozen_path = PHASE16 / "挑战集列表/Phase16_manifest.json"
    frozen = json.loads(frozen_path.read_text())
    record(frozen_path, "PHASE16_FROZEN_SPEC")
    test_path = PHASE16 / "挑战集列表/Phase16_TEST_manifest.csv"
    dev_path = PHASE16 / "挑战集列表/Phase16_DEV_manifest.csv"
    test_hash = record(test_path, "FROZEN_TEST_MANIFEST", frozen["sha256_test_manifest"])
    dev_hash = record(dev_path, "FROZEN_DEV_MANIFEST", frozen["sha256_dev_manifest"])
    tests, dev = read_csv(test_path), read_csv(dev_path)
    unique_index(tests, "challenge_id")
    if len(tests) != 32 or len({row["sequence"] for row in tests}) != 8:
        raise ValueError("Frozen TEST32 32/8 coverage mismatch")
    dev_samples = {row["sample"] for row in dev}
    dev_sequences = {row["sequence"] for row in dev}
    if {row["sample"] for row in tests} & dev_samples or {row["sequence"] for row in tests} & dev_sequences:
        raise ValueError("DEV/TEST sample/sequence overlap; stop")
    prior_inputs = unique_index(read_csv(FIX / "TEST32_INPUT_AUDIT.csv"), "challenge_id")
    prior_masks = unique_index(read_csv(FIX / "TEST32_CANDIDATE_AUDIT.csv"), "path")
    protected = {}
    for name in ("ranking_policy.py", "candidate_metadata.py", "生成RAP-FSAM3掩膜.py"):
        path = PROD / name
        if git_bytes(path) != path.read_bytes():
            raise ValueError(f"Frozen production source changed from {BASELINE}: {name}")
        protected[name] = record(path, "BYTE_FROZEN_PRODUCTION_SOURCE")
    historical_hashes = json.loads((FIX / "PROTECTED_SOURCE_HASHES.json").read_text())
    if protected["ranking_policy.py"] != historical_hashes["ranking_policy_sha256"]:
        raise ValueError("R1 protected hash mismatch")
    record(ROOT / "sam3/sam3.pt", "UNUSED_FROZEN_CHECKPOINT", historical_hashes["checkpoint_sha256"])
    record(ROOT / "第三方源码/SAM3-latest/sam3/model/sam3_image_processor.py", "SAM3_SOURCE",
           historical_hashes["sam3_processor_sha256"])
    for name in ("PHASE20C_FIX_FINAL_REPORT.md", "FINAL_FOREGROUND_METHOD_DECISION.md",
                 "CANDIDATE_METADATA_PROVENANCE.md", "PHASE20D_INDEPENDENT_VALIDATION_PLAN.md",
                 "ARTIFACT_RECOVERY.md", "TEST32_INPUT_AUDIT.csv", "TEST32_CANDIDATE_AUDIT.csv",
                 "scripts/evaluate_independent_test.py", "PROTECTED_SOURCE_HASHES.json"):
        record(FIX / name, "PHASE20C_FIX_EVIDENCE")
    parameter_path = FIX / "small_run_outputs/V1_GT10_all/参数.json"
    parameter_hash = record(parameter_path, "FROZEN_V1_ALL_PARAMETERS")
    parameters = json.loads(parameter_path.read_text())
    off_flags = ["use_spnp_refinement", "use_residual_repair", "use_corrective_geometry",
                 "use_cross_view_consensus", "use_memory_propagation", "use_prompt_ensemble"]
    if any(parameters[flag] for flag in off_flags) or parameters["ranking_policy"] != "r1":
        raise ValueError("V1 source parameters violate frozen method")
    for policy, name in (("r1", "V1"), ("base", "R0")):
        snapshot = {**parameters, "ranking_policy": policy}
        write_json(output / f"configs/FROZEN_{name}_PARAMETERS.json",
                   {"status": "ARCHIVED_RECIPE_ONLY_NOT_EXECUTED", "source_commit": BASELINE,
                    "source_path": relative(parameter_path), "source_sha256": parameter_hash,
                    "parameters": snapshot, "runtime_paths": "Archived DEV paths, not an executable TEST32 recipe",
                    "only_algorithmic_change_from_V1": "ranking_policy=base" if name == "R0" else "NONE"})
    old_parameters_path = SOURCE / "参数.json"
    old_parameters = json.loads(old_parameters_path.read_text())
    record(old_parameters_path, "ORIGINAL_U00_PARAMETERS")
    score_path = SOURCE / "提示词评分.csv"
    score_hash = record(score_path, "ORIGINAL_CANDIDATE_SCORE_TABLE")
    historical_scores = read_csv(score_path)
    source_index_path = SOURCE / "图像索引.csv"
    source_index = unique_index(read_csv(source_index_path), "图像名")
    record(source_index_path, "ORIGINAL_GENERATION_IMAGE_INDEX")
    if len(source_index) != 32:
        raise ValueError("Original generation input index is not TEST32")
    historical_path = PHASE17C / "Phase17C_test32_safety.csv"
    historical_changes = read_csv(historical_path)
    historical_map = unique_index([{**row, "frame_id": f"{row['sample']}_{row['frame']}"} for row in historical_changes], "frame_id")
    record(historical_path, "HISTORICAL_BEHAVIOR_ONLY_NOT_ACCURACY")
    loader = load_module("phase20d_frozen_metadata_loader", PROD / "candidate_metadata.py")
    manifest_rows, candidate_rows, review_rows, difficulty_rows, location_rows, change_rows = [], [], [], [], [], []
    reference_rgb = []
    for frame in dev:
        path = Path(frame["image_path"])
        info = rgb_info(path)
        record(path, "PHASE16_DEV_RGB", frozen["image_sha256"][frame["challenge_id"]])
        reference_rgb.append({"frame_id": "Phase16_DEV:" + f"{frame['sample']}_{frame['frame']}", **info})
    old_manifest_path = BASE / "阶段二十C_R1与完整细化链联合验证/configs/PHASE20C_INPUT_MANIFEST.csv"
    record(old_manifest_path, "EASY21_GT10_DEV_MANIFEST")
    old_manifest = read_csv(old_manifest_path)
    development_samples = {row["sample_id"] for row in old_manifest} | dev_samples
    if development_samples & {row["sample"] for row in tests}:
        raise ValueError("Easy21/GT10 DEV sequence overlaps TEST32")
    for frame in old_manifest:
        path = ROOT / frame["input_path"]
        info = rgb_info(path)
        record(path, "EASY21_GT10_DEV_RGB", frame["input_sha256"])
        reference_rgb.append({"frame_id": frame["dataset"] + ":" + frame["stem"], **info})
    for frame in tests:
        stem = f"{frame['sample']}_{frame['frame']}"
        row = {"challenge_id": frame["challenge_id"], "frame_id": stem, "split": "TEST",
               "sample": frame["sample"], "sequence": frame["sequence"], "frame": frame["frame"],
               "rgb_path": relative(Path(frame["image_path"])), "input_status": "BLOCKED",
               "candidate_count_expected": int(prior_inputs[frame["challenge_id"]]["candidate_count"]),
               "candidate_count_verified": 0, "gt_status": "USER_ANNOTATION_REQUIRED",
               "human_target_identity": "NOT_MANUALLY_CONFIRMED",
               "heuristic_primary_NOT_GT": frame["primary_difficulty"],
               "heuristic_secondary_NOT_GT": frame["secondary_difficulty"], "errors": ""}
        try:
            rgb = Path(frame["image_path"])
            info = rgb_info(rgb)
            row.update(info)
            record(rgb, "ORIGINAL_TEST_RGB", frozen["image_sha256"][frame["challenge_id"]])
            if info["rgb_sha256"] != prior_inputs[frame["challenge_id"]]["rgb_sha256"]:
                raise ValueError("Prior TEST32 RGB identity mismatch")
            alias = Path(old_parameters["input_dir"]) / f"{stem}.jpg"
            if source_index[f"{stem}.jpg"]["图像路径"] != str(alias):
                raise ValueError("Original generation image-index/parameters mismatch")
            record(alias, "ORIGINAL_GENERATION_RGB", info["rgb_sha256"])
            target = PHASE16 / f"annotation_package/{frame['challenge_id']}/target.jpg"
            target_hash = record(target, "PHASE16_ANNOTATION_TARGET_COPY_NOT_INPUT")
            row["phase16_annotation_target_sha256"] = target_hash
            row["phase16_annotation_target_matches_input"] = int(target_hash == info["rgb_sha256"])
            record(FIX / f"annotation_package/rgb/{stem}.jpg", "PHASE20C_FIX_RGB_LINK", info["rgb_sha256"])
            detail_path = SOURCE / f"候选掩膜/候选评分明细_{stem}.csv"
            detail_hash = record(detail_path, "EXPLICIT_ID_SCORE_BOX_METADATA")
            details = read_csv(detail_path)
            detail_map = unique_index([{**detail, "identity": (detail["图像"], detail["提示词编号"], int(detail["实例编号"]))} for detail in details], "identity")
            raw_paths = list((SOURCE / "候选掩膜/raw_instance_P6").glob(f"mask_{stem}_*.png"))
            raw_ids = {int(path.stem.rsplit("_", 1)[1]) for path in raw_paths}
            expected_ids = {identity[2] for identity in detail_map if identity[:2] == (f"{stem}.jpg", "P6")}
            if len(raw_ids) != len(raw_paths) or expected_ids != raw_ids or len(raw_paths) != row["candidate_count_expected"] or len(details) != len(raw_paths):
                raise ValueError("Missing/duplicate/unmatched original candidates; affected frame stopped")
            entries = []
            for raw in sorted(raw_paths, key=lambda path: int(path.stem.rsplit("_", 1)[1])):
                instance = int(raw.stem.rsplit("_", 1)[1])
                detail = detail_map[(f"{stem}.jpg", "P6", instance)]
                mask_hash = record(raw, "ORIGINAL_P6_CANDIDATE", prior_masks[relative(raw)]["sha256"])
                if detail_hash != prior_masks[relative(raw)]["detail_sha256"]:
                    raise ValueError("Original candidate detail hash mismatch")
                with Image.open(raw) as handle:
                    mask = np.asarray(handle)
                joined = verify_candidate(detail, historical_scores, mask, info["width"], info["height"], len(raw_paths))
                entry = {"prompt_id": "P6", "instance_id": instance, "sam_score": joined["sam_score"],
                         "box": joined["box"], "mask_threshold": 0.5, "candidate_order": instance,
                         "mask_sha256": mask_hash, "scores": joined["scores"], "raw_detection_count": len(raw_paths),
                         "prompt_mode": "per_instance", "metadata_status": "VERIFIED",
                         "score_precision": "ARCHIVED_6_DECIMALS", "source_csv": relative(detail_path),
                         "source_csv_sha256": detail_hash}
                entries.append(entry)
                candidate_rows.append({"challenge_id": frame["challenge_id"], "frame_id": stem,
                                       "candidate_id": f"{stem}_{instance:02d}", "prompt_id": "P6",
                                       "instance_id": instance, "candidate_order": instance,
                                       "sam_score_archived_text": detail["SAM3分数"], "box": json.dumps(joined["box"]),
                                       "mask_threshold": detail["掩膜阈值"], "rgb_sha256": info["rgb_sha256"],
                                       "mask_path": relative(raw), "mask_sha256": mask_hash,
                                       "width": info["width"], "height": info["height"], "area_ratio": joined["area_ratio"],
                                       "empty_candidate": joined["empty_mask"], "source_detail": relative(detail_path),
                                       "source_detail_sha256": detail_hash, "source_score_sha256": score_hash,
                                       "status": "VERIFIED_ARCHIVED_PRECISION", "unrounded_confidence": "NOT_SAVED"})
            metadata_path = output / f"metadata/TEST32/候选元数据_{stem}.json"
            write_json(metadata_path, {"schema_version": 1, "image_name": f"{stem}.jpg", "image_sha256": info["rgb_sha256"],
                                       "source_image_path": relative(alias), "candidates": entries})
            original_link(rgb, output / f"inputs/TEST32/{stem}.jpg")
            original_link(rgb, output / f"annotation_package/rgb/{stem}.jpg")
            for direction in ("prev", "next"):
                context_path = PHASE16 / f"annotation_package/{frame['challenge_id']}/context_{direction}.jpg"
                record(context_path, "ANNOTATION_CONTEXT_ONLY_NOT_TEST_EVALUATION_FRAME")
                original_link(context_path, output / f"annotation_package/context/{stem}_{direction}.jpg")
            indexed = loader.load_cache_metadata(output / f"inputs/TEST32/{stem}.jpg", SOURCE, metadata_path.parent)
            for raw in raw_paths:
                instance = int(raw.stem.rsplit("_", 1)[1])
                loader.metadata_for_mask(indexed, "P6", instance, raw, strict=True)
                original_link(raw, output / f"candidates/TEST32/候选掩膜/raw_instance_P6/{raw.name}")
            row.update({"input_status": "VERIFIED", "candidate_count_verified": len(entries),
                        "candidate_metadata_path": relative(metadata_path), "candidate_metadata_sha256": sha256(metadata_path),
                        "strict_metadata_status": "VERIFIED", "trackB_rgb_sha256": sha256(alias),
                        "phase16_annotation_rgb_sha256": sha256(target)})
        except (ValueError, KeyError, FileNotFoundError, OSError) as error:
            row["errors"] = str(error)
        locations = annotation_locations(frame, output)
        found = [path for path in locations if path.is_file()]
        row["annotation_files_found_known_locations"] = len(found)
        row["annotation_json"] = f"annotation_package/annotations/{stem}.json"
        manifest_rows.append(row)
        for path in locations:
            location_rows.append({"frame_id": stem, "annotation_path": relative(path), "exists": int(path.is_file()),
                                  "annotation_sha256": sha256(path) if path.is_file() else "", "approval_inferred": "NO"})
        review_rows.append({"challenge_id": frame["challenge_id"], "frame_id": stem, "sample": frame["sample"],
                            "frame": frame["frame"], "rgb_sha256": row.get("rgb_sha256", ""),
                            "annotation_json": row["annotation_json"], "annotation_sha256": "", "annotation_origin": "",
                            "annotator": "", "reviewer": "", "second_reviewer": "", "annotator_is_human": "",
                            "reviewer_is_human": "", "second_reviewer_is_human": "", "manual_annotation_confirmed": "",
                            "target_identity_description": "", "target_identity_confirmed": "", "plant_confirmed": "",
                            "pot_confirmed": "", "blue_cube_present": "", "difficulty_manually_confirmed": "",
                            "human_difficulty_labels": "", "difficulty_evidence_json": "", "reviewed_at": "",
                            "second_reviewed_at": "", "approval": "PENDING"})
        difficulty_rows.append({"challenge_id": frame["challenge_id"], "frame_id": stem,
                                "rgb_sha256": row.get("rgb_sha256", ""), "annotation_sha256": "",
                                **{code: "" for code in DIFFICULTIES}, "human_difficulty_labels": "",
                                "difficulty_evidence_json": "", "reviewer": "", "second_reviewer": "",
                                "status": "PENDING_HUMAN_VISUAL_REVIEW"})
        historical = historical_map[stem]
        known_ids = {entry["candidate_id"] for entry in candidate_rows if entry["frame_id"] == stem}
        for field in ("baseline_selected", "r1_selected"):
            if row["input_status"] == "VERIFIED" and historical[field] not in known_ids:
                raise ValueError("Historical selection refers to an unverified candidate ID")
        change_rows.append({"frame_id": stem, "historical_r0_selected": historical["baseline_selected"],
                            "historical_r1_selected": historical["r1_selected"], "historical_changed": historical["changed"],
                            "historical_behavior_only": "YES", "new_r0_r1_run": "NOT_EXECUTED",
                            "improvement_degradation_judgment": "BLOCKED_HUMAN_GT_REQUIRED",
                            "target_identity_judgment": "NOT_MANUALLY_CONFIRMED", "historical_source": relative(historical_path)})
    duplicate_rows = []
    for frame in manifest_rows:
        duplicates = {}
        for kind, field in (("file", "rgb_sha256"), ("decoded_rgb", "decoded_rgb_sha256")):
            duplicates[f"test_{kind}_duplicates"] = [other["frame_id"] for other in manifest_rows
                                                      if other["frame_id"] != frame["frame_id"] and
                                                      frame.get(field) and frame.get(field) == other.get(field)]
            duplicates[f"dev_{kind}_matches"] = [other["frame_id"] for other in reference_rgb
                                                 if frame.get(field) and frame.get(field) == other.get(field)]
        duplicate_rows.append({"frame_id": frame["frame_id"], **{key: json.dumps(value) for key, value in duplicates.items()},
                               "exact_duplicate_or_dev_match": int(any(duplicates.values())),
                               "near_duplicate_status": "NOT_MANUALLY_VERIFIED; adjacent frames are correlated"})
    verified = sum(row["input_status"] == "VERIFIED" for row in manifest_rows)
    duplicates_found = sum(row["exact_duplicate_or_dev_match"] for row in duplicate_rows)
    input_pass = verified == 32 and len(candidate_rows) == 71 and not duplicates_found
    write_csv(output / "TEST32_INPUT_MANIFEST.csv", manifest_rows)
    write_csv(output / "TEST32_CANDIDATE_PROVENANCE.csv", candidate_rows, fields=[
        "challenge_id", "frame_id", "candidate_id", "prompt_id", "instance_id", "candidate_order",
        "sam_score_archived_text", "box", "mask_threshold", "rgb_sha256", "mask_path", "mask_sha256",
        "width", "height", "area_ratio", "empty_candidate", "source_detail", "source_detail_sha256",
        "source_score_sha256", "status", "unrounded_confidence"])
    write_csv(output / "TEST32_ANNOTATION_REVIEW.csv", review_rows)
    write_csv(output / "DIFFICULTY_HUMAN_VERIFICATION.csv", difficulty_rows)
    write_csv(output / "ANNOTATION_LOCATION_AUDIT.csv", location_rows)
    write_csv(output / "DUPLICATE_FRAME_AUDIT.csv", duplicate_rows)
    write_csv(output / "TEST32_HISTORICAL_SELECTION_CHANGE.csv", change_rows)
    write_csv(output / "SOURCE_HASHES.csv", list(source_hashes.values()))
    write_json(output / "configs/FROZEN_TEST_SPEC.json", {
        "source_commit": BASELINE, "test_manifest_sha256": test_hash, "dev_manifest_sha256": dev_hash,
        "input_manifest_sha256": sha256(output / "TEST32_INPUT_MANIFEST.csv"),
        "frames": 32, "sequences": sorted({row["sequence"] for row in tests}),
        "expected_challenge_ids": [row["challenge_id"] for row in tests], "original_candidates": 71,
        "target_definition": "potted_clean = potted_plant AND NOT blue_cube",
        "labels": ["potted_plant", "plant", "pot", "blue_cube"], "difficulty_codes": DIFFICULTIES,
        "human_annotation_and_distinct_second_review_required": True, "tuning_forbidden": True,
        "new_inference_forbidden": True, "source_ranking_policy_sha256": protected["ranking_policy.py"],
        "phase20d_evaluation_status": "BLOCKED", "confidence_precision": "ARCHIVED_6_DECIMALS; extra digits NOT_SAVED"})
    write_json(output / "INPUT_AUDIT_STATUS.json", {
        "input_audit": "PASS" if input_pass else "BLOCKED", "frames": 32, "verified_input_frames": verified,
        "sequence_count": 8, "candidates_verified": len(candidate_rows), "exact_duplicate_or_dev_match_frames": duplicates_found,
        "dev_sequence_ids": sorted(development_samples), "test_sequence_ids": sorted({row["sequence"] for row in tests}),
        "phase16_dev_reference_frames": len(dev), "easy21_gt10_dev_reference_frames": len(old_manifest),
        "human_annotation_files_found": sum(row["annotation_files_found_known_locations"] > 0 for row in manifest_rows),
        "human_gt": "USER_ANNOTATION_REQUIRED", "evaluation": "BLOCKED", "protected_code_sha256": protected,
        "metadata_snapshot": "IDENTITY_VERIFIED_ARCHIVED_PRECISION; no score computation performed",
        "historical_churn_frames": sum(int(row["changed"]) for row in historical_changes),
        "new_scoring_or_inference": False, "new_predictions_or_accuracy": False,
        "physical_target_identity_and_near_duplicates": "Human confirmation pending",
        "original_inputs_preserved": True,
        "baseline_commit": subprocess.check_output(["git", "rev-parse", BASELINE], cwd=ROOT, text=True).strip()})
    print(json.dumps({"input_audit": "PASS" if input_pass else "BLOCKED", "verified_frames": verified,
                      "candidates": len(candidate_rows), "annotation_frames_found": sum(bool(row["annotation_files_found_known_locations"]) for row in manifest_rows),
                      "human_gt": "USER_ANNOTATION_REQUIRED", "accuracy_evaluation": "BLOCKED"}, ensure_ascii=False))
    return 0 if input_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
