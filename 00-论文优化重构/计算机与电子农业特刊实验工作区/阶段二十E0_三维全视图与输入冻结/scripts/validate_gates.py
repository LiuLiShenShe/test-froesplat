from __future__ import annotations

import json

from phase20e_common import DATA, FINAL, SAMPLES, WORK, frozen_parameters, read_csv, relative, sha256, write_csv, write_json


def run():
    identity = read_csv(WORK / "FULL_VIEW_IDENTITY_MANIFEST.csv")
    candidates = read_csv(WORK / "FULL_VIEW_CANDIDATE_MANIFEST.csv")
    selections = read_csv(WORK / "FULL_VIEW_R0_R1_SELECTION.csv")
    datasets = read_csv(WORK / "MATCHED_2DGS_DATASET_MANIFEST.csv")
    splits = read_csv(WORK / "TRAIN_VALIDATION_SPLIT.csv")
    commands = read_csv(WORK / "reports/PHASE20E1_COMMANDS.csv")
    actual_cameras = read_csv(WORK / "reports/ACTUAL_LOADER_CAMERA_PROVENANCE.csv")
    delivery = json.loads((WORK / "logs/DELIVERY_VERIFICATION.json").read_text())
    config_hashes = json.loads((WORK / "reports/FROZEN_TRAIN_CONFIG_HASHES.json").read_text())
    acceptance = json.loads((WORK / "reports/FINAL_ACCEPTANCE.json").read_text())
    snapshot = json.loads((FINAL / "logs/full_r1_runtime_parameters.json").read_text())
    corrupt_identity = json.loads((WORK / "reports/WanNianQing1_0136_SIFT_KEYPOINT_IDENTITY.json").read_text())
    gates = []
    for sample in SAMPLES:
        ids = [row for row in identity if row["sample"] == sample]
        cs = [row for row in candidates if row["sample"] == sample]
        ss = [row for row in selections if row["sample"] == sample]
        ds = [row for row in datasets if row["sample"] == sample]
        sp = [row for row in splits if row["sample"] == sample]
        ranking = json.loads((WORK / f"reports/{sample}_RANKING_PARAMETERS.json").read_text())
        build = json.loads((WORK / f"reports/{sample}_BUILD.json").read_text())
        preflight = json.loads((WORK / f"reports/{sample}_PREFLIGHT.json").read_text())
        scores = read_csv(WORK / f"reports/{sample}_ALL_SCORES.csv")
        by_policy = {policy: {row["candidate_id"]: row for row in scores if row["policy"] == policy} for policy in ("R0", "R1")}
        mapped = read_csv(WORK / f"reports/{sample}_MAPPED_MASK_DIFFERENCES.csv")
        comparisons = [row for row in read_csv(WORK / f"reports/{sample}_RGB_COMPARISONS.csv") if row["reference"].startswith("NATIVE_")]
        nearest_verified = build.get("native_nearest_mask_verified", False)
        nearest_path = WORK / "reports/NATIVE_NEAREST_MASK_COMPARISON.json"
        if not nearest_verified and nearest_path.exists():
            nearest = json.loads(nearest_path.read_text())
            nearest_verified = nearest["sample"] == sample and nearest["pixelwise_identity"] and nearest["native_nearest_changed_pixels"] == 0
        params_equal = ranking["R1"] == frozen_parameters() and ranking["R0"] == frozen_parameters("base")
        same_candidate_scores = by_policy["R0"].keys() == by_policy["R1"].keys() and all(
            all(by_policy["R0"][candidate][field] == by_policy["R1"][candidate][field] for field in
                ("sam_score_full_precision", "box_json", "mask_sha256", "candidate_order", "total_score"))
            for candidate in by_policy["R0"])
        complete = len(ids) == len(ss) == len(sp) == len(mapped) and len(ds) == len(ids) * 3
        loaded_cameras = [row for row in actual_cameras if row["sample"] == sample]
        camera_contract = len(loaded_cameras) == len(ids) * 3 and all(row["status"] == "PASS" and
            row["production_CameraInfo_uid"] == row["camera_id"] for row in loaded_cameras)
        if sample == "WanNianQing1":
            camera_contract = camera_contract and corrupt_identity["status"] == "PASS" and corrupt_identity["raw_RGB_camera_identity_verified"]
        common_fields = ("registered_image_id", "registered_name", "camera_id", "source_rgb_sha256", "common_rgb_sha256", "split", "sparse_bundle")
        matched = all(len({tuple(row[field] for field in common_fields) for row in ds if row["frame_id"] == view}) == 1
                      for view in {row["frame_id"] for row in ds})
        checked_source = delivery["status"] == "PASS" and not delivery["outside_stage_changes"]
        current_commands = [row for row in commands if row["sample"] == sample]
        configs = [WORK / f"reports/{sample}_{group}_TRAIN_CONFIG.json" for group in build["groups"]]
        checks = [
            ("G1", snapshot == frozen_parameters() and params_equal, "Full Phase20D snapshot; only ranking_policy differs in R0/R1"),
            ("G2", all(row["set_type"] in ("NEW_FULL_VIEW_CANDIDATE_SET", "VERIFIED_HISTORICAL_TEST32_CANDIDATES") for row in cs), "Old frozen artifacts and new inference explicitly separated; old checkpoint tensor digest was not archived"),
            ("G3", all(row["rgb_sha256"] and row["mask_sha256"] and row["metadata_sha256"] and row["model_provenance_status"] and row["box"] and row["sam_score"] for row in cs)
             and all(row["score_precision"] == "FULL_AVAILABLE_TENSOR_PRECISION" for row in cs if row["set_type"] == "NEW_FULL_VIEW_CANDIDATE_SET"), "New instances retain full available score/box tensor precision and complete hash provenance"),
            ("G4", same_candidate_scores, "Identical candidate IDs, masks, SAM scores, boxes, order and base raw scores"),
            ("G5", complete and all(row["missing"] == "False" for row in ss), "All registered views covered; no silent drop"),
            ("G6", camera_contract and all(row["common_raw_domain_proven"] == "True" and not row["raw_rgb_error"] for row in ids), "RGB joins and actual camera identities verified; image_id/camera_id not conflated; corrupt originals preserved"),
            ("G7", build["status"] == "PASS" and nearest_verified and build["sparse_checks"]["conversion_max_px"] <= 1e-6 and len(comparisons) == 2
             and all(float(row["mean_abs_color_error"]) <= 0.1 and int(row["max_abs_color_error"]) <= 1 for row in comparisons), "Official camera/half-pixel map, native RGB/nearest mask and sparse projection validation"),
            ("G8", matched and complete, "One immutable target camera/sparse bundle, common RGB and registered-name split"),
            ("G9", all(row["legacy_A_status"] == "NEW_RAW_RGB_CONTROL_NOT_HISTORICAL_FORESPLAT" for row in ds if row["group"] == "A_raw_RGB_control_v1"), "Incomplete historical A is explicitly replaced by versioned raw-RGB control, not called a historical reproduction"),
            ("G10", checked_source and len(ss) == len(ids), "Delivery checker confirms every selected mask equals its source candidate"),
            ("G11", len(mapped) == len(ids) and sum(row["switched"] == "True" for row in ss) == acceptance["scenes"][sample]["R0_R1_switches"], "Actual raw and mapped mask differences plus consumed train RGB hashes retained"),
            ("G12", preflight["status"] == "PASS" and all(group["loaded_views"] == len(ids) and group["one_real_forward"] for group in preflight["groups"]), "All three real production loaders and CUDA no-optimizer forwards passed"),
            ("G13", checked_source, "Protected production sources unchanged; only new stage files changed"),
            ("G14", not preflight["formal_training"] and delivery["optimizer_steps"] == delivery["checkpoints_created"] == 0
             and all(not group["optimizer_created"] and group["optimizer_steps"] == 0 for group in preflight["groups"]), "No training_setup, backward, optimization, checkpoint or mesh"),
            ("G15", len(current_commands) == acceptance["scenes"][sample]["training_runs"] and all(row["executed"] == "False" for row in current_commands)
             and all(sha256(path) == config_hashes[relative(path)] for path in configs) and not (DATA / "phase20e1_outputs").exists(), "Frozen full configs, unique unexecuted commands and output paths traced"),
        ]
        for gate, passed, evidence in checks:
            gates.append({"sample": sample, "gate": gate, "status": "PASS" if passed else "FAIL", "evidence": evidence})
    write_csv(WORK / "reports/ACCEPTANCE_GATES_G1_G15.csv", gates)
    result = {"status": "PASS" if all(row["status"] == "PASS" for row in gates) else "FAIL", "checks": len(gates),
              "failed": [row for row in gates if row["status"] != "PASS"], "formal_training": False}
    write_json(WORK / "logs/G1_G15_VALIDATION.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    if result["failed"]:
        raise ValueError("Phase20E0 acceptance gate failed")


if __name__ == "__main__":
    run()
