from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from fix_common import OLD, PROD, PYTHON, ROOT, WORK, link, manifest, read_csv, relative, sha256, write_csv, write_json


def main():
    provenance = []
    for frame in manifest():
        stem, dataset = frame["stem"], frame["dataset"]
        source = ROOT / frame["candidate_source_root"]
        source_parameters = json.loads((source / "参数.json").read_text())
        source_image = Path(source_parameters["input_dir"]) / f"{stem}.jpg"
        rgb = ROOT / frame["input_path"]
        if sha256(rgb) != frame["input_sha256"] or sha256(source_image) != frame["input_sha256"]:
            raise ValueError(f"Frozen RGB identity mismatch: {stem}")
        if sha256(ROOT / frame["gt_path"]) != frame["gt_sha256"]:
            raise ValueError(f"Frozen GT identity mismatch: {stem}")
        details = read_csv(ROOT / frame["score_record_path"])
        keyed = {(row["图像"], row["提示词编号"], int(row["实例编号"])): row for row in details}
        if len(keyed) != len(details):
            raise ValueError(f"Duplicate historical candidate identity: {stem}")
        score_rows = [row for row in read_csv(source / "提示词评分.csv") if row["图像"] == f"{stem}.jpg" and row["提示词编号"] == "P6"]
        entries = []
        files = sorted((ROOT / frame["candidate_alias_dir"]).glob(f"mask_{stem}_*.png"))
        if len(files) != int(frame["candidate_count"]):
            raise ValueError(f"Candidate coverage mismatch: {stem}")
        if sorted(sha256(path) for path in files) != sorted(frame["candidate_hashes"].split(";")):
            raise ValueError(f"Frozen candidate hash mismatch: {stem}")
        for alias in files:
            instance = int(alias.stem.rsplit("_", 1)[1])
            row = keyed[(f"{stem}.jpg", "P6", instance)]
            raw = source / "候选掩膜/raw_instance_P6" / alias.name
            if sha256(raw) != sha256(alias):
                raise ValueError(f"Alias/mask identity mismatch: {alias}")
            with Image.open(raw) as image:
                mask = np.asarray(image)
            if set(np.unique(mask)) - {0, 255}:
                raise ValueError(f"Nonbinary candidate format: {raw}")
            area = np.count_nonzero(mask) / mask.size
            if abs(area - float(row["面积比例"])) > 1e-14:
                raise ValueError(f"Candidate metadata area mismatch: {raw}")
            matched = [value for value in score_rows if abs(float(value["面积比例"]) - area) < 1e-14]
            if len(matched) != 1:
                raise ValueError(f"Historical score row not uniquely identifiable by image/prompt/mask area: {raw}")
            score_row = matched[0]
            scores = [float(value) for value in score_row["SAM3原始分数"].split(";") if value]
            score = float(row["SAM3分数"])
            if instance >= len(scores) or abs(scores[instance] - score) > 5.1e-7:
                raise ValueError(f"Independent archived score-list check failed: {raw}")
            box = [int(value) for value in row["外接框"].split(",")]
            entry = {"prompt_id": "P6", "instance_id": instance, "sam_score": score,
                     "box": box, "mask_threshold": float(row["掩膜阈值"]),
                     "candidate_order": instance, "mask_sha256": sha256(raw),
                     "scores": scores, "raw_detection_count": len(scores), "prompt_mode": "per_instance",
                     "metadata_status": "VERIFIED", "score_precision": "ARCHIVED_6_DECIMALS",
                     "source_csv": frame["score_record_path"], "source_csv_sha256": sha256(ROOT / frame["score_record_path"])}
            entries.append(entry)
            provenance.append({"dataset": dataset, "sample": frame["sample_id"], "frame": frame["frame"],
                               "candidate_id": f"{stem}_{instance:02d}", "prompt_id": "P6", "instance_id": instance,
                               "sam_score": score, "box": json.dumps(box), "mask_threshold": entry["mask_threshold"],
                               "candidate_order": instance, "source_image_name": f"{stem}.jpg",
                               "source_image_sha256": frame["input_sha256"], "mask_sha256": entry["mask_sha256"],
                               "mask_path": relative(raw), "source_detail_path": frame["score_record_path"],
                               "source_detail_sha256": entry["source_csv_sha256"],
                               "source_score_path": relative(source / "提示词评分.csv"),
                               "source_score_sha256": sha256(source / "提示词评分.csv"),
                               "source_parameters_sha256": sha256(source / "参数.json"),
                               "historical_total": score_row["总分"], "historical_q_contrast": score_row["前背景对比得分"],
                               "status": "VERIFIED_ARCHIVED_PRECISION", "unrounded_tensor_score": "NOT_SAVED"})
            link(raw, WORK / "candidates" / dataset / "候选掩膜/raw_instance_P6" / raw.name)
        write_json(WORK / "metadata" / dataset / f"候选元数据_{stem}.json",
                   {"schema_version": 1, "image_name": f"{stem}.jpg", "image_sha256": frame["input_sha256"],
                    "source_image_path": relative(source_image), "candidates": entries})
        link(rgb, WORK / "inputs" / dataset / f"{stem}.jpg")
    write_csv(WORK / "CANDIDATE_METADATA_PROVENANCE.csv", provenance)
    for dataset in ("Easy21", "GT10"):
        for group in ("V0", "V1", "V4", "V5"):
            use_spnp = group in ("V4", "V5")
            original = json.loads((OLD / "configs" / f"{dataset}_{'V3' if use_spnp else 'V1'}.json").read_text())
            config = {key: value for key, value in original.items() if key != "command"}
            config.update({"phase": "20C-FIX", "group": group, "ranking_policy": "base" if group in ("V0", "V5") else "r1",
                           "refinement": "SPNP_ONLY" if use_spnp else "OFF", "use_residual_repair": False, "thin_repair_mode": "none",
                           "use_corrective_geometry": False, "corrective_geometry_backend": "none", "colmap_dir": None,
                           "candidate_score_metadata_replayed": True, "candidate_metadata_strict": True,
                           "candidate_score_precision": "ARCHIVED_6_DECIMALS_NOT_UNROUNDED_TENSOR"})
            for key, folder in [("input_dir", "inputs"), ("candidate_source_dir", "candidates"), ("candidate_score_metadata_source", "metadata")]:
                config[key] = str(WORK / folder / dataset)
            config["output_dir"] = str(WORK / "runs" / group / dataset)
            config["device"] = "cuda:0" if use_spnp else "cpu"
            command = [PYTHON, str(PROD / "生成RAP-FSAM3掩膜.py")]
            for key in ["input_dir", "output_dir", "sam3_repo", "sam3_checkpoint", "device", "candidate_mode", "prompt_list", "default_prompt_id", "prompt_selection_mode", "score_weights", "ranking_policy", "candidate_source_dir"]:
                command.extend([f"--{key}", str(config[key])])
            command.extend(["--reuse_existing_candidates", "--candidate_metadata_dir", config["candidate_score_metadata_source"], "--candidate_metadata_strict", "--force"])
            if use_spnp:
                command.extend(["--use_spnp_refinement", "--spnp_backend", "sam3_if_supported"])
            config["command"] = command
            folder = "V4_V5_CONFIGS" if use_spnp else "REPLAY_CONFIGS"
            write_json(WORK / folder / f"{dataset}_{group}.json", config)
    write_json(WORK / "PROTECTED_SOURCE_HASHES.json", {"ranking_policy_sha256": sha256(PROD / "ranking_policy.py"),
               "phase20c_commit": "970082da", "old_manifest_sha256": sha256(OLD / "configs/PHASE20C_INPUT_MANIFEST.csv"),
               "checkpoint_sha256": sha256(ROOT / "sam3/sam3.pt"),
               "sam3_processor_sha256": sha256(ROOT / "第三方源码/SAM3-latest/sam3/model/sam3_image_processor.py"),
               "source_candidates": len(provenance), "source_frames": 31})
    print(f"Prepared {len(provenance)} verified archived-precision candidates / 31 frames; no confidence inferred")


if __name__ == "__main__":
    main()
