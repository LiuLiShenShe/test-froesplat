from __future__ import annotations

import json

from phase20e_common import BASE, DATA, FINAL, ROOT, SAMPLES, WORK, frozen_parameters, read_csv, relative, sha256, write_csv, write_json


def run():
    parameters = frozen_parameters()
    old_config = BASE / "01-算法模块修改与验证/阶段十七A_10GT困难样本Pilot与无GT测试审计/03_trackB_variants/U00_Control/参数.json"
    archived = json.loads(old_config.read_text())
    for field in ("sam3_repo", "sam3_checkpoint", "confidence_threshold", "sam3_mask_threshold", "candidate_mode", "prompt_list", "prompt_selection_mode"):
        if archived[field] != parameters[field]:
            raise ValueError(f"Historical model/recipe path drift: {field}")
    protected = {row["path"]: row for row in read_csv(FINAL / "SOURCE_PROTECTION_MANIFEST.csv")}
    checkpoint = ROOT / "sam3/sam3.pt"
    processor = ROOT / "第三方源码/SAM3-latest/sam3/model/sam3_image_processor.py"
    for path in (checkpoint, processor):
        if sha256(path) != protected[relative(path)]["sha256"]:
            raise ValueError("Phase20D frozen model/source differs")
    rows = []
    for sample in SAMPLES:
        provenance_path = DATA / sample / "candidates/GENERATION_PROVENANCE.json"
        provenance = json.loads(provenance_path.read_text())
        for entry in read_csv(WORK / f"reports/{sample}_FULL_CANDIDATES.csv"):
            historical = entry["set_type"] != "NEW_FULL_VIEW_CANDIDATE_SET"
            rows.append({"sample": sample, "frame_id": entry["frame_id"], "candidate_id": entry["candidate_id"],
                         "set_type": entry["set_type"], "candidate_metadata_sha256": entry["metadata_sha256"],
                         "checkpoint_path": str(checkpoint), "checkpoint_frozen_phase20d_sha256": protected[relative(checkpoint)]["sha256"],
                         "processor_path": str(processor), "processor_frozen_phase20d_sha256": protected[relative(processor)]["sha256"],
                         "new_generation_provenance_path": "" if historical else str(provenance_path),
                         "new_generation_provenance_sha256": "" if historical else sha256(provenance_path),
                         "generation_checkpoint_sha256": "NOT_ARCHIVED_AT_PHASE17A_GENERATION" if historical else provenance["checkpoint_sha256"],
                         "archive_generation_config_path": relative(old_config) if historical else "",
                         "archive_generation_config_sha256": sha256(old_config) if historical else "",
                         "phase20d_model_freeze_manifest_path": relative(FINAL / "SOURCE_PROTECTION_MANIFEST.csv"),
                         "recipe_fields_match": True, "score_precision": entry["score_precision"],
                         "provenance_limitation": "Immutable archive mask/score/box and Phase20D model lineage verified; original Phase17A tensor checkpoint digest and runtime were not independently archived" if historical else "NONE"})
    write_csv(WORK / "reports/FULL_CANDIDATE_MODEL_LINEAGE.csv", rows)
    write_json(WORK / "logs/CANDIDATE_MODEL_LINEAGE.json", {"status": "PASS", "instances": len(rows),
               "historical_instances": sum(row["generation_checkpoint_sha256"].startswith("NOT_ARCHIVED") for row in rows),
               "original_historical_generation_digest_recovered": False,
               "phase20d_checkpoint_processor_hashes_match": True})
    print("CANDIDATE_MODEL_LINEAGE", len(rows), "PASS", flush=True)


if __name__ == "__main__":
    run()
