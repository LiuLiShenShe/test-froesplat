from __future__ import annotations

import argparse
import dataclasses
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
from PIL import Image

from phase20e_common import DATA, PROD, ROOT, WORK, binary_mask, frozen_parameters, namespace, pixel_hash, production, relative, safe_copy, scene_rows, sha256, view_key, write_csv, write_json


class RecordingProcessor:
    def __init__(self, processor):
        self.processor = processor
        self.raw_scores = []
        self.raw_boxes = []
        self.score_dtype = ""
        self.box_dtype = ""

    def set_image(self, image):
        return self.processor.set_image(image)

    def reset_all_prompts(self, state):
        return self.processor.reset_all_prompts(state)

    def set_text_prompt(self, **kwargs):
        output = self.processor.set_text_prompt(**kwargs)
        scores, boxes = output.get("scores"), output.get("boxes")
        self.raw_scores = scores.detach().float().cpu().reshape(-1).tolist() if scores is not None else []
        self.raw_boxes = boxes.detach().float().cpu().reshape(-1, 4).tolist() if boxes is not None else []
        self.score_dtype = str(scores.dtype) if scores is not None else "NONE"
        self.box_dtype = str(boxes.dtype) if boxes is not None else "NONE"
        return output


def cache_paths(sample, key):
    root = DATA / sample / "candidates"
    return root, root / "metadata" / f"候选元数据_{key}.json"


def verify_cache(row, prod, parameters):
    key = view_key(row)
    root, metadata_path = cache_paths(row["sample"], key)
    image_path = DATA / row["sample"] / "inputs" / f"{key}.jpg"
    if sha256(image_path) != row["raw_rgb_sha256"]:
        raise ValueError(f"Input image mismatch: {key}")
    meta = json.loads(metadata_path.read_text())
    if meta["candidates"] and all(entry.get("empty_fallback", False) for entry in meta["candidates"]):
        loaded = []
        for entry in meta["candidates"]:
            path = root / "候选掩膜/raw_instance_P6" / f"mask_{key}_{entry['instance_id']:02d}.png"
            mask = binary_mask(path)
            if mask.any() or sha256(path) != entry["mask_sha256"] or entry["raw_detection_count"] != 0 or entry["box"] is not None:
                raise ValueError("Invalid explicitly recorded zero-detection placeholder")
            loaded.append(prod.Candidate(prompt_id="P6", prompt_text=parameters["prompt_texts"]["P6"], mask=mask,
                                         scores=entry["scores"], raw_detection_count=0, instance_id=entry["instance_id"],
                                         box=None, sam_score=0.0, mask_threshold=entry["mask_threshold"], prompt_mode="per_instance",
                                         candidate_order=entry["candidate_order"], mask_sha256=entry["mask_sha256"],
                                         metadata_status="EXPLICIT_ZERO_DETECTION_NOT_AN_INSTANCE"))
    else:
        loaded = prod.load_existing_candidates(image_path, ["P6"], parameters["prompt_texts"], root,
                                               root, root / "metadata", True)
    if meta["image_sha256"] != row["raw_rgb_sha256"] or len(loaded) != len(meta["candidates"]):
        raise ValueError("Incomplete cache identity")
    for candidate, entry in zip(loaded, sorted(meta["candidates"], key=lambda item: item["candidate_order"]), strict=True):
        if candidate.mask.shape != (int(row["raw_rgb_height"]), int(row["raw_rgb_width"])):
            raise ValueError("Wrong candidate grid")
        if candidate.sam_score != entry["sam_score"] or candidate.instance_id != entry["instance_id"] or list(candidate.box or []) != list(entry["box"] or []):
            raise ValueError("Cache lost score/box/instance")
        if candidate.mask_threshold != parameters["sam3_mask_threshold"]:
            raise ValueError("Threshold differs")
    return loaded, meta


def import_historical(row, prod, parameters):
    source = ROOT / row["historical_candidate_metadata"]
    meta = json.loads(source.read_text())
    if meta["image_sha256"] != row["raw_rgb_sha256"]:
        raise ValueError("Historical RGB mismatch")
    key = view_key(row)
    root, metadata_path = cache_paths(row["sample"], key)
    mask_root = source.parents[2] / "candidates/TEST32/候选掩膜/raw_instance_P6"
    for entry in meta["candidates"]:
        source_mask = mask_root / f"mask_{Path(meta['image_name']).stem}_{entry['instance_id']:02d}.png"
        if sha256(source_mask) != entry["mask_sha256"] or entry["prompt_id"] != "P6" or entry["mask_threshold"] != parameters["sam3_mask_threshold"]:
            raise ValueError("Historical candidate not frozen")
        safe_copy(source_mask, root / "候选掩膜/raw_instance_P6" / f"mask_{key}_{entry['instance_id']:02d}.png")
    meta["image_name"] = f"{key}.jpg"
    meta["cache_set_type"] = "VERIFIED_HISTORICAL_TEST32_CANDIDATES"
    meta["historical_metadata_path"] = relative(source)
    meta["historical_metadata_sha256"] = sha256(source)
    meta["model_provenance_status"] = "ARCHIVED_GENERATION_CONFIG_SAME_CHECKPOINT_PATH_ORIGINAL_TENSOR_HASH_NOT_ARCHIVED"
    write_json(metadata_path, meta)
    verify_cache(row, prod, parameters)


def provenance(parameters, torch_module):
    repo = Path(parameters["sam3_repo"])
    files = [PROD / "生成RAP-FSAM3掩膜.py", PROD / "ranking_policy.py", PROD / "candidate_metadata.py"]
    files.extend(sorted((repo / "sam3").rglob("*.py")))
    checkpoint_hash = sha256(parameters["sam3_checkpoint"])
    processor_hash = sha256(repo / "sam3/model/sam3_image_processor.py")
    if checkpoint_hash != "9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e" or processor_hash != "cb52a75ff84d46e62fbf8d18cd080b77814ab54d9e99b44763676ad194246853":
        raise ValueError("Frozen checkpoint/processor changed")
    return {"checkpoint_path": parameters["sam3_checkpoint"], "checkpoint_sha256": checkpoint_hash,
            "source_sha256": {relative(path): sha256(path) for path in files},
            "python": platform.python_version(), "torch": torch_module.__version__, "cuda": torch_module.version.cuda,
            "gpu": torch_module.cuda.get_device_name(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "parameters": parameters, "set_type": "NEW_FULL_VIEW_CANDIDATE_SET", "random_seed": 0,
            "allow_tf32_matmul": torch_module.backends.cuda.matmul.allow_tf32,
            "allow_tf32_cudnn": torch_module.backends.cudnn.allow_tf32}


def run(sample, smoke=False):
    rows = scene_rows(sample)
    if not rows or any(row["common_raw_domain_proven"] != "True" for row in rows):
        raise ValueError("Scene has unproven original identity")
    prod = production()
    parameters = frozen_parameters(device="cuda")
    parameters.update(input_dir=str(DATA / sample / "inputs"), output_dir=str(DATA / sample / "candidates"),
                      reuse_existing_candidates=False, candidate_source_dir=None, candidate_metadata_dir=None,
                      force=False, limit=0, dry_run=False)
    args = namespace(parameters)
    for row in rows:
        _, meta_path = cache_paths(sample, view_key(row))
        if not meta_path.exists() and row["historical_candidate_metadata"]:
            import_historical(row, prod, parameters)
    missing = [row for row in rows if not cache_paths(sample, view_key(row))[1].exists()]
    smoke_file = WORK / f"reports/{sample}_CANDIDATE_SMOKE.json"
    if not smoke and not smoke_file.exists():
        raise ValueError("Two-view smoke must precede full inference")
    chosen = [missing[0], missing[len(missing) // 2]] if smoke and len(missing) >= 2 else missing
    if smoke and len(chosen) != 2:
        raise ValueError("Smoke requires exactly two missing frames")
    if chosen:
        import torch
        torch.manual_seed(0)
        np.random.seed(0)
        torch.set_num_threads(4)
        precheck = provenance(parameters, torch)
        import shutil
        if shutil.disk_usage(DATA.parent).free < 20 * 1024 ** 3:
            raise ValueError("Insufficient managed-data disk space")
        free_memory, total_memory = torch.cuda.mem_get_info()
        if free_memory < 16 * 1024 ** 3:
            raise ValueError("Insufficient GPU memory for frozen SAM3")
        write_json(WORK / f"reports/{sample}_{'SMOKE' if smoke else 'FULL'}_GPU_PRECHECK.json",
                   {"status": "PASS", "free_memory_bytes": free_memory, "total_memory_bytes": total_memory,
                    "disk_free_bytes": shutil.disk_usage(DATA.parent).free, "checkpoint_sha256": precheck["checkpoint_sha256"]})
        torch_module, processor = prod.load_sam3(args)
        recorder = RecordingProcessor(processor)
        run_provenance = {**precheck, "allow_tf32_matmul": torch_module.backends.cuda.matmul.allow_tf32,
                          "allow_tf32_cudnn": torch_module.backends.cudnn.allow_tf32}
        provenance_path = DATA / sample / "candidates/GENERATION_PROVENANCE.json"
        if provenance_path.exists():
            previous = json.loads(provenance_path.read_text())
            if previous != run_provenance:
                raise ValueError("Model/code/runtime changed within frozen full-view set")
        else:
            write_json(provenance_path, run_provenance)
        write_json(WORK / f"reports/{sample}_SAM3_PROVENANCE.json", run_provenance)
        for index, row in enumerate(chosen):
            started = time.monotonic()
            key = view_key(row)
            root, metadata_path = cache_paths(sample, key)
            path = DATA / sample / "inputs" / f"{key}.jpg"
            with Image.open(path) as image, torch_module.inference_mode():
                candidates = prod.infer_candidates(image.convert("RGB"), path.name, ["P6"], parameters["prompt_texts"],
                                                  recorder, torch_module, args)
            entries = []
            for order, candidate in enumerate(candidates):
                mask_path = root / "候选掩膜/raw_instance_P6" / f"mask_{key}_{candidate.instance_id:02d}.png"
                if mask_path.exists():
                    raise ValueError("Partial mask must be audited before rerun; refusing overwrite")
                from phase20e_common import save_mask
                save_mask(mask_path, candidate.mask)
                entries.append({"prompt_id": "P6", "instance_id": candidate.instance_id,
                                "candidate_order": order, "sam_score": candidate.sam_score,
                                "box": list(candidate.box) if candidate.box else None,
                                "raw_box_float": recorder.raw_boxes[order] if order < len(recorder.raw_boxes) else None,
                                "score_tensor_dtype": recorder.score_dtype, "box_tensor_dtype": recorder.box_dtype,
                                "scores": candidate.scores, "raw_detection_count": candidate.raw_detection_count,
                                "mask_threshold": candidate.mask_threshold, "mask_sha256": sha256(mask_path),
                                "mask_pixel_sha256": pixel_hash(candidate.mask.astype(np.uint8) * 255),
                                "prompt_mode": "per_instance", "metadata_status": "VERIFIED",
                                "score_precision": "FULL_AVAILABLE_TENSOR_PRECISION",
                                "empty_fallback": candidate.raw_detection_count == 0})
            meta = {"schema_version": 1, "image_name": path.name, "image_sha256": row["raw_rgb_sha256"],
                    "source_image_path": row["raw_rgb_path"], "candidates": entries,
                    "cache_set_type": "NEW_FULL_VIEW_CANDIDATE_SET", "generation_provenance": str(provenance_path),
                    "generation_provenance_sha256": sha256(provenance_path), "mask_threshold": parameters["sam3_mask_threshold"],
                    "wall_seconds": time.monotonic() - started}
            write_json(metadata_path, meta)
            verify_cache(row, prod, parameters)
            print("INFER", sample, index + 1, "/", len(chosen), key, "instances", len(entries),
                  "seconds", round(meta["wall_seconds"], 3), flush=True)
    checked = chosen if smoke else rows
    manifest = []
    for row in checked:
        _, meta = verify_cache(row, prod, parameters)
        root, metadata_path = cache_paths(sample, view_key(row))
        for entry in meta["candidates"]:
            mask = root / "候选掩膜/raw_instance_P6" / f"mask_{view_key(row)}_{entry['instance_id']:02d}.png"
            manifest.append({"sample": sample, "frame_id": view_key(row), "registered_image_id": row["registered_image_id"],
                             "registered_name": row["registered_name"], "rgb_path": row["raw_rgb_path"],
                             "rgb_sha256": row["raw_rgb_sha256"], "set_type": meta["cache_set_type"],
                             "candidate_id": f"{view_key(row)}_{entry['instance_id']:02d}", "mask_path": str(mask),
                             "metadata_path": str(metadata_path), "metadata_sha256": sha256(metadata_path),
                             "model_provenance_status": meta.get("model_provenance_status", "FULL_CODE_AND_CHECKPOINT_HASH"),
                             **{key: json.dumps(value) if isinstance(value, (list, dict)) else value for key, value in entry.items()}})
    write_csv(WORK / f"reports/{sample}_{'SMOKE' if smoke else 'FULL'}_CANDIDATES.csv", manifest)
    if smoke:
        write_json(smoke_file, {"status": "PASS", "views": [view_key(row) for row in checked],
                               "strict_production_cache_reuse": True, "instances": len(manifest), "optimization": False})
    print("CANDIDATES", sample, "SMOKE" if smoke else "FULL", len(checked), "instances", len(manifest), "PASS", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", required=True)
    parser.add_argument("--smoke", action="store_true")
    options = parser.parse_args()
    run(options.sample, options.smoke)
