from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

from direct_common import OLD, PHASEC, PROD, ROOT, SWITCHES, WORK, frames, local_path, pixel_hash, production_module, read_csv, read_mask, relative, sha256, write_csv, write_json

NEGATABLE = {"save_raw_instance_masks", "save_semantic_gate_debug", "fill_holes", "keep_largest_component",
             "geometry_enable_negative_correction", "save_corrective_geometry_debug", "use_spnp_evidence_guidance",
             "memory_bidirectional", "save_candidate_masks", "save_intermediate_masks", "save_foreground_rgb",
             "save_rgba", "save_overlay", "log_json"}
OFF = ["use_spnp_refinement", "use_residual_repair", "use_geometry_feedback", "use_corrective_geometry",
       "use_cross_view_consensus", "use_memory_propagation", "use_prompt_ensemble", "use_semantic_gate",
       "use_temporal_alignment", "use_reprompt_detection", "use_spnp_evidence_guidance"]


def frozen_parameters(policy, input_dir, output_dir):
    parameters = json.loads((OLD / "configs/FROZEN_V1_PARAMETERS.json").read_text())["parameters"]
    parameters.update(input_dir=str(input_dir), output_dir=str(output_dir),
                      candidate_source_dir=str(OLD / "candidates/TEST32"), candidate_metadata_dir=str(OLD / "metadata/TEST32"),
                      ranking_policy=policy)
    if any(parameters[name] for name in OFF):
        raise ValueError("Frozen optional module unexpectedly enabled")
    return parameters


def command_for(parameters):
    command = [sys.executable, str(PROD / "生成RAP-FSAM3掩膜.py")]
    for key, value in parameters.items():
        if key in ("prompt_texts", "prompt_ids") or value is None:
            continue
        if isinstance(value, bool):
            if value:
                command.append("--" + key)
            elif key in NEGATABLE:
                command.append("--no-" + key)
        else:
            command.extend(["--" + key, str(value)])
    return command


def validate_outputs(output, expected_frames, policy):
    parameters = json.loads((output / "参数.json").read_text())
    expected_parameters = frozen_parameters(policy, Path(parameters["input_dir"]), output)
    if parameters != expected_parameters:
        differences = {name: [parameters.get(name), expected_parameters.get(name)] for name in set(parameters) | set(expected_parameters)
                       if parameters.get(name) != expected_parameters.get(name)}
        raise ValueError(f"Runtime configuration differs from frozen recipe: {differences}")
    scores = read_csv(output / "提示词评分.csv")
    choices = read_csv(output / "提示词选择.csv")
    source = read_csv(OLD / "TEST32_CANDIDATE_PROVENANCE.csv")
    if {row["图像"] for row in choices} != {frame["frame_id"] + ".jpg" for frame in expected_frames}:
        raise ValueError("Production selection log coverage mismatch")
    production = production_module()
    ranking = sys.modules["ranking_policy"]
    rows = []
    for frame in expected_frames:
        stem = frame["frame_id"]
        selected_path = output / "选择后掩膜" / f"mask_{stem}.png"
        final_path = output / "最终掩膜" / f"mask_{stem}.png"
        selected, final = read_mask(selected_path), read_mask(final_path)
        if selected.shape != (3840, 2160) or final.shape != selected.shape:
            raise ValueError("Production output grid mismatch")
        candidates = [row for row in source if row["frame_id"] == stem]
        matched = [row for row in candidates if np.array_equal(read_mask(local_path(row["mask_path"])), selected)]
        if len(matched) != 1:
            raise ValueError(f"Nonunique raw selected candidate identity: {stem}")
        chosen = matched[0]
        frame_scores = [row for row in scores if row["图像"] == stem + ".jpg"]
        if len(frame_scores) != len(candidates):
            raise ValueError("Candidate score coverage mismatch")
        details = read_csv(output / "候选掩膜" / f"候选评分明细_{stem}.csv")
        loaded = production.load_existing_candidates(OLD / "inputs/TEST32" / f"{stem}.jpg", ["P6"],
                                                     parameters["prompt_texts"], output, OLD / "candidates/TEST32",
                                                     OLD / "metadata/TEST32", True)
        arguments = argparse.Namespace(**parameters)
        weights = production.parse_key_values(parameters["score_weights"], {})
        with Image.open(local_path(frame["rgb_path"])) as image:
            records = [production.score_candidate(OLD / "inputs/TEST32" / f"{stem}.jpg", image.convert("RGB"),
                       candidate, {}, weights, arguments) for candidate in loaded]
        best = ranking.select_best_policy(records, policy)
        if best.instance_id != int(chosen["instance_id"]):
            raise ValueError("Pure frozen ranking replay disagrees with real production mask")
        for row, detail, record in zip(frame_scores, details, records, strict=True):
            candidate_id = f"{stem}_{int(detail['实例编号']):02d}"
            metadata = next(value for value in candidates if value["candidate_id"] == candidate_id)
            if abs(float(detail["SAM3分数"]) - float(metadata["sam_score_archived_text"])) > 1e-12:
                raise ValueError("Archived confidence mismatch")
            if record.instance_id != int(detail["实例编号"]):
                raise ValueError("Explicit candidate order mismatch")
            for key in ("总分", "主评分", "面积得分", "前背景对比得分", "平局决胜分"):
                if abs(float(row[key]) - float(production.score_to_dict(record)[key])) > 1e-12:
                    raise ValueError("Saved score decomposition disagrees with unchanged scorer")
        selected_score = vars(best)
        if not np.array_equal(selected, final):
            raise ValueError("Frozen V0/V1 must preserve raw selected mask with all refinementOFF")
        rows.append({"frame_id": stem, "policy": policy, "selected_candidate_id": chosen["candidate_id"],
                     "source_candidate_sha256": chosen["mask_sha256"], "metadata_sha256": frame["candidate_metadata_sha256"],
                     "source_rgb_sha256": frame["rgb_sha256"], "selected_mask_path": relative(selected_path),
                     "selected_mask_sha256": sha256(selected_path), "selected_mask_pixels_sha256": pixel_hash(selected.astype(np.uint8) * 255),
                     "final_mask_path": relative(final_path), "output_mask_sha256": sha256(final_path),
                     "score_decomposition_json": json.dumps(selected_score, ensure_ascii=False, sort_keys=True),
                     "full_parameters_sha256": sha256(output / "参数.json"), "raw_equals_final": True,
                     "empty_output": int(not final.any())})
    return rows


def run(policy, smoke):
    tag = f"smoke_{policy}_{smoke}" if smoke else f"full_{policy}"
    output = WORK / "runs" / tag
    if output.exists():
        raise FileExistsError(f"Refusing pre-existing production output: {output}")
    if not smoke:
        for stem in SWITCHES:
            for smoke_policy in ("base", "r1"):
                status = WORK / "logs" / f"smoke_{smoke_policy}_{stem}_status.json"
                if not status.exists() or not json.loads(status.read_text()).get("verified", False):
                    raise ValueError("All four named smoke runs must pass before full evaluation")
    expected = [frame for frame in frames() if not smoke or frame["frame_id"] == smoke]
    input_dir = OLD / "inputs/TEST32"
    if smoke:
        input_dir = WORK / "smoke_inputs" / smoke
        input_dir.mkdir(parents=True, exist_ok=True)
        source = local_path(expected[0]["rgb_path"])
        link = input_dir / f"{smoke}.jpg"
        if not link.exists():
            link.symlink_to(source)
        if sha256(link) != expected[0]["rgb_sha256"]:
            raise ValueError("SmokeRGB source mismatch")
    parameters = frozen_parameters(policy, input_dir, output)
    command = command_for(parameters)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(PHASEC / "scripts/runtime_compat") + os.pathsep + environment.get("PYTHONPATH", "")
    environment["OMP_NUM_THREADS"] = "4"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["CUDA_VISIBLE_DEVICES"] = ""
    snapshot = {"command": command, "cwd": str(ROOT), "parameters": parameters,
                "production_sha256": sha256(PROD / "生成RAP-FSAM3掩膜.py"),
                "ranking_code_sha256": sha256(PROD / "ranking_policy.py"),
                "metadata_loader_sha256": sha256(PROD / "candidate_metadata.py"),
                "wrapper_sha256": sha256(Path(__file__)), "spec_sha256": sha256(WORK / "CONTOUR_RASTERIZATION_SPEC.md"),
                "PYTHONPATH": environment["PYTHONPATH"], "OMP_NUM_THREADS": "4", "device": "cpu",
                "no_new_candidate_inference": True, "frames": [frame["frame_id"] for frame in expected]}
    write_json(WORK / "logs" / f"{tag}_execution.json", snapshot)
    log = WORK / "logs" / f"{tag}_stdout.txt"
    started = time.monotonic()
    with log.open("x", encoding="utf-8") as handle:
        result = subprocess.run(command, cwd=ROOT, env=environment, stdout=handle, stderr=subprocess.STDOUT)
    status = {"policy": policy, "smoke": smoke or "", "returncode": result.returncode,
              "wall_sec": time.monotonic() - started, "output": relative(output), "log": relative(log), "verified": False}
    if result.returncode == 0:
        rows = validate_outputs(output, expected, policy)
        write_csv(WORK / "logs" / f"{tag}_selections.csv", rows)
        write_csv(WORK / "logs" / f"{tag}_artifacts.csv", [{"path": relative(path), "bytes": path.stat().st_size,
                   "sha256": sha256(path)} for path in sorted(output.rglob("*")) if path.is_file()])
        status["verified"] = True
        write_json(WORK / "logs" / f"{tag}_runtime_parameters.json", json.loads((output / "参数.json").read_text()))
        write_csv(WORK / "logs" / f"{tag}_all_candidate_scores.csv", read_csv(output / "提示词评分.csv"))
        write_csv(WORK / "logs" / f"{tag}_production_selection.csv", read_csv(output / "提示词选择.csv"))
        write_json(WORK / "logs" / f"{tag}_runtime_log.json", json.loads((output / "运行日志.json").read_text()))
    write_json(WORK / "logs" / f"{tag}_status.json", status)
    print(json.dumps(status, ensure_ascii=False), flush=True)
    if not status["verified"]:
        raise RuntimeError(f"Frozen production failed; inspect {log}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=["base", "r1"], required=True)
    parser.add_argument("--smoke", choices=SWITCHES)
    args = parser.parse_args()
    run(args.policy, args.smoke)


if __name__ == "__main__":
    main()
