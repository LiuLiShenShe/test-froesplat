from __future__ import annotations

import collections
import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

from phase20e_common import DATA, FINAL, GS, PROD, ROOT, SAMPLES, WORK, binary_mask, colmap_io, frozen_parameters, image_array, pixel_hash, read_csv, relative, safe_copy, sha256, write_csv, write_json


def inspect(path):
    path = Path(path)
    result = {"path": relative(path), "exists": path.is_file(), "sha256": "", "width": "", "height": "", "mode": "", "error": ""}
    if not path.is_file():
        result["error"] = "MISSING"
        return result
    result["sha256"] = sha256(path)
    try:
        with Image.open(path) as image:
            image.load()
            result.update(width=image.width, height=image.height, mode=image.mode)
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
    return result


def prefix(record, name):
    return {f"{name}_{key}": value for key, value in record.items()}


def build_fft_map(sample, raw_hashes):
    folder = ROOT / "02-FFT" / sample
    log_path = folder / "filter_log.json"
    entries = [entry for entry in json.loads(log_path.read_text())["per_frame"] if entry["kept"]]
    mapping = {}
    for index, entry in enumerate(entries):
        fft = folder / f"{index:04d}.jpg"
        raw = ROOT / "01-FFmepg" / sample / entry["file"]
        fft_hash = sha256(fft) if fft.exists() else ""
        matches = raw_hashes.get(fft_hash, [])
        if len(matches) == 1:
            if matches[0] != raw:
                raise ValueError(f"FFT log/hash conflict {fft}")
            status = "LOG_AND_SHA256_IDENTICAL"
        else:
            status = "LOG_ONLY_REQUIRES_CROP_PIXEL_PROOF"
        mapping[raw.name] = {"fft_path": fft, "fft_frame": fft.stem, "status": status,
                             "log_path": log_path, "fft_sha256": fft_hash}
    return mapping


def run():
    io = colmap_io()
    identity, provenance, protection, summary, archive_search = [], [], [], [], []
    frozen = frozen_parameters()
    write_json(WORK / "reports/FROZEN_V1_PARAMETERS.json", frozen)
    tracked = subprocess.check_output(["git", "ls-files", "-z", str(GS.relative_to(ROOT)), str(PROD.relative_to(ROOT))], cwd=ROOT)
    for value in tracked.decode().split("\0"):
        if value.endswith(".py"):
            path = ROOT / value
            protection.append({"path": relative(path), "sha256": sha256(path), "kind": "PRODUCTION_CODE"})
    for sample in SAMPLES:
        print("AUDIT", sample, flush=True)
        root = ROOT / "04-COLMAP" / sample
        source = root / "distorted/sparse/0"
        target = root / "sparse/0"
        distorted = io.read_extrinsics_binary(source / "images.bin")
        undistorted = io.read_extrinsics_binary(target / "images.bin")
        dist_cameras = io.read_intrinsics_binary(source / "cameras.bin")
        cameras = io.read_intrinsics_binary(target / "cameras.bin")
        if set(distorted) != set(undistorted):
            raise ValueError(f"Registered ID mismatch {sample}")
        for model in (source, target):
            for filename in ("cameras.bin", "images.bin", "points3D.bin"):
                protection.append({"path": relative(model / filename), "sha256": sha256(model / filename), "kind": "CAMERA_SOURCE"})
        for camera_id, camera in cameras.items():
            original = dist_cameras[camera_id]
            provenance.append({"sample": sample, "camera_version": "HISTORICAL_POSES_CANONICAL_RAW_RGB_V1",
                               "pose_conditioning": "HISTORICAL_FOREGROUND_CONDITIONED" if sample == "CaoMei2" else "RAW_RGB_REGISTRATION",
                               "camera_id": camera_id, "source_model": original.model,
                               "source_width": original.width, "source_height": original.height,
                               "source_parameters": json.dumps(original.params.tolist()),
                               "target_model": camera.model, "target_width": camera.width, "target_height": camera.height,
                               "target_parameters": json.dumps(camera.params.tolist()),
                               "source_model_path": relative(source), "target_model_path": relative(target),
                               "cameras_sha256": sha256(target / "cameras.bin"), "images_sha256": sha256(target / "images.bin"),
                               "points_sha256": sha256(target / "points3D.bin")})
        raw_info, raw_hashes = {}, collections.defaultdict(list)
        for path in sorted((ROOT / "01-FFmepg" / sample).glob("*.jpg")):
            info = inspect(path)
            raw_info[path.name] = info
            raw_hashes[info["sha256"]].append(path)
        fft_map = build_fft_map(sample, raw_hashes)
        fft_by_frame = {item["fft_frame"]: (raw_name, item) for raw_name, item in fft_map.items()}
        old_candidates = {}
        for metadata_path in sorted((ROOT / "00-论文优化重构/计算机与电子农业特刊实验工作区").glob("阶段*/metadata/*/候选元数据_*.json")):
            metadata = json.loads(metadata_path.read_text())
            if sample not in metadata_path.name:
                continue
            archive_search.append({"sample": sample, "path": relative(metadata_path), "sha256": sha256(metadata_path),
                                   "image_sha256": metadata.get("image_sha256", ""), "candidate_count": len(metadata.get("candidates", []))})
            if "阶段二十D_独立GT与困难场景验证" in str(metadata_path) and "/TEST32/" in str(metadata_path):
                old_candidates[metadata["image_sha256"]] = metadata_path
        eligible, old_coverage, historical_coverage = 0, 0, 0
        for order, image in enumerate(sorted(undistorted.values(), key=lambda value: Path(value.name).stem)):
            original = distorted[image.id]
            if image.name != original.name or image.camera_id != original.camera_id or not np.array_equal(image.qvec, original.qvec) or not np.array_equal(image.tvec, original.tvec):
                raise ValueError(f"Pose/name changed {sample}/{image.id}")
            camera = cameras[image.camera_id]
            crop_domain = image.name.startswith("crop_")
            registered_stem = Path(image.name).stem
            if crop_domain:
                raw_name, mapping = fft_by_frame[registered_stem.removeprefix("crop_")]
            else:
                raw_name = image.name
                mapping = fft_map.get(raw_name)
            raw_path = ROOT / "01-FFmepg" / sample / raw_name
            raw = raw_info[raw_name]
            historical_mask = ROOT / "03-SAM" / sample / f"mask_{mapping['fft_frame']}.png" if mapping else ROOT / "NOT_AVAILABLE"
            crop = ROOT / "03-SAM" / sample / f"crop_{mapping['fft_frame']}.png" if mapping else ROOT / "NOT_AVAILABLE"
            crop_proof, mask_valid, legacy_error = False, False, ""
            try:
                mask = binary_mask(historical_mask)
                crop_pixels = image_array(crop)
                raw_pixels = image_array(raw_path)
                if mask.shape != raw_pixels.shape[:2] or crop_pixels.shape != raw_pixels.shape:
                    raise ValueError("Historical mask/RGB dimensions differ")
                crop_proof = np.array_equal(crop_pixels[mask], raw_pixels[mask]) and not crop_pixels[~mask].any()
                mask_valid = crop_proof
                if not crop_proof:
                    legacy_error = "CROP_MASK_RAW_PIXEL_MISMATCH"
            except Exception as error:
                legacy_error = f"{type(error).__name__}: {error}"
            colmap_input = inspect(root / "input" / image.name)
            undistorted_rgb = inspect(root / "images" / image.name)
            rgba = root / "images_rgba" / (registered_stem + ".png")
            old_rgba = inspect(rgba)
            input_proof = (colmap_input["sha256"] == raw["sha256"] if not crop_domain else
                           colmap_input["sha256"] == (sha256(crop) if crop.is_file() else ""))
            reliable_raw = not raw["error"] and raw["width"] == dist_cameras[image.camera_id].width and raw["height"] == dist_cameras[image.camera_id].height
            domain_proof = reliable_raw and (crop_proof if crop_domain else input_proof or bool(colmap_input["error"]))
            candidate = old_candidates.get(raw["sha256"])
            old_coverage += bool(candidate)
            historical_coverage += mask_valid
            eligible += domain_proof
            row = {"sample": sample, "registered_image_id": image.id, "registered_name": image.name,
                   "camera_id": image.camera_id, "raw_frame": Path(raw_name).stem,
                   "actual_raw_frame_id": f"{sample}_{Path(raw_name).stem}",
                   "historical_fft_frame": mapping["fft_frame"] if mapping else "",
                   "fft_lineage_status": mapping["status"] if mapping else "NO_FFT_MASK",
                   "fft_log_path": relative(mapping["log_path"]) if mapping else "",
                   "same_stem_is_same_raw_view": Path(raw_name).stem == registered_stem.removeprefix("crop_"),
                   "crop_pixel_identity_verified": crop_proof, "historical_mask_verified": mask_valid,
                   "historical_mask_path": relative(historical_mask) if mapping else "",
                   "historical_mask_sha256": sha256(historical_mask) if historical_mask.is_file() else "",
                   "historical_crop_path": relative(crop) if mapping else "", "legacy_error": legacy_error,
                   "colmap_input_identity_verified": input_proof, "common_raw_domain_proven": domain_proof,
                   "split": "validation" if order % 8 == 0 else "train", "split_order_index": order,
                   "camera_width": camera.width, "camera_height": camera.height,
                   "qvec_json": json.dumps(image.qvec.tolist()), "tvec_json": json.dumps(image.tvec.tolist()),
                   "camera_params_json": json.dumps(camera.params.tolist()),
                   "historical_candidate_metadata": relative(candidate) if candidate else "",
                   "reliable_historical_P6": bool(candidate),
                   **prefix(raw, "raw_rgb"), **prefix(colmap_input, "colmap_input"),
                   **prefix(undistorted_rgb, "historical_undistorted"), **prefix(old_rgba, "historical_rgba")}
            identity.append(row)
            for path in [raw_path, historical_mask, crop, root / "input" / image.name, root / "images" / image.name, rgba]:
                if path.is_file():
                    protection.append({"path": relative(path), "sha256": sha256(path), "kind": "REGISTERED_INPUT"})
            if order % 25 == 0:
                print(sample, order, "/", len(undistorted), "raw", raw_name, "legacy", mask_valid, flush=True)
        summary.append({"sample": sample, "registered": len(undistorted), "raw_domain_proven": eligible,
                        "historical_P6_views": old_coverage, "historical_mask_coverage": historical_coverage,
                        "sparse_points": len(io.read_points3D_binary(target / "points3D.bin")[0])})
    write_csv(WORK / "FULL_VIEW_IDENTITY_MANIFEST.csv", identity)
    write_csv(WORK / "CAMERA_PROVENANCE_MATRIX.csv", provenance)
    write_csv(WORK / "reports/HISTORICAL_CANDIDATE_SEARCH.csv", archive_search)
    write_csv(WORK / "reports/SOURCE_PROTECTION.csv", list({row["path"]: row for row in protection}.values()))
    previous = read_csv(FINAL / "SOURCE_PROTECTION_MANIFEST.csv")
    merged = {row["path"]: row for row in protection}
    for row in previous:
        if row["path"] not in merged:
            merged[row["path"]] = {"path": row["path"], "sha256": row["sha256"], "kind": "PREVIOUS_STAGE_FULL_SOURCE_PROTECTION"}
    write_csv(WORK / "reports/SOURCE_PROTECTION.csv", list(merged.values()))
    write_json(WORK / "reports/SCENE_AUDIT.json", summary)
    for sample in SAMPLES:
        for row in [row for row in identity if row["sample"] == sample]:
            target = DATA / sample / "inputs" / f"{sample}_{row['raw_frame']}.jpg"
            safe_copy(ROOT / row["raw_rgb_path"], target)
        safe_copy(FINAL / "logs/full_r1_runtime_parameters.json", DATA / sample / "FROZEN_PHASE20D_SNAPSHOT.json")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    run()
