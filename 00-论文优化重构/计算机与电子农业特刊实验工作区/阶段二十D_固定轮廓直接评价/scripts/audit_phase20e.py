from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from PIL import Image

from direct_common import OLD, PROD, ROOT, WORK, frames, load_module, read_csv, relative, sha256, write_csv, write_json


def image_files(folder):
    return sorted(path for path in folder.iterdir() if path.suffix.lower() in (".png", ".jpg", ".jpeg")) if folder.exists() else []


def image_info(path):
    if path is None:
        return "", "", "", ""
    try:
        with Image.open(path) as image:
            width, height = image.size
            mode = image.mode
    except OSError:
        return "", "", "UNREADABLE_IMAGE", sha256(path)
    return width, height, mode, sha256(path)


def exact_or_extension(folder, name):
    path = folder / name
    if path.exists():
        return path
    matches = [folder / (Path(name).stem + suffix) for suffix in (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG")]
    return next((path for path in matches if path.exists()), None)


def main():
    loader = load_module("phase20e_readonly_colmap", ROOT / "2d-gaussian-splatting-main/scene/colmap_loader.py")
    view_rows, camera_rows, summaries, source_rows = [], [], [], []
    source_candidate_ids = {row["frame_id"] for row in read_csv(OLD / "TEST32_CANDIDATE_PROVENANCE.csv")}
    for sample in ("CaoMei2", "WanNianQing1", "HongZhang"):
        raw_root = ROOT / "01-FFmepg" / sample
        model_root = ROOT / "04-COLMAP" / sample
        undistorted_root = model_root / "sparse/0"
        distorted_root = model_root / "distorted/sparse/0"
        undistorted_images = loader.read_extrinsics_binary(str(undistorted_root / "images.bin"))
        distorted_images = loader.read_extrinsics_binary(str(distorted_root / "images.bin"))
        undistorted_cameras = loader.read_intrinsics_binary(str(undistorted_root / "cameras.bin"))
        distorted_cameras = loader.read_intrinsics_binary(str(distorted_root / "cameras.bin"))
        distorted_by_name = {entry.name: entry for entry in distorted_images.values()}
        if set(distorted_by_name) != {entry.name for entry in undistorted_images.values()}:
            raise ValueError("Original/undistorted model registration identity mismatch")
        for kind, sparse, cameras in (("distorted", distorted_root, distorted_cameras), ("undistorted", undistorted_root, undistorted_cameras)):
            for name in ("cameras.bin", "images.bin", "points3D.bin"):
                path = sparse / name
                source_rows.append({"sample": sample, "kind": kind, "path": relative(path), "bytes": path.stat().st_size, "sha256": sha256(path)})
            for camera in cameras.values():
                camera_rows.append({"sample": sample, "kind": kind, "camera_id": camera.id, "model": camera.model,
                                    "width": camera.width, "height": camera.height, "params_json": json.dumps(camera.params.tolist()),
                                    "model_path": relative(sparse), "cameras_sha256": sha256(sparse / "cameras.bin")})
        config = ROOT / "05-2DGS-new" / sample / "cfg_args"
        source_rows.append({"sample": sample, "kind": "historical_2dgs_config", "path": relative(config),
                            "bytes": config.stat().st_size, "sha256": sha256(config)})
        sample_rows = []
        for index, entry in enumerate(sorted(undistorted_images.values(), key=lambda row: Path(row.name).stem)):
            match = re.fullmatch(r"(?:crop_)?(\d{4})\.(?:png|jpg|jpeg)", entry.name)
            if not match:
                raise ValueError(f"No explicit full-view raw identity mapping for{entry.name}")
            number = match[1]
            raw_path = raw_root / f"{number}.jpg"
            if not raw_path.exists():
                raise ValueError("Registered view cannot be joined to original rawRGB")
            input_path = exact_or_extension(model_root / "input", entry.name)
            undistorted_path = exact_or_extension(model_root / "images", entry.name)
            historical_path = exact_or_extension(model_root / "images_rgba", entry.name)
            raw_width, raw_height, raw_mode, raw_hash = image_info(raw_path)
            input_width, input_height, input_mode, input_hash = image_info(input_path)
            width, height, mode, image_hash = image_info(undistorted_path)
            alpha_width, alpha_height, alpha_mode, alpha_hash = image_info(historical_path)
            undistorted_camera = undistorted_cameras[entry.camera_id]
            distorted_entry = distorted_by_name[entry.name]
            distorted_camera = distorted_cameras[distorted_entry.camera_id]
            undistorted_matches_camera = (width, height) == (undistorted_camera.width, undistorted_camera.height)
            raw_matches_camera = (raw_width, raw_height) == (distorted_camera.width, distorted_camera.height)
            raw_candidate_frame = f"{sample}_{number}" in source_candidate_ids
            row = {"sample": sample, "registered_image_id": entry.id, "registered_name": entry.name,
                   "registered_camera_id": entry.camera_id, "raw_frame_id": f"{sample}_{number}", "raw_frame": number,
                   "raw_rgb_path": relative(raw_path), "raw_rgb_sha256": raw_hash, "raw_width": raw_width, "raw_height": raw_height,
                   "colmap_input_path": relative(input_path) if input_path else "", "colmap_input_sha256": input_hash,
                   "colmap_input_width": input_width, "colmap_input_height": input_height, "colmap_input_mode": input_mode,
                   "colmap_input_equal_raw_bytes": int(input_hash == raw_hash),
                   "undistorted_rgb_path": relative(undistorted_path) if undistorted_path else "", "undistorted_rgb_sha256": image_hash,
                   "undistorted_width": width, "undistorted_height": height,
                   "undistorted_mode": mode, "undistorted_grid_matches_camera": int(undistorted_matches_camera),
                   "camera_width": undistorted_camera.width, "camera_height": undistorted_camera.height,
                   "raw_grid_matches_camera": int(raw_matches_camera),
                   "historical_rgba_path": relative(historical_path) if historical_path else "",
                   "historical_rgba_sha256": alpha_hash, "historical_rgba_width": alpha_width, "historical_rgba_height": alpha_height,
                   "historical_rgba_exists": int(historical_path is not None),
                   "historical_rgba_grid_matches_undistorted": int((alpha_width, alpha_height) == (width, height)),
                   "frozen_split": "validation" if index % 8 == 0 else "train", "split_order_index": index,
                   "full_view_P6_candidate_verified": int(raw_candidate_frame),
                   "frozen71_candidate_source": "TEST32_ARCHIVE" if raw_candidate_frame else "MISSING_VERSIONED_FULL_VIEW_CANDIDATES",
                   "mapping_status": "OPENCV_TO_PINHOLE_REMAP_REQUIRED_NOT_PERFORMED",
                   "original_gate_certification_not_required_for_engineering": True}
            view_rows.append(row)
            sample_rows.append(row)
        model_numbers = {row["raw_frame"] for row in sample_rows}
        test_views = [frame for frame in frames() if frame["sample"] == sample]
        summaries.append({"sample": sample, "raw_rgb_count": len(image_files(raw_root)), "registered_views": len(sample_rows),
                          "train_views": sum(row["frozen_split"] == "train" for row in sample_rows),
                          "validation_views": sum(row["frozen_split"] == "validation" for row in sample_rows),
                          "historical_rgba_exists": sum(row["historical_rgba_exists"] for row in sample_rows),
                          "historical_rgba_matching_grid": sum(row["historical_rgba_grid_matches_undistorted"] for row in sample_rows),
                          "undistorted_rgb_matching_grid": sum(row["undistorted_grid_matches_camera"] for row in sample_rows),
                          "undistorted_rgb_unreadable": sum(row["undistorted_mode"] == "UNREADABLE_IMAGE" for row in sample_rows),
                          "colmap_input_unreadable": sum(row["colmap_input_mode"] == "UNREADABLE_IMAGE" for row in sample_rows),
                          "verified_frozen_candidate_views": sum(row["full_view_P6_candidate_verified"] for row in sample_rows),
                          "missing_full_view_candidate_views": sum(not row["full_view_P6_candidate_verified"] for row in sample_rows),
                          "colmap_input_equal_raw_bytes_views": sum(row["colmap_input_equal_raw_bytes"] for row in sample_rows),
                          "test32_registered_views": ";".join(frame["frame_id"] for frame in test_views if frame["frame"] in model_numbers),
                          "test32_unregistered_views": ";".join(frame["frame_id"] for frame in test_views if frame["frame"] not in model_numbers),
                          "actual_undistorted_dimensions": json.dumps(sorted({f"{row['undistorted_width']}x{row['undistorted_height']}" for row in sample_rows})),
                          "model_undistorted_dimensions": json.dumps(sorted({f"{row['camera_width']}x{row['camera_height']}" for row in sample_rows})),
                          "historical_cfg_args": config.read_text().strip(), "pilot_ready": "CONDITIONAL"})
        print(f"auditedPhase20E {sample}: {len(sample_rows)} registered, {summaries[-1]['missing_full_view_candidate_views']} candidate views missing", flush=True)
    candidate_search = []
    for root in (ROOT / "00-论文优化重构", ROOT / "03-SAM", ROOT / "sam3"):
        for path in root.rglob("候选元数据_*.json"):
            if path.is_relative_to(WORK) or "__pycache__" in path.parts:
                continue
            if any(sample in path.name for sample in ("CaoMei2", "WanNianQing1", "HongZhang")):
                candidate_search.append({"path": relative(path), "sha256": sha256(path),
                                         "frame_id": path.stem.removeprefix("候选元数据_")})
    write_csv(WORK / "PHASE20E_VIEW_MANIFEST.csv", view_rows)
    write_csv(WORK / "PHASE20E_CAMERA_MANIFEST.csv", camera_rows)
    write_csv(WORK / "PHASE20E_SEQUENCE_READINESS.csv", summaries)
    write_csv(WORK / "PHASE20E_SOURCE_HASHES.csv", source_rows)
    write_csv(WORK / "PHASE20E_CANDIDATE_ARCHIVE_SEARCH.csv", candidate_search)
    write_json(WORK / "logs/PHASE20E_AUDIT_STATUS.json", {"status": "CONDITIONAL", "training_started": False,
               "registered_views": len(view_rows), "full_sequence_candidates_missing": True,
               "historical_rgba_coordinate_or_coverage_gaps": True,
               "split": "Sort exact registered image stem lexicographically; index%8==0 heldout, else train",
               "source_search_roots": [relative(ROOT / name) for name in ("00-论文优化重构", "03-SAM", "sam3")],
               "loader_sha256": sha256(ROOT / "2d-gaussian-splatting-main/scene/colmap_loader.py")})


if __name__ == "__main__":
    main()
