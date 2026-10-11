from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from fix_common import ALG, ROOT, WORK, link, load_module, manifest, read_csv, relative, sha256, write_csv, write_json


def geometry_audit():
    loader = load_module("phase20c_fix_colmap", ROOT / "2d-gaussian-splatting-main/scene/colmap_loader.py")
    rows = []
    samples = {value["sample_id"] for value in manifest()}
    for sample in sorted(samples):
        root = ROOT / "04-COLMAP" / sample
        for sparse in [root / "sparse/0", root / "distorted/sparse/0"]:
            if not (sparse / "images.bin").exists():
                continue
            images = loader.read_extrinsics_binary(str(sparse / "images.bin"))
            cameras = loader.read_intrinsics_binary(str(sparse / "cameras.bin"))
            named = {Path(image.name).stem: image for image in images.values()}
            for frame in [value for value in manifest() if value["sample_id"] == sample]:
                image = named.get(frame["frame"]) or named.get(frame["stem"]) or named.get(f"crop_{frame['frame']}")
                with Image.open(ROOT / frame["input_path"]) as handle:
                    rgb_width, rgb_height = handle.size
                source = root / "images" / f"{frame['frame']}.jpg"
                raw_source = ROOT / "00-论文优化重构/数据管理/01-输入图像/01-raw_frames" / sample / f"{frame['frame']}.jpg"
                candidate_source = source if source.exists() else raw_source
                camera = cameras[image.camera_id] if image is not None else next(iter(cameras.values()))
                same_dimensions = (camera.width, camera.height) == (rgb_width, rgb_height)
                registered = image is not None
                count = int((image.point3D_ids >= 0).sum()) if registered else 0
                rows.append({"dataset": frame["dataset"], "sample": sample, "frame": frame["frame"],
                             "model": relative(sparse), "images_bin_sha256": sha256(sparse / "images.bin"),
                             "cameras_bin_sha256": sha256(sparse / "cameras.bin"), "registered_image_count": len(images),
                             "registered": int(registered), "image_name": image.name if registered else "",
                             "image_id": image.id if registered else "", "camera_id": camera.id,
                             "camera_model": camera.model, "camera_width": camera.width, "camera_height": camera.height,
                             "rgb_width": rgb_width, "rgb_height": rgb_height, "coordinate_dimensions_match": int(same_dimensions),
                             "points_with_3d_ids": count, "source_image_path": relative(candidate_source),
                             "source_rgb_sha256": sha256(candidate_source) if candidate_source.exists() else "",
                             "input_rgb_sha256": frame["input_sha256"],
                             "rgb_identity_match": int(candidate_source.exists() and sha256(candidate_source) == frame["input_sha256"]),
                             "decision": "GEOMETRY_NOT_AVAILABLE" if not registered or not count else
                                         "REGISTERED_RAW_COORDINATES" if same_dimensions and "distorted" in sparse.parts else
                                         "COORDINATE_MAPPING_UNVERIFIED"})
    write_csv(WORK / "COLMAP_MODEL_IMAGE_AUDIT.csv", rows)
    missing = [value for value in rows if value["sample"] == "DouBanLv2" and value["frame"] in ("0037", "0039", "0041")]
    if any(value["registered"] for value in missing):
        raise ValueError("Unexpected registration requires additional identity/coordinate verification")
    path = WORK / "GT10_COLMAP_OBSERVATION_DECISION.md"
    path.write_text("""# GT10 COLMAP observation decision

DouBanLv2 0037、0039、0041 在冻结原始 OPENCV distorted sparse 模型和
PINHOLE undistorted sparse 模型中均未注册（两个模型各 210 registered images）。
逐条核对真实 image.name / image_id / camera_id，而非只依赖生产前缀匹配。
0038、0040 能找到真实 image_id 35、36 及非负 point3D_id；不得借给缺失帧。
原始输入 RGB 存在，并与冻结候选输入 SHA256 匹配；不是输入路径丢失或前缀错误。

缺失帧保留 `GEOMETRY_NOT_AVAILABLE`。不生成伪稀疏点、不用 GT 反馈、
不借邻帧观测、不重建 COLMAP、不改 COLMAP/2DGS 核心算法。

额外坐标审计：DouBanLv2 undistorted 模型 PINHOLE 为 2119×3779，
distorted OPENCV 为 2160×3840，冻结 RGB 为 2160×3840。
旧生产 sparse 优先路径的 undistorted xys 不能仅凭名称直接认定与原始 RGB
同坐标。完整表 `COLMAP_MODEL_IMAGE_AUDIT.csv` 记录所有 Easy21/GT10 注册、
camera dimensions、RGB identity 和模型 hash。历史 V3 可作同帧保存掩膜的
描述性比较；在坐标映射未验证前，不作为 GCFC 机制收益或最终默认依据。

SPNP-only 未启用 evidence_guidance、geometry_feedback、corrective_geometry、
A6/A7；仅用当前 RGB、已选 mask、冻结 SAM3 checkpoint 和 mask 派生正负框。
因此上述几何缺失不阻断 V4/V5。

来源：原始 `04-COLMAP/DouBanLv2/{sparse/0,distorted/sparse/0}/{images,cameras}.bin`；
只读官方格式解析器 `2d-gaussian-splatting-main/scene/colmap_loader.py`；
本阶段 `COLMAP_MODEL_IMAGE_AUDIT.csv`。未修改历史几何。
""", encoding="utf-8")


def independent_package():
    phase16 = ALG / "阶段十六_HardCase_GT构建与锁定"
    phase17a = ALG / "阶段十七A_10GT困难样本Pilot与无GT测试审计"
    test_path = phase16 / "挑战集列表/Phase16_TEST_manifest.csv"
    dev_path = phase16 / "挑战集列表/Phase16_DEV_manifest.csv"
    tests, dev = read_csv(test_path), read_csv(dev_path)
    if {value["sample"] for value in tests} & {value["sample"] for value in dev}:
        raise ValueError("Frozen sample-level DEV/TEST leakage")
    source = phase17a / "03_trackB_variants/U00_Control"
    rows, candidates, annotation_queue = [], [], []
    for frame in tests:
        sample, number = frame["sample"], frame["frame"]
        stem = f"{sample}_{number}"
        rgb = Path(frame["image_path"])
        if not rgb.exists():
            raise FileNotFoundError(rgb)
        alias = phase17a / "01_input_trackB" / f"{stem}.jpg"
        with Image.open(rgb) as handle:
            width, height = handle.size
        detail_path = source / "候选掩膜" / f"候选评分明细_{stem}.csv"
        details = read_csv(detail_path) if detail_path.exists() else []
        raw_masks = sorted((source / "候选掩膜/raw_instance_P6").glob(f"mask_{stem}_*.png"))
        expected_ids = {int(value["实例编号"]) for value in details if value["提示词编号"] == "P6"}
        actual_ids = {int(path.stem.rsplit("_", 1)[1]) for path in raw_masks}
        gt_json = ROOT / "03-GT-区分_challenge" / sample / f"{number}.json"
        rows.append({"challenge_id": frame["challenge_id"], "split": "TEST", "sample": sample, "frame": number,
                     "rgb_path": relative(rgb), "rgb_sha256": sha256(rgb), "width": width, "height": height,
                     "frozen_trackB_rgb_sha256": sha256(alias), "input_identity_match": int(sha256(alias) == sha256(rgb)),
                     "gt_status": frame["gt_status"], "annotation_found": int(gt_json.exists()),
                     "plant_label_status": "NOT_MANUALLY_CONFIRMED", "pot_label_status": "NOT_MANUALLY_CONFIRMED",
                     "candidate_source": relative(source), "candidate_count": len(raw_masks),
                     "candidate_complete": int(bool(raw_masks) and expected_ids == actual_ids),
                     "heuristic_primary_NOT_GT": frame["primary_difficulty"],
                     "heuristic_secondary_NOT_GT": frame["secondary_difficulty"]})
        for raw in raw_masks:
            candidates.append({"challenge_id": frame["challenge_id"], "candidate_id": raw.stem.removeprefix("mask_"),
                               "path": relative(raw), "sha256": sha256(raw), "detail_path": relative(detail_path),
                               "detail_sha256": sha256(detail_path)})
        link(rgb, WORK / "annotation_package/rgb" / f"{stem}.jpg")
        annotation_queue.append({"challenge_id": frame["challenge_id"], "sample": sample, "frame": number,
                                 "rgb_sha256": sha256(rgb), "annotation_json": f"annotation_package/annotations/{stem}.json",
                                 "reviewer": "", "second_reviewer": "", "target_identity_confirmed": "",
                                 "plant_confirmed": "", "pot_confirmed": "", "blue_cube_present": "",
                                 "difficulty_manually_confirmed": "", "human_difficulty_labels": "", "approval": "PENDING"})
    write_csv(WORK / "TEST32_INPUT_AUDIT.csv", rows)
    write_csv(WORK / "TEST32_CANDIDATE_AUDIT.csv", candidates)
    write_csv(WORK / "annotation_package/ANNOTATION_REVIEW_QUEUE.csv", annotation_queue)
    write_json(WORK / "annotation_package/FROZEN_TEST_SPEC.json", {
        "test_manifest_sha256": sha256(test_path), "dev_manifest_sha256": sha256(dev_path),
        "frames": len(tests), "samples": sorted({value["sample"] for value in tests}),
        "labels": ["potted_plant", "plant", "pot", "blue_cube"],
        "target_definition": "potted_clean = target_potted_plant & ~blue_cube; one manually confirmed target, exclude neighbors",
        "human_review_required": True, "tuning_forbidden": True,
        "metrics": ["f1", "iou_fg", "hd95_px", "boundary_f1_3px", "leakage_fp_per_image"],
        "aggregation": ["sample_equal_mean", "frame_level_paired_difference"],
        "candidate_generation": "P6/per_instance/single, frozen confidence/checkpoint; U00_Control source kept distinct"})
    print(f"Geometry audited; TEST32 package: {len(rows)} frames, {sum(row['annotation_found'] for row in rows)} annotations found, {len(candidates)} candidate masks")


if __name__ == "__main__":
    geometry_audit()
    independent_package()
