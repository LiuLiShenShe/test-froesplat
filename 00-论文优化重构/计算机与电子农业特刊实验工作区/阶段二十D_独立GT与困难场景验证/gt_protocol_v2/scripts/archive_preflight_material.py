from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from PIL import Image

from protocol_v2 import (DIFFICULTIES, PACKAGE, PROTOCOL, ROOT, STAGE, derive_difficulty,
                         json_digest, read_csv, sha256, shape_geometry, unique_index, verify_original_sources)
from phase20d_common import relative, write_csv, write_json, write_text


VISIBILITY_FIELDS = ("pot_visibility", "plant_visible", "blue_cube_present", "visibility_confirmed_by",
                     "visibility_confirmed_is_human", "visibility_confirmed_at", "visibility_confirmed_source", "visibility_status")
REVIEW_FIELDS = ("annotator", "reviewer", "second_reviewer", "annotator_is_human", "reviewer_is_human",
                 "second_reviewer_is_human", "reviewed_at", "second_reviewed_at", "target_identity_description",
                 "difficulty_evidence_json", "manual_annotation_confirmed", "target_identity_confirmed",
                 "visible_components_confirmed", "difficulty_manually_confirmed", "annotation_origin", "approval", "annotation_sha256")


def archive(stage, records, conversion, details, output):
    frames = read_csv(stage / "TEST32_INPUT_MANIFEST.csv")
    verify_original_sources(stage, frames)
    changed = subprocess.check_output(["git", "diff", "--name-only", "6092e2ea", "--"], cwd=ROOT, text=True).splitlines()
    if any(not path.startswith(relative(PACKAGE) + "/") for path in changed):
        raise ValueError("Protected tracked files outside this V2 work scope changed")
    identities = unique_index(read_csv(records / "SOURCE_IDENTITIES.csv"), "frame_id")
    visibility = unique_index(read_csv(records / "GT_VISIBILITY_MANIFEST.csv"), "frame_id")
    reviews = unique_index(read_csv(records / "HUMAN_REVIEW_V2.csv"), "frame_id")
    old_difficulty = unique_index(read_csv(stage / "DIFFICULTY_HUMAN_VERIFICATION.csv"), "frame_id")
    outlines = read_csv(records / "LINESTRIP_CONVERSION_AUDIT.csv")
    converted = read_csv(conversion / "CONVERTED_GT_MANIFEST.csv")
    expected = {frame["frame_id"] for frame in frames}
    if any(set(index) != expected for index in (identities, visibility, reviews, unique_index(converted, "frame_id"))):
        raise ValueError("All32 frame tables must match frozen identities")
    missing, invalid, hashes = [], [], []
    for frame in frames:
        stem = frame["frame_id"]
        source = stage / "annotation_package/annotations" / (stem + ".json")
        annotation = json.loads(source.read_text(encoding="utf-8-sig"))
        shape_rows = {int(row["shape_index"]): row for row in outlines if row["frame_id"] == stem}
        if set(shape_rows) != set(range(len(annotation["shapes"]))):
            raise ValueError("Shape audit must retain every source shape exactly once")
        invalid_indices = []
        for index, shape in enumerate(annotation["shapes"]):
            row = shape_rows[index]
            if row["source_annotation_sha256"] != sha256(source) or row["source_shape_sha256"] != json_digest(shape):
                raise ValueError("Objective shape/source identity changed")
            fresh = shape_geometry(shape, int(frame["width"]), int(frame["height"]))
            if any(row[key] != str(value) for key, value in fresh.items()):
                raise ValueError("Objective geometry audit disagrees with the current detector")
            if fresh["geometry_status"] != "VALID_CLOSED_GEOMETRY_ONLY":
                invalid_indices.append(index)
                invalid.append({"frame_id": stem, "shape_index": index, "label": shape["label"],
                                "geometry_errors_json": fresh["geometry_errors_json"],
                                "closed_intersection_edge_pairs_json": fresh["closed_intersection_edge_pairs_json"]})
        visible, review = visibility[stem], reviews[stem]
        if any(review[code] != old_difficulty[stem][code] for code in DIFFICULTIES) or review["human_difficulty_labels"] != derive_difficulty(review):
            raise ValueError("Historical frame-specific manual difficulty decisions must remain unchanged")
        missing.append({"frame_id": stem, "source_annotation_sha256": sha256(source),
                        "existing_component_counts_json": json.dumps({label: sum(shape["label"] == label for shape in annotation["shapes"])
                                                                       for label in ("potted_plant", "plant", "pot", "blue_cube")}),
                        "pot_visibility": visible["pot_visibility"], "human_difficulty_labels": review["human_difficulty_labels"],
                        "visibility_fields_to_record": ";".join(field for field in VISIBILITY_FIELDS
                            if not visible.get(field) or visible.get(field) in ("UNVERIFIED", "HUMAN_REVIEW_REQUIRED")),
                        "shape_indices_needing_closure_confirmation": ";".join(str(index) for index, row in shape_rows.items()
                            if row.get("confirmation_status") != "CONFIRMED_CLOSED_REGION"),
                        "invalid_geometry_shape_indices": ";".join(map(str, invalid_indices)),
                        "human_review_fields_to_record": ";".join(field for field in REVIEW_FIELDS
                            if not review.get(field) or review.get(field) == "PENDING_V2"),
                        "plant_derivation_confirmation": "CONDITIONAL_ONLY_IF_HUMAN_CONFIRMS_NOT_VISIBLE_POT_AND_PLANT_ONLY"
                            if not any(shape["label"] == "plant" for shape in annotation["shapes"]) else "NOT_NEEDED_EXISTING_PLANT",
                        "status": "HUMAN_INPUT_REQUIRED_NOT_FULL_REANNOTATION"})
        hashes.append({"frame_id": stem, "source_annotation_sha256": sha256(source), "rgb_sha256": frame["rgb_sha256"],
                       "source_unchanged": True})
    backup_rows = read_csv(stage / "completion/GT32_LOCAL_BACKUP_INDEX.csv")
    if len(backup_rows) != 32 or any(sha256(ROOT / row["local_backup"]) != row["source_sha256"] for row in backup_rows):
        raise ValueError("Original byte-exact private backup is missing or changed")
    previews = read_csv(records / "OVERLAY_SOURCE_INDEX.csv")
    for row in previews:
        path = ROOT / row["output_path"]
        if sha256(path) != row["output_sha256"]:
            raise ValueError("Review overlay SHA256 mismatch")
        with Image.open(path) as image:
            if image.size != (int(row["display_width"]), int(row["display_height"])):
                raise ValueError("Overlay dimensions mismatch")
    detail_rows = read_csv(details / "INTERSECTION_DETAIL_INDEX.csv")
    if any(sha256(ROOT / row["detail_image"]) != row["detail_image_sha256"] for row in detail_rows):
        raise ValueError("Intersection detail SHA256 mismatch")
    write_csv(output / "CONVERTED_GT_MANIFEST.csv", converted)
    write_csv(output / "HUMAN_REVIEW_REQUIRED_ITEMS.csv", missing)
    write_csv(output / "GEOMETRY_EXCEPTION_AUDIT.csv", invalid, fields=["frame_id", "shape_index", "label", "geometry_errors_json", "closed_intersection_edge_pairs_json"])
    write_csv(output / "INTERSECTION_DETAIL_INDEX.csv", detail_rows)
    result = {"protocol": PROTOCOL, "baseline_commit": "6092e2ea", "frames": 32, "frozen_candidates_verified": 71,
              "source_shapes": len(outlines), "original_json_backups_verified": 32, "unchanged_original_gt": hashes,
              "protected_tracked_files_outside_v2_unchanged": True, "historical_d_decisions_preserved": 160,
              "overlays_verified": len(previews), "intersection_detail_images_verified": len(detail_rows),
              "invalid_geometry_shapes": len(invalid), "invalid_geometry_frames": len({row['frame_id'] for row in invalid}),
              "converted_gt_frames": sum(bool(row['derived_annotation_sha256']) for row in converted),
              "human_signature_or_evidence_generated": False, "accuracy_metrics_computed": False,
              "conversion_result_sha256": sha256(conversion / "CONVERSION_RESULT.json"),
              "conversion_manifest_sha256": sha256(conversion / "CONVERTED_GT_MANIFEST.csv")}
    write_json(output / "logs/INPUT_AND_PROTECTION_CHECKS.json", result)
    write_json(output / "logs/CONVERSION_RESULT.json", json.loads((conversion / "CONVERSION_RESULT.json").read_text()))
    text = ["# 最少人工补录清单（不是重新绘制32张GT）", "",
            "已接受可见区域协议：缺pot标签本身不是错误，不应画不存在的花盆。", "",
            "1. 32帧：将已有目标身份/实际pot可见性/plant可见性/blue_cube存在性填入V2表，记录实际确认人和时间。",
            "2. 75个linestrip：借助REVIEW_INDEX.html确认是否完整区域边界；共享人员/时间可按真实已完成范围录入。",
            "3. 保留现有160个D类YES/NO；补已有逐图视觉依据、独立真人复核、最终派生JSON哈希。",
            "4. 11帧没有plant标签，仅在真人确认pot不可见且整体仅植物时补等值派生授权；不要求补不可见pot。", "",
            f"额外数值几何异常：{len(invalid)}个shape/{len({row['frame_id'] for row in invalid})}帧，"
            f"{len(detail_rows)}张自交局部图。边序号/坐标可定位；未自动修复。", "",
            "自交不能靠签字把坐标变为有效polygon。请只核查以下局部异常；原JSON永久保留。",
            "若需要人为改变坐标，必须另立人工修订版本及来源记录；当前转换器不会修改顶点，也不会自动接受修订版。", "",
            "| frame | shape（0-based）| label | 交叉边 |", "| --- | --- | --- | --- |"]
    for row in invalid:
        text.append(f"| {row['frame_id']} | {row['shape_index']} | {row['label']} | {row['closed_intersection_edge_pairs_json']} |")
    text += ["", "精确逐帧缺字段见HUMAN_REVIEW_REQUIRED_ITEMS.csv；填写说明见HUMAN_INPUT_GUIDE.md。",
             "源SHA256预填不代表人类审批。无确定帧映射，不会把全部缺pot的帧推断为NOT_VISIBLE。"]
    write_text(output / "REMAINING_HUMAN_CONFIRMATIONS.md", "\n".join(text))
    return result


def main():
    parser = argparse.ArgumentParser(description="Archive actual protocol-only preparation and protection checks; no predictions")
    parser.add_argument("--stage-dir", type=Path, default=STAGE)
    parser.add_argument("--records-dir", type=Path, default=PACKAGE)
    parser.add_argument("--conversion-dir", type=Path, required=True)
    parser.add_argument("--intersection-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=PACKAGE)
    args = parser.parse_args()
    result = archive(args.stage_dir.resolve(), args.records_dir.resolve(), args.conversion_dir.resolve(),
                     args.intersection_dir.resolve(), args.output.resolve())
    print(json.dumps({key: value for key, value in result.items() if key != "unchanged_original_gt"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
