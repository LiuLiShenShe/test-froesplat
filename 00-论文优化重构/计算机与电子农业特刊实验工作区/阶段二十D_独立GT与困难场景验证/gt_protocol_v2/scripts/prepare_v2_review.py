from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from protocol_v2 import (CONFIRM_FIELDS, DERIVATION_RULE, DIFFICULTIES, PACKAGE, PROTOCOL, ROOT, STAGE,
                         derive_difficulty, json_digest, read_csv, shape_geometry, sha256,
                         unique_index, validate_source, verify_original_sources)
from phase20d_common import relative, write_csv, write_json, write_text


PALETTE = {"potted_plant": (255, 210, 20), "plant": (50, 230, 100),
           "pot": (255, 135, 60), "blue_cube": (80, 200, 255)}


def dashed_line(draw, first, last):
    distance = math.hypot(last[0] - first[0], last[1] - first[1])
    steps = max(1, math.ceil(distance / 10))
    for step in range(0, steps, 2):
        start, stop = step / steps, min((step + 1) / steps, 1)
        points = [(first[0] + (last[0] - first[0]) * value,
                   first[1] + (last[1] - first[1]) * value) for value in (start, stop)]
        draw.line(points, fill=(255, 30, 235), width=4)


def render_overlay(rgb, shapes, records, frame_id, destination, preview_width):
    scale = min(1, preview_width / rgb.width)
    preview = rgb.resize((round(rgb.width * scale), round(rgb.height * scale)), Image.Resampling.LANCZOS)
    header = 132
    canvas = Image.new("RGB", (preview.width, preview.height + header), (16, 18, 24))
    canvas.paste(preview, (0, header))
    draw = ImageDraw.Draw(canvas)
    font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    font = ImageFont.truetype(str(font_path), 17) if font_path.is_file() else ImageFont.load_default()
    lines = [f"{frame_id} | ORIGINAL HUMAN OUTLINES | HUMAN REVIEW REQUIRED",
             "Colored paths=source vertices; green circle=start; red circle=end",
             "Magenta dashed edge=proposed closure ONLY, not GT; no filled region",
             f"Source grid {rgb.width}x{rgb.height}; display scale {scale:.6f}; shape indices are zero-based"]
    for line_index, line in enumerate(lines):
        draw.text((10, 8 + line_index * 27), line, fill=(245, 245, 245), font=font)
    for shape, record in zip(shapes, records):
        try:
            points = [(float(point[0]) * scale, float(point[1]) * scale + header) for point in shape["points"]]
        except (KeyError, ValueError, TypeError):
            continue
        if len(points) < 2 or not all(math.isfinite(value) for point in points for value in point):
            continue
        color = PALETTE.get(shape.get("label"), (255, 255, 255))
        draw.line(points, fill=color, width=3)
        dashed_line(draw, points[-1], points[0])
        for point, endpoint_color in ((points[0], (10, 255, 40)), (points[-1], (255, 35, 35))):
            draw.ellipse((point[0] - 5, point[1] - 5, point[0] + 5, point[1] + 5), fill=endpoint_color)
        caption = f"#{record['shape_index']} {shape['label']} gap={record['endpoint_distance_px']:.2f}px"
        location = (min(max(points[0][0] + 9, 4), preview.width - 300), max(points[0][1] - 22, header))
        draw.text(location, caption, fill=color, font=font, stroke_width=1, stroke_fill=(0, 0, 0))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as handle:
        canvas.save(handle, format="JPEG", quality=92, subsampling=0)
    return {"display_width": canvas.width, "display_height": canvas.height, "rgb_display_scale": scale,
            "header_height_px": header, "output_sha256": sha256(destination)}


def prepare(stage, output, preview_width):
    frames = read_csv(stage / "TEST32_INPUT_MANIFEST.csv")
    verify_original_sources(stage, frames)
    originals = unique_index(read_csv(stage / "completion/GT32_SOURCE_ARCHIVE.csv"), "frame_id")
    old_review = unique_index(read_csv(stage / "TEST32_ANNOTATION_REVIEW.csv"), "frame_id")
    old_difficulty = unique_index(read_csv(stage / "DIFFICULTY_HUMAN_VERIFICATION.csv"), "frame_id")
    if len(frames) != 32 or set(originals) != {frame["frame_id"] for frame in frames}:
        raise ValueError("Preserve all32 frozen frames and source annotation versions")
    for path in ("GT_VISIBILITY_MANIFEST.csv", "LINESTRIP_CONVERSION_AUDIT.csv", "HUMAN_REVIEW_V2.csv", "SOURCE_IDENTITIES.csv", "OVERLAY_SOURCE_INDEX.csv", "REVIEW_INDEX.html"):
        if (output / path).exists():
            raise FileExistsError(f"Refusing to overwrite existing audit/review material: {path}")
    visibility, outlines, reviews, identities, overlay_index, gallery = [], [], [], [], [], []
    for frame in frames:
        stem = frame["frame_id"]
        source = ROOT / originals[stem]["source_json"]
        source_hash = originals[stem]["source_sha256_before"]
        annotation = json.loads(source.read_text(encoding="utf-8-sig"))
        validate_source(annotation, frame, source, source_hash)
        review, difficulty = old_review[stem], old_difficulty[stem]
        labels = derive_difficulty(difficulty)
        present = {shape.get("label") for shape in annotation["shapes"]}
        shared = {"protocol": PROTOCOL, "challenge_id": frame["challenge_id"], "frame_id": stem,
                  "source_annotation_json": relative(source), "source_annotation_sha256": source_hash,
                  "rgb_path": frame["rgb_path"], "rgb_sha256": frame["rgb_sha256"],
                  "width": frame["width"], "height": frame["height"]}
        identities.append(shared)
        visibility.append({**shared, "sample": frame["sample"],
                           **{f"{label}_shape_count": sum(shape.get("label") == label for shape in annotation["shapes"])
                              for label in sorted(PALETTE)}, "pot_visibility": "UNVERIFIED", "plant_visible": "",
                           "blue_cube_present": review.get("blue_cube_present", ""),
                           "visibility_confirmed_by": "", "visibility_confirmed_is_human": "", "visibility_confirmed_at": "",
                           "visibility_confirmed_source": "", "visibility_status": "HUMAN_REVIEW_REQUIRED",
                           "potted_only_plant_confirmed": "", "plant_equal_potted_derivation_confirmed": "",
                           "plant_derivation_confirmed_by": "", "plant_derivation_confirmed_is_human": "",
                           "plant_derivation_confirmed_at": "", "plant_derivation_confirmed_source": "",
                           "plant_derivation_rule": "", "expected_plant_derivation_rule": DERIVATION_RULE if "plant" not in present else "NOT_NEEDED",
                           "missing_pot_interpretation": "VISIBILITY_UNVERIFIED_NOT_AUTOMATIC_MISSING_ANNOTATION",
                           "user_protocol_explanation": "Some pots genuinely not visible; frame identities not supplied"})
        review_fields = ("annotator", "reviewer", "second_reviewer", "annotator_is_human", "reviewer_is_human",
                         "second_reviewer_is_human", "manual_annotation_confirmed", "target_identity_description",
                         "target_identity_confirmed", "difficulty_manually_confirmed", "difficulty_evidence_json",
                         "reviewed_at", "second_reviewed_at", "blue_cube_present")
        reviews.append({**shared, "annotation_sha256": "", "human_difficulty_labels": labels,
                        **{code: difficulty[code] for code in DIFFICULTIES},
                        **{field: review.get(field, "") for field in review_fields},
                        "visible_components_confirmed": "", "annotation_origin": "", "approval": "PENDING_V2",
                        "existing_review_approval_not_v2_approval": review.get("approval", ""),
                        "batch_record_id": "", "public_reviewer_codes_confirmed": ""})
        frame_shapes = []
        for shape_index, shape in enumerate(annotation["shapes"]):
            record = {**shared, "shape_index": shape_index, "label": shape.get("label", ""),
                      "source_shape_type": shape.get("shape_type", "polygon"), "source_shape_sha256": json_digest(shape),
                      "source_points_sha256": json_digest(shape.get("points", [])),
                      **shape_geometry(shape, int(frame["width"]), int(frame["height"])),
                      **{field: "" for field in CONFIRM_FIELDS}, "human_review_status": "HUMAN_REVIEW_REQUIRED",
                      "conversion_rule": "IMPLICIT_LAST_TO_FIRST_EDGE_NO_VERTEX_CHANGE",
                      "derived_annotation_sha256": "", "conversion_status": "NOT_CONVERTED"}
            frame_shapes.append(record)
            outlines.append(record)
        rgb_path = ROOT / frame["rgb_path"]
        with Image.open(rgb_path) as handle:
            rgb = handle.convert("RGB")
        previews = [("FRAME", annotation["shapes"], frame_shapes, f"{stem}_all.jpg")]
        previews += [(str(record["shape_index"]), [shape], [record], f"{stem}_shape_{record['shape_index']:02d}.jpg")
                     for shape, record in zip(annotation["shapes"], frame_shapes)]
        paths = []
        for shape_id, shapes, records, name in previews:
            destination = output / "review_overlays" / name
            dimensions = render_overlay(rgb, shapes, records, stem, destination, preview_width)
            overlay_index.append({**shared, "shape_index_or_frame": shape_id, "output_path": relative(destination),
                                  "purpose": "HUMAN_REVIEW_ONLY_NO_APPROVAL_NO_GT", **dimensions})
            paths.append(name)
        gallery.append((stem, paths))
        if sha256(source) != source_hash or sha256(rgb_path) != frame["rgb_sha256"]:
            raise ValueError("Source JSON/RGB changed during display-only rendering")
    write_csv(output / "SOURCE_IDENTITIES.csv", identities)
    write_csv(output / "GT_VISIBILITY_MANIFEST.csv", visibility)
    write_csv(output / "LINESTRIP_CONVERSION_AUDIT.csv", outlines)
    write_csv(output / "HUMAN_REVIEW_V2.csv", reviews)
    write_csv(output / "OVERLAY_SOURCE_INDEX.csv", overlay_index)
    batch = {"batch_record_id": "", "scope_source_identities_sha256": sha256(output / "SOURCE_IDENTITIES.csv"),
             "actual_human_review_completed": "", "human_record_source": "", "scope_frame_ids": [frame["frame_id"] for frame in frames],
             "scope_final_derived_sha256_by_frame": {},
             "annotator": "", "reviewer": "", "second_reviewer": "", "annotator_is_human": "",
             "reviewer_is_human": "", "second_reviewer_is_human": "", "reviewed_at": "", "second_reviewed_at": "",
             "public_reviewer_codes_confirmed": "", "approval": "PENDING_V2"}
    write_json(output / "HUMAN_BATCH_RECORD.json", batch)
    blocks = ["<!doctype html><meta charset='utf-8'><title>TEST32 V2 outline review</title>",
              "<h1>Human review only — no converted GT</h1>",
              "<p>Source RGB/open paths; dashed magenta is a proposed closure, not approved GT. Shape indices are zero-based.</p>"]
    for stem, paths in gallery:
        blocks.append(f"<h2>{html.escape(stem)}</h2>")
        for name in paths:
            blocks.append(f"<a href='review_overlays/{html.escape(name)}'><img width='216' src='review_overlays/{html.escape(name)}' alt='{html.escape(name)}'></a>")
    write_text(output / "REVIEW_INDEX.html", "\n".join(blocks))
    summary = {"protocol": PROTOCOL, "frames": len(frames), "source_shapes": len(outlines), "review_overlays": len(overlay_index),
               "source_json_modified": False, "pot_visibility_inferred_from_missing_label": False,
               "closed_geometry_invalid_shapes": sum(row["geometry_status"] == "INVALID" for row in outlines),
               "self_intersecting_closed_shapes": sum(row["closed_self_intersection_count"] != "" and row["closed_self_intersection_count"] > 0 for row in outlines),
               "complete_region_boundary_human_confirmed": 0, "pot_visibility_verified_frames": 0,
               "converted_gt_created": 0, "accuracy_metrics_computed": False,
               "source_identities_sha256": sha256(output / "SOURCE_IDENTITIES.csv"), "preview_width": preview_width}
    write_json(output / "logs/PREPARATION_RESULT.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Read-only TEST32 V2 audit and actual-RGB human review previews")
    parser.add_argument("--stage-dir", type=Path, default=STAGE)
    parser.add_argument("--output-dir", type=Path, default=PACKAGE)
    parser.add_argument("--preview-width", type=int, default=864)
    args = parser.parse_args()
    if args.preview_width < 400:
        raise ValueError("Preview too small for human outline review")
    prepare(args.stage_dir.resolve(), args.output_dir.resolve(), args.preview_width)


if __name__ == "__main__":
    main()
