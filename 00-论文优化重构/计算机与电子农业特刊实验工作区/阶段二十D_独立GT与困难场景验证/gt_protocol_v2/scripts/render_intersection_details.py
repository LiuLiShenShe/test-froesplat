from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from protocol_v2 import PACKAGE, ROOT, read_csv, sha256, shape_geometry
from phase20d_common import relative, write_csv, write_text


def render(records_dir, output):
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a fresh intersection-detail directory; no existing review image is overwritten")
    output.mkdir(parents=True, exist_ok=True)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 17)
    index = []
    for row in read_csv(records_dir / "LINESTRIP_CONVERSION_AUDIT.csv"):
        source = ROOT / row["source_annotation_json"]
        rgb = ROOT / row["rgb_path"]
        if sha256(source) != row["source_annotation_sha256"] or sha256(rgb) != row["rgb_sha256"]:
            raise ValueError("Review source SHA256 mismatch")
        annotation = json.loads(source.read_text(encoding="utf-8-sig"))
        shape = annotation["shapes"][int(row["shape_index"])]
        geometry = shape_geometry(shape, int(row["width"]), int(row["height"]))
        if geometry["closed_intersection_edge_pairs_json"] != row["closed_intersection_edge_pairs_json"]:
            raise ValueError("Stored geometry audit differs from current detector; create a new preparation run")
        pairs = json.loads(geometry["closed_intersection_edge_pairs_json"])
        if not pairs:
            continue
        points = shape["points"]
        count = len(points) - 1 if points[0] == points[-1] else len(points)
        with Image.open(rgb) as original:
            original = original.convert("RGB")
            for pair_index, (first_edge, second_edge) in enumerate(pairs):
                selected = [points[first_edge], points[(first_edge + 1) % count],
                            points[second_edge], points[(second_edge + 1) % count]]
                left = max(0, math.floor(min(point[0] for point in selected)) - 48)
                top = max(0, math.floor(min(point[1] for point in selected)) - 48)
                right = min(original.width, math.ceil(max(point[0] for point in selected)) + 49)
                bottom = min(original.height, math.ceil(max(point[1] for point in selected)) + 49)
                crop = original.crop((left, top, right, bottom))
                scale = min(4.0, 864 / crop.width, 864 / crop.height)
                size = (max(1, round(crop.width * scale)), max(1, round(crop.height * scale)))
                preview = crop.resize(size, Image.Resampling.NEAREST)
                canvas = Image.new("RGB", (max(864, size[0]), size[1] + 120), "white")
                canvas.paste(preview, (0, 120))
                draw = ImageDraw.Draw(canvas)
                draw.text((12, 8), f"{row['frame_id']} shape {row['shape_index']} {row['label']}", fill="black", font=font)
                draw.text((12, 34), f"Source edge indices {first_edge} (cyan), {second_edge} (yellow); HUMAN REVIEW REQUIRED", fill="black", font=font)
                draw.text((12, 60), f"RGB crop [{left},{top},{right},{bottom}]; display scale {scale:.3f}; original points unchanged", fill="black", font=font)
                draw.text((12, 86), "Numeric crossing diagnostic only: no contour correction, GT fill or human approval", fill="black", font=font)
                for edge_index, color in ((first_edge, "cyan"), (second_edge, "yellow")):
                    endpoints = [points[edge_index], points[(edge_index + 1) % count]]
                    display = [((point[0] - left) * scale, (point[1] - top) * scale + 120) for point in endpoints]
                    draw.line(display, fill=color, width=3)
                    for point in display:
                        draw.ellipse((point[0] - 4, point[1] - 4, point[0] + 4, point[1] + 4), outline="red", width=2)
                destination = output / f"{row['frame_id']}__shape_{row['shape_index']}__pair_{pair_index}.png"
                with destination.open("xb") as stream:
                    canvas.save(stream, format="PNG")
                index.append({"frame_id": row["frame_id"], "shape_index": row["shape_index"], "label": row["label"],
                              "source_annotation_sha256": row["source_annotation_sha256"], "rgb_sha256": row["rgb_sha256"],
                              "intersection_edges_json": json.dumps([first_edge, second_edge]),
                              "rgb_crop_bounds_json": json.dumps([left, top, right, bottom]), "display_scale": scale,
                              "detail_image": relative(destination), "detail_image_sha256": sha256(destination),
                              "human_confirmation_status": "HUMAN_REVIEW_REQUIRED"})
    write_csv(output / "INTERSECTION_DETAIL_INDEX.csv", index)
    sections = ["<!doctype html><meta charset='utf-8'><title>V2 intersection details</title>",
                "<h1>Geometry diagnostics — actual human review required</h1>",
                "<p>Real RGB crops; no automatic contour correction or approval. Indices are zero-based source edges.</p>"]
    for row in index:
        name = Path(row["detail_image"]).name
        sections.append(f"<h2>{html.escape(row['frame_id'])} shape {row['shape_index']} edges {row['intersection_edges_json']}</h2>"
                        f"<a href='{html.escape(name)}'><img src='{html.escape(name)}' style='max-width:95%'></a>")
    write_text(output / "INTERSECTION_REVIEW.html", "\n".join(sections))
    return index


def main():
    parser = argparse.ArgumentParser(description="Real RGB self-crossing crops for humans; no annotation mutation")
    parser.add_argument("--records-dir", type=Path, default=PACKAGE)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    index = render(args.records_dir.resolve(), args.output.resolve())
    print(json.dumps({"intersection_detail_images": len(index), "original_gt_modified": False, "human_approval_generated": False}))


if __name__ == "__main__":
    main()
