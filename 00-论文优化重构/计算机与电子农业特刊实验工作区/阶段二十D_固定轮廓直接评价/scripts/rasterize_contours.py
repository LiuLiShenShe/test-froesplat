from __future__ import annotations

import argparse
import copy
import json
from collections import defaultdict

import cv2
import numpy as np
import PIL
from PIL import Image, ImageDraw

from direct_common import OLD, WORK, frames, local_path, pixel_hash, protocol_module, read_csv, relative, sha256, write_csv, write_json

ROUNDING = "float64_half_away_from_zero_sign(x)*floor(abs(x)+0.5)_int32_no_clipping"
RASTERIZERS = {"opencv": f"cv2.fillPoly/{cv2.__version__}/LINE_8/shift=0",
               "pillow": f"Pillow.ImageDraw.polygon/{PIL.__version__}/L/fill=255/outline=None/width=1"}


def noncollinear(points):
    distinct = np.unique(points, axis=0)
    if len(distinct) < 3:
        return False
    origin, direction = distinct[0], distinct[1] - distinct[0]
    displacement = distinct - origin
    return bool(np.any(direction[0] * displacement[:, 1] - direction[1] * displacement[:, 0] != 0))


def round_vertices(points):
    try:
        coordinates = np.asarray(points, dtype=np.float64)
    except (ValueError, TypeError) as error:
        raise ValueError("Non-numeric contour vertices") from error
    if coordinates.ndim != 2 or coordinates.shape[1] != 2 or len(coordinates) < 3:
        raise ValueError("At least three2D vertices required")
    if not np.isfinite(coordinates).all() or not noncollinear(coordinates):
        raise ValueError("Nonfinite, insufficient or collinear contour vertices")
    rounded = np.sign(coordinates) * np.floor(np.abs(coordinates) + 0.5)
    bounds = np.iinfo(np.int32)
    if rounded.min() < bounds.min or rounded.max() > bounds.max:
        raise ValueError("Vertices cannot be represented in int32")
    if not noncollinear(rounded):
        raise ValueError("Contour degenerates after fixed rounding")
    return rounded.astype(np.int32)


def rasterize(annotation, rasterizer, width=2160, height=3840):
    if (annotation.get("imageWidth"), annotation.get("imageHeight")) != (width, height):
        raise ValueError("Original JSON grid mismatch; resampling forbidden")
    if rasterizer not in RASTERIZERS:
        raise ValueError("Unknown fixed rasterizer")
    canvases = {label: np.zeros((height, width), dtype=np.uint8) for label in ("potted_plant", "blue_cube")}
    counts = defaultdict(int)
    mappings = []
    for index, shape in enumerate(annotation.get("shapes", [])):
        label = shape.get("label")
        if label not in canvases:
            continue
        if shape.get("shape_type") not in ("linestrip", "polygon"):
            raise ValueError(f"Unsupported target shape_type at shape{index}")
        vertices = round_vertices(shape.get("points", []))
        if rasterizer == "opencv":
            cv2.fillPoly(canvases[label], [vertices], 255, lineType=cv2.LINE_8, shift=0)
        else:
            image = Image.fromarray(canvases[label])
            ImageDraw.Draw(image).polygon([tuple(int(value) for value in point) for point in vertices], fill=255, outline=None, width=1)
            canvases[label] = np.asarray(image).copy()
        counts[label] += 1
        mappings.append({"shape_index": index, "label": label, "source_shape_type": shape["shape_type"],
                         "original_points": copy.deepcopy(shape["points"]), "rounded_points": vertices.tolist(),
                         "implicit_closure": True, "vertex_count": len(vertices)})
    if not counts["potted_plant"]:
        raise ValueError("No original potted_plant shape")
    target = (canvases["potted_plant"] > 0) & ~(canvases["blue_cube"] > 0)
    if not target.any():
        raise ValueError("Filled target region is empty; no surrogateGT produced")
    return target, mappings, counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default=str(WORK))
    args = parser.parse_args()
    from pathlib import Path
    output = Path(args.output).resolve()
    if (output / "RASTERIZED_MASK_MANIFEST.csv").exists():
        raise FileExistsError("Refusing to overwrite rasterization evidence")
    protocol = protocol_module()
    source_archive = {row["frame_id"]: row for row in read_csv(OLD / "completion/GT32_SOURCE_ARCHIVE.csv")}
    historical_geometry = {(row["frame_id"], int(row["shape_index"])): row for row in read_csv(OLD / "gt_protocol_v2/LINESTRIP_CONVERSION_AUDIT.csv")}
    protocol.verify_original_sources(OLD, frames())
    manifests, geometry_rows, sensitivity = [], [], []
    for frame in frames():
        stem = frame["frame_id"]
        source = local_path(source_archive[stem]["source_json"])
        digest = sha256(source)
        rgb = local_path(frame["rgb_path"])
        if digest != source_archive[stem]["source_sha256_before"] or sha256(rgb) != frame["rgb_sha256"]:
            raise ValueError(f"Immutable source mismatch: {stem}")
        annotation = json.loads(source.read_text())
        if (annotation["imageWidth"], annotation["imageHeight"]) != (2160, 3840):
            raise ValueError(f"Invalid original dimensions: {stem}")
        affected, target_affected = [], []
        for index, shape in enumerate(annotation["shapes"]):
            result = protocol.shape_geometry(shape, 2160, 3840)
            old_result = historical_geometry[(stem, index)]
            if protocol.json_digest(shape) != old_result["source_shape_sha256"]:
                raise ValueError("Historical geometry source mismatch")
            warning = bool(result["closed_self_intersection_count"])
            if warning:
                affected.append(index)
                if shape["label"] in ("potted_plant", "blue_cube"):
                    target_affected.append(index)
            geometry_rows.append({"frame_id": stem, "source_json_sha256": digest, "shape_index": index,
                                  "label": shape["label"], "source_shape_type": shape["shape_type"],
                                  "source_shape_sha256": protocol.json_digest(shape),
                                  "source_points_sha256": protocol.json_digest(shape["points"]),
                                  "main_gt_contributor": int(shape["label"] in ("potted_plant", "blue_cube")),
                                  "geometry_warning": "GEOMETRY_AMBIGUITY" if warning else "NUMERIC_GEOMETRY_ONLY",
                                  **result})
        warning = "GEOMETRY_AMBIGUITY" if affected else "NUMERIC_GEOMETRY_ONLY"
        masks = {}
        for rasterizer, name in RASTERIZERS.items():
            target, mappings, counts = rasterize(annotation, rasterizer)
            pixels = target.astype(np.uint8) * 255
            path = output / "derived_masks" / rasterizer / f"mask_{stem}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                raise FileExistsError(path)
            Image.fromarray(pixels).save(path)
            mapping_path = output / "contour_mappings" / rasterizer / f"{stem}.json"
            write_json(mapping_path, {"frame_id": stem, "source_json": relative(source), "source_json_sha256": digest,
                                     "rgb_sha256": frame["rgb_sha256"], "rasterizer": name, "rounding_rule": ROUNDING,
                                     "geometry_warning": warning, "all_source_shapes": annotation["shapes"],
                                     "evaluation_shapes": mappings, "mask_path": relative(path), "mask_sha256": sha256(path),
                                     "protocol": "fixed-contour-v1", "not_original_gate_certified_gt": True})
            manifests.append({"frame_id": stem, "sample": frame["sample"], "rasterizer": rasterizer,
                              "rasterizer_name_version": name, "coordinate_rounding_rule": ROUNDING,
                              "source_json": relative(source), "source_json_sha256": digest, "rgb_path": relative(rgb),
                              "rgb_sha256": frame["rgb_sha256"], "width": 2160, "height": 3840,
                              "shape_count": len(annotation["shapes"]), "potted_plant_shape_count": counts["potted_plant"],
                              "blue_cube_shape_count": counts["blue_cube"], "filled_area_px": int(target.sum()),
                              "mask_path": relative(path), "mask_sha256": sha256(path), "mask_pixels_sha256": pixel_hash(pixels),
                              "mapping_path": relative(mapping_path), "mapping_sha256": sha256(mapping_path),
                              "contour_geometry_warning": warning, "ambiguous_shape_indices": json.dumps(affected),
                              "ambiguous_main_gt_shape_indices": json.dumps(target_affected), "status": "VALID_RASTERIZED_REGION"})
            masks[rasterizer] = target
        intersection = int((masks["opencv"] & masks["pillow"]).sum())
        union = int((masks["opencv"] | masks["pillow"]).sum())
        area0, area1 = (int(masks[key].sum()) for key in ("opencv", "pillow"))
        sensitivity.append({"frame_id": stem, "sample": frame["sample"], "geometry_warning": warning,
                            "main_gt_shape_ambiguous": int(bool(target_affected)), "raster_mask_iou": intersection / union,
                            "opencv_area_px": area0, "pillow_area_px": area1, "area_difference_px": area1 - area0,
                            "relative_area_difference": (area1 - area0) / area0,
                            "symmetric_difference_px": union - intersection,
                            "noticeable_raster_disagreement": int(intersection / union < 0.99 or abs(area1 - area0) / area0 > 0.01)})
        if sha256(source) != digest:
            raise ValueError("Original JSON changed during read-only rasterization")
        print(f"rasterized {stem}: area={area0}, cross-rasterIoU={intersection / union:.9f}, {warning}", flush=True)
    write_csv(output / "RASTERIZED_MASK_MANIFEST.csv", manifests)
    write_csv(output / "CONTOUR_GEOMETRY_AUDIT.csv", geometry_rows)
    write_csv(output / "RASTERIZATION_SENSITIVITY.csv", sensitivity)
    write_json(output / "logs/RASTERIZATION_STATUS.json", {"primary_masks": 32, "secondary_masks": 32,
               "geometry_ambiguity_frames": sum(row["geometry_warning"] == "GEOMETRY_AMBIGUITY" for row in sensitivity),
               "main_gt_geometry_ambiguity_frames": sum(row["main_gt_shape_ambiguous"] for row in sensitivity),
               "original_shapes": len(geometry_rows), "original_gate_not_executed": True})


if __name__ == "__main__":
    main()
