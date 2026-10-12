from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import pycolmap
from PIL import Image, ImageDraw

from phase20e_common import DATA, ROOT, WORK, binary_mask, colmap_io, image_array, pixel_hash, relative, safe_copy, save_mask, scene_rows, sha256, view_key, write_csv, write_json


def native_camera(camera):
    return pycolmap.Camera(model=camera.model, width=camera.width, height=camera.height, params=camera.params)


def inverse_maps(source, target):
    map_x = np.empty((target.height, target.width), np.float64)
    map_y = np.empty_like(map_x)
    for start in range(0, target.height, 64):
        stop = min(start + 64, target.height)
        columns, rows = np.meshgrid(np.arange(target.width, dtype=np.float64) + 0.5,
                                   np.arange(start, stop, dtype=np.float64) + 0.5)
        rays = target.cam_from_img(np.column_stack((columns.ravel(), rows.ravel())))
        projected = source.img_from_cam(np.column_stack((rays, np.ones(len(rays))))) - 0.5
        map_x[start:stop] = projected[:, 0].reshape(stop - start, target.width)
        map_y[start:stop] = projected[:, 1].reshape(stop - start, target.width)
    valid = np.isfinite(map_x) & np.isfinite(map_y) & (map_x >= 0) & (map_y > 0) & (map_x < source.width - 1) & (map_y <= source.height - 1)
    return map_x, map_y, valid


def remap_nearest(mask, map_x, map_y, valid):
    result = np.zeros(map_x.shape, dtype=bool)
    columns = np.floor(map_x[valid] + 0.5).astype(np.int64)
    rows = np.floor(map_y[valid] + 0.5).astype(np.int64)
    result[valid] = mask[rows, columns]
    return result


def remap_bilinear(rgb, map_x, map_y, valid):
    result = np.zeros((*map_x.shape, 3), np.uint8)
    for start in range(0, map_x.shape[0], 128):
        stop = min(start + 128, map_x.shape[0])
        mask = valid[start:stop]
        xcoord, ycoord = map_x[start:stop][mask], map_y[start:stop][mask]
        xlow, ylow = np.floor(xcoord).astype(np.int64), np.floor(ycoord).astype(np.int64)
        xhigh, yhigh = np.minimum(xlow + 1, rgb.shape[1] - 1), np.minimum(ylow + 1, rgb.shape[0] - 1)
        wx, wy = (xcoord - xlow)[:, None], (ycoord - ylow)[:, None]
        colors = (rgb[ylow, xlow] * ((1 - wx) * (1 - wy)) + rgb[ylow, xhigh] * (wx * (1 - wy)) +
                  rgb[yhigh, xlow] * ((1 - wx) * wy) + rgb[yhigh, xhigh] * (wx * wy))
        result[start:stop][mask] = np.clip(np.floor(colors + 0.5), 0, 255).astype(np.uint8)
    return result


def sparse_checks(io, sample, source_camera, target_camera):
    root = ROOT / "04-COLMAP" / sample
    before = io.read_extrinsics_binary(root / "distorted/sparse/0/images.bin")
    after = io.read_extrinsics_binary(root / "sparse/0/images.bin")
    point_ids, positions, colors, errors, tracks = io.read_points3D_binary_with_tracks(root / "sparse/0/points3D.bin")
    points = dict(zip(point_ids.tolist(), positions))
    track_errors, projected_errors, rows = [], [], []
    for image_id, image in after.items():
        original = before[image_id]
        mapped = target_camera.img_from_cam(np.column_stack((source_camera.cam_from_img(original.xys), np.ones(len(original.xys)))))
        conversion_error = np.linalg.norm(mapped - image.xys, axis=1)
        track_errors.extend(conversion_error.tolist())
        valid = [index for index, point_id in enumerate(image.point3D_ids) if int(point_id) in points]
        coords = np.array([points[int(image.point3D_ids[index])] for index in valid])
        if len(coords):
            projected = target_camera.img_from_cam(coords @ io.qvec2rotmat(image.qvec).T + image.tvec)
            reprojection_error = np.linalg.norm(projected - image.xys[valid], axis=1)
            projected_errors.extend(reprojection_error.tolist())
        else:
            reprojection_error = np.array([])
        rows.append({"sample": sample, "image_id": image_id, "registered_name": image.name,
                     "observations": len(image.xys), "conversion_max_px": float(conversion_error.max()),
                     "conversion_mean_px": float(conversion_error.mean()), "real_sparse_projections": len(valid),
                     "reprojection_mean_px": float(reprojection_error.mean()) if len(valid) else ""})
    if not track_errors or max(track_errors) > 1e-6:
        raise ValueError("Sparse observations do not match camera-domain conversion")
    return {"status": "PASS", "converted_observations": len(track_errors), "conversion_max_px": max(track_errors),
            "real_sparse_projections": len(projected_errors), "reprojection_median_px": float(np.median(projected_errors)),
            "reprojection_p95_px": float(np.percentile(projected_errors, 95)), "sparse_points": len(positions)}, rows


def qa_figure(sample, row, rgb, masks, source_rgb, historical=None):
    panels = []
    for name, mask in masks.items():
        edge = cv2.dilate(mask.astype(np.uint8), np.ones((7, 7), np.uint8)) - cv2.erode(mask.astype(np.uint8), np.ones((7, 7), np.uint8))
        overlay = rgb.copy()
        overlay[edge.astype(bool)] = [255, 0, 0]
        panel = Image.fromarray(overlay).resize((360, 640))
        ImageDraw.Draw(panel).text((8, 8), f"{sample} raw {row['raw_frame']} {name}", fill="white", stroke_width=1, stroke_fill="black")
        panels.append(panel)
    canvas = Image.new("RGB", (360 * len(panels), 640))
    for index, panel in enumerate(panels):
        canvas.paste(panel, (index * 360, 0))
    folder = WORK / "ALIGNMENT_QA_FIGURES"
    folder.mkdir(parents=True, exist_ok=True)
    canvas.save(folder / f"{sample}_{row['raw_frame']}.jpg", quality=92)


def run(sample):
    rows = scene_rows(sample)
    if any(row["common_raw_domain_proven"] != "True" for row in rows):
        raise ValueError("Unproven camera/raw RGB")
    io = colmap_io()
    root = ROOT / "04-COLMAP" / sample
    source_cameras = io.read_intrinsics_binary(root / "distorted/sparse/0/cameras.bin")
    target_cameras = io.read_intrinsics_binary(root / "sparse/0/cameras.bin")
    if len(source_cameras) != 1 or len(target_cameras) != 1:
        raise ValueError("This run requires per-image map routing for multi-camera scenes")
    source, target = native_camera(next(iter(source_cameras.values()))), native_camera(next(iter(target_cameras.values())))
    options = pycolmap.UndistortCameraOptions()
    options.warp_options.direct_warp_min_scale = 0.0
    official_target = pycolmap.undistort_camera(options, source)
    if (official_target.width, official_target.height) != (target.width, target.height) or not np.allclose(official_target.params, target.params, atol=1e-12, rtol=0):
        raise ValueError("Native undistortion camera differs from frozen model")
    map_x, map_y, valid = inverse_maps(source, target)
    common = DATA / sample / "common"
    common.mkdir(parents=True, exist_ok=True)
    map_path = common / "camera_remap.npz"
    np.savez(map_path, map_x=map_x, map_y=map_y, valid=valid)
    map_hash, map_x_hash, map_y_hash = sha256(map_path), pixel_hash(map_x), pixel_hash(map_y)
    sparse, track_rows = sparse_checks(io, sample, source, target)
    write_json(WORK / f"reports/{sample}_SPARSE_PROJECTION.json", sparse)
    write_csv(WORK / f"reports/{sample}_SPARSE_PROJECTION.csv", track_rows)
    historical_complete = all(row["historical_mask_verified"] == "True" for row in rows)
    group_a = "A_historical_reconstructed_v1" if historical_complete else "A_raw_RGB_control_v1"
    groups = [group_a, "B_R0", "C_R1"]
    camera_bundle = common / "camera_bundle/sparse/0"
    for filename in ("cameras.bin", "images.bin", "points3D.bin", "points3D.ply"):
        safe_copy(root / "sparse/0" / filename, camera_bundle / filename)
    from plyfile import PlyData
    vertices = PlyData.read(camera_bundle / "points3D.ply")["vertex"]
    ply_points = np.column_stack((vertices["x"], vertices["y"], vertices["z"]))
    bin_points = io.read_points3D_binary(camera_bundle / "points3D.bin")[0]
    if ply_points.shape != bin_points.shape or not np.array_equal(ply_points, bin_points.astype(np.float32)):
        raise ValueError("Historical initialization PLY differs from common binary sparse points")
    for group in groups:
        directory = DATA / sample / group
        directory.mkdir(parents=True, exist_ok=True)
        sparse_link = directory / "sparse"
        if not sparse_link.exists():
            sparse_link.symlink_to(camera_bundle.parent, target_is_directory=True)
    dataset_rows, remap_rows, split_rows, comparisons, mask_rows = [], [], [], [], []
    for index, row in enumerate(rows):
        key = view_key(row)
        raw = image_array(ROOT / row["raw_rgb_path"])
        rgb = remap_bilinear(raw, map_x, map_y, valid)
        if index < 2:
            bitmap, output_camera = pycolmap.undistort_image(options, pycolmap.Bitmap.from_array(raw), source)
            reference = bitmap.to_array()
            errors = np.abs(reference.astype(np.int16) - rgb)
            comparisons.append({"sample": sample, "frame_id": key, "reference": "NATIVE_PYCOLMAP_DIRECT_BILINEAR",
                                "mean_abs_color_error": float(errors.mean()), "max_abs_color_error": int(errors.max()),
                                "different_pixel_fraction": float(np.any(errors, axis=2).mean())})
            if errors.max() > 1 or errors.mean() > 0.1:
                raise ValueError(f"Native RGB undistortion differs {key}: {errors.max()}/{errors.mean()}")
        common_image = common / "rgb" / f"{Path(row['registered_name']).stem}.png"
        common_image.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgb).save(common_image, compress_level=1)
        common_image_hash = sha256(common_image)
        common_pixels_hash = pixel_hash(rgb)
        source_masks = {"B_R0": binary_mask(DATA / sample / "R0" / f"mask_{key}.png"),
                        "C_R1": binary_mask(DATA / sample / "R1" / f"mask_{key}.png")}
        masks = {name: remap_nearest(mask, map_x, map_y, valid) for name, mask in source_masks.items()}
        if index == 0:
            nearest_options = pycolmap.UndistortCameraOptions()
            nearest_options.warp_options.direct_warp_min_scale = 0.0
            nearest_options.warp_options.interpolation = type(nearest_options.warp_options.interpolation).NEAREST_NEIGHBOR
            native_mask, _ = pycolmap.undistort_image(nearest_options,
                              pycolmap.Bitmap.from_array(source_masks["C_R1"].astype(np.uint8) * 255), source)
            if not np.array_equal(masks["C_R1"], native_mask.to_array().astype(bool) & valid):
                raise ValueError("Native COLMAP nearest-neighbor mask disagrees with custom map")
        masks[group_a] = remap_nearest(binary_mask(ROOT / row["historical_mask_path"]), map_x, map_y, valid) if historical_complete else valid.copy()
        for group in groups:
            directory = DATA / sample / group
            filename = f"{Path(row['registered_name']).stem}.png"
            mask = masks[group]
            image_path, alpha_path = directory / "images_rgba" / filename, directory / "masks" / f"mask_{Path(filename).stem}.png"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            save_mask(alpha_path, mask)
            composite = rgb.copy()
            composite[~mask] = 0
            if group == "C_R1" and np.array_equal(masks["B_R0"], mask):
                base_path = DATA / sample / "B_R0/images_rgba" / filename
                if image_path.exists():
                    if sha256(image_path) != sha256(base_path):
                        raise ValueError("Existing identical-treatment image differs")
                else:
                    os.link(base_path, image_path)
            else:
                Image.fromarray(np.dstack((composite, mask.astype(np.uint8) * 255))).save(image_path, compress_level=1)
            dataset_rows.append({"sample": sample, "group": group, "frame_id": key, "registered_image_id": row["registered_image_id"],
                                 "registered_name": row["registered_name"], "camera_id": row["camera_id"], "split": row["split"],
                                 "dataset_root": str(directory), "loader_image_path": str(image_path), "image_sha256": sha256(image_path),
                                 "common_rgb_path": str(common_image), "common_rgb_sha256": common_image_hash,
                                 "common_rgb_pixels_sha256": common_pixels_hash, "source_rgb_sha256": row["raw_rgb_sha256"],
                                 "mask_path": str(alpha_path), "mask_sha256": sha256(alpha_path), "alpha_pixel_sha256": pixel_hash(mask.astype(np.uint8) * 255),
                                 "width": target.width, "height": target.height, "train_width": round(target.width / 2), "train_height": round(target.height / 2),
                                 "rgba_semantics": "EXPLICIT_BLACK_RGB_COMPOSITE_ALPHA_STORED_ONLY", "sparse_bundle": str(camera_bundle),
                                 "legacy_A_status": "RECONSTRUCTED_METHOD_NOT_EXACT_OLD_TRAINING_VERSION" if historical_complete else "NEW_RAW_RGB_CONTROL_NOT_HISTORICAL_FORESPLAT"})
        changed = masks["B_R0"] != masks["C_R1"]
        mask_rows.append({"sample": sample, "frame_id": key, "split": row["split"], "pixelwise_identity": not changed.any(),
                          "changed_pixel_ratio": float(changed.mean()), "B_pixels_sha256": pixel_hash(masks["B_R0"].astype(np.uint8) * 255),
                          "C_pixels_sha256": pixel_hash(masks["C_R1"].astype(np.uint8) * 255)})
        if index < 2 or changed.any() and sum(item["pixelwise_identity"] is False for item in mask_rows) <= 4:
            qa_figure(sample, row, rgb, masks, raw)
        legacy_path = ROOT / row["historical_undistorted_path"]
        if not row["historical_undistorted_error"]:
            historical = image_array(legacy_path)
            difference = np.abs(historical.astype(np.int16) - rgb)
            comparisons.append({"sample": sample, "frame_id": key, "reference": "EXISTING_HISTORICAL_IMAGES_NOT_ASSUMED_RAW",
                                "mean_abs_color_error": float(difference.mean()), "max_abs_color_error": int(difference.max()),
                                "different_pixel_fraction": float(np.any(difference, axis=2).mean())})
        remap_rows.append({"sample": sample, "frame_id": key, "image_id": row["registered_image_id"], "camera_id": row["camera_id"],
                           "source_rgb": row["raw_rgb_path"], "source_sha256": row["raw_rgb_sha256"], "target_rgb": str(common_image),
                           "target_sha256": common_image_hash, "map_path": str(map_path), "map_sha256": map_hash,
                           "map_x_pixels_sha256": map_x_hash, "map_y_pixels_sha256": map_y_hash,
                           "valid_pixels": int(valid.sum()), "valid_ratio": float(valid.mean()), "invalid_mask_pixels": 0,
                           "half_pixel_convention": "TARGET_xy+0.5_SOURCE_projected_xy-0.5", "RGB_interpolation": "COLMAP_DIRECT_BILINEAR",
                           "mask_interpolation": "NEAREST_NEIGHBOR", "width": target.width, "height": target.height})
        split_rows.append({"sample": sample, "image_id": row["registered_image_id"], "registered_name": row["registered_name"],
                           "loader_image_name": Path(row["registered_name"]).stem, "camera_id": row["camera_id"], "split": row["split"],
                           "split_order_index": row["split_order_index"], "source_rgb_sha256": row["raw_rgb_sha256"],
                           "common_rgb_sha256": common_image_hash, "camera_params_json": row["camera_params_json"],
                           "qvec_json": row["qvec_json"], "tvec_json": row["tvec_json"]})
        if index % 20 == 0:
            print("BUILD", sample, index + 1, "/", len(rows), "mapped switch", changed.any(), flush=True)
    write_csv(WORK / f"reports/{sample}_DATASETS.csv", dataset_rows)
    write_csv(WORK / f"reports/{sample}_REMAP.csv", remap_rows)
    write_csv(WORK / f"reports/{sample}_SPLIT.csv", split_rows)
    write_csv(WORK / f"reports/{sample}_RGB_COMPARISONS.csv", comparisons)
    write_csv(WORK / f"reports/{sample}_MAPPED_MASK_DIFFERENCES.csv", mask_rows)
    write_csv(common / "view_manifest/identity.csv", rows)
    write_csv(common / "split_manifest/split.csv", split_rows)
    write_json(WORK / f"reports/{sample}_BUILD.json", {"status": "PASS", "views": len(rows), "groups": groups,
                "historical_A_masks_complete": historical_complete, "map_sha256": map_hash, "valid_ratio": float(valid.mean()),
                "native_reference_views": 2, "native_nearest_mask_verified": True, "sparse_checks": sparse, "formal_training": False})
    print("BUILD_DONE", sample, groups, len(rows), "PASS", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", required=True)
    run(parser.parse_args().sample)
