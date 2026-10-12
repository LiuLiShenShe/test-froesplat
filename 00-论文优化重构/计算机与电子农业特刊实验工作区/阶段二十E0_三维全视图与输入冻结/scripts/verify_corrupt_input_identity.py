from __future__ import annotations

import time

import cv2
import numpy as np
import pycolmap
from scipy.spatial import cKDTree

from phase20e_common import DATA, ROOT, WORK, colmap_io, image_array, read_csv, sha256, write_json


def run():
    photometric = []
    rows = [row for row in read_csv(WORK / "FULL_VIEW_IDENTITY_MANIFEST.csv") if row["colmap_input_error"]]
    for row in rows:
        historical = image_array(ROOT / row["historical_undistorted_path"])
        current = image_array(DATA / row["sample"] / "common/rgb" / f"{row['registered_name'].split('.')[0]}.png")
        interior = cv2.erode(np.any(historical > 10, axis=2).astype(np.uint8), np.ones((11, 11), np.uint8)).astype(bool)
        delta = np.abs(current.astype(np.int16) - historical)[interior]
        photometric.append({"sample": row["sample"], "registered_image_id": row["registered_image_id"],
                            "registered_name": row["registered_name"], "raw_frame": row["raw_frame"],
                            "foreground_interior_pixels": int(interior.sum()), "mean_abs_color_error": float(delta.mean()),
                            "p95_abs_color_error": float(np.percentile(delta, 95)),
                            "fraction_channel_error_le_5": float((delta <= 5).mean()),
                            "method": "Historical RGB nonblack interior; QA only, no foreground-mask inference"})
    write_json(WORK / "reports/CORRUPT_REGISTERED_INPUT_RGB_PHOTOMETRIC_PROOF.json", photometric)
    io = colmap_io()
    images = io.read_extrinsics_binary(ROOT / "04-COLMAP/WanNianQing1/distorted/sparse/0/images.bin")
    views = {image.name: image for image in images.values()}
    target = views["0136.jpg"]
    options = pycolmap.FeatureExtractionOptions()
    options.use_gpu = False
    options.num_threads = 4
    sift = pycolmap.Sift(options, pycolmap.Device.cpu)
    results = []
    for frame in ("0136", "0135", "0137", "0138"):
        path = ROOT / f"01-FFmepg/WanNianQing1/{frame}.jpg"
        gray = image_array(path, "L")
        started = time.monotonic()
        keys, descriptors = sift.extract(gray)
        distances = cKDTree(keys[:, :2]).query(target.xys)[0]
        tracked = distances[target.point3D_ids >= 0]
        own_view = views.get(f"{frame}.jpg")
        own = cKDTree(keys[:, :2]).query(own_view.xys)[0] if own_view else np.array([])
        results.append({"source_raw_frame": frame, "raw_sha256": sha256(path), "target_registered_image_id": target.id,
                        "target_registered_name": target.name, "target_observations": len(target.xys), "extracted_keypoints": len(keys),
                        "target_nn_median_px": float(np.median(distances)), "target_fraction_le_0_5px": float((distances <= .5).mean()),
                        "target_fraction_le_1px": float((distances <= 1).mean()),
                        "tracked_sparse_observations": len(tracked),
                        "tracked_nn_median_px": float(np.median(tracked)),
                        "tracked_fraction_le_1px": float((tracked <= 1).mean()),
                        "own_image_median_px": float(np.median(own)) if len(own) else None,
                        "own_fraction_le_1px": float((own <= 1).mean()) if len(own) else None,
                        "seconds": time.monotonic() - started})
        print("SIFT_IDENTITY", frame, results[-1]["target_nn_median_px"], results[-1]["target_fraction_le_1px"], flush=True)
    positive, negatives = results[0], results[1:]
    verified = positive["target_fraction_le_1px"] > 0.5 and positive["target_nn_median_px"] < 0.5 and positive["tracked_fraction_le_1px"] > 0.5 and all(
        row["target_fraction_le_1px"] < 0.05 and row["target_nn_median_px"] > 5 for row in negatives)
    write_json(WORK / "reports/WanNianQing1_0136_SIFT_KEYPOINT_IDENTITY.json", {"status": "PASS" if verified else "FAIL",
               "method": "Actual installed native CPU SIFT against immutable original distorted COLMAP observations; adjacent wrong frames are negative controls",
               "options": options.summary(), "pycolmap_version": pycolmap.__version__, "results": results,
               "original_camera_id": target.camera_id, "original_image_id": target.id,
               "original_images_bin_sha256": sha256(ROOT / "04-COLMAP/WanNianQing1/distorted/sparse/0/images.bin"),
               "raw_RGB_camera_identity_verified": verified, "historical_undistorted_RGB_same_version": False,
               "original_descriptors_unavailable": True, "formal_training": False, "new_camera_registration": False})
    if not verified:
        raise ValueError("Registered raw RGB identity remains unresolved; scene must not train")
    print("CORRUPT_INPUT_IDENTITY", len(rows), "PASS; old 0136 RGB conflict retained", flush=True)


if __name__ == "__main__":
    run()
