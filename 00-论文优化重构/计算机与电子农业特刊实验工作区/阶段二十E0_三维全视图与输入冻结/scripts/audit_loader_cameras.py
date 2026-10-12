from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
WORK = Path(__file__).resolve().parents[1]
GS = ROOT / "2d-gaussian-splatting-main"
sys.dont_write_bytecode = True
sys.path.insert(0, str(GS / "submodules/simple-knn"))
sys.path.insert(0, str(GS / "submodules/diff-surfel-rasterization"))
sys.path.insert(0, str(GS))


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def run():
    from scene.colmap_loader import qvec2rotmat
    from scene.dataset_readers import readColmapSceneInfo
    from utils.graphics_utils import focal2fov
    rows = []
    for sample in ("CaoMei2", "WanNianQing1", "HongZhang"):
        split = {row["loader_image_name"]: row for row in read_csv(WORK / f"reports/{sample}_SPLIT.csv")}
        build = json.loads((WORK / f"reports/{sample}_BUILD.json").read_text())
        for group in build["groups"]:
            manifest = [row for row in read_csv(WORK / f"reports/{sample}_DATASETS.csv") if row["group"] == group]
            scene = readColmapSceneInfo(manifest[0]["dataset_root"], "images_rgba", True)
            infos = scene.train_cameras + scene.test_cameras
            if {info.image_name for info in infos} != set(split) or len(infos) != len(split):
                raise ValueError("Production camera reader changed registered coverage")
            for info in infos:
                source = split[info.image_name]
                intrinsics = json.loads(source["camera_params_json"])
                rotation = qvec2rotmat(np.array(json.loads(source["qvec_json"]))).T
                translation = np.array(json.loads(source["tvec_json"]))
                max_rotation = float(np.max(np.abs(rotation - info.R)))
                max_translation = float(np.max(np.abs(translation - info.T)))
                max_fov = max(abs(focal2fov(intrinsics[0], info.width) - info.FovX), abs(focal2fov(intrinsics[1], info.height) - info.FovY))
                if max_rotation != 0 or max_translation != 0 or max_fov != 0 or info.uid != int(source["camera_id"]):
                    raise ValueError("Actual loaded camera does not equal frozen COLMAP view")
                rows.append({"sample": sample, "group": group, "registered_image_id": source["image_id"],
                             "registered_name": source["registered_name"], "loader_image_name": info.image_name,
                             "camera_id": source["camera_id"], "production_CameraInfo_uid": info.uid,
                             "uid_semantics": "CAMERA_ID_NOT_IMAGE_ID_JOIN_BY_UNIQUE_REGISTERED_NAME",
                             "split": source["split"], "rotation_max_abs_error": max_rotation,
                             "translation_max_abs_error": max_translation, "fov_max_abs_error": max_fov,
                             "decoded_full_RGB_alpha_checked_separately": "REAL_ALL_VIEW_PREFLIGHT",
                             "status": "PASS"})
                info.image.close()
    with (WORK / "reports/ACTUAL_LOADER_CAMERA_PROVENANCE.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (WORK / "logs/ACTUAL_LOADER_CAMERA_AUDIT.json").write_text(json.dumps({"status": "PASS", "loaded_camera_views": len(rows),
          "image_id_camera_id_not_conflated": True, "formal_training": False}, indent=2) + "\n")
    print("ACTUAL_CAMERA_CONTRACT", len(rows), "PASS", flush=True)


if __name__ == "__main__":
    run()
