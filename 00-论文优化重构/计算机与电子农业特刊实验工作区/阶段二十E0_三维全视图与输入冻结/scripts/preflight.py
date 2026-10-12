from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[4]
WORK = Path(__file__).resolve().parents[1]
GS = ROOT / "2d-gaussian-splatting-main"
DATA = Path("/data/fj/F2DMAS_phase20e0_data/f53bf8f3_v1")
sys.dont_write_bytecode = True
sys.path.insert(0, str(GS / "submodules/simple-knn"))
sys.path.insert(0, str(GS / "submodules/diff-surfel-rasterization"))
sys.path.insert(0, str(GS))


def pixel_hash(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def frozen_config(source, output):
    from arguments import ModelParams, PipelineParams, OptimizationParams
    parser = argparse.ArgumentParser()
    model, pipeline, optimization = ModelParams(parser), PipelineParams(parser), OptimizationParams(parser)
    params = parser.parse_args([])
    params.source_path = str(source)
    params.model_path = str(output)
    params.images = "images_rgba"
    params.resolution = 2
    params.white_background = False
    params.eval = True
    params.data_device = "cpu"
    params.iterations = 30000
    if params.mask_mode != "none" or params.init_pcd_mode != "none" or params.use_mask_loss or params.use_foreground_rgb_loss or params.use_bg_opacity_loss:
        raise ValueError("Unintended plant-aware training mechanism")
    return model.extract(params), optimization.extract(params), pipeline.extract(params), vars(params)


def run(sample):
    import torch
    import simple_knn._C
    import diff_surfel_rasterization
    from scene.dataset_readers import readColmapSceneInfo
    from scene.gaussian_model import GaussianModel
    from utils.camera_utils import loadCam
    from gaussian_renderer import render
    started = time.monotonic()
    build = json.loads((WORK / f"reports/{sample}_BUILD.json").read_text())
    expected = read_csv(WORK / f"reports/{sample}_SPLIT.csv")
    manifest = read_csv(WORK / f"reports/{sample}_DATASETS.csv")
    if not torch.cuda.is_available():
        raise ValueError("CUDA not available")
    torch.set_num_threads(4)
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(0)
    reports = []
    rgb_identity, camera_identity, config_identity, initial_identity = {}, {}, {}, {}
    for group in build["groups"]:
        source = DATA / sample / group
        dataset, optimization, pipeline, config = frozen_config(source, DATA / "phase20e1_outputs" / sample / group)
        write_json(WORK / f"reports/{sample}_{group}_TRAIN_CONFIG.json", {"all_parameters": config,
                    "model": vars(dataset), "optimization": vars(optimization), "pipeline": vars(pipeline), "seed": 0,
                    "CUDA_VISIBLE_DEVICES": "1", "training_resolution_policy": "EXPLICIT_PRODUCTION_resolution_2_PIL_resize",
                    "background": [0, 0, 0], "mask_effect": "BLACK_RGB_PRECOMPOSITION_ONLY",
                    "additional_mask_filtering_of_inherited_sparse_points": False})
        comparable = {key: value for key, value in config.items() if key not in ("source_path", "model_path")}
        if config_identity and config_identity != comparable:
            raise ValueError("Group optimization/preprocessing drift")
        config_identity = comparable
        scene = readColmapSceneInfo(str(source), dataset.images, True)
        views = scene.train_cameras + scene.test_cameras
        actual_train = {view.image_name for view in scene.train_cameras}
        actual_validation = {view.image_name for view in scene.test_cameras}
        if actual_train != {row["loader_image_name"] for row in expected if row["split"] == "train"} or actual_validation != {row["loader_image_name"] for row in expected if row["split"] == "validation"}:
            raise ValueError("Production loader split or view coverage differs")
        if scene.point_cloud is None or not np.isfinite(scene.point_cloud.points).all() or not len(scene.point_cloud.points):
            raise ValueError("Invalid sparse points")
        expected_by_name = {Path(row["registered_name"]).stem: row for row in manifest if row["group"] == group}
        first_camera = None
        consumed_train = hashlib.sha256()
        for index, info in enumerate(views):
            row = expected_by_name[info.image_name]
            if info.image.size != (int(row["width"]), int(row["height"])) or (info.width, info.height) != info.image.size:
                raise ValueError("Silent image/camera sizing mismatch")
            rgba = np.array(info.image.convert("RGBA"))
            common = np.array(Image.open(row["common_rgb_path"]).convert("RGB"))
            alpha = np.array(Image.open(row["mask_path"]).convert("L"))
            reference = common.copy()
            reference[alpha == 0] = 0
            if not np.array_equal(rgba[..., :3], reference) or not np.array_equal(rgba[..., 3], alpha):
                raise ValueError("Actual RGBA pixel semantics mismatch")
            key = info.image_name
            camera_hash = pixel_hash(np.concatenate((info.R.ravel(), info.T, [info.FovX, info.FovY])))
            if key in camera_identity and camera_identity[key] != camera_hash:
                raise ValueError("Group camera changed")
            camera_identity[key] = camera_hash
            common_hash = pixel_hash(common)
            if key in rgb_identity and rgb_identity[key] != common_hash:
                raise ValueError("Group base RGB changed")
            rgb_identity[key] = common_hash
            camera = loadCam(dataset, index, info, 1.0)
            if (camera.image_width, camera.image_height) != (int(row["train_width"]), int(row["train_height"])):
                raise ValueError("Undeclared resolution scaling")
            expected_rgb = torch.from_numpy(np.stack([np.array(Image.fromarray(rgba[..., channel]).resize((camera.image_width, camera.image_height))) for channel in range(3)])) / 255.0
            expected_alpha = torch.from_numpy(np.array(Image.fromarray(alpha).resize((camera.image_width, camera.image_height))))[None] / 255.0
            if not torch.equal(camera.original_image.cpu(), expected_rgb) or not torch.equal(camera.gt_alpha_mask.cpu(), expected_alpha):
                raise ValueError("Production RGB/alpha resize diverges")
            if info.image_name in actual_train:
                consumed_train.update(info.image_name.encode())
                consumed_train.update(camera.original_image.cpu().numpy().tobytes())
            if index == 0:
                first_camera = camera
            info.image.close()
            del common, reference, rgba, alpha, expected_rgb, expected_alpha
            if index % 40 == 0:
                print("LOAD", sample, group, index + 1, "/", len(views), flush=True)
        torch.manual_seed(0)
        gaussians = GaussianModel(dataset.sh_degree)
        gaussians.create_from_pcd(scene.point_cloud, scene.nerf_normalization["radius"])
        current_init = {name: pixel_hash(value.detach().cpu().numpy()) for name, value in
                        [("xyz", gaussians._xyz), ("scale", gaussians._scaling), ("rotation", gaussians._rotation), ("opacity", gaussians._opacity)]}
        if initial_identity and initial_identity != current_init:
            raise ValueError("Initial Gaussians differ between groups")
        initial_identity = current_init
        with torch.no_grad():
            result = render(first_camera, gaussians, pipeline, torch.zeros(3, dtype=torch.float32, device="cuda"))
            for field in ("render", "rend_alpha"):
                if not torch.isfinite(result[field]).all():
                    raise ValueError("Nonfinite actual forward render")
            if tuple(result["render"].shape) != (3, first_camera.image_height, first_camera.image_width):
                raise ValueError("Forward render shape mismatch")
        if gaussians.optimizer is not None:
            raise ValueError("Optimizer must never be created in Phase20E0")
        forward = (result["render"].detach().cpu().clamp(0, 1).numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        figure_path = WORK / "ALIGNMENT_QA_FIGURES" / f"{sample}_{group}_NO_OPTIMIZER_FORWARD.jpg"
        Image.fromarray(forward).resize((360, 640)).save(figure_path, quality=90)
        reports.append({"group": group, "status": "PASS", "train_views": len(actual_train), "validation_views": len(actual_validation),
                        "loaded_views": len(views), "initial_sparse_points": len(scene.point_cloud.points), "initialization_sha256": current_init,
                        "one_real_forward": True, "optimizer_created": False, "optimizer_steps": 0, "checkpoint_created": False,
                        "forward_render_sha256": pixel_hash(forward), "gpu_peak_bytes": torch.cuda.max_memory_allocated()})
        reports[-1]["actual_consumed_train_rgb_sha256"] = consumed_train.hexdigest()
        del gaussians, scene, result, first_camera, camera
        torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info()
    write_json(WORK / f"reports/{sample}_PREFLIGHT.json", {"status": "PASS", "groups": reports, "python": sys.version,
                "torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""), "rasterizer_path": diff_surfel_rasterization._C.__file__,
                "knn_path": simple_knn._C.__file__, "gpu_free_bytes": free, "gpu_total_bytes": total,
                "wall_seconds": time.monotonic() - started, "formal_training": False})
    print("PREFLIGHT", sample, "PASS", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", required=True)
    run(parser.parse_args().sample)
