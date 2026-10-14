from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import time
from argparse import Namespace
from pathlib import Path

from common import E0, GS, MATRIX, ROOT, STAGE, config_for, import_paths, now, read_csv, sha256, write_csv, write_json


def resize_reference(path, width, height):
    import numpy as np
    import torch
    from PIL import Image
    with Image.open(path) as image:
        array = np.array(image.convert("RGB"))
    channels = [np.array(Image.fromarray(array[..., channel]).resize((width, height)))
                for channel in range(3)]
    return torch.from_numpy(np.stack(channels)).float() / 255.0


def proxy_metrics(rgb, reference, alpha, mask):
    import cv2
    import numpy as np
    import torch
    foreground = mask > 0.5
    background = ~foreground
    prediction = alpha > 0.5
    fg_mse = ((rgb - reference) ** 2)[:, foreground].mean().item() if foreground.any() else None
    union = torch.logical_or(prediction, foreground).sum().item()
    intersection = torch.logical_and(prediction, foreground).sum().item()
    outside = torch.logical_and(prediction, background).cpu().numpy().astype(np.uint8)
    component_count, _, stats, _ = cv2.connectedComponentsWithStats(outside, connectivity=8)
    counts = int(np.sum(stats[1:, cv2.CC_STAT_AREA] >= 16)) if component_count > 1 else 0
    return {"R0_proxy_foreground_psnr": -10 * math.log10(fg_mse) if fg_mse and fg_mse > 0 else None,
            "R0_proxy_background_rgb_energy": rgb[:, background].pow(2).mean().item() if background.any() else None,
            "R0_proxy_background_alpha": alpha[background].mean().item() if background.any() else None,
            "R0_proxy_silhouette_iou": intersection / union if union else 1.0,
            "R0_proxy_missing_alpha_fraction": (~prediction)[foreground].float().mean().item() if foreground.any() else None,
            "R0_proxy_outside_components_ge16px": counts,
            "R0_proxy_area_px": foreground.sum().item()}


def tensor_sha(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", required=True)
    parser.add_argument("--group", required=True)
    args = parser.parse_args()
    assert (args.sample, args.group) in MATRIX
    import_paths()
    import torch
    import numpy as np
    import lpips
    from PIL import Image
    from plyfile import PlyData
    from scene import Scene, GaussianModel
    from gaussian_renderer import render
    from utils.image_utils import psnr
    from utils.loss_utils import ssim

    started = time.perf_counter()
    run_id = f"{args.sample}_{args.group}"
    config = config_for(args.sample, args.group)
    output = Path(config["model"]["model_path"])
    runtime = json.loads((STAGE / f"reports/{run_id}_RUNTIME.json").read_text())
    assert runtime["exit_code"] == 0 and runtime["observed_iteration"] == 30000
    final_ply = output / "point_cloud/iteration_30000/point_cloud.ply"
    checkpoint = output / "chkpnt30000.pth"
    vertex = PlyData.read(final_ply)["vertex"].data
    assert len(vertex) > 0
    for field in vertex.dtype.names:
        assert np.isfinite(vertex[field]).all(), f"Nonfinite PLY {field}"
    model_args, iteration = torch.load(checkpoint, map_location="cuda", weights_only=False)
    assert iteration == 30000
    for tensor in model_args[1:10]:
        assert torch.isfinite(tensor).all()
    gaussians = GaussianModel(config["model"]["sh_degree"])
    scene = Scene(Namespace(**config["model"]), gaussians, load_iteration=30000, shuffle=False)
    tensor_fields = ("_xyz", "_features_dc", "_features_rest", "_scaling", "_rotation", "_opacity")
    checkpoint_hashes = {}
    for index, field in enumerate(tensor_fields, start=1):
        actual = getattr(gaussians, field)
        assert torch.equal(actual, model_args[index]), f"Checkpoint/PLY mismatch: {field}"
        checkpoint_hashes[field] = tensor_sha(actual)
    for tensor in (gaussians.get_xyz, gaussians.get_scaling, gaussians.get_rotation,
                   gaussians.get_opacity, gaussians.get_features):
        assert torch.isfinite(tensor).all()
    assert (gaussians.get_scaling > 0).all()
    assert (gaussians.get_opacity >= 0).all() and (gaussians.get_opacity <= 1).all()
    assert (torch.linalg.vector_norm(gaussians._rotation, dim=1) > 0).all()
    restored = GaussianModel(config["model"]["sh_degree"])
    restored.restore(model_args, Namespace(**config["optimization"]))
    pipeline = Namespace(**config["pipeline"])
    background_color = torch.zeros(3, dtype=torch.float32, device="cuda")
    reference_rows = [row for row in read_csv(E0 / "reports/EVALUATION_REFERENCE_MANIFEST.csv")
                      if row["sample"] == args.sample]
    camera_map = {camera.image_name: camera for camera in scene.getTestCameras()}
    assert set(camera_map) == {row["loader_image_name"] for row in reference_rows}
    actual_train_names = {camera.image_name for camera in scene.getTrainCameras()}
    assert not actual_train_names & set(camera_map)
    with torch.no_grad():
        camera = camera_map[reference_rows[0]["loader_image_name"]]
        ply_image = render(camera, gaussians, pipeline, background_color)["render"]
        checkpoint_image = render(camera, restored, pipeline, background_color)["render"]
        reload_max_error = (ply_image - checkpoint_image).abs().max().item()
        assert reload_max_error == 0, f"Reload rendering mismatch {reload_max_error}"
    del restored, model_args, ply_image, checkpoint_image
    gc.collect()
    torch.cuda.empty_cache()
    perceptual = lpips.LPIPS(net="vgg", version="0.1", verbose=False).eval().cuda()
    lpips_weight_path = Path(lpips.__file__).parent / "weights/v0.1/vgg.pth"
    vgg_weight_path = Path(torch.hub.get_dir()) / "checkpoints/vgg16-397923af.pth"
    metric_versions = {name: importlib.metadata.version(name) for name in
                       ("torch", "torchvision", "lpips", "Pillow", "numpy", "plyfile")}
    metric_versions.update({"psnr_source_sha256": sha256(GS / "utils/image_utils.py"),
                            "ssim_source_sha256": sha256(GS / "utils/loss_utils.py"),
                            "lpips_weights_sha256": sha256(lpips_weight_path),
                            "vgg16_weights_sha256": sha256(vgg_weight_path),
                            "evaluator_sha256": sha256(Path(__file__)),
                            "lpips_input_range": "[-1,1] via normalize=True", "lpips_network": "vgg",
                            "reference_policy": "same canonical RAW RGB, independent of treatment",
                            "resize_policy": "production channelwise PIL default BICUBIC resolution=2"})
    render_dir = output / "heldout_shared_raw"
    assert not render_dir.exists(), "Preserve previous evaluation artifacts"
    render_dir.mkdir()
    datasets = [row for row in read_csv(E0 / "MATCHED_2DGS_DATASET_MANIFEST.csv")
                if row["sample"] == args.sample and row["group"] == "B_R0"]
    dataset_map = {Path(row["registered_name"]).stem: row for row in datasets}
    results = []
    failures = []
    renders = []
    with torch.no_grad():
        for row in reference_rows:
            name = row["loader_image_name"]
            camera = camera_map[name]
            entry = {"sample": args.sample, "group": args.group, "registered_image_id": row["image_id"],
                     "registered_name": row["registered_name"], "loader_image_name": name,
                     "reference_original_sha256": row["common_rgb_sha256"], "independent_geometry_GT": "NONE",
                     "reference_path": row["reference_rgb_path"], "render_width": camera.image_width,
                     "render_height": camera.image_height, "status": "PASS", "shared_model_alias": ""}
            try:
                assert sha256(row["reference_rgb_path"]) == row["common_rgb_sha256"]
                reference_cpu = resize_reference(row["reference_rgb_path"], camera.image_width, camera.image_height)
                entry["reference_resized_tensor_sha256"] = tensor_sha(reference_cpu)
                reference = reference_cpu.cuda()
                torch.cuda.synchronize()
                render_start = time.perf_counter()
                package = render(camera, gaussians, pipeline, background_color)
                torch.cuda.synchronize()
                entry["render_seconds"] = time.perf_counter() - render_start
                raw_rgb, alpha = package["render"], package["rend_alpha"][0]
                assert torch.isfinite(raw_rgb).all() and torch.isfinite(alpha).all()
                entry["unclamped_rgb_min"] = raw_rgb.min().item()
                entry["unclamped_rgb_max"] = raw_rgb.max().item()
                rgb = raw_rgb.clamp(0, 1)
                entry["empty_render"] = bool(alpha.max().item() <= 1e-6)
                entry["PSNR"] = psnr(rgb[None], reference[None]).mean().item()
                entry["SSIM"] = ssim(rgb, reference).item()
                entry["LPIPS"] = perceptual(rgb[None], reference[None], normalize=True).item()
                assert all(math.isfinite(entry[field]) for field in ("PSNR", "SSIM", "LPIPS"))
                width, height = camera.image_width, camera.image_height
                crop = (slice(height // 4, 3 * height // 4), slice(width // 4, 3 * width // 4))
                rgb_crop, reference_crop = rgb[:, crop[0], crop[1]], reference[:, crop[0], crop[1]]
                entry["fixed_center_crop_PSNR"] = psnr(rgb_crop[None], reference_crop[None]).mean().item()
                entry["fixed_center_crop_SSIM"] = ssim(rgb_crop, reference_crop).item()
                entry["fixed_center_crop_LPIPS"] = perceptual(rgb_crop[None], reference_crop[None], normalize=True).item()
                with Image.open(dataset_map[name]["mask_path"]) as image:
                    mask = torch.from_numpy(np.array(image.resize((width, height), Image.Resampling.NEAREST))).cuda().float() / 255
                entry.update(proxy_metrics(rgb, reference, alpha, mask))
                entry["R0_proxy_mask_sha256"] = dataset_map[name]["mask_sha256"]
                entry["independent_manual_contour_available"] = False
                rgb_path = render_dir / f"{name}_RGB.png"
                alpha_path = render_dir / f"{name}_ALPHA.png"
                raw_path = render_dir / f"{name}_RAW_REFERENCE.png"
                Image.fromarray((rgb.permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)).save(rgb_path)
                Image.fromarray((alpha.clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8)).save(alpha_path)
                Image.fromarray((reference_cpu.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)).save(raw_path)
                tensor_path = render_dir / f"{name}_FLOAT_RENDER.pt"
                torch.save({"rgb": rgb.cpu(), "alpha": alpha.cpu()}, tensor_path)
                entry.update({"render_rgb_path": str(rgb_path), "render_rgb_sha256": sha256(rgb_path),
                              "render_alpha_path": str(alpha_path), "render_alpha_sha256": sha256(alpha_path),
                              "float_render_path": str(tensor_path), "float_render_sha256": sha256(tensor_path),
                              "reference_display_path": str(raw_path)})
                renders.append(entry["render_seconds"])
                del package, raw_rgb, rgb, alpha, reference, reference_cpu, rgb_crop, reference_crop, mask
                print(f"EVALUATED {run_id} {name} PSNR={entry['PSNR']:.6f} SSIM={entry['SSIM']:.6f} LPIPS={entry['LPIPS']:.6f}", flush=True)
            except Exception as error:
                entry["status"] = "FAILED"
                entry["error"] = repr(error)
                failures.append({"registered_name": row["registered_name"], "error": repr(error)})
            results.append(entry)
    fields = list(dict.fromkeys(key for entry in results for key in entry))
    write_csv(STAGE / f"metrics/{run_id}_HELDOUT.csv", results, fields)
    validation = {"sample": args.sample, "group": args.group, "iteration": iteration,
                  "Gaussian_count": len(vertex), "all_PLY_parameters_finite": True,
                  "all_activated_parameters_finite": True, "checkpoint_restore": "PASS",
                  "positive_scales_and_rotation_norms": True, "opacity_bounds_valid": True,
                  "checkpoint_PLY_tensor_identity": "PASS", "reload_render_max_abs_error": reload_max_error,
                  "checkpoint_path": str(checkpoint), "checkpoint_bytes": checkpoint.stat().st_size,
                  "checkpoint_sha256": sha256(checkpoint), "final_PLY_path": str(final_ply),
                  "final_PLY_bytes": final_ply.stat().st_size, "final_PLY_sha256": sha256(final_ply),
                  "parameter_tensor_sha256": checkpoint_hashes, "expected_heldout": len(reference_rows),
                  "successful_heldout": len(results) - len(failures), "failed_heldout": failures,
                  "empty_renders": sum(row.get("empty_render", False) for row in results),
                  "mean_render_seconds": float(np.mean(renders)) if renders else None,
                  "evaluation_wall_seconds": time.perf_counter() - started,
                  "metric_versions": metric_versions, "timestamp": now(),
                  "status": "PASS" if not failures else "PARTIAL"}
    opacity = gaussians.get_opacity.detach().cpu().numpy().ravel()
    scales = gaussians.get_scaling.detach().cpu().numpy().ravel()
    sparse = PlyData.read(output / "input.ply")["vertex"]
    centroid = np.stack([sparse[field] for field in ("x", "y", "z")], axis=1).mean(axis=0)
    distances = np.linalg.norm(gaussians.get_xyz.detach().cpu().numpy() - centroid, axis=1)
    validation["Gaussian_distribution_descriptive"] = {
        field: dict(zip(("p01", "p50", "p95", "p99"), np.quantile(values, [0.01, 0.5, 0.95, 0.99]).tolist()))
        for field, values in (("opacity", opacity), ("scale", scales), ("distance_from_sparse_centroid", distances))}
    write_json(STAGE / f"reports/{run_id}_VALIDATION.json", validation)
    print(json.dumps(validation, indent=2, ensure_ascii=False))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
