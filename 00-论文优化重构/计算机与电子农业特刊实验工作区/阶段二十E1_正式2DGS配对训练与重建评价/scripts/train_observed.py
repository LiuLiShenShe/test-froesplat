from __future__ import annotations

import argparse
import csv
import hashlib
import json
import runpy
import sys
import time
import traceback
from pathlib import Path

from common import E0, MATRIX, STAGE, config_for, import_paths, now, read_csv, sha256, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", required=True)
    parser.add_argument("--group", required=True)
    args = parser.parse_args()
    assert (args.sample, args.group) in MATRIX
    assert json.loads((STAGE / "reports/PRETRAINING_INPUT_RECHECK.json").read_text())["status"] == "PASS"
    config = config_for(args.sample, args.group)
    run_id = f"{args.sample}_{args.group}"
    output = Path(config["model"]["model_path"])
    assert not output.exists(), "Never overwrite an earlier attempt"
    import_paths()
    import torch
    import train
    from plyfile import PlyData
    import numpy as np
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    if not hasattr(FigureCanvasAgg, "tostring_rgb"):
        def tostring_rgb(canvas):
            return np.asarray(canvas.buffer_rgba())[..., :3].tobytes()
        FigureCanvasAgg.tostring_rgb = tostring_rgb
    original_scene = train.Scene
    original_report = train.training_report
    state = {"iteration": 0, "optimizer_steps": 0}
    runtime = {"sample": args.sample, "group": args.group, "start_time": now(),
               "config_sha256": sha256(E0 / f"reports/{run_id}_TRAIN_CONFIG.json"),
               "output_path": str(output), "seed": 0, "iterations": 30000,
               "physical_gpu": 1, "native_final_optimizer_update": 29999,
               "logging_only_compatibility": "Matplotlib removed Agg.tostring_rgb; exact RGB bytes from buffer_rgba; no core-source/package changes",
               "observer_source_sha256": sha256(Path(__file__))}
    log_path = STAGE / f"metrics/{run_id}_TRAINING_CURVE.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    curve = log_path.open("x", newline="")
    writer = csv.writer(curve)
    writer.writerow(["iteration", "l1", "rgb_loss", "gaussians", "native_iteration_ms", "wall_seconds", "cuda_peak_allocated_bytes", "cuda_peak_reserved_bytes"])
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()

    def observed_scene(dataset, gaussians, *positional, **keywords):
        scene = original_scene(dataset, gaussians, *positional, **keywords)
        splits = [row for row in read_csv(E0 / "TRAIN_VALIDATION_SPLIT.csv") if row["sample"] == args.sample]
        for split, cameras in (("train", scene.getTrainCameras()), ("validation", scene.getTestCameras())):
            expected = {Path(row["registered_name"]).stem for row in splits if row["split"] == split}
            assert {camera.image_name for camera in cameras} == expected, f"Loaded {split} changed"
        initial = output / "input.ply"
        inherited = Path(config["model"]["source_path"]) / "sparse/0/points3D.ply"
        assert sha256(initial) == sha256(inherited)
        vertex = PlyData.read(inherited)["vertex"]
        expected_xyz = np.stack([vertex[field] for field in ("x", "y", "z")], axis=1)
        assert np.array_equal(gaussians.get_xyz.detach().cpu().numpy(), expected_xyz)
        frozen = json.loads((E0 / f"reports/{args.sample}_PREFLIGHT.json").read_text())
        frozen_group = next(row for row in frozen["groups"] if row["group"] == args.group)
        for field, tensor in (("xyz", gaussians.get_xyz), ("scale", gaussians._scaling),
                              ("rotation", gaussians._rotation), ("opacity", gaussians._opacity)):
            digest = hashlib.sha256(tensor.detach().cpu().numpy().tobytes()).hexdigest()
            assert digest == frozen_group["initialization_sha256"][field], f"Initialization {field} changed"
        runtime.update({"train_views": len(scene.getTrainCameras()), "validation_views": len(scene.getTestCameras()),
                        "initial_gaussians": len(expected_xyz), "initial_ply_sha256": sha256(inherited),
                        "loaded_split_verified_before_optimizer": True})
        return scene

    def observed_report(tb_writer, iteration, l1, loss, loss_function, elapsed, testing_iterations, scene, render_function, render_args):
        state.update({"iteration": iteration, "scene": scene})
        writer.writerow([iteration, l1.item(), loss.item(), len(scene.gaussians.get_xyz), elapsed,
                         time.perf_counter() - started, torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()])
        if iteration % 100 == 0:
            curve.flush()
        return original_report(tb_writer, iteration, l1, loss, loss_function, elapsed, testing_iterations,
                               scene, render_function, render_args)

    train.Scene = observed_scene
    train.training_report = observed_report
    failure = None
    try:
        sys.argv = [str(E0 / "scripts/run_phase20e1.py"), "--sample", args.sample, "--group", args.group,
                    "--execute-optimization"]
        runpy.run_path(sys.argv[0], run_name="__main__")
        torch.cuda.synchronize()
        assert state["iteration"] == 30000
        runtime["training_wall_seconds"] = time.perf_counter() - started
        model = state["scene"].gaussians
        torch.save((model.capture(), state["iteration"]), output / "chkpnt30000.pth")
        runtime["final_gaussians"] = len(model.get_xyz)
        runtime["status"] = "TRAINING_COMPLETED_PENDING_RELOAD_VALIDATION"
        write_json(output / "FULL_FROZEN_TRAIN_CONFIG.json", config)
    except BaseException as error:
        failure = error
        runtime.update({"status": "FAILED", "exception": repr(error)})
        traceback.print_exc()
    finally:
        curve.close()
        runtime.update({"end_time": now(), "observed_iteration": state["iteration"],
                        "process_wall_seconds": time.perf_counter() - started,
                        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                        "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                        "exit_code": 0 if failure is None else 1,
                        "curve_sha256": sha256(log_path)})
        write_json(STAGE / f"reports/{run_id}_RUNTIME.json", runtime)
        print(json.dumps(runtime, indent=2, ensure_ascii=False))
    if failure is not None:
        raise failure


if __name__ == "__main__":
    main()
