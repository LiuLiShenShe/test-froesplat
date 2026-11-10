# Native QGS interactive local viewer

The local viewer now supplies an HTTP UI rendering the **actual native QGS quadratic primitives**. Mouse drag or yaw/pitch controls rotate the camera; wheel/slider zoom changes camera distance. RGB, pure-gray primitive rendering, alpha, camera-normal and camera-Z depth modes call the original quadratic CUDA renderer. No conversion to ordinary Gaussians or center points is involved.

Launch from E13_QGS after a real checkpoint exists:

```bash
uv run --no-project --python .venv/bin/python python scripts/native_qgs_viewer.py --iteration 30000 --port 8767 --width 540
```

Open `http://127.0.0.1:8767` on the same host. Native PLY path is displayed in the UI. `--iteration 7000` or `15000` opens the actual saved partial model and labels it partial. The server binds localhost by default; `--host` controls the listening address. Physical GPU1 is selected explicitly. Background defaults to gray RGB(.25,.25,.25) so black or dark peripheral primitives remain visible; --background controls display compositing. Start it when the experiment GPU queue is finished.

The optional 180° flip is a **view-only rotation about the original world X axis at the shared plant center**, applied as an inverse camera transformation. The original PLY geometry is unchanged. Displayed normals are normalized native camera directions; any nonfinite visible direction is colored magenta and counted in /api/state instead of silently converted. The raw native model is retained unfiltered, and its nonfinite primitive count is displayed in state. Depth is camera Z in the uncalibrated standard-VGGT gauge. RGB SH appearance is native QGS and no EO correction is applied.

The actual native30K HTTP smoke passed; its RGB/alpha/normal/depth/gray/orbit/zoom/180° responses are recorded in native_viewer/smoke/complete.json. The smoke service was stopped before E14. The actual532,735 primitive PLY is /home/test/foresplat_model_files/KongQueZhuYu/E13_QGS/Mip/checkpoints/point_cloud/iteration_30000/point_cloud.ply. Its3,456 nonfinite native primitives are retained and displayed in state; successful loading does not certify geometric quality.


This repository exposes quadratic native rendering, rather than a released interactive SuperSplat viewer. The native PLY stores six signed scale parameters. Standard Gaussian viewers cannot reproduce its curved primitive geometry. A center XYZRGB point cloud shows center positions only, and must not be presented as a QGS surface.

The experiment supplies `scripts/view_native_qgs.py` using the official `gaussian_renderer.render` and official `utils.render_utils.generate_path`. It loads the actual native checkpoint PLY, applies the frozen camera adapter, and exports normal/RGB/alpha PNG and original float depth NPY. The original coordinate gauge is preserved; all transformation during orbit is a camera view transformation.

From this E13_QGS directory, after actual checkpoint creation:

```bash
uv run --no-project --python .venv/bin/python python scripts/view_native_qgs.py --iteration 30000
```

To inspect a partial checkpoint, use its **actual** saved iteration, such as 7000 or 15000. Partial checkpoints remain labelled partial and are not compared as 30K.

```bash
uv run --no-project --python .venv/bin/python python scripts/view_native_qgs.py --iteration 15000 --ids 0,105
```

For native free-camera views:

```bash
uv run --no-project --python .venv/bin/python python scripts/view_native_qgs.py --iteration 30000 --orbit-frames 24
```

Outputs are saved under `native_viewer/iteration_<actual iteration>/`. Open generated RGB/alpha/normal PNGs with an image viewer. Gray Mesh, vertex colors and wireframe use the actual native_TSDF raw/post PLY in Blender/MeshLab/Open3D.

## Why a small local launch wrapper is needed

Official [render.py](https://github.com/will-zzy/QGS/blob/74d05c945e99fcaef7afe5a8831903be71ad9b55/render.py) obtains a `load_model_path` configuration key absent from official base.yaml, then reloads model/config.yaml. Its skip flags are parsed but behavior is driven by configuration values. Its GaussianExtractor export_image currently indexes saved 2D depth as though it had a channel dimension (`depthmaps[idx].permute(1,2,0)`), which fails. The local launch wrapper calls the same native renderer directly and saves each frame without these export errors. It does not approximate quadratic primitives or change the model.

Sources: pinned official render.py/gaussian_renderer/utils and local adapter implementation. Bocha search returnedHTTP403; the response does not establish the reason. No training-data claims used.


## Archived base models and current Mip paths

The default iteration selector opens the current Mip output tree. An archived base file must be selected explicitly with its own actual configuration. These are retained partial/failed states, excluded from the primary30K comparison. The archived7K file exists:

```bash
uv run --no-project --python .venv/bin/python python scripts/native_qgs_viewer.py --native-ply "/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/E13_QGS/runs/base_expandable_oom_13301/checkpoints/point_cloud/iteration_7000/point_cloud.ply" --config "/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/E13_QGS/runs/base_expandable_oom_13301/actual_config.yaml" --iteration 7000 --model-stage archived_base_partial --port 8767 --width 540
```

The failed13301 model is listed in the actual artifact index. Its capture metadata iteration13300 describes the preceding step at a backward failure, not formal30K completion. Select its exact PLY and the archived base config with `--model-stage failed_native_state --iteration 13300`. The new explicit-file interface preserves native parameters and displays the selected stage; model-backed rendering of the archive will be checked after the formal GPU queue. The default current30K terminal viewer smoke remains an independent required stage.

Current Mip Checkpoint paths resolve to `/home/test/foresplat_model_files/KongQueZhuYu/E13_QGS/Mip/checkpoints/`; no archived base model is moved into that tree.

The failed13301 PTH is unreadable in both Python ZIP and the native PyTorch archive reader. It is retained as an explicitly unusable failed export, not a resume checkpoint. Its separate native PLY passed exact expected byte-length validation and can be selected independently; original PROVENANCE capture-phase flags are superseded for actual file usability by VALIDATED_DELIVERY_STATUS.json.

## Actual retained Mip1K native model

The first Mip capture is retained by a hard link and exported with the official save_ply. Its591,494 native primitives and all exported parameter bits match the actual1K capture. Seven nonfinite position/rotation primitives are retained unchanged; active SH degree1 is read from actual export metadata. This is a partial passive capture before the iteration optimizer step. Its actual native PLY is `/home/test/foresplat_model_files/KongQueZhuYu/E13_QGS/Mip/audited_snapshots/iteration_1000/native_quadratic.ply`.

```bash
uv run --no-project --python .venv/bin/python python scripts/native_qgs_viewer.py --native-ply "/home/test/foresplat_model_files/KongQueZhuYu/E13_QGS/Mip/audited_snapshots/iteration_1000/native_quadratic.ply" --config "/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/ForeSplat_VGGT_Omega_Frontend/experiments/KongQueZhuYu/gaussian_mesh_algorithm_selection/E13_QGS/actual_config.yaml" --iteration 1000 --model-stage retained_Mip_partial --port 8767 --width 540
```

The actual native CUDA viewer smoke for this retained model will run after formal training; CPU PLY/checkpoint provenance has already passed. Use the artifact index stage labels and never promote this1K file to the30K result.
