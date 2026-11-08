# ScanNet++ preprocessing (training + evaluation)

[ScanNet++](https://kaldir.vc.in.tum.de/scannetpp/) DSLR captures, rectified to a
pinhole model with per-frame metric depth rendered from the aligned GT mesh.
Train split `nvs_sem_train`, eval split `nvs_sem_val`.

## 1. Download

Register at <https://kaldir.vc.in.tum.de/scannetpp/>, accept the license, and use
the official downloader to fetch the DSLR data and meshes. Expected raw layout
(`--data_root`):

```text
scannetpp_raw/
├── splits/
│   ├── nvs_sem_train.txt
│   └── nvs_sem_val.txt
└── data/
    └── <SCENE_ID>/
        ├── scans/mesh_aligned_0.05.ply        # GT mesh (read)
        └── dslr/
            ├── resized_images/*.JPG           # fisheye RGB (read)
            ├── colmap/images.txt              # poses (read)
            └── nerfstudio/transforms.json     # intrinsics + distortion (read)
```

## 2. Install (preprocess-only)

Depth rendering needs PyTorch3D. It compiles against the `torch` already in your
env, so build with `--no-build-isolation` and install a matching CUDA toolchain
first (torch 2.5.1 → CUDA 12.4):

```bash
conda install -c "nvidia/label/cuda-12-4" cuda-toolkit
pip install ninja fvcore iopath
FORCE_CUDA=1 pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable" --no-build-isolation
```

`open3d` and `opencv-python` come with the core install. `cuda-toolkit` provides
the `nvcc` the build needs; `FORCE_CUDA=1` compiles the CUDA kernels even on a node
without a visible GPU.

## 3. Run

One command processes both splits end to end (discovers scenes from the split
files, renders depth, undistorts, and writes everything in the layout the configs
expect):

```bash
python -m dvlt.scripts.preprocess.scannetpp.preprocess \
    --data_root datasets/scannetpp_raw \
    --output_root datasets
```

- `--data_root`: the raw release (step 1).
- `--output_root`: same as the `user.data_root`

Options: `--splits nvs_sem_val` (eval only), `--scene_ids <id> ...` (subset),
`--overwrite` (re-render), `--device` (default `cuda`).

### Rasterization speed (optional)

Depth rendering defaults to PyTorch3D's safe rasterization heuristic: correct but
slow, because its per-bin face budget is heavily oversized. `--bin_size` and
`--max_faces_per_bin` trade that for speed; good values depend on mesh size, image
resolution, and GPU. For the ScanNet++ DSLR meshes at 1752×1168, a good starting
point is:

```bash
python -m dvlt.scripts.preprocess.scannetpp.preprocess \
    --data_root datasets/scannetpp_raw \
    --output_root datasets \
    --bin_size 128 --max_faces_per_bin 200000
```

## 4. Output

With `--output_root datasets`:

```text
datasets/
├── train/scannetpp/
│   ├── nvs_sem_train.txt
│   └── scannetpp_undistort/<SCENE_ID>/
│       ├── undistorted_images/*.JPG
│       ├── undistorted_depth/*.png                 # 16-bit metric depth
│       └── nerfstudio/transforms_undistorted.json  # PINHOLE intrinsics
└── test/scannetpp/
    ├── nvs_sem_val.txt
    └── scannetpp_undistort/<SCENE_ID>/...
```

Then set the configs' `user.data_root` to `--output_root` (here `datasets/`).
`nvs_sem_train` lands under `train/`, `nvs_sem_val` under `test/`, matching
[`train_datasets/scannetpp.yaml`](../../config/experiments/data/train_datasets/scannetpp.yaml)
and
[`test_datasets/scannetpp.yaml`](../../config/experiments/data/test_datasets/scannetpp.yaml).

## Notes

- Depth is rasterized through the original fisheye camera from the GT mesh, then
  rectified to pinhole with nearest-neighbour resampling (matching the images'
  rectification).
- Depth PNGs store the float16 bit-pattern read by
  [`common.io.read_depth`](../../common/io.py); read them back with that function.
