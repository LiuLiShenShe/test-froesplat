# Gaussian Compression Plugin

This directory is an independent, post-hoc compressor for frozen 2DGS Gaussian PLY files. It does not import the training entry point and does not mutate a checkpoint, the source PLY, the dataset split, or the renderer CUDA source.

## Public entry point

```python
from plugins.gaussian_compression import compress_model

result = compress_model(
    input_model="/path/to/point_cloud.ply",
    output_dir="/path/outside/source/model",
    method="opacity",                 # or random / external
    prune_ratio="0.4",
    config={"enabled": True, "seed": 20261016, "active_sh_degree": 3},
)
```

`input_model` may be a frozen PLY or a model directory containing `point_cloud/iteration_<N>/point_cloud.ply`. The output directory must not exist and must be outside the protected source directory. The compressor preserves every scalar PLY property and writes `retained_indices.npy`, `removed_indices.npy`, `importance_scores.npy`, `compression_config.json`, `source_model_sha256`, `output_model_sha256`, and `compression_log.jsonl`.

`enabled=false` is an identity pass: it copies the PLY byte-for-byte, emits identity indices, and does not initialize a scorer. Recovery optimization is intentionally unsupported in this phase; `recovery=true` fails before any training code can run.

The implemented methods are deliberately limited to deterministic Random and stored-opacity baselines plus a provenance-bound external score. PUP and LightGaussian are registered as blocked until their 3D CUDA contracts are adapted to the 2DGS surfel rasterizer. They are not silently approximated.

Run the focused tests with:

```bash
2d-gaussian-splatting-main/venv/bin/python -m unittest discover \
  -s plugins/gaussian_compression/tests -p 'test_*.py' -v
```

Large PLYs and render tensors are kept outside Git under `/data/fj/ForeSplat_Phase20G0_assets/723c02c2_v1`; manifests in the phase directory point to them by absolute path and SHA256.
