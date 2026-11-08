# AMB3R isolated uv environment

Use project `.venv-amb3r`; only physical GPU1 is authorized. Never run these commands in the VGGT/GGGS or global environments.

```bash
CUDA_VISIBLE_DEVICES=1 .venv-amb3r/bin/python -m frontends.amb3r.base --mode both
```

The default attempts all 208 usable KongQueZhuYu frames at an aspect-preserving long edge of 518, pad alignment 14, seed 42. OFF runs `AMB3R.forward(frames,iters=0)` and ON runs `iters=1`. Existing output directories are preserved and cause an explicit refusal to overwrite. For a fixed, clearly labeled subset after full-sequence failure:

```bash
CUDA_VISIBLE_DEVICES=1 .venv-amb3r/bin/python -m frontends.amb3r.base --mode both --representative 16 --suffix _representative16
```

The full installed environment is in `requirements.lock.txt`; this is a frozen requirements snapshot, not a fabricated uv project lock. Dependencies retain Git commit URLs for PyTorch3D and utils3d. The snapshot names the official FlashAttention release URL rather than the transient downloaded file.

Environment repairs performed with uv:

```bash
uv pip install --python .venv-amb3r/bin/python --no-deps --reinstall /tmp/flash_attn-2.7.3+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl 'numpy==1.26.4'
uv pip install --python .venv-amb3r/bin/python --no-deps 'pandas==2.2.3' 'plyfile==1.1.2' 'pytz==2025.2'
uv pip check --python .venv-amb3r/bin/python
```

The 182 MiB FlashAttention wheel was fetched directly from the [official v2.7.3 release](https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl), SHA256 `b1243e9b86687348a5ab03a073abacdf8e3d5e9e4b7e5326a183f47348c5dfba`. It replaces the inherited CUDA11 binary, which failed to import with `libcudart.so.11.0`. Torch remains `2.5.0+cu121`. No runtime symlink, conda environment or global package change was used.

Verified: `uv pip check` passes for all 165 installed packages; FlashAttention and the official AMB3R core import successfully. PyTorch3D KNN, torch-scatter, and spconv imports also pass. Extension GPU functionality is checked by real Base ON inference; import success alone is not sufficient.

Actual validation completed with all 208 usable frames on physical GPU1: OFF took 43.652 s (23.671 GiB peak allocated); ON executed the original sparse backend and took 120.108 s (30.826 GiB peak allocated, 38.602 GiB reserved). Neither full run ran out of memory. Both use exactly the same seed 42, RGB/mask tensors, transformations and stage-0 pointmap SHA256. This validates dependency execution, without establishing geometric accuracy. No representative fallback was run.

The official 3.9 GiB checkpoint additionally loads into the unmodified AMB3R model on CPU with `strict=True`; its loaded `backend.interp_v2` is false. This probe does not run inference or use a GPU.

Dependencies that were inherited but unused include CUDA11 runtime packages alongside CUDA12 packages. They were not substituted into Torch/FlashAttention. The active spconv package is `spconv-cu120==2.3.6`, torch-scatter is `2.1.2+pt25cu121`, and PyTorch3D source is v0.7.8 commit `75ebeeaea0908c5527e7b1e305fbc7681382db47`. See the source audit for the distinction between normalized raw geometry and predicted metric-depth derivatives.
