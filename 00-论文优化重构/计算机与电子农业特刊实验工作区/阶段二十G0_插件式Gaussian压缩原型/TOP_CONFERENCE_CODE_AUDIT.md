# Top-Conference Code Audit

## PUP 3D-GS

- Official repository: https://github.com/j-alex-hanson/gaussian-splatting-pup
- Pinned commit: `e971ea4802908c69eab7e8601bb1eddd2e443754`
- CVPR 2025 paper: https://openaccess.thecvf.com/content/CVPR2025/html/Hanson_PUP_3D-GS_Principled_Uncertainty_Pruning_for_3D_Gaussian_Splatting_CVPR_2025_paper.html
- Official implementation entry points: `fisher_pool_xyz_scaling.py`, `fisher_renderer/__init__.py`, `prune_finetune.py`, and the `rasterization_and_pup_fisher` submodule.
- Pinned Fisher submodule: `eaef2faba21f076867c1db84c3f998e5f673762f`.
- Pinned compression submodule: `79b24db6fe801a5ad3f23a440780d11884d5bc00`.
- Key behavior: the Python reference computes per-color/per-patch gradients with respect to XYZ plus three scaling parameters and accumulates 6x6 outer products. The CUDA path stores 21 symmetric entries per Gaussian. Pruning ranks by the sum of log singular values. The upstream model code uses a percentile threshold whose ties can remove more than the requested fraction; this plugin uses exact stable top-k selection instead.
- Recovery: `prune_finetune.py` computes Fisher on training cameras, prunes, then continues the training loop. That optimizer recovery is not run here.

## LightGaussian

- Official repository: https://github.com/VITA-Group/LightGaussian
- Pinned commit: `6676b983e77baadd909effc56a6aaadafa964dcc`
- NeurIPS 2024 reference: https://arxiv.org/abs/2311.17245
- Official implementation entry points: `prune.py`, `prune_finetune.py`, `gaussian_renderer/__init__.py:count_render`, and `compress-diff-gaussian-rasterization`.
- Pinned compression submodule: `240618bbc5fb21c270a0c5babb292615011a2f15`.
- Key behavior: `count_render` returns `gaussians_count` and `important_score`; the pinned CUDA kernel increments the count for accepted visible pixel/Gaussian pairs and adds `con_o.w` (opacity) to the importance score. `calculate_v_imp_score` multiplies that score by a power of the product of three activated scales. `prune_finetune.py` supports pruning and recovery training. SH distillation and quantization are excluded here.

## License and dependency audit

Both repositories ship the original Inria/MPII Gaussian-Splatting custom license (`LICENSE.md`) for non-commercial research/evaluation use. They are not MIT-licensed. Exact license hashes, environment files, submodule pins, source-file hashes and pinned blob URLs are in `reports/UPSTREAM_AUDIT.json` and `configs/UPSTREAM_CODE_INDEX.csv`.

PUP declares Python 3.8, PyTorch 2.4.1, torchvision 0.19.1, CUDA toolkit 11.8, and its Fisher/diff-Gaussian/simple-knn submodules. LightGaussian declares Python 3.9, PyTorch 1.12.1, torchvision 0.13.1, CUDA toolkit 11.6, plyfile 0.8.1, and its compression/simple-knn submodules. Neither environment was substituted into the production 2DGS process; the local plugin uses the existing frozen inference environment.

## Audit result

The source-level audit is complete. The upstream algorithms are not direct drop-ins for this 2DGS rasterizer. The phase records this as an explicit blocked adaptation, not as a heuristic reproduction.
