# E2_Omega_RawDepth_NoFusion - RUN REPORT

- label: VGGT-Omega raw depth, no fusion (N_E1 pts)
- plant: KongQueZhuYu
- EO: off | foreground loss: off (ORIGINAL GGGS recipe)
- iterations: 30000 | resolution: 2 | seed: 0
- training wall time: 38.2 min
- final Gaussian count: 573373
- peak VRAM allocated: 8944 MiB
- coordinate frame: VGGT-Omega native + E1 scene alignment (no scan registration)

## Reproduction

```bash
CUDA_VISIBLE_DEVICES=1 python gggs_community/train_foresplat.py --source_path <dataset> --resolution 2 --iterations 30000
```
