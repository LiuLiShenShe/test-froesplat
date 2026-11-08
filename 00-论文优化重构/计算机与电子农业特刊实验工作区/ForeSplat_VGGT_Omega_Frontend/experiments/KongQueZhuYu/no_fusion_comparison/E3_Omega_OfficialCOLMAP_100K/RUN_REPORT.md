# E3_Omega_OfficialCOLMAP_100K - RUN REPORT

- label: Official-style COLMAP init, 100K pts
- plant: KongQueZhuYu
- EO: off | foreground loss: off (ORIGINAL GGGS recipe)
- iterations: 30000 | resolution: 2 | seed: 0
- training wall time: 35.8 min
- final Gaussian count: 601482
- peak VRAM allocated: 8779 MiB
- coordinate frame: VGGT-Omega native + E1 scene alignment (no scan registration)

## Reproduction

```bash
CUDA_VISIBLE_DEVICES=1 python gggs_community/train_foresplat.py --source_path <dataset> --resolution 2 --iterations 30000
```
