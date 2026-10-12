# Managed data recovery

Managed root: `/data/fj/F2DMAS_phase20e0_data/f53bf8f3_v1`. It is OUTSIDE Git; no large mask/RGBA/model archive is committed.
CSV manifests contain exact source paths and SHA256, every candidate instance, score,
box, precision, code/checkpoint, output RGB, alpha, camera bundle and map hashes.
Copy/restore the managed root and verify `reports/MANAGED_ARTIFACT_HASHES.csv` first.
Preserve directory symlinks and file bytes; identical B/C images may be hardlinked.
Run the read-only managed-cache verifier after restoration:

```bash
PYTHONDONTWRITEBYTECODE=1 /home/test/biosoft/enter/bin/python '/data/fj/F2DMAS/00-论文优化重构/计算机与电子农业特刊实验工作区/阶段二十E0_三维全视图与输入冻结/scripts/verify_managed_cache.py'
```

This also validates each group's common-camera directory symlink. Do not flatten
symlinks or replace sparse initialization PLY merely because points3D.bin matches.
Historical checkpoint: `/data/fj/F2DMAS/sam3/sam3.pt`, SHA256
`9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e`.

For full rebuild use exact original assets at the recorded baseline, `audit_inputs.py`,
then `recover_verified_sources.py`, each scene's `generate_candidates.py --sample NAME --smoke`, full generation,
`rank_candidates.py`, `build_datasets.py`, and real `preflight.py`. Complete the first
scene before starting remaining scene inference. SAM runtime is
`/home/test/biosoft/enter/envs/sam3/bin/python`; CPU/remap runtime is
`/home/test/biosoft/enter/bin/python`; loader runtime is `/data/fj/F2DMAS/2d-gaussian-splatting-main/venv/bin/python`.
Scripts import only local frozen SAM3 and 2DGS sources; no CLI parameters are guessed.
GPU is selected externally with CUDA_VISIBLE_DEVICES=1. Set OMP/OPENBLAS threads=4.

Exact cache restore is preferred. A new GPU regeneration may not byte-match across
runtime/hardware; label it a NEW candidate version, never silently replace this cache.
Archive confidences have six-decimal serialization limitations and are preserved,
not reconstructed digits. New scores retain full available tensor precision.
No GT directory is read by inference, ranking or dataset construction.
Inference needs the recorded local SAM checkpoint/processor; loader uses its local
CPython3.12 CUDA extension binaries. Hashes, versions, environment paths and physical
GPU UUID are in candidate provenance, `reports/*_PREFLIGHT.json` and the hardware
identity JSON. Never silently install a newer library to simulate an exact restore.
