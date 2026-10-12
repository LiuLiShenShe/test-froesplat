# External retrieval and environment audit

- Bocha API was actually called for COLMAP camera-domain mapping references and
  returned HTTP403, `You do not have enough money or package quota`. No credentials
  are saved in any script/report/log. No claim of successful Bocha search is made.
- Context7 CLI resolution: `npx ctx7@latest library COLMAP` with the task-specific
  OPENCV/PINHOLE, pixel-center, undistortion and sparse-projection question.
  Selected official `/colmap/colmap`; documentation fetched with `docs` afterward.
- Official source was retrieved directly:
  https://github.com/colmap/colmap/blob/main/src/colmap/image/warp.cc
  https://github.com/colmap/colmap/blob/main/src/colmap/image/undistortion.cc
  https://github.com/colmap/colmap/blob/main/python/README.md
- Local installed `pycolmap=4.2.0` native function docstrings and actual smoke were
  used to confirm bindings. Tested output dimensions exactly match all three frozen
  cameras. Pixel centers `(0.5,0.5)` and source-address subtraction `-0.5` are verified
  against the official implementation, not assumed from generic OpenCV defaults.
- Requested `/home/test/.codex/RTK.md` was unavailable (`No such file or directory`).
  No replacement instructions were invented. No `.codegraph/` exists in the repo.
- Initial worktree clean on `main` at `f53bf8f3`, remote origin is the requested repo.
- CPU: `/home/test/biosoft/enter/bin/python`, Python3.13, Torch2.9.1+cu128.
- SAM: `/home/test/biosoft/enter/envs/sam3/bin/python`, Python3.12, Torch2.10+cu128.
  Frozen local processor/checkpoint hash is checked before GPU initialization.
- 2DGS: local `2d-gaussian-splatting-main/venv/bin/python`, Python3.12.3,
  Torch2.5.1+cu121, **original local** CPython3.12 surfel rasterizer/simple_knn binaries.
  Both imports and full production modules load. No environment upgrade or extension
  source edit was needed. Per-scene true loader/forward checks retain actual paths.
- NVIDIA RTX A6000×2, driver590.48.01, 49GB each. Existing other users' GPU processes
  were not killed or changed. Physical GPU1 is fixed for all planned treatments.
  CPU threads=4. Per-run GPU/disk available capacity is logged before inference.

Late damaged-input identity audit used a second Context7 library/docs pair for
native CPU SIFT: `logs/CTX7_SIFT_LIBRARY.txt` and `logs/CTX7_SIFT_DOCS.txt`.
Current upstream documentation uses FeatureExtractor; installed pycolmap4.2.0
still exposes deprecated `Sift`, whose native constructor/extract docstrings were
checked. No environment upgrade occurred. Raw0136 keypoints match original COLMAP
observations while neighboring wrong-frame controls do not; no new registration
or reconstruction was executed. This independently resolves a historical derived
RGB version conflict, not a guessed filename join.

Scientific input/statistical claims are sourced from actual local files and new
executions, not external documentation or training-data assumptions.
