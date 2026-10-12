# Phase20E1 acceptance gates

Current readiness is measured in `reports/FINAL_ACCEPTANCE.json`, not inferred
from configuration files. 3/3 complete scenes; 8 distinct runs.

G1 full V1 snapshot and ranking/checkpoint/processor hashes unchanged; G2/G3 new
versus old cache labels, full new tensor score/box precision and per-instance hashes;
G4/G10 identical cache replay, all scores and mask identity; G5 complete mask coverage;
G6/G7 RGB-camera join plus official mapping, native RGB and real sparse projections;
G8 identical bundle and immutable split; G9 historical A differences and missing RGBA
explicit; G11 real train-mask difference counts; G12 true loader and CUDA render;
G13 no production 2DGS/R1/M2M3 edits; G14 optimizer steps=0/checkpoints=0; G15 explicit
commands, frozen configs, data manifest and outputs. Final automated checks are
in `logs/DELIVERY_VERIFICATION.json`.
The 45 scene-specific G1–G15 checks are in
`reports/ACCEPTANCE_GATES_G1_G15.csv` and `logs/G1_G15_VALIDATION.json`.
The predeclared Phase20D view/camera IDs, names, order and splits are replayed exactly
in `reports/PREDECLARED_SPLIT_COMPARISON.csv`; corrected raw identity does not change
which registered cameras belong to train or validation.

Before any training, re-hash all datasets/configs, confirm same physical GPU and
available resources, preserve split names/IDs and source/model points. Never enable
A6/A7, SPNP/Residual/GCFC/semantic/temporal/geometry gates or tune R1 on TEST data.

Evaluation separates held-out shared raw RGB PSNR/SSIM/LPIPS; independently reliable
fixed silhouette/ROI; background contamination/visual inspection; Gaussian counts,
time/VRAM; descriptive mesh/point-cloud quality in a later authorized phase. Training
masks are NOT independent GT. No independent geometry exists; no true 3D Chamfer or
geometric F-score claim. Unregistered TEST32 images cannot become held-out cameras.
