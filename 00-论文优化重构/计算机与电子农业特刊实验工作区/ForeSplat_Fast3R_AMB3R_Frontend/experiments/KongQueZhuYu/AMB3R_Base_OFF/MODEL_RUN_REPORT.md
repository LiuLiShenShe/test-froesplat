# AMB3R_Base_OFF

Official `AMB3R.forward(frames, iters=0)`, 208 frames, scope `full_sequence`. Input size: [518, 294].

Inference: 43.652 s; peak allocated VRAM: 23.671 GiB; peak reserved VRAM: 31.703 GiB. Only physical GPU1 used.

Raw `world_points`, `pts3d_by_unprojection` and `pose` retain the official normalized scene scale. `depth_metric` is the official predicted metric depth. Metric pointmaps and camera translations are saved as explicitly named derived arrays using the exact official median scale; no scan registration or scale correction is applied. Native local pointmap head is absent.

Masks are external filters only; the original network does not consume masks. All input frames retain source order; geometry uses aspect-preserving resize and padding, with nearest-neighbor masks and saved pixel transforms.

Checkpoint SHA256: `2d9342282cce0a20e1074ad1513d9a18c3d100dc06c734a4a11f2d1baeea5239`. Official commit: `92c4081f910f98e683503092b85301861519175e`.

Foreground mask+confidence point count: 2010474.

[Official source](https://github.com/HengyiWang/amb3r/blob/92c4081f910f98e683503092b85301861519175e/amb3r/model.py), [paper](https://arxiv.org/abs/2511.20343).

Actual seed: 42 in both OFF and ON. Shared input SHA256: `1bb8c1d2d24b6173f7529edb39d97a776f573b5bc40c7ea02c6d0ffcdca962b6`.

Valid foreground pixel coverage: 20.676516%; mean per-frame foreground coverage: 19.910121%; mean foreground silhouette boundary coverage: 8.287570%. These are model confidence validity and silhouette coverage diagnostics, not ground-truth 3D accuracy.

The q85 derived cloud was repaired by the evaluation workflow to require confidence > 1 before the quantile threshold; the original q85 PLY was preserved and raw predictions remain unchanged. Confidence near 1.0 is floating-point saturation or very low model confidence and is not interpreted as high-quality geometry.

Median across views of foreground point-head vs camera reprojection medians: 273.6383 px. This compares predictions from two heads and does not establish camera accuracy against ground truth.
