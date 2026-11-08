# Full-sequence batch report

Plants processed: 19

| plant | frames | processed | res | mode | infer s | peak VRAM (MiB) | raw pts | fused pts | ratio | cameras | GGGS read | Gauss init |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ChangShouHua2 | 212 | 211 | 512 | balanced | 85.329 | 22436.9 | 1000000 | 1289801 | 1.2898 | 211 | ok | ok |
| WanNianQing1 | 213 | 211 | 512 | balanced | 57.43 | 22436.9 | 1000000 | 1212162 | 1.2122 | 211 | ok | ok |
| DouBanLv2 | 210 | 208 | 512 | balanced | 56.18 | 22202.7 | 1000000 | 1333251 | 1.3333 | 208 | ok | ok |
| DouBanLv3 | 210 | 208 | 512 | balanced | 57.57 | 22202.5 | 1000000 | 1369270 | 1.3693 | 208 | ok | ok |
| KongQueZhuYu | 211 | 208 | 512 | balanced | 59.607 | 22202.5 | 1000000 | 1252383 | 1.2524 | 208 | ok | ok |
| WanNianQing2 | 215 | 207 | 512 | balanced | 55.515 | 22124.0 | 1000000 | 1296493 | 1.2965 | 207 | ok | ok |
| XianKeLai1 | 208 | 207 | 512 | balanced | 55.702 | 22124.0 | 1000000 | 1257760 | 1.2578 | 207 | ok | ok |
| XiangPiShu2 | 211 | 207 | 512 | balanced | 55.257 | 22124.0 | 1000000 | 1174423 | 1.1744 | 207 | ok | ok |
| ChangShouHua1 | 213 | 205 | 512 | balanced | 54.657 | 21968.8 | 1000000 | 1239199 | 1.2392 | 205 | ok | ok |
| XiangPiShu1 | 208 | 204 | 512 | balanced | 54.223 | 21892.0 | 1000000 | 1199409 | 1.1994 | 204 | ok | ok |
| CaoMei1 | 212 | 203 | 512 | balanced | 53.348 | 21821.6 | 1000000 | 1257110 | 1.2571 | 203 | ok | ok |
| WangWenCao1 | 211 | 203 | 512 | balanced | 53.454 | 21821.6 | 1000000 | 1243680 | 1.2437 | 203 | ok | ok |
| WangWenCao2 | 210 | 202 | 512 | balanced | 53.251 | 21742.7 | 1000000 | 1152834 | 1.1528 | 202 | ok | ok |
| XianKeLai2 | 208 | 201 | 512 | balanced | 52.617 | 21667.2 | 1000000 | 1360808 | 1.3608 | 201 | ok | ok |
| XianKeLai3 | 206 | 201 | 512 | balanced | 52.685 | 21667.2 | 1000000 | 1293191 | 1.2932 | 201 | ok | ok |
| DouBanLv1 | 215 | 200 | 512 | balanced | 52.075 | 21588.5 | 1000000 | 1321587 | 1.3216 | 200 | ok | ok |
| HongZhang | 213 | 199 | 512 | balanced | 51.725 | 21507.2 | 1000000 | 1229530 | 1.2295 | 199 | ok | ok |
| ChangShouHua3 | 210 | 198 | 512 | balanced | 51.615 | 21429.8 | 1000000 | 1262403 | 1.2624 | 198 | ok | ok |
| CaoMei2 | 210 | 195 | 512 | balanced | 49.936 | 21203.6 | 1000000 | 1232244 | 1.2322 | 195 | ok | ok |

## Method notes

- One plant = one scene; the **entire** valid sequence runs through a single joint
  VGGT-Omega forward pass (no sampling / striding / windowing).
- On OOM the ladder is `balanced@512 -> max_size@512 -> balanced@448 -> max_size@448 -> balanced@384 -> max_size@384`; frames are never dropped.
- Fusion is confidence-weighted, normal-aware (voxel hash + spherical normal bins,
  so the two sides of a thin leaf and different leaves are never merged) and
  edge-preserving (finer voxel and larger weight on depth discontinuities).
- `points3D.ply` is the fused cloud; per-view raw points remain in `points3D.bin`
  with their single-observation tracks. Reprojection error is not a geometric
  accuracy metric for depth back-projection points.
