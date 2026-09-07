# Phase 16 — Input Audit

## 0. Starting state

- HEAD: `ad1c739526de9b9cc25c8ab6126a1c1d462b9ffb`
- Branch: `main`
- Status:
```
## main...origin/main
?? 00-论文优化重构/计算机与电子农业特刊实验工作区/01-算法模块修改与验证/阶段十六_HardCase_GT构建与锁定/
```

```
ad1c7395 Phase 15: 零计算难度分层复分析(挑战集与外部验证)
79471fe7 Add Phase14.1 A7 validity closure: fix session lifecycle, per-sample diagnostics, rerun V01/V11
f9f091d8 Add 阶段十四 A6A7真实数据析因消融 (V00-V11 variants, 919MB) + update 掩膜脚本
5c6e4ab8 Update Phase13 tests & report: add integration test, refine runtime checks
e939396e Add 阶段十三 A6A7增强分支强制验证 (6 files: report, audit, 4 tests)
```

- 5 GT samples (frozen): CaoMei1, ChangShouHua2, DouBanLv1, KongQueZhuYu, XianKeLai1
- 15 non-GT samples: BaiZhang, CaoMei2, ChangShouHua1, ChangShouHua3, DouBanLv2, DouBanLv3, HongZhang, WangWenCao1, WangWenCao2, WanNianQing1, WanNianQing2, XiangPiShu1, XiangPiShu2, XianKeLai2, XianKeLai3

## 1. difficulty_ranking.csv (Phase 15)

- Rows: 60
- Columns: rank, sample, frame, diff, laplacian_var, pixel_entropy, exg_sep, green_ratio, dyn_range, c_laplacian, c_entropy, c_exg_sep, c_green_ratio, c_dyn_range, path
- diff range: 0.5513 – 0.7746
- Samples represented: 14/15
  - BaiZhang: 5
  - CaoMei2: 3
  - ChangShouHua1: 5
  - ChangShouHua3: 5
  - DouBanLv2: 3
  - HongZhang: 1
  - WanNianQing1: 5
  - WanNianQing2: 5
  - WangWenCao1: 5
  - WangWenCao2: 5
  - XianKeLai2: 3
  - XianKeLai3: 5
  - XiangPiShu1: 5
  - XiangPiShu2: 5
- **Samples with 0 candidates**: DouBanLv3
- Ranking source: raw-image attributes only (laplacian_var, pixel_entropy, exg_sep, green_ratio, dyn_range)
- **Model-independence: model-independent / model-agnostic selection** (no A6/A7/V10/V01/V11 metric used in ranking)

## 2. dataset_index.json (sample metadata)

- Samples indexed: 20
| sample | raw_frames | fft_frames | has_gt | colmap_status | reg_rate | points3d | selected_path |
|---|---|---|---|---|---|---|---|
| BaiZhang | 250 | 214 | no | fail | 0.9% | 116 | `/data/fj/F2DMAS/04-COLMAP-rerun-original/BaiZhang/sequential` |
| CaoMei1 | 250 | 210 | yes | ok | 100.0% | 24226 | `/data/fj/F2DMAS/04-COLMAP/CaoMei1` |
| CaoMei2 | 250 | 210 | no | warn | 50.5% | 26199 | `/data/fj/F2DMAS/04-COLMAP-rerun-original/CaoMei2/sequential` |
| ChangShouHua1 | 250 | 213 | no | fail | 0.9% | 16 | `/data/fj/F2DMAS/04-COLMAP-rerun-original/ChangShouHua1/exhaustive` |
| ChangShouHua2 | 250 | 212 | yes | ok | 100.0% | 20796 | `/data/fj/F2DMAS/04-COLMAP/ChangShouHua2` |
| ChangShouHua3 | 250 | 210 | no | missing_or_fail | 0.0% | 0 | `/data/fj/F2DMAS/04-COLMAP-rerun-original/ChangShouHua3/exhaustive` |
| DouBanLv1 | 250 | 215 | yes | ok | 100.0% | 10764 | `/data/fj/F2DMAS/04-COLMAP/DouBanLv1` |
| DouBanLv2 | 250 | 210 | no | ok | 100.0% | 21653 | `/data/fj/F2DMAS/04-COLMAP/DouBanLv2` |
| DouBanLv3 | 250 | 210 | no | ok | 95.2% | 47352 | `/data/fj/F2DMAS/04-COLMAP/DouBanLv3` |
| HongZhang | 250 | 213 | no | ok | 100.0% | 24912 | `/data/fj/F2DMAS/04-COLMAP/HongZhang` |
| KongQueZhuYu | 250 | 210 | yes | fail | 1.4% | 162 | `/data/fj/F2DMAS/04-COLMAP-rerun-original/KongQueZhuYu/sequential` |
| WanNianQing1 | 252 | 213 | no | ok | 100.0% | 23259 | `/data/fj/F2DMAS/04-COLMAP/WanNianQing1` |
| WanNianQing2 | 250 | 215 | no | ok | 100.0% | 17715 | `/data/fj/F2DMAS/04-COLMAP/WanNianQing2` |
| WangWenCao1 | 250 | 211 | no | ok | 70.6% | 35160 | `/data/fj/F2DMAS/04-COLMAP/WangWenCao1` |
| WangWenCao2 | 250 | 210 | no | ok | 100.0% | 33317 | `/data/fj/F2DMAS/04-COLMAP/WangWenCao2` |
| XianKeLai1 | 250 | 208 | yes | ok | 100.0% | 47148 | `/data/fj/F2DMAS/04-COLMAP/XianKeLai1` |
| XianKeLai2 | 250 | 208 | no | ok | 100.0% | 26405 | `/data/fj/F2DMAS/04-COLMAP/XianKeLai2` |
| XianKeLai3 | 250 | 206 | no | ok | 100.0% | 40140 | `/data/fj/F2DMAS/04-COLMAP/XianKeLai3` |
| XiangPiShu1 | 250 | 208 | no | ok | 97.6% | 36875 | `/data/fj/F2DMAS/04-COLMAP/XiangPiShu1` |
| XiangPiShu2 | 250 | 211 | no | fail | 0.9% | 224 | `/data/fj/F2DMAS/04-COLMAP-rerun-original/XiangPiShu2/sequential_lowmem` |

## 3. COLMAP completeness (03-final_locked)

| sample | has_sparse/0 | images_count | registered |
|---|---|---|---|
| BaiZhang | yes | 209 | 2 |
| CaoMei2 | yes | 210 | 106 |
| ChangShouHua1 | yes | 212 | 2 |
| ChangShouHua3 | yes | 205 | 0 |
| DouBanLv2 | yes | 210 | 210 |
| DouBanLv3 | yes | 0 | 200 |
| HongZhang | yes | 213 | 213 |
| WangWenCao1 | yes | 0 | 149 |
| WangWenCao2 | yes | 210 | 210 |
| WanNianQing1 | yes | 213 | 213 |
| WanNianQing2 | yes | 215 | 215 |
| XiangPiShu1 | yes | 0 | 203 |
| XiangPiShu2 | yes | 211 | 2 |
| XianKeLai2 | yes | 208 | 208 |
| XianKeLai3 | yes | 0 | 206 |

## 4. Frame readability spot-check (cv2.imread, 5 evenly-spaced frames)

| sample | total_files | readable | unreadable |
|---|---|---|---|
| BaiZhang | 250 | 5/5 | - |
| CaoMei2 | 250 | 5/5 | - |
| ChangShouHua1 | 250 | 5/5 | - |
| ChangShouHua3 | 250 | 5/5 | - |
| DouBanLv2 | 250 | 4/5 | 0124.jpg |
| DouBanLv3 | 250 | 4/5 | 0000.jpg |
| HongZhang | 250 | 5/5 | - |
| WangWenCao1 | 250 | 4/5 | 0187.jpg |
| WangWenCao2 | 250 | 5/5 | - |
| WanNianQing1 | 252 | 5/5 | - |
| WanNianQing2 | 250 | 5/5 | - |
| XiangPiShu1 | 250 | 5/5 | - |
| XiangPiShu2 | 250 | 5/5 | - |
| XianKeLai2 | 250 | 4/5 | 0124.jpg |
| XianKeLai3 | 250 | 5/5 | - |

## 5. Candidate path existence

- All 60 candidate paths exist (via raw_frames symlink layer).

## 6. Contact graph / adjacency

- **No contact graph / adjacency file exists in the project.**
- Temporal adjacency is implicit from filename order (0000.jpg → 0001.jpg → …).
- COLMAP view graph is derivable from `sparse/0/images.bin` but not materialized.
- Phase 16 uses filename order + per-sample isolation for temporal context.

## 7. Audit conclusions

- Data pool: 15 non-GT samples, ~3,752 raw frames, COLMAP sparse models present for all 15.
- Caveats: WangWenCao1 registration only ~70.6%; DouBanLv3 contributed 0 Phase-15 candidates.
- Selection protocol is model-independent (raw-image attributes only).