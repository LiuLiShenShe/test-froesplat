# Final analysis — KongQueZhuYu no-fusion initialisation comparison

## A. Experiment setup (all three share everything except the initialisation cloud)

- Dataset: same 208 views (identical `images/`, `masks/`, camera intrinsics/extrinsics —
  **max |Δcamera-params| = 0.0** vs E1; verified GGGS `readColmapSceneInfo` reads all three).
- Training: **original GGGS recipe** — EO off, foreground loss off, seed 0, 30000 iters,
  resolution divisor 2, identical optimizer/densification; GPU1 sequential.
- Initialisation clouds (only difference):
  - **E1 fused (ForeSplat)**: 1,252,383 points (existing E1 run, reused, not retrained).
  - **E2 raw depth (no fusion)**: 1,252,383 points — per-view stratified sampling (seed 0)
    of the raw VGGT-Ω depth back-projection; every point is a single-view unprojection,
    identical filters (foreground ∧ finite ∧ conf ≥ percentile-20-in-foreground; the
    resulting threshold 1.2527 and the 13,640,967-point valid candidate pool match E1's
    pre-fusion numbers exactly).
  - **E3 official-style 100K**: 100,000 points — global deterministic sample (seed 0) of the
    same pool (official `demo_colmap.py` semantics: single-observation points, no BA, no track).

## B. Results (Training-view Metrics over the 208 trained views)

| Method | Init pts | PSNR | SSIM | LPIPS | FG-PSNR | FG-SSIM | Final pts | Time |
|---|---|---|---|---|---|---|---|---|
| E1 fused (ForeSplat) | 1,252,383 | 17.8947 | 0.7858 | 0.2406 | 15.2787 | 0.6117 | 600,458 | 40.3 m |
| E2 raw depth | 1,252,383 | **17.9293** | 0.7849 | 0.2414 | **15.3148** | 0.6103 | 573,373 | 37.2 m |
| E3 official 100K | 100,000 | 17.9153 | 0.7852 | 0.2409 | 15.2979 | 0.6109 | 601,482 | 34.8 m |

E1 re-evaluation reproduced its earlier numbers exactly (deterministic pipeline).

## C. Point-cloud evidence (before training)

- Plant-scale views (front/back/left/right/top): E1 and E2 clouds are visually near-identical;
  E3 is a sparser version of the same geometry. **No leaf-edge smoothing, no leaf adhesion,
  no surface thickening visible between E1 and E2 at plant scale.**
- Identical-box zooms quantify what fusion did: in the leaf-edge box E1 holds 106,556 points
  vs E2's 174,097 — fusion **compacted/merged** ~39% of the raw points in that box — but the
  surviving structure follows the same leaf surfaces. Fusion is a compaction, not a geometric
  distortion, at this scene scale.

## D. Attribution (data-driven, maps to 情况 C of the plan)

1. **Fusion is NOT the cause of the blur.** At identical point count E2 ≥ E1 everywhere
   (PSNR +0.035 dB, FG-PSNR +0.036 dB, SSIM/LPIPS tied to the 4th decimal). The margin is
   within run-to-run noise, so the honest statement is *statistically indistinguishable, with
   no evidence that fusion helps*. Removing fusion neither fixes nor worsens the blur.
2. **Initial point count is NOT the cause.** E3 with 100K points (12.5x fewer) converges to
   the same metrics (PSNR 17.9153) and even the largest final Gaussian count (601,482).
   GGGS densification compensates for the initial budget within this range.
3. **All three initialisations converge to equally soft renders.** The fixed-view montages
   (`comparison_renders/leaf_zoom_gt_E1_E2_E3.png` etc.) show the GT with sharp leaf
   serrations while E1/E2/E3 renders are all similarly blurred — the initialisation is not
   the bottleneck.
4. Therefore the blur originates in the **shared factors**: the GGGS optimization/rendering
   (resolution divisor 2, mip-splatting 3D filter, surfel rendering, densification budget,
   per-view exposure mismatch — recall the earlier E2+EO experiment reached EO-corrected
   **18.23 dB vs 17.89 raw**, i.e. ~0.34 dB of the residual is per-view exposure) or the
   VGGT-Ω depth/pose quality common to all three runs.

## E. Answers to the six questions

1. **不融合后原始深度点云是否更清晰？** 点云层面：原始点在叶缘局部更致密（融合把该
   局部 174K 点压缩到 106K），结构一致；植株尺度上两者几乎无差别。不是"更清晰"，
   是"更致密、同样正确"。
2. **相同点数下 E2 是否优于 E1？** PSNR/FG-PSNR 略高（+0.035 dB），SSIM/LPIPS 持平，
   属噪声级差异——**E2 ≥ E1，融合没有带来光度收益**。
3. **100K 的 E3 是否更好？** 与 E1/E2 持平（PSNR 17.9153，介于两者之间）。初始点数在
   100K–1.25M 区间内不构成瓶颈。
4. **模糊是否主要来自融合模块？** **否。** 去掉融合（E2）与 100K 初始化（E3）都没有
   改善清晰度；三种初始化的渲染同样模糊。
5. **是否有必要继续使用 VGGT-Ω？** 目前没有弃用理由：VGGT-Ω 原始深度初始化取得了与
   融合版相同的结果，前端深度足以支撑 GGGS 初始化。
6. **下一步是否应该切换标准 VGGT？** 还不是首选。证据指向共享因素（GGGS 训练/渲染侧
   与 VGGT-Ω 深度质量）。建议顺序：
   a. EO 逐视角曝光的完整评估（已证 +0.34 dB，先把这部分残差拿掉）；
   b. resolution divisor 1（全分辨率训练）与更大 densification 预算的对照；
   c. 若以上仍模糊，再做标准 VGGT 前端对照（情况 C 的最后一步），并直接对比
      VGGT-Ω 与 VGGT 的逐帧深度图质量。

## F. Practical recommendation

- The fusion module should lose its mandatory status: the raw-depth path (E2) is simpler,
  equally accurate, and cheaper (37.2 vs 40.3 min); keep fusion as an optional research
  baseline. This does NOT modify the ForeSplat mainline — it is a comparison result only.
- The reconstruction blur must be attacked on the GGGS optimization/rendering side (resolution,
  densification, exposure) or by a frontend depth-quality comparison (VGGT-Ω vs standard VGGT),
  not by reworking the fusion.

## G. Artifacts

- `source_data_audit.md` — input audit (all VGGT-Ω per-frame data present; no re-inference).
- `E2_Omega_RawDepth_NoFusion/`, `E3_Omega_OfficialCOLMAP_100K/` — dataset/ (COLMAP, 208
  views), initial_pointcloud/ (init cloud + provenance + stats), checkpoints/, renders/,
  metrics/, RUN_REPORT.md.
- `initialization_comparison/` — 8 fixed-camera point-cloud comparisons (E1|E2|E3).
- `comparison_renders/` — GT|E1|E2|E3 render montages (fixed views + leaf zoom).
- `comparison.csv` / `comparison.md` — the table in §B.
