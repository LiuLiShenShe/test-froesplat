# ForeSplat frontend decision: KongQueZhuYu

Decisions use saved real predictions only. Missing or failed workflows remain missing; another model does not substitute for them.

Incomplete/failed required workflows: none.

## Questions and evidence

1. **Fast3R versus VGGT-Ω:** compare foreground and leaf-edge support, neighboring-view depth agreement, normalized map distance, trajectory and the common-frame figures. The legacy VGGT-Ω artifact is depth-based and has higher input resolution. A score advantage is only a predicted-geometry consistency advantage; it is not proof of GT accuracy.
2. **AMB3R backend OFF versus ON:** the OFF/ON pair must have exactly identical frame names, transforms and RGB/masks. Evaluate pointmap changes, leaf-edge support, depth jumps and adjacent consistency. A shared full-input OFF/ON artifact is required before declaring a backend gain.
3. **AMB3R-SfM global registration:** use registered/unmapped frame counts and official registration logs. Full global mapping success means organization/registration completed; it does not itself validate every thin leaf surface.
4. **Fast3R + AMB3R-SfM:** compare mapping failures and matched-frame consistency against one-pass Fast3R. Each cluster reference is local until official registration; concatenation of raw cluster outputs is not accepted as global geometry.
5. **Leaf-edge retention:** the original-pixel rectangular leaf/edge ROIs of `crop_0000.png` are shared across methods. They are evaluation ROIs chosen by directly viewing the source RGB, not leaf instance annotations. Edge coverage does not determine which sheet is reconstructed correctly.
6. **3DGS initialization:** choose only after valid camera convention, sufficient foreground support, acceptable registration rate, adjacent geometry and thin-leaf diagnostics agree. No 3DGS training was started; current bundles expose camera/pointmap inputs for subsequent validation.
7. **Metric measurement:** AMB3R metric head provides learned metric-scale predictions when saved. There is no external length measurement in this evaluation. Fast3R and VGGT-Ω scales are ambiguous. Metric output is a measurement candidate, not a demonstrated measurement accuracy.

## Current decision

All required outputs are available for review. A main-frontend decision still requires inspecting the leaf/edge figures alongside the numeric diagnostics; no external 3D GT or metric accuracy has been established. EO remains a separate next-stage decision.

## Actual Fast3R / existing VGGT-Ω evidence

| Diagnostic | Existing VGGT-Ω depth reference | Fast3R single global pass |
| --- | --- | --- |
| views | 208 | 208 |
| edge_q85_coverage | 0.2703 | 0.1191 |
| adjacent_depth_agreement_fraction | 0.9606 | 0.8234 |
| adjacent_point_distance_p50_normalized | 0.02586 | 0.05513 |
| rotation_step_p90_deg | 10.25 | 99.25 |
| low_confidence_frames_ratio | 0.1298 | 0.5769 |

**Current answer: Fast3R does not improve on the existing VGGT-Ω reference in these full-sequence consistency diagnostics.** Its lower adjacent depth agreement and larger normalized pointmap discrepancy point in the same direction. This comparison retains the existing VGGT input resolution and the prescribed Fast3R settings; it is not a matched-resolution GT accuracy claim.

Fast3R PnP solver success is 1, but foreground map/camera reprojection median is 27.38px and rotation-step p90 is 99.25degrees. Solver success does not establish reliable camera geometry. Keep VGGT-Ω as the reference until another frontend demonstrates a useful geometric gain.

A very small Fast3R depth-jump ratio must be read alongside map discrepancy, reprojection and confidence: excessive smoothing or collapsed sheets can also be continuous. No claim of superior thin-leaf geometry follows from continuity alone.

OFF/ON frame order identical: True; pixel transforms identical: True; RGB pixels identical: True; mask pixels identical: True; seed identical: True.

| Diagnostic | OFF | ON |
| --- | --- | --- |
| foreground_coverage | 0.2068 | 0.9482 |
| edge_coverage | 0.08107 | 0.9063 |
| edge_q85_coverage | 0.07394 | 0.2886 |
| confidence_floor_fg_fraction | 0.8009 | 0.05417 |
| depth_neighbor_jump_ratio | 0 | 0.0004909 |
| adjacent_depth_agreement_fraction | — | 0.8834 |
| adjacent_point_distance_p50_normalized | — | 0.4696 |
| pointhead_camera_reprojection_all_foreground_p50_px | 273.6 | 231.9 |
| depth_unprojection_self_reprojection_p50_px | 0.8722 | 0.6836 |
| pointhead_over_unprojection_worldnorm_p50 | 1.632 | 2.296 |

The higher ON confidence-valid coverage is a real change in the saved confidence output, strongly affected by OFF rounding at1. It is not calibrated confidence or a measured geometric accuracy gain. Large all-foreground point-head/camera disagreement remains after the backend; depth-unprojection self-reprojection is algebraic. Separate prediction gauges and point/depth inconsistency prevent assigning this discrepancy entirely to camera error.

The official depth_metric field retains its learned metre prediction, but transferring its factor to the direct point-head is gauge-unvalidated. Frame-to-frame metric factors and head mismatch limit current measurement claims; no measured length or scan-scale correction was used.

## OFF/ON identical-pixel geometry controls

Primary confidence-valid pixels differ between OFF and ON. These two controls use exactly the same selected pixels in both methods: the intersection of baseline-valid geometry, and all mask+finite geometry without a confidence threshold. They separate geometry changes from confidence-selection effects.

| Method | Control | Selected FG | Adjacent depth agreement | Normalized map distance | Projection overlap | World NN |
| --- | --- | --- | --- | --- | --- | --- |
| AMB3R_Base_OFF | shared_baseline_valid | 0.2019 | — | — | 0 | 0.05711 |
| AMB3R_Base_ON | shared_baseline_valid | 0.2019 | 0.8214 | 0.5948 | 0.04016 | 0.05504 |
| AMB3R_Base_OFF | mask_finite_without_confidence | 1 | 0.8908 | 0.4294 | 0.0618 | 0.03187 |
| AMB3R_Base_ON | mask_finite_without_confidence | 1 | 0.8669 | 0.4672 | 0.1934 | 0.03194 |

World NN is a pose-independent proximity proxy. Occlusions, density and collapsed surfaces can yield a low distance without correct geometry; it is never a sole accuracy or winner criterion.

## Full-sequence precision control

The official precision string32 disables outer autocast; official internal FlashAttention still uses BF16. Both runs use all208 identical original frames and unchanged weights. The BF16 primary result is preserved.

| Diagnostic | BF16 primary | Outer FP32 control |
| --- | --- | --- |
| self_reprojection_p50_px | 27.38 | 28.44 |
| edge_q85_coverage | 0.1191 | 0.1183 |
| adjacent_depth_agreement_fraction | 0.8234 | 0.7844 |
| adjacent_point_distance_p50_normalized | 0.05513 | 0.05768 |
| rotation_step_p90_deg | 99.25 | 109.1 |
| total_frontend_seconds | 29.28 | 38.39 |

Inspect `visualizations/precision_control` together with this table. Outer precision is a tested control; the result does not isolate the official internal BF16 attention. Neither a similar reprojection result nor numeric continuity alone establishes reliable thin-leaf reconstruction.

## Existing five-view manual2D GT

Actual manual masks and photos from03-GT are used for source0000,0025,0050,0075,0100. The GT column and model projections share original-view coordinates. The mask also includes the pot and separate blue object; it is not a leaf-only instance annotation. Support recall measures valid RAP foreground geometry at manual-GT pixels. No registered3D scan or real-scale accuracy is asserted.

| Method | Manual2D GT support | Manual2D GT edge support |
| --- | --- | --- |
| VGGT_Omega_Existing | 0.9708 | 0.209 |
| Fast3R | 0.9708 | 0.222 |
| AMB3R_Base_OFF | 0.06541 | 0.01054 |
| AMB3R_Base_ON | 0.9402 | 0.2016 |
| AMB3R_SfM | 0.9707 | 0.2154 |
| Fast3R_AMB3R_SfM | 0.9708 | 0.222 |
| Fast3R_FP32 | 0.9708 | 0.222 |