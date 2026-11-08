# ForeSplat frontend decision: KongQueZhuYu

Decisions use saved real predictions only. Missing or failed workflows remain missing; another model does not substitute for them.

Incomplete/failed required workflows: VGGT_Omega_Existing, Fast3R, AMB3R_Base_OFF, AMB3R_Base_ON, AMB3R_SfM, Fast3R_AMB3R_SfM.

## Questions and evidence

1. **Fast3R versus VGGT-Ω:** compare foreground and leaf-edge support, neighboring-view depth agreement, normalized map distance, trajectory and the common-frame figures. The legacy VGGT-Ω artifact is depth-based and has higher input resolution. A score advantage is only a predicted-geometry consistency advantage; it is not proof of GT accuracy.
2. **AMB3R backend OFF versus ON:** the OFF/ON pair must have exactly identical frame names, transforms and RGB/masks. Evaluate pointmap changes, leaf-edge support, depth jumps and adjacent consistency. A shared full-input OFF/ON artifact is required before declaring a backend gain.
3. **AMB3R-SfM global registration:** use registered/unmapped frame counts and official registration logs. Full global mapping success means organization/registration completed; it does not itself validate every thin leaf surface.
4. **Fast3R + AMB3R-SfM:** compare mapping failures and matched-frame consistency against one-pass Fast3R. Each cluster reference is local until official registration; concatenation of raw cluster outputs is not accepted as global geometry.
5. **Leaf-edge retention:** the original-pixel rectangular leaf/edge ROIs of `crop_0000.png` are shared across methods. They are evaluation ROIs chosen by directly viewing the source RGB, not leaf instance annotations. Edge coverage does not determine which sheet is reconstructed correctly.
6. **3DGS initialization:** choose only after valid camera convention, sufficient foreground support, acceptable registration rate, adjacent geometry and thin-leaf diagnostics agree. No 3DGS training was started; current bundles expose camera/pointmap inputs for subsequent validation.
7. **Metric measurement:** AMB3R metric head provides learned metric-scale predictions when saved. There is no external length measurement in this evaluation. Fast3R and VGGT-Ω scales are ambiguous. Metric output is a measurement candidate, not a demonstrated measurement accuracy.

## Current decision

Keep the frontend decision provisional until all required workflows have either produced real outputs or have a documented concrete failure. Do not start EO exposure optimization yet.