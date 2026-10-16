# Phase20G1 Handoff

## Ready state

Phase20G0 leaves a real, independent post-hoc 2DGS PLY compressor with exact row budgets, source-index provenance, native reload validation, two baseline scorers, common held-out evaluation, synchronized renderer benchmarking, and one TSDF compatibility run. The production training mathematics and historical assets are unchanged.

## Required before a conference-derived scorer

1. Implement a dedicated 2D surfel Fisher CUDA path that accumulates the five spatial parameters `(x,y,z,log_scale_0,log_scale_1)` per observation into 15 symmetric entries. Validate it against the tiny Python/finite-difference oracle before scene-scale use.
2. Define and validate a LightGaussian surface support score rather than reusing three-scale volume or alpha heuristics.
3. Add recovery training only as a separate, explicitly controlled experiment with optimizer/checkpoint provenance.
4. Run multiple scenes only after the scorer passes those contracts.

## Recommendation

Do not introduce foreground support as a new score in G1 until an all-scene capacity curve exists. The current ROI is a frozen SAM3 proxy, explicitly not independent GT. If the baseline curves show leaf-edge loss at equal capacity, test foreground support as a registered ablation with the same frozen ratios, not as an unannounced replacement.

Current strongest baseline by raw LPIPS is `random 0`; strongest by foreground PSNR proxy is `random 0`. These are descriptive results only and do not establish a new algorithm.
