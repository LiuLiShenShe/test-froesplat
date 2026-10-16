# Qualitative Compression Observations

This report records a fixed visual diagnostic for the WanNianQing1 / B_R0 / Seed 0 / M0 experiment. It is not an independent geometry or foreground ground-truth evaluation.

## Protocol

- Cameras: `0000`, `0107`, and `0244`.
- Runs: `random_p00`, `random_p40`, `opacity_p40`, `random_p60`, and `opacity_p60`.
- Reference images were bilinearly resized from the frozen E1 image shape `3801x2121` to the evaluation shape `1900x1060` only for visual alignment.
- Frozen ROI masks were nearest-neighbor resized to the same evaluation shape.
- Error sheets show absolute RGB error. The ROI and inner-edge statistics are proxy diagnostics, not independent labels.
- Source data and per-view values are recorded in `reports/QUALITATIVE_OBSERVATIONS.json` and `reports/HELDOUT_PER_VIEW_METRICS.csv`.

## Observations

The 40% and 60% removals produce visible error concentrations around leaf boundaries and other high-frequency plant details in the fixed camera sheets. The effect is more apparent at 60% than at 40%. The two scorers produce different error patterns at the same capacity: random pruning preserves some foreground proxy signal better in the reported p40/p60 views, while opacity pruning retains higher alpha coverage in the same views. These observations do not establish a superior algorithm because the ROI is a frozen proxy and the experiment contains one scene.

At camera `0244`, the fixed foreground PSNR proxy difference between random and opacity pruning is larger than at camera `0107`; inner-edge behavior is not uniformly ordered between the two methods. This supports registering foreground-aware support as a controlled future ablation, while keeping the current baselines and capacity ratios fixed.

## Figures

- `figures/qualitative/0000_rgb_roi.png`
- `figures/qualitative/0000_error.png`
- `figures/qualitative/0107_rgb_roi.png`
- `figures/qualitative/0107_error.png`
- `figures/qualitative/0244_rgb_roi.png`
- `figures/qualitative/0244_error.png`
