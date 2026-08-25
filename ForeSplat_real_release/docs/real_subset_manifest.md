# Real Non-Full Subset Manifest

This repository includes selected real data products from the ForeSplat manuscript workspace. The subset is small by design and is intended for review of data organisation, prompt/mask outputs, workflow interfaces and paper table source data.

## `examples/real_subset_doubanlv1`

Purpose: document RAP-FSAM3 foreground-prior generation on a real manuscript example frame.

Files:

- `images/0000_raw_rgb.jpg`: real RGB frame used in the RAP-FSAM3 figure material.
- `masks/0000_P1_green_region.png`: real P1 candidate mask.
- `masks/0000_P2_plant_instance.png`: real P2 candidate/selected mask.
- `masks/0000_P3_organs.png`: real P3 candidate mask.
- `masks/0000_P4_seedling.png`: real P4 candidate mask.
- `masks/0000_P5_background_excluding.png`: real P5 candidate mask.
- `masks/0000_RAP_FSAM3_final_prior.png`: final RAP-FSAM3 foreground prior after refinement/correction.
- `foreground_rgb/0000_foreground_checkerboard.png`: real foreground-only visualisation.
- `metadata/prompt_selection.csv`: real prompt-selection log.
- `metadata/semantic_gate_scores.csv`: real semantic-gate scoring log.
- `metadata/posneg_prompt_points.csv`: real structured positive/negative prompt-point log.
- `metadata/geometry_support_prompt.csv`: real geometry-support prompt log.
- `metadata/corrective_geometry_delta.csv`: real geometry-correction delta log.

## `examples/real_subset_kongquezhu`

Purpose: document the real ForeSplat workflow around a representative reconstruction sample.

Files:

- `images/0000_raw_rgb.jpg`, `images/0025_raw_rgb.jpg`, `images/0075_raw_rgb.jpg`: real multiview RGB frames.
- `masks/0000_P2_candidate_mask.png`: real selected plant-instance candidate mask.
- `masks/0000_posneg_refined_mask.png`: real structured positive/negative refinement mask.
- `masks/0000_geometry_consistent_mask.png`: real geometry-consistency corrected mask.
- `masks/0000_RAP_FSAM3_final_prior.png`: final foreground prior.
- `reconstruction/RAP_FSAM3_2DGS_render_00000.png`: real foreground-object 2DGS render visual output.
- `reconstruction/foreground_track_render.png`: real foreground-track reconstruction visualisation.
- `reconstruction/post_pruning_gaussians.png`: real post-pruning Gaussian visualisation.
- `reconstruction/tsdf_mesh_foreground.png`: real foreground mesh visualisation.

## `examples/paper_tables`

Purpose: provide real source tables/figures behind manuscript-level experiments.

Files:

- `table6_prior_injection_ablation.csv`: real source table for the VFM-prior injection ablation.
- `table6_prior_injection_ablation_figure.png`: real figure generated from the prior-injection source table.
- `table8_phenotype_summary.csv`: real manual-virtual phenotype agreement summary.
- `phenotype_error_by_mask_quality.csv`: real mask/trait error table.
- `trait_sensitivity_ranking.csv`: real trait sensitivity ranking.
- `strict_multimask_status.csv`: real strict multimask status table.
- `table8_trait_error_profile.png`: real trait-error profile figure.

## Excluded Data

The full dataset is not included in this lightweight repository:

- raw videos and all extracted frames;
- complete sequence-level mask folders;
- complete Gaussian training outputs;
- external model weights;
- server logs and machine-specific absolute paths.

Those files will be released separately after curation, subject to storage and sharing constraints.
