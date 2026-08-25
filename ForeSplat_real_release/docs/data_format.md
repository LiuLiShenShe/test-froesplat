# Data Format

ForeSplat uses sequence-level RGB frames, foreground-prior masks, camera/sparse-track information, reconstruction outputs and phenotype measurement tables.

## RGB Frames

Example:

```text
examples/real_subset_kongquezhu/images/
├── 0000_raw_rgb.jpg
├── 0025_raw_rgb.jpg
└── 0075_raw_rgb.jpg
```

These are real non-full frames. Full experiments use complete multiview sequences.

## Foreground-Prior Masks

Example:

```text
examples/real_subset_doubanlv1/masks/
├── 0000_P1_green_region.png
├── 0000_P2_plant_instance.png
├── 0000_P3_organs.png
├── 0000_P4_seedling.png
├── 0000_P5_background_excluding.png
└── 0000_RAP_FSAM3_final_prior.png
```

Masks are single-frame foreground-prior outputs aligned to their RGB frame. Candidate masks correspond to the prompt set, and the final prior is used by foreground-object 2DGS optimisation.

## RAP-FSAM3 Logs

Example:

```text
examples/real_subset_doubanlv1/metadata/
├── prompt_selection.csv
├── semantic_gate_scores.csv
├── posneg_prompt_points.csv
├── geometry_support_prompt.csv
└── corrective_geometry_delta.csv
```

These logs show how prompt candidates were scored, selected and corrected.

## Reconstruction Outputs

Example:

```text
examples/real_subset_kongquezhu/reconstruction/
├── RAP_FSAM3_2DGS_render_00000.png
├── foreground_track_render.png
├── post_pruning_gaussians.png
└── tsdf_mesh_foreground.png
```

These are visual outputs from real reconstruction runs and are included to document the downstream object that is used for measurement.

## Paper Table Data

The `examples/paper_tables/` directory contains source CSV files used for manuscript-level tables and figures. The files retain the original measured values but exclude full local output directories and raw intermediate folders.
