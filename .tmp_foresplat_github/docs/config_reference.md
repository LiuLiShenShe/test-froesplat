# Configuration Reference

The `configs/` directory contains templates for the main paper workflow and table-level experiments.

## `default_foresplat.yaml`

Defines the foreground-object 2DGS workflow:

- foreground-track initialisation from mask-consistent sparse points;
- foreground RGB supervision;
- alpha-mask and background-opacity losses;
- view-quality soft weighting;
- M2M3-GO primitive-level refinement;
- mask-guided Gaussian pruning;
- TSDF-style mesh extraction settings.

## `rap_fsam3_prompts.yaml`

Defines the RAP-FSAM3 foreground-prior interface:

- FFT-based frame-quality filtering;
- five plant-related prompts;
- semantic-gated prompt selection;
- structure-balanced positive/negative refinement prompts;
- geometry-consistency foreground correction;
- output logs for prompt selection and semantic gate scores.

## `m2m3_go.yaml`

Controls primitive-level refinement:

- mask-to-model foreground support;
- model-to-mask coverage deficit;
- boundary support;
- utility thresholds for keep/densify/prune decisions;
- optional Floor40 local retention safeguard.

## `view_quality_weights.yaml`

Defines the soft weighting of views using:

- high-frequency saliency;
- mask reliability;
- geometry reliability;
- clamped min-max normalisation.

## `pruning.yaml`

Defines mask-guided Gaussian pruning:

- outside-mask ratio threshold;
- low-opacity threshold;
- weak foreground-support threshold;
- scheduled pruning iterations;
- boundary/sparse-track preservation rules.

## Table configuration files

- `table6_ablation.yaml`: VFM-prior injection and M2M3-GO ablation
- `table7_workflow_comparison.yaml`: COLMAP/3DGS/2DGS/ForeSplat workflow comparison
- `table8_phenotype_measurement.yaml`: manual-virtual phenotype agreement analysis

These files document the experimental design and expected metrics. They are not substitutes for the full curated dataset.
