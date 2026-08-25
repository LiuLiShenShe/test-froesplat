# Configuration Reference

The configuration files are compact templates matching the manuscript workflow.

## `default_foresplat.yaml`

Defines the ForeSplat workflow settings:

- foreground-track initialisation from mask-consistent sparse points;
- foreground RGB supervision;
- alpha-mask and background-opacity losses;
- view-quality weighting;
- M2M3-GO primitive refinement;
- mask-guided pruning;
- TSDF-style mesh extraction.

## `rap_fsam3_prompts.yaml`

Defines the RAP-FSAM3 prompt and refinement settings:

- five plant-related prompts;
- semantic-gated prompt selection;
- structured positive/negative refinement;
- geometry-consistency correction;
- links to the included real subset logs.

## `m2m3_go.yaml`

Documents the mask-to-model/model-to-mask Gaussian optimisation controller.

## `view_quality_weights.yaml`

Documents soft weighting based on frame quality, mask reliability and geometry reliability.

## `pruning.yaml`

Documents mask-guided Gaussian pruning thresholds and schedule.

## Table Configurations

- `table6_ablation.yaml`: prior-injection ablation design.
- `table7_workflow_comparison.yaml`: workflow-level comparison design.
- `table8_phenotype_measurement.yaml`: phenotype agreement design and source files.
