# Reproducibility Notes

This repository is meant to support manuscript review by exposing the structure and real evidence chain of the method without uploading the full experiment workspace.

## Why The Data Are Non-Full

The full project contains raw videos, complete multiview sequences, complete mask folders, Gaussian optimisation outputs and large intermediate files. Uploading them directly would be noisy and would include uncurated files. This repository therefore contains a curated real subset.

## How To Interpret The Included Data

- The image and mask files are real outputs used in manuscript figure preparation.
- The CSV files under `metadata/` are real RAP-FSAM3 logs.
- The CSV files under `paper_tables/` are real source tables for manuscript-level results.
- The reconstruction images are real visual outputs from the ForeSplat workflow.

The subset is sufficient to inspect data formats, prompt/mask behaviour, table values and the workflow interface. It is not intended to reproduce all manuscript numbers alone.

## External Dependencies

Full reproduction requires:

- COLMAP for camera pose and sparse-track estimation;
- SAM-series or equivalent VFM backends for promptable segmentation;
- a 2DGS/3DGS training backend;
- the full curated multiview RGB dataset and annotations.

This repository does not redistribute third-party model weights.
