# Reproducibility Notes

This repository is designed to make the ForeSplat workflow inspectable without uploading uncurated raw videos, large intermediate files or third-party model weights.

## What is included

- A clear repository structure for code, configs, examples and docs
- Configuration templates corresponding to the paper workflow and main tables
- Demo RGB/mask/camera data showing expected input formats
- Lightweight scripts for the workflow entry points
- Demo-mode outputs for checking evaluation and trait-measurement interfaces

## What is not included

- Large raw multiview videos
- Uncurated experiment intermediates
- Server logs or machine-specific absolute paths
- SAM-series or other third-party model weights
- Full paper-number reproduction data before curation is complete

## External dependencies

Full experiments require external installations:

- COLMAP for camera poses and sparse tracks
- A 2DGS/3DGS implementation for Gaussian optimisation
- SAM-series or other VFM backends for promptable foreground priors
- Optional perceptual metric backends for LPIPS

The scripts in this repository keep the interfaces to those systems explicit. Demo mode writes manifests and compact placeholder outputs so that reviewers can inspect the data organisation and execution flow.

## Recommended release path

1. Keep this lightweight package as the public first version.
2. Add a curated demo sequence with real RGB frames and masks when publication policy allows.
3. Add Zenodo or institutional repository links for the complete multiview RGB dataset.
4. Pin third-party commit hashes and environment versions for the camera, VFM and Gaussian backends.
5. Update the citation entry after the paper receives final bibliographic information.
