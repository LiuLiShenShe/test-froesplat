# ForeSplat

**ForeSplat: Vision-Foundation-Model-Guided Foreground-Object 2D Gaussian Splatting for Low-Cost 3D Plant Phenotyping**

This repository is a paper-facing reproducibility package for ForeSplat. It contains code-entry templates, configuration files and a **real but non-full** subset of the data used to prepare the manuscript figures and tables.

The included example data are not synthetic placeholders. They are selected from the project's real RAP-FSAM3, ForeSplat, reconstruction and phenotypic-measurement outputs. The subset is intentionally small so reviewers can inspect file formats, prompts, masks, metrics and workflow entry points without downloading the full multiview RGB dataset or uncurated experiment workspace.

## What Is Included

- Real RGB frames and RAP-FSAM3 foreground-prior masks from selected manuscript examples.
- Real prompt-selection, semantic-gate, positive/negative prompt and geometry-correction CSV logs for one RAP-FSAM3 sample.
- Real ForeSplat/2DGS reconstruction visual outputs from one representative workflow sample.
- Real source tables used for prior-injection ablation and manual-virtual phenotype agreement.
- Configuration templates matching the paper workflow.
- Lightweight scripts for inspecting data, validating paired image/mask files, summarising metrics and running the documented workflow interfaces.

## What Is Not Included

- Full raw videos or complete multiview RGB sequences.
- SAM-series model weights or other third-party model weights.
- Uncurated server logs, absolute local paths or large intermediate training folders.
- A one-command full reproduction of all manuscript numbers.

Full reproduction requires the complete curated dataset, external VFM/SAM installation, COLMAP and the selected 2DGS/3DGS backend.

## Repository Structure

```text
ForeSplat/
├── configs/
│   ├── default_foresplat.yaml
│   ├── rap_fsam3_prompts.yaml
│   ├── m2m3_go.yaml
│   ├── view_quality_weights.yaml
│   ├── pruning.yaml
│   ├── table6_ablation.yaml
│   ├── table7_workflow_comparison.yaml
│   └── table8_phenotype_measurement.yaml
├── docs/
│   ├── data_format.md
│   ├── real_subset_manifest.md
│   ├── config_reference.md
│   └── reproducibility_notes.md
├── examples/
│   ├── real_subset_doubanlv1/
│   ├── real_subset_kongquezhu/
│   └── paper_tables/
├── scripts/
│   ├── inspect_real_subset.py
│   ├── validate_image_mask_pairs.py
│   ├── summarize_table_metrics.py
│   ├── prepare_colmap_manifest.py
│   ├── train_foresplat_entry.py
│   ├── extract_mesh_entry.py
│   └── measure_traits_entry.py
├── environment.yml
├── requirements.txt
└── LICENSE
```

## Real Subsets

### `examples/real_subset_doubanlv1/`

This subset documents the RAP-FSAM3 foreground-prior generation process for a real frame:

- `images/0000_raw_rgb.jpg`
- five prompt candidate masks in `masks/`
- final RAP-FSAM3 foreground prior in `masks/0000_RAP_FSAM3_final_prior.png`
- foreground-only visual output in `foreground_rgb/`
- prompt-selection, semantic-gate and geometry-correction logs in `metadata/`

### `examples/real_subset_kongquezhu/`

This subset documents the ForeSplat workflow around one representative real sample:

- three real multiview RGB frames in `images/`
- foreground-prior and refinement masks in `masks/`
- real reconstruction/render/mesh visual outputs in `reconstruction/`

### `examples/paper_tables/`

This folder contains real source data behind manuscript-level tables/figures:

- `table6_prior_injection_ablation.csv`
- `table6_prior_injection_ablation_figure.png`
- `table8_phenotype_summary.csv`
- `phenotype_error_by_mask_quality.csv`
- `trait_sensitivity_ranking.csv`
- `strict_multimask_status.csv`
- `table8_trait_error_profile.png`

## Environment

```bash
conda env create -f environment.yml
conda activate foresplat
```

or:

```bash
pip install -r requirements.txt
```

The inspection scripts only need common Python packages. Full ForeSplat training additionally requires external COLMAP, SAM/VFM and 2DGS/3DGS installations.

## Quick Checks

Inspect the included real subset:

```bash
python scripts/inspect_real_subset.py --root examples
```

Validate image/mask dimensions:

```bash
python scripts/validate_image_mask_pairs.py \
  --image examples/real_subset_doubanlv1/images/0000_raw_rgb.jpg \
  --mask examples/real_subset_doubanlv1/masks/0000_RAP_FSAM3_final_prior.png
```

Summarise manuscript table data:

```bash
python scripts/summarize_table_metrics.py \
  --table examples/paper_tables/table6_prior_injection_ablation.csv
```

## Workflow Entry Points

The scripts under `scripts/` are intentionally lightweight and explicit:

- `prepare_colmap_manifest.py` prepares a small image/mask manifest for COLMAP or a COLMAP-converted backend.
- `train_foresplat_entry.py` writes a reproducible training command manifest for an external ForeSplat/2DGS backend.
- `extract_mesh_entry.py` writes a mesh-extraction command manifest.
- `measure_traits_entry.py` documents the trait-measurement input/output interface.

They are entry points and format checks, not bundled third-party training code.

## Data Availability

The full multiview RGB sequences, complete manually annotated evaluation frames, full RAP-FSAM3 masks, complete phenotypic measurement tables, view-weight files and run configurations will be released after curation through this repository or an associated data repository. Before public release, they are available from the corresponding author upon reasonable request.

## Citation

```bibtex
@article{foresplat2026,
  title   = {ForeSplat: Vision-Foundation-Model-Guided Foreground-Object 2D Gaussian Splatting for Low-Cost 3D Plant Phenotyping},
  author  = {Author list to be updated},
  journal = {Computers and Electronics in Agriculture},
  year    = {2026}
}
```
