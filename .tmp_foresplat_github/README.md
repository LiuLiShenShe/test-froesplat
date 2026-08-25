# ForeSplat

**ForeSplat: Vision-Foundation-Model-Guided Foreground-Object 2D Gaussian Splatting for Low-Cost 3D Plant Phenotyping**

This repository provides a lightweight reproducibility package for ForeSplat, a foreground-object 2D Gaussian Splatting workflow for low-cost 3D plant phenotyping from multiview RGB images.

ForeSplat uses VFM-derived foreground priors to reformulate plant reconstruction as a foreground-object optimisation problem. The workflow includes RAP-FSAM3 foreground-prior generation, foreground-track initialisation, foreground RGB supervision, alpha/background opacity constraints, view-quality soft weighting, lightweight M2M3-GO primitive refinement, mask-guided Gaussian pruning, TSDF-style mesh extraction and virtual phenotypic measurement.

## Repository status

This repository is released as a paper-facing reproducibility package. It contains:

- core execution scripts for the ForeSplat workflow interface;
- configuration files for foreground-prior generation, loss settings, view weighting, M2M3-GO and pruning;
- lightweight demo data showing the expected input/output format;
- scripts for rendering-quality evaluation and phenotypic measurement on demo-style outputs;
- table-level configuration files corresponding to the main experiments in the paper.

The demo data is intentionally small. It is meant to document file formats and execution entry points, not to reproduce the exact paper numbers. Full reproduction requires the complete multiview RGB dataset, manually annotated evaluation frames and third-party model installations.

## Directory structure

```text
ForeSplat/
├── README.md
├── LICENSE
├── environment.yml
├── requirements.txt
├── configs/
│   ├── default_foresplat.yaml
│   ├── rap_fsam3_prompts.yaml
│   ├── m2m3_go.yaml
│   ├── view_quality_weights.yaml
│   ├── pruning.yaml
│   ├── table6_ablation.yaml
│   ├── table7_workflow_comparison.yaml
│   └── table8_phenotype_measurement.yaml
├── scripts/
│   ├── 00_extract_frames.py
│   ├── 01_run_rap_fsam3_prior.py
│   ├── 02_prepare_colmap_inputs.py
│   ├── 03_train_foresplat.py
│   ├── 04_extract_mesh.py
│   ├── 05_evaluate_rendering.py
│   ├── 06_measure_traits.py
│   └── 07_reproduce_tables.py
├── examples/
│   ├── demo_sequence/
│   │   ├── images/
│   │   ├── masks/
│   │   ├── cameras/
│   │   └── scale_marker.json
│   ├── demo_traits.csv
│   └── demo_results/
└── docs/
    ├── data_format.md
    ├── config_reference.md
    └── reproducibility_notes.md
```

## Environment

The experiments in the paper were run using two NVIDIA RTX A6000 GPUs with 48 GB memory each and 128 GB system memory. Neural rendering methods were trained for 30,000 iterations under a matched computing environment.

Install dependencies with:

```bash
conda env create -f environment.yml
conda activate foresplat
```

or:

```bash
pip install -r requirements.txt
```

External dependencies such as COLMAP, 2DGS/3DGS implementations and SAM-series models should be installed according to their official instructions. This repository does not redistribute third-party model weights.

## Demo data

The `examples/demo_sequence/` folder provides a lightweight example of the expected data format:

```text
demo_sequence/
├── images/          # RGB frames
├── masks/           # foreground-prior masks
├── cameras/         # camera parameters or COLMAP-converted transforms
└── scale_marker.json
```

The demo data is synthetic and compact. It is intended for checking the workflow interface, configuration format and metric scripts.

## Running the workflow

### 1. Extract frames from RGB video

```bash
python scripts/00_extract_frames.py \
  --video path/to/input_video.mp4 \
  --output examples/demo_sequence/images \
  --fps 3
```

If OpenCV is unavailable, the script can also index an existing image directory:

```bash
python scripts/00_extract_frames.py \
  --image_dir examples/demo_sequence/images \
  --output examples/demo_sequence/extracted_frames
```

### 2. Generate RAP-FSAM3 foreground priors

```bash
python scripts/01_run_rap_fsam3_prior.py \
  --image_dir examples/demo_sequence/images \
  --config configs/rap_fsam3_prompts.yaml \
  --output_dir examples/demo_sequence/masks_generated
```

The prompt list, selection thresholds and refinement parameters are stored in `configs/rap_fsam3_prompts.yaml`. In full experiments this step calls a SAM-series/VFM backend. In demo mode it can generate deterministic placeholder masks so the downstream file interface can be checked.

### 3. Prepare camera poses and foreground tracks

```bash
python scripts/02_prepare_colmap_inputs.py \
  --image_dir examples/demo_sequence/images \
  --mask_dir examples/demo_sequence/masks \
  --output_dir examples/demo_sequence/cameras
```

This step prepares a camera manifest and foreground-aware sparse-track manifest. In full experiments, this step wraps or imports COLMAP outputs.

### 4. Train ForeSplat

```bash
python scripts/03_train_foresplat.py \
  --data examples/demo_sequence \
  --config configs/default_foresplat.yaml \
  --output outputs/demo_foresplat \
  --demo
```

The default configuration includes foreground RGB supervision, alpha/background opacity constraints, view-quality weighting, M2M3-GO primitive refinement and mask-guided pruning. The `--demo` flag writes a compact run manifest instead of launching a full 2DGS training backend.

### 5. Extract mesh

```bash
python scripts/04_extract_mesh.py \
  --run outputs/demo_foresplat/run_manifest.json \
  --config configs/default_foresplat.yaml \
  --output outputs/demo_foresplat/mesh
```

### 6. Evaluate rendering quality

```bash
python scripts/05_evaluate_rendering.py \
  --renders examples/demo_sequence/images \
  --references examples/demo_sequence/images \
  --masks examples/demo_sequence/masks \
  --output outputs/demo_foresplat/rendering_metrics.csv
```

### 7. Measure phenotypic traits

```bash
python scripts/06_measure_traits.py \
  --mesh outputs/demo_foresplat/mesh/plant_mesh.ply \
  --scale examples/demo_sequence/scale_marker.json \
  --output outputs/demo_foresplat/traits.csv
```

## Reproducing paper tables

Configuration templates are provided for the main table-level experiments:

```bash
python scripts/07_reproduce_tables.py --config configs/table6_ablation.yaml
python scripts/07_reproduce_tables.py --config configs/table7_workflow_comparison.yaml
python scripts/07_reproduce_tables.py --config configs/table8_phenotype_measurement.yaml
```

The released demo data is not intended to reproduce the exact numerical values in the paper. Full reproduction requires the complete dataset and manually annotated evaluation frames.

## Configuration files

The main reproducibility parameters are stored in `configs/`:

- `rap_fsam3_prompts.yaml`: foreground-prior prompt settings, prompt groups and selection rules;
- `default_foresplat.yaml`: foreground loss weights, training iterations and rendering settings;
- `m2m3_go.yaml`: mask-to-model support, model-to-mask deficit, boundary support and utility thresholds;
- `view_quality_weights.yaml`: view-quality soft weighting settings;
- `pruning.yaml`: mask-guided Gaussian pruning parameters;
- `table6_ablation.yaml`: VFM-prior injection and M2M3-GO ablation settings;
- `table7_workflow_comparison.yaml`: workflow-level reconstruction comparison settings;
- `table8_phenotype_measurement.yaml`: phenotypic measurement evaluation settings.

## Data availability

The multiview RGB sequences, manually annotated foreground-prior evaluation frames, RAP-FSAM3 masks, phenotypic measurement tables, view-weight files and main running configurations supporting this study will be released through this project repository or a data repository after data curation is complete. Before public release, the data and configuration files are available from the corresponding author upon reasonable request.

## Citation

If you use this repository, please cite:

```bibtex
@article{foresplat2026,
  title   = {ForeSplat: Vision-Foundation-Model-Guided Foreground-Object 2D Gaussian Splatting for Low-Cost 3D Plant Phenotyping},
  author  = {Author list to be updated},
  journal = {Computers and Electronics in Agriculture},
  year    = {2026}
}
```

## License

The code in this repository is released for academic research use under the MIT License. Third-party dependencies, including COLMAP, 2DGS/3DGS implementations and SAM-series models, follow their original licenses.
