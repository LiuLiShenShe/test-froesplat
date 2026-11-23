# ForeSplat TomatoPGT batch run

Archived camera-reuse pilot. The current end-to-end experiment recomputes COLMAP from original RGB and writes to [ForeSplat_TomatoPGT_Independent](../ForeSplat_TomatoPGT_Independent/README.md). It does not resume this archived queue.

Source dataset: TomatoPGT, DOI [10.17632/72md54c7n7.1](https://doi.org/10.17632/72md54c7n7.1).
The local batch contains **42 time-point scans from three cultivars**, not 42
independent plants. All 42 contain RGB and registered native COLMAP; 40 have
the official structural/phenotype reference. `CH_07012025` and
`CH_07112025` do not have both reference files. Scans run serially on physical
GPU 1 and completed stages are reused. Progress is in
[`summary/batch_index.html`](summary/batch_index.html),
[`summary/batch_results.csv`](summary/batch_results.csv) and
[`summary/batch_progress.json`](summary/batch_progress.json).

Every completed scan contains metric reconstruction and leaflet-level mesh
candidates. SAM3 also proposes compound-leaf masks and meshes. A SAM3 candidate
is not considered identical to an official instance until explicitly matched
in that scan's `phenotype/compound_review.csv`.

The plant-height value follows the requested visible-mesh extent along the
estimated vertical, `max(dot(vertex, vertical)) - min(dot(vertex, vertical))`.
Height comparison is against the official CloudGraph reference and is
method-to-method benchmark agreement, not an independent tape-measured error.
Official compound-leaf length follows its graph-path protocol; compare it only
after explicit instance matching and confirmation of the same path definition.
Keep leaflet and whole-compound-leaf traits separate.

Primary scan: `BFS_06262025` (Beefsteak, June 26, 2025). This is a one-scan
feasibility run. The May 8 scan (`BFS_05082025`) is a separate marker check.

The primary scan completed through validation. Open its
[interactive results](BFS_06262025/summary/index.html),
[RGB-to-mesh comparison](BFS_06262025/summary/overview.png) or
[held-out silhouette comparison](BFS_06262025/summary/heldout_silhouettes.png).
The interactive index lists each candidate's source image, 3D viewer and PLY.
The mesh vertex coordinates are already in centimetres.

| Stage | Verified result |
| --- | --- |
| COLMAP | 67/67 registered, 16,030 sparse points, mean sparse reprojection error 0.577 px |
| SAM3 | 67 semantic views; 439 2D leaf proposals from 16 key views |
| GGGS | 30,000 iterations; 439,922 Gaussians; PSNR 28.34 dB on 9 held-out training views |
| Scale | 4.201591 cm per COLMAP unit; 3 targets with 13–17 inlier views each; worst target median reprojection error 1.75 px, maximum relative baseline scale deviation 1.75% |
| TSDF | 58 views integrated; 0.643 mm voxels; 696,554 scene faces |
| Semantics | 51,507 plant-only faces; 12,054 separate pot faces |
| Leaf association | 34 candidate meshes supported by 3–16 views each, with 31,575 assigned plant faces; 25 automatic surface-length candidates and no verified complete anatomical leaves |
| Validation | All 289 data checks passed; mean silhouette IoU 0.816 against SAM3 in 9 images excluded from training and fusion, but used for semantic voting |

Main files for manual inspection:

- [Plant-only centimetre mesh](BFS_06262025/plant_only/plant_only_cm.ply)
- [Full scene centimetre mesh](BFS_06262025/mesh/mesh_cm.ply)
- [Pot centimetre mesh](BFS_06262025/plant_only/pot_cm.ply)
- [Centimetre Gaussian model](BFS_06262025/gaussian/metric_gaussians.ply)
- [Leaf_001 viewer](BFS_06262025/leaf_instances/Leaf_001/viewer.html) and [PLY](BFS_06262025/leaf_instances/Leaf_001/leaf_cm.ply)
- [Leaf candidate measurements](BFS_06262025/phenotype/leaf_measurements.csv)
- [Plant measurements](BFS_06262025/phenotype/plant_measurements.csv)
- [Official point-cloud reference and provenance](BFS_06262025/summary/official_reference.json)

The main stem and most of the canopy can be seen in the plant-only comparison.
Some tips and boundaries are incomplete; two leaf candidates have fewer than
50 faces. The official reference labels 7 compound leaves (each includes its
leaflets and rachis), so the 34 candidates must not be used as the count of
compound leaves. No manual dimensions were paired and no trait errors were
computed.

The pilot originally produced a 7.79 cm pot-baseline height. The batch now
reports the requested 12.90 cm visible mesh extent for this scan. Its
comparison to the official 12.95 cm CloudGraph height is not an independently
measured error. The prior pot-baseline value is preserved in
`phenotype/plant_measurements_pot_baseline.csv`.

- `summary/progress.json`: current or most recent stage and its log path.
- `BFS_06262025/common_input/complete.json`: registered-image count, native
  feature-track provenance, and sparse reprojection statistics.
- `BFS_06262025/semantic_masks/`: SAM3 plant, pot and background masks.
- `BFS_06262025/calibration/scale_calibration.json`: scale decision and evidence.
- `BFS_06262025/gaussian/original_gaussians.ply`: final native GGGS in COLMAP units.
- `BFS_06262025/summary/index.html`: generated final overview and individual 3D
  leaf candidate viewers and PLY downloads.
- `BFS_06262025/summary/validation.json`: geometry and delivery checks.
- `attempts/`: retained results from the default undistortion and initial
  segmentation. These attempts are not included in the current run.

The benchmark's actual COLMAP feature-track poses are reused. Original JPEGs
are not modified. Fixed-canvas undistortion avoids the aggressive crop produced
by the default undistorter on this scan. GGGS uses the existing 30K/SH3/SG0
settings. The tomato SAM3 configuration keeps the independently detected
plant mask in the foreground and uses both `leaf` and `tomato leaflet` prompts.
Bounded TSDF and multi-view leaf association are reused without parameter changes.
Three-dimensional leaf candidates are not verified compound-leaf identities.
Files named `*_units.ply` have no confirmed centimetre scale.

The Metashape archive is used only as a source of target pixel observations and
reference baselines for the May 8 marker check. Its camera poses and reconstructed
mesh are not used. Board transfer to the June scan must pass independent
triangulation and reprojection checks before centimetre outputs are enabled.
Official traits derived from the reference point cloud are not paired as manual
physical measurements.

The current entry point starts or resumes the independent batch after checking
`../ForeSplat_TomatoPGT_Independent/summary/batch_progress.json`:

```bash
uv run --no-project --python experiments/COLMAP_Gaussian_Mesh_Selection/.venv-rap/bin/python \
  run_tomatopgt_pilot.py --all
```
