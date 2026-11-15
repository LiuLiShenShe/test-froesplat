# C1 GGGS → NKSR → exploratory phenotype geometry

This adapter runs only `DouBanLv1`, `HongZhang`, and `ChangShouHua3`. It reads
the existing `experiments/C1_C2_all19/<plant>/C1_raw/cleaned_gaussians.ply` and
never repeats training, segmentation, frontend reconstruction, or cleaning.

Results remain on the server under `experiments/C1_NKSR_Phenotype/`; its
`summary/index.html` links the six raw/post meshes, images, numerical tables,
resource measurements, and final conclusions. Only adapter source and the
small report/index files under `reports/C1_NKSR_Phenotype/` belong in Git.

The adapters import the validated historical implementation from
`../experiments/KongQueZhuYu/geometry_mesh_final_selection/scripts/`:

- `surface_protocol.normals_from_grid`, `unproject`, `unique_observations`
  preserve depth differences, camera-facing normal orientation, and the
  voxel + dominant-normal-direction observation definition.
- `mesh_artifacts.colour_from_observations` transfers the same nearest
  observation colours. `save_meshes` performs exactly the existing component
  operation: remove components with fewer than 50 triangles. Its writers are
  adapted to paced binary PLY/NPZ streams, without replacing geometry code.
- `evaluate_meshes.cast`, `metrics`, and `structure` retain the existing
  integer-centre raycast and structural statistics.
- `phenotype_core` supplies PCA extents, axis height, orthogonal canopy spans,
  guarded physical-unit conversion, and raw/post coordinate-subset checks.

The three plants inherit the Omega exported GGGS gauge. Every saved original
geometry camera is checked against the corrected GGGS camera through the
already recorded rigid alignment. Original depth observations are transformed
once solely to calculate the historical robust reference scale. GGGS median
depth is backprojected directly into its existing world gauge: it receives no
additional alignment or Sim(3) fit.

The reference scale uses finite original foreground depth observations with
the already saved frontend confidence threshold. Deterministic samples of
about 4,000 observations per frame give coordinate quantiles 0.005/0.995.
Their robust bbox diagonal determines sampling `/1024` and NKSR `/512`, with
no clipping and no plant-specific threshold choices. NKSR input itself uses
the existing foreground masks, native GGGS median depth, alpha >=0.5, and
relative depth discontinuity <=0.05. Original confidence does not become an
additional filter on the rendered GGGS surface.

All existing training cameras are used at the recorded training resolution
(divisor 2; max side 1920 here). Each view is processed and released. The
incremental sorted-key sampler selects the same earliest observations as the
historical full concatenation, including separate opposite normal directions.
It stores only newly selected xyz/normal/sensor/rgb, rather than every full
rendered frame or every duplicate surface observation. Only one final
`surface/nksr_input.npz` and four representative depth/normal/alpha previews
are saved. The equivalence has an executable regression check.

NKSR uses the existing private uv environment `.venv-nksr`, official `ks`
weights, and the already validated ATen/cuBLAS compatibility binary. No
installation, compilation, network/weight changes, or new meshing method are
performed. Fixed parameters are `detail_level=None`, `voxel_size=diagonal/512`,
`solver_max_iter=2000`, `solver_tol=1e-5`, `approx_kernel_grad=False`,
`fused_mode=True`, and `extract_dual_mesh(mise_iter=1, trim=True,
max_points=1000000)`. The existing expandable-segment allocator setting is
recorded as memory management, not a geometry parameter.

Every stage runs in a fresh process. GPU stages use physical GPU1 only;
plants and stages run serially. One shared 20 MiB/s application limiter per
stage covers large PLY/NPY/NPZ reads/writes, weight loading, and the image
streams used by the adapter. Kernel writeback/device peaks and other processes
are outside this application limiter and appear separately in `/proc` data.
Surface processing checks a 16 GiB RSS guard; other stages use 24 GiB, and
both require at least 10 GiB system-available memory at check boundaries.
These guards are not OS quotas inside NKSR's native solver. Torch allocated/
reserved GPU peaks and sampled PID-specific GPU use are reported separately.

Mesh comparison uses four actual observed cameras from the prior all19
evaluation, at max side 960. All rows share those cameras; the rendered rows
use gray 0.5 and the original capture RGB retains its actual background.
`references.py` attaches the matching original `02-FFT` photographs after
checking saved full-canvas preprocessing metadata and foreground RGB agreement.
It preserves the black-background training references separately and records
its own CPU/I/O usage. Mesh colouring comes from historical observation transfer;
supplementary normal-lit images expose geometry. A raycast is opaque first-hit
geometry, whereas GGGS is alpha compositing. Neither projection is a 3D scan.

Exploratory dimensions include the whole foreground mesh and may include pot,
blue reference objects, or reconstruction artifacts. Whole-mesh area is not
leaf area. PCA major-axis height and minor-plane spans are
explicit proxies, not gravity-aligned plant height/canopy width. Without
plant-only labels, a vertical direction, plant base, specimen-linked physical
calibration, or leaf IDs/landmarks, formal and centimetre traits stay empty
with the corresponding status. No pot size or historic workbook value is used
as an assumed calibration.

From this directory, run the resumable three-plant driver:

```bash
uv run --no-project --python ../.venv/bin/python python run_c1_nksr.py
```

After reviewing the actual matched images, measured tables and the final
index are assembled without any reconstruction:

```bash
uv run --no-project --python ../.venv/bin/python python -m nksr_pipeline.report
```

The final conclusions also require `summary/visual_reviews.json`, containing
direct image inspection notes for these three plants; numerical statistics
alone do not generate a geometry-quality verdict.

Individual surface adapters accept the requested path interface:

```bash
uv run --no-project --python ../.venv/bin/python python -m nksr_pipeline.surface \
  --scene configs/scenes/DouBanLv1.json \
  --input-ply experiments/C1_C2_all19/DouBanLv1/C1_raw/cleaned_gaussians.ply \
  --output experiments/C1_NKSR_Phenotype/DouBanLv1
```

The execution protocol/input stamps are recorded before running. Finished
stages are retained. Actual commands and per-stage logs/resources remain in
the experiment directory. Numerical and visual conclusions refer to this run,
not to historical KongQueZhuYu timing or earlier C1 cleanup costs.

Validation:

```bash
../.venv/bin/python -m pytest tests/nksr_pipeline/test_adapters.py -q
```

Checks cover exact historical first-observation equivalence across duplicate
and opposing observations, corrected world-coordinate backprojection, and
byte-preserving mesh geometry through paced PLY and native Open3D readers.
