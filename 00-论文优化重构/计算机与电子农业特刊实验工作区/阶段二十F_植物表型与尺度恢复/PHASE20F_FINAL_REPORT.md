# ForeSplat Phase20F final report

## Final status

```text
PHASE20F STATUS:
PARTIAL

REAL_MESH_ANALYZED:
3/3

RAW_POST_ANALYSIS:
PASS

PHYSICAL_SCALE:
NOT_AVAILABLE

VERTICAL_REFERENCE:
NOT_AVAILABLE

PLANT_HEIGHT:
GEOMETRIC_ONLY

CANOPY_WIDTH:
GEOMETRIC_ONLY

LEAF_LENGTH:
BLOCKED

LEAF_WIDTH:
BLOCKED

HISTORICAL_DATA_PAIRING:
PARTIAL

TABLE8_STATUS:
REPLACEMENT_REQUIRED

NEXT_PHASE:
USER_DATA_REQUIRED
```

## What was actually measured

The runner analyzed the fixed M0 models: WanNianQing1/B_R0/Seed0, CaoMei2/B_R0/Seed0,
and HongZhang/B-C shared C_R1/Seed0. All six raw/post Mesh hashes matched the
Phase20E-3 structure index. Raw/post coordinate alignment passed, and each post
face count matched the largest raw face-connected component.

| Scene | Raw vertices/faces | Post vertices/faces | Components | Removed faces | Removed vertices |
|---|---:|---:|---:|---:|---:|
| WanNianQing1 | 1,271,632 / 2,388,092 | 1,135,076 / 2,241,526 | 20,147 | 6.137% | 10.739% |
| CaoMei2 | 792,827 / 1,226,040 | 438,283 / 815,120 | 41,555 | 33.516% | 44.719% |
| HongZhang | 2,575,549 / 4,296,140 | 1,672,621 / 3,293,400 | 126,275 | 23.340% | 35.058% |

The per-component records and hashes are in `REMOVED_STRUCTURE_AUDIT.csv`. The
projection figures use the first, middle, and last frozen RGB camera selected by
the fixed protocol. They show removed regions overlapping foreground-alpha
pixels, but alpha does not identify leaf organs or distinguish plant from pot.

The geometry CSV contains 12 exploratory rows. They are PCA extents of the whole
reconstructed Mesh, in inherited COLMAP arbitrary units, with
`physical_trait_claim=False`. They are not plant height, canopy width, or
physical measurements.

## Answers to the required questions

1. **Are the three Meshes suitable?** They are suitable for reproducible raw/post
   structural and measurement-interface checks. None is presently suitable for
   validated centimetre phenotyping. Raw preserves more structure; post is a
   connected-body sensitivity condition.
2. **Which retains more plant structure?** Raw, by vertices and faces in all
   scenes. The biological identity of deleted components is unresolved.
3. **Was a true scale found?** No. Phase20B audits, workbook schema, E3
   manifests/logs, and Mesh records contain no verified marker, pot dimension,
   correspondence, or COLMAP-to-physical factor.
4. **Was vertical direction found?** No. Camera poses do not define gravity; no
   ground, pot-rim, or recorded gravity plane was recovered.
5. **Can height be measured?** Only as an exploratory geometric proxy. Formal
   height is blocked by missing plant-only points, base, vertical axis, and scale.
6. **Can canopy width be measured?** Only as an exploratory PCA proxy. Formal
   horizontal projection and manual direction pairing are blocked.
7. **Which leaves can be measured?** No leaf has a verified persistent organ ID,
   base/tip landmark, or boundary ownership in the recovered evidence. The
  leaf-length/width methods are implemented and guarded, but no organ-level
  measurement is currently possible; all six scene-stage reports are blocked.
8. **Can 21 plants/63 leaves be paired?** No exact pairing. The workbook has two
   WanNianQing candidates, two CaoMei candidates, and one HongZhang cultivar-only
   candidate. Matching is `AMBIGUOUS` for WanNianQing/CaoMei and
   `SAMPLE_MATCH_ONLY` for HongZhang, never exact plant/time identity.
9. **How many valid centimetre values were computed?** Zero.
10. **Which MAE/RMSE/R2 have new evidence?** None. The new error CSV is header
    only. Existing Table 8 numbers remain legacy recorded workbook statistics.
11. **Which traits cannot yet be quantified?** All four as validated physical
    traits; leaf traits additionally lack organ identity and landmarks.
12. **What should happen to Table 8?** Keep old values separate as legacy
    statistics with their provenance caveat; require a new replacement table for
    Mesh-based validation. Do not silently replace them with geometry proxies.
13. **What data is needed?** Specimen/time bridge, physical reference and
    uncertainty, gravity/base frame, trait definitions, leaf IDs/landmarks, and
    independent same-time manual validation values.
14. **What can be written next?** The guarded measurement method, raw/post
    component audit, no-scale limitation, and provenance findings can be written
    into Methods/Limitations. New accuracy claims require the missing data.

## Acceptance gates

G1, G2, G3, G6, G7, G8, G9, G10, G11, G12, G15, G16, G17, G18, and G19 are
satisfied as evidence or protection gates. G4, G5, G13, and G14 correctly stop
at `NOT_AVAILABLE`/`BLOCKED` rather than generating illegal physical values.
The overall phase is therefore `PARTIAL`, not a four-trait validation pass.

## Reproduction and protected inputs

Run:

```bash
/data/fj/F2DMAS/2d-gaussian-splatting-main/venv/bin/python \
  '00-论文优化重构/计算机与电子农业特刊实验工作区/阶段二十F_植物表型与尺度恢复/scripts/run_phase20f.py'
```

The complete input hash map and status fields are in `logs/EXECUTION.json`.
The historical workbook was read and hash-checked but not modified. No source
Mesh, Gaussian PLY, camera JSON, COLMAP pose, model/training file, or `.tex`
file was modified by this phase.

## Verification

The phase test suite passed with 12 tests: guarded geometry math, rotation and
scale invariance, invalid-reference rejection, leaf-identity enforcement,
raw/post coordinate identity, output row counts, component-audit hashes, and
the no-physical-claim execution log. The command used was:

```bash
PYTHONPATH='00-论文优化重构/计算机与电子农业特刊实验工作区/阶段二十F_植物表型与尺度恢复' \
/data/fj/F2DMAS/2d-gaussian-splatting-main/venv/bin/python -m unittest discover \
  -s '00-论文优化重构/计算机与电子农业特刊实验工作区/阶段二十F_植物表型与尺度恢复/tests' -p 'test_*.py' -v
```
