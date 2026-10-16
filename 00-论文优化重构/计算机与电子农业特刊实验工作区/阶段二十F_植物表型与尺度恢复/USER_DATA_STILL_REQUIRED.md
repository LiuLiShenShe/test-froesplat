# User data still required

Phase20F is executable for geometry and provenance checks but cannot produce
validated physical phenotypes with the current repository. The following
existing-record recovery items are required; no TEST32 image relabelling is
required by this phase.

## 1. Identity and time bridge

For each reconstructed scene, provide a stable specimen ID, capture date/time,
the exact image sequence or camera manifest, and the corresponding manual
measurement record. The bridge must identify the exact Mesh/checkpoint or a
deterministic mapping to it. Cultivar text alone is insufficient because the
workbook has two WanNianQing rows and two CaoMei rows.

## 2. Physical scale

Provide at least one independently known distance for each applicable scene:

- marker endpoints with real length and their image/3D correspondence;
- a verified pot rim diameter with the exact rim points used in the Mesh; or
- another rigid reference whose endpoints are preserved in the reconstruction.

Also provide the recovered scale factor, unit, endpoint-selection uncertainty,
and the source file or notebook page. A manual plant height is not acceptable as
the scale source for validating that same plant height.

## 3. Reference frame

Provide gravity, ground/bench, pot-rim, or equivalent reference geometry and the
transform into the frozen COLMAP frame. State whether height starts at ground,
pot rim, or plant base. State the two canopy directions if manual canopy width
is directional.

## 4. Trait definitions

Freeze whether height is a top-to-base projection, whether canopy width is a
maximum span or two named orthogonal spans, whether leaf length is straight or
curved, whether the petiole is included, and how leaf width's cross-section is
chosen. Record manual tool and repeatability where available.

## 5. Leaf identity evidence

For every measured leaf, provide a persistent leaf ID, image/3D correspondence,
base and tip landmarks, and the boundary or width-plane evidence. Existing
archived labels/landmarks are sufficient; the current phase must not invent IDs
from visually similar surfaces.

## 6. Independent validation records

Provide same-plant, same-time manual measurements with units, missing-value
reasons, and repeated measurements if available. The same values used to define
the scale cannot be the only validation values. With only three valid pairs,
report descriptive errors and avoid broad generalisation; with no valid pairs,
leave the metrics empty.

The current blocker inventory is recorded in
`PHYSICAL_SCALE_PROVENANCE.csv`, `GRAVITY_AND_REFERENCE_FRAME.csv`,
`LEAF_MEASURABILITY_REPORT.csv`, and `HISTORICAL_PHENOTYPE_MATCHING.csv`.
