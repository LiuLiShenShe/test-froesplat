# Raw/post phenotype suitability

## Decision

Raw retains more reconstructed structure in all three scenes. Post has one
connected component and a cleaner main body, but E3 did not establish that the
discarded components are background. The removed-component projections fall on
plant, pot, and background regions, so post is not automatically the more
phenotype-correct Mesh.

| Scene | Raw components | Removed faces | Removed vertices | Candidate use | Main blocker |
|---|---:|---:|---:|---|---|
| WanNianQing1 | 20,147 | 6.137% | 10.739% | Raw exploratory height/canopy coverage; post main-body comparison | No scale, gravity, base, or plant-only mask |
| CaoMei2 | 41,555 | 33.516% | 44.719% | Raw is the safer coverage candidate; post is only a reduced-body comparison | Strong camera/world tilt; no gravity or plant-only mask |
| HongZhang | 126,275 | 23.340% | 35.058% | Raw exploratory coverage including leaves/inflorescences; post main-body comparison | No scale, base, organ identity, or plant-only mask |

The exact per-component audit is in
`REMOVED_STRUCTURE_AUDIT.csv`. Each scene's `SUMMARY` row stores the raw and
post SHA256 values; component rows store the raw face-connected `component_id`,
triangle count, centroid and centroid bounding box. RGB projection records are
in `logs/REMOVED_COMPONENT_RGB_PROJECTION.csv`.

## Scene observations

### WanNianQing1

The raw Mesh has 1,271,632 vertices and 2,388,092 faces. Post has 1,135,076
vertices and 2,241,526 faces. Removed face-centroid samples land partly on
visible leaf regions and partly on pot/background regions in the three fixed
RGB views. The raw Mesh is the more complete structure candidate. Post can be
used for a main-body stability comparison, but its top and edge coverage is not
validated as complete.

### CaoMei2

The raw Mesh has 792,827 vertices and 1,226,040 faces. Post has 438,283
vertices and 815,120 faces, the largest proportional deletion among the three
scenes. The fixed source views show a strongly tilted camera-oriented plant.
The removed regions include projections that overlap foreground alpha, so the
deletion cannot be interpreted as only floating noise. Raw is preferred for
coverage exploration; neither raw nor post supports formal height or canopy
measurement without a recovered frame and plant mask.

### HongZhang

The raw Mesh has 2,575,549 vertices and 4,296,140 faces. Post has 1,672,621
vertices and 3,293,400 faces. The source RGB contains leaves, inflorescences,
and the pot. Removed projections overlap visible plant and pot areas in the
fixed views. Post is useful as a connected-body sensitivity view, not as a
validated complete plant representation.

## Trait-specific suitability

- **Plant height:** no formal candidate is accepted. PCA major-axis values are
  exploratory proxies only; the axis is not gravity and the whole Mesh includes
  pot/artifacts.
- **Canopy width:** no formal candidate is accepted. Horizontal projection and
  direction are not established; PCA minor-plane extents are proxies only.
- **Leaf length:** leaf-like surfaces are visible, but no instance IDs,
  base/tip landmarks, or boundary ownership are available.
- **Leaf width:** the same identity and boundary deficiencies apply, with the
  additional need for a declared maximum-width cross-section.

The six generated figures show raw, post, and deleted components in a common
scene view, plus deleted face-centroid samples projected with frozen RGB camera
poses. They are diagnostic evidence, not manual leaf annotations.
