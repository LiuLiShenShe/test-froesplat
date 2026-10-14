# Next-stage decision (not execution authorization)

Status: PASS; R1 pilot evidence: MIXED.

Before adopting R1 as a formal reconstruction component, add independent
changed-input scenes and at least several predeclared seeds. Reuse frozen input
recipes, add independently defined held-out plant contours, and distinguish
scene-level effects from related validation views. Do not automatically retrain
all historical scenes; prioritize the two changed-input scenes plus additional
independent difficult scenes under a prospective protocol.

Technically valid final models permit planning a separate M2M3-GO/mesh stage,
but do not establish that R1 must be its default input. If the pilot is mixed or
negative, keep R0 as control and carry the uncertainty forward. Geometry/mesh
quality requires new independent assessment, not appearance metrics alone.
No M2M3-GO change, TSDF/mesh extraction or phenotype measurement runs here.

Recovery: never resume from native incomplete checkpoints without random-state
and camera-stack evidence. Native loop checkpoints omit those, so an interruption
requires a preserved failed attempt plus a full same-config seed-0 rerun only
after documented authorization/cost; no alternative seed or tuning is allowed.

Reproduction: checkout the source/config commit in TRAINING_RUN_MANIFEST.csv;
restore external assets by reports/LARGE_ARTIFACT_MANIFEST.csv relative paths
and hashes; restore frozen camera symlinks; mount/link the data at the recorded
root if necessary. Absolute paths are conveniences, not sole provenance.
Rebind a different physical GPU only under an explicitly recorded new-hardware
reproduction protocol, never silently bypass the original UUID gate.
Full float renders allow shared-reference metric recomputation without retraining.

Three scenes and one fixed seed form a feasibility pilot, not broad generalization.
Held-out views are correlated and are not independent experimental repetitions.
No independent 3D GT exists: no truth Chamfer/F-score, physical dimensions or mesh.
All 68 frozen validation views lack matching existing manual fixed contours.
Frozen R0 proxy ROIs are predicted labels, not GT; fixed central rectangles are
method-independent descriptive spatial ROIs, not guaranteed plant-only regions.
CaoMei2 uses historically foreground-conditioned COLMAP cameras, shared by A/B/C.
CaoMei2 TEST32 raw0146 and WanNianQing1 raw0227 are not registered improvement cases.
Actual treatment switches: WanNianQing1 raw0037/raw0038/raw0165; CaoMei2 raw0139.
HongZhang B/C uses the same checkpoint; it cannot establish an R1-versus-R0 benefit.
Input treatment is explicit black-RGB precomposition only; no alpha-weighted RGB
loss or mask-guided sparse initialization is enabled. A is not historical ForeSplat.
CUDA rasterization/densification may not be bitwise deterministic across repeated
runs; one-seed differences cannot estimate this variability.
