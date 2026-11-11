# Development version record

v1 candidates reduced total background mass but failed measured leaf-boundary coverage and local dark-foreground/geometry constraints on KongQueZhuYu. No heldout cleaner was run. These candidates were rejected and retained.

v2 applies one shared change: A2 must have no measured boundary/thin protection; background-dominant Gaussians with actual sensitive foreground contribution become A3 for bounded covariance optimization. This is a feature rule for every plant, not a Kong ROI, ID or world-space threshold. Six meaningful tests pass, including the high-background boundary-support regression. Source/config hashes are re-frozen under all_19_plants_v2. Previous v1 artifacts remain unchanged.

Independent heldout set remains the same 15 plants; no heldout outputs informed this change. Method benefit is still pending measured C/B and cross-plant results.


v3: among v2 A2, 44 Gaussians had >=2 actual footprint depth-support views. Whole-point invalidity was inconsistent with this evidence. The same global A2 rule now excludes multi-view depth-supported/occluded contributors; after a quality-rejected prune, actual foreground contributors are offered constrained local refinement. Thresholds and all plant input data stay unchanged. Seven native CUDA tests pass. No heldout output has been read. v2 remains archived under its original output root.
