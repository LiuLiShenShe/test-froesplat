# Raw/post phenotype comparison

The E3 post rule retains the single largest face-connected component. This
comparison treats that rule as an algorithmic transformation, not as a ground
truth segmentation.

| Scene | Raw vertices/faces | Post vertices/faces | Face deletion | Vertex deletion | Interpretation |
|---|---:|---:|---:|---:|---|
| WanNianQing1 | 1,271,632 / 2,388,092 | 1,135,076 / 2,241,526 | 6.137% | 10.739% | Moderate deletion; possible edge/top loss remains unresolved |
| CaoMei2 | 792,827 / 1,226,040 | 438,283 / 815,120 | 33.516% | 44.719% | Large deletion; post cannot be assumed phenotype-complete |
| HongZhang | 2,575,549 / 4,296,140 | 1,672,621 / 3,293,400 | 23.340% | 35.058% | Large deletion with plant and inflorescence overlap in projection evidence |

Raw/post alignment passed for all three scenes: every post vertex is an exact
coordinate member of raw, and the post face count equals the raw largest
component count. No affine transform was introduced between stages.

The raw Mesh is structurally more complete by direct counts in every scene.
Post may be more convenient for algorithms requiring a single connected body,
and E3 structural diagnostics show cleaner connectivity, but that convenience
does not establish better plant phenotyping. In particular:

1. A deleted component can be a real disconnected leaf, leaf tip, petiole,
   inflorescence, or a background artifact.
2. Alpha-foreground projection fractions do not distinguish those cases because
   the input RGB is not an organ-level segmentation.
3. A reduced PCA extent after post-processing is not evidence of a more accurate
   biological height or canopy width.
4. M0/M1/M2 differences are algorithm sensitivity evidence, not measurement
   error or independent phenotype truth.

Accordingly, this phase reports raw/post geometric proxies and deletion audits,
but does not select a post Mesh as a validated phenotype reference. A future
measurement run may use raw for recall and post for a connected-body sensitivity
condition, provided plant-only segmentation and physical/frame references are
available.
