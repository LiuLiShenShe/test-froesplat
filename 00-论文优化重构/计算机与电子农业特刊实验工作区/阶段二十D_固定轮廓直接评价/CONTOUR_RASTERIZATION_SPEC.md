# Fixed-contour rasterization evaluation — protocol v1

Frozen before TEST32 accuracy computation on 2026-10-12. This is an evaluation using
deterministically rasterized manual contours, not certification by the original
two-person human GT Gate. The user's statement of completed human review is retained
as a user statement; missing archived signatures are not supplied or inferred.

## Coordinates and region

- Grid: width2160, height3840. Read original JSON bytes and original vertex order.
- Only exact `potted_plant` and real `blue_cube` labels contribute; `plant`/`pot` do not.
- Accept linestrip/polygon point sequences for this interpretation only. Require at
  least three finite distinct non-collinear 2D vertices, also after rounding. A zero
  signed shoelace area is NOT grounds to reject a non-collinear self-intersecting shape.
- Rounding: float64; half-away-from-zero, `sign(x)*floor(abs(x)+0.5)`; int32 range
  checked. No coordinate clipping, point deletion, reordering, simplification or repair.
  The drawing library naturally limits coverage to the image canvas.
- Primary: OpenCV4.13.0 `cv2.fillPoly(canvas,[points],255,lineType=cv2.LINE_8,shift=0)`.
  Draw every shape in a separate call onto the same zero uint8 canvas: this is union,
  not the even/odd cancellation that a single multi-contour call could introduce.
- Secondary: Pillow12.3.0 `ImageDraw.polygon(points,fill=255,outline=None,width=1)` on
  an L-mode canvas, separate calls per shape. Same integer vertices, implicit closure,
  no antialiasing, smoothing, convex hull or morphological operation.
- `GT_eval = filled(potted_plant) AND NOT filled(blue_cube_if_present)`. No cube label
  means no subtraction. An empty/invalid target is an error, never an empty surrogate GT.
- Store PNG0/255, byte SHA256, decoded uint8 pixel SHA256, source/RGB SHA256, areas,
  warnings, full original and rounded vertex mappings; source JSON remains byte-identical.

## Geometry and sensitivity

Recompute the existing V2 numeric geometry audit read-only for all75 shapes and bind
its historical source hashes. Mark any previously/recurrently intersecting or backtracking
shape's frame `GEOMETRY_AMBIGUITY`, including ignored component labels. Report separately
whether a potted_plant/blue_cube shape itself is affected. Keep all32 in the primary table;
the13/19 subsets are descriptive, never replacement validation sets.

Predeclared descriptive "noticeable raster disagreement": cross-raster maskIoU<0.99
OR absolute area difference / OpenCV area >0.01. This is not GT certification.
Report every frame even below this threshold. Record R1−R0 direction per metric,
aggregate/subset, switch and frame. Any reversal in F1/IoU method ranking or switch
direction => `RASTERIZATION_SENSITIVE`; no such reversal => `STABLE` for these two
implementations only. Numerical differences remain reported. Stable algorithms do
not demonstrate that a self-intersecting contour represents the intended target.

## Frozen metrics and algorithm

Use existing Phase20B `evaluate_arrays` via Phase20C `load_segmentation_metrics`;
F1, IoU_fg, mIoU, approximate HD95 and Boundary-F1(3 original pixels) unchanged.
Leakage=FP/(2160*3840), not RGB radiation energy. Oracle=max raw candidateF1;
oracle_regret=oracle−raw selectedF1 (not cleanupF1). Also report finalF1 and raw metrics.
Severe failure proxy is F1<0.5, not an invented human identity approval.
SD is population ddof0. Main mean is sample-equal; frame-equal, median, SD and paired
R1−R0 are separate outputs. All32=8 sequences×4 frames; no DEV GT10 mixing.

Production configurations are identical except ranking_policy=base/r1 and necessary
output paths. P6/per_instance/single, original71 candidates, strict archived metadata,
frozen score weights; SPNP/Residual/GCFC/A6/A7 OFF. Do not change historical cleanup
settings or any other frozen parameter. Save raw selection and final production output.

## Engineering decision rule (predeclared)

An engineering lock is feasible, not a certification of accurate contours, when frozen
runs/provenance are complete, both rasterizers' full32 F1/IoU paired means are not
negative, no new F1<0.5 failures or empty outputs are introduced, and rankings are
stable for both switches and the13-frame subset. Other metrics and all regressions
must still be disclosed. If feasibility checks fail, use PROVISIONAL or BLOCKED;
never retune the same R1 and call it independent validation.

## Documentation sources

Current documentation retrieved with Context7:
- https://docs.opencv.org/4.13.0/d1/d4f/imgproc_2include_2opencv2_2imgproc_8hpp.html
- https://docs.opencv.org/4.13.0/d3/d96/tutorial_basic_geometric_drawing.html
- https://github.com/python-pillow/pillow/blob/main/docs/reference/ImageDraw.rst

Bocha search returnedHTTP403; online search did not supply experiment evidence.
Numerical conclusions come only from preserved local data and actual runs.
