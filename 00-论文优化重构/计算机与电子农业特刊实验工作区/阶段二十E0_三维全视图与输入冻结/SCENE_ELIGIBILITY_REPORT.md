# Scene eligibility — current measured local assets

| sample | registered | candidate_views | new_instances | R0_R1_switches | train_mask_differences | ready | training_runs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| WanNianQing1 | 213 | 213 | 339 | 3 | 3 | True | 3 |
| CaoMei2 | 106 | 106 | 111 | 1 | 1 | True | 3 |
| HongZhang | 213 | 213 | 472 | 0 | 0 | True | 2 |

Selection priority is camera/RGB provenance, not future reconstruction scores:
WanNianQing1 first, then CaoMei2, then HongZhang; every scene retains all registered
views. Raw camera poses, IDs and sparse points are immutable. Common camera version:
`HISTORICAL_POSES_CANONICAL_RAW_RGB_V1`. This version reconstructs RGB on the exact
undistorted domain without mixing any new COLMAP model into the experiment.

## CaoMei2 identity correction

`03-SAM/segment_v2.py:185` does **not** spatially crop the frame. It copies the whole
RGB and sets background pixels to zero. FFT filtering **does** renumber retained
frames (`02-FFT/blur_filter.py:245`). Full per-view hash/log/pixel joins are in
`FULL_VIEW_IDENTITY_MANIFEST.csv`. Registered `crop_0146.png` is raw `0170.jpg`;
TEST32 raw `0146.jpg` would be `crop_0129.png`, but that image is NOT registered
in the actual binary model. The previously reported 0146 registration was an
equal-stem identity error. Reusing these foreground-conditioned poses is a limitation, not a
claim that the old model was full-raw-RGB registration. Pixel coordinate mapping
is identity before true undistortion; there is no resize or bbox inversion.

WanNianQing1 raw 0223 is registered. Raw 0227 remains unregistered, so its TEST32
improvement is not a three-dimensional benefit claim in this camera collection.
HongZhang/WanNianQing1 raw registrations retain raw names; legacy masks must join
the FFT index to raw SHA256, not same-name numbers. No unreadable view is dropped.

Historical damaged paths and restoration status: `reports/RECOVERABLE_HISTORICAL_DAMAGE.csv`.
Twelve unreadable HongZhang originals are recovered into managed data from exact-name
readable registered raw COLMAP inputs; original corrupt files/hashes remain unchanged.
Four CaoMei2 coordinate identities use secondary registered input crops with exact
nonblack raw-pixel equality. `reports/SOURCE_RECOVERY_MANIFEST.csv` records this proof.
All canonical originals, including verified recovery copies, are fully read and checked; rebuilt RGB lives in managed
data, not overwritten historical files. Detailed per-view source hashes are retained.
Readable legacy RGBA on the exact camera grid: CaoMei2 2/106 (104 are raw-sized),
WanNianQing1 193/213, HongZhang 185/213. Proven legacy raw/mask/crop pairs at the
initial audit: 102/106, 211/213, 188/213 respectively. These are different coverage
definitions and neither proves a complete historical training version. The new A
control is explicitly raw RGB; it cannot substantiate a reproduced-ForeSplat claim.
Readable legacy RGBA was tested as an alternate missing-mask source, not dismissed
only by file count: all four damaged CaoMei2 frames' alpha composites disagree
with registered historical crops (17199, 19330, 29483 and 31773 pixels). A known
readable CaoMei2 alpha differs from original SAM by 3715 pixels; a WanNianQing1
alpha differs from official-nearest mapped original SAM by 50541 pixels.
Without an identified construction/version recipe these are not recoverable
historical-mask identities. `reports/LEGACY_RGBA_RECOVERY_PROOF.json` and
`scripts/audit_legacy_alpha.py` retain the evidence. No alpha is inferred from black RGB.
WanNianQing1 damaged input 0136 has an additional **historical RGB version conflict**:
its existing undistorted color image does not match current raw0136 (mean interior
error 61.53 intensity units). The raw RGB's actual native CPU SIFT locations match
original image_id133 distorted COLMAP observations: median 0.195px, 79.53% within
1px. Adjacent wrong-frame controls match <0.76% within 1px (median 13–24px).
Therefore raw0136 belongs to this immutable camera; the existing derived RGB is
not reused. `reports/WanNianQing1_0136_SIFT_KEYPOINT_IDENTITY.json` preserves actual
evidence and missing original descriptor limitations. No camera registration was run.
