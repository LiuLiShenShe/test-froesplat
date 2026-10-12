# Phase20E-0 implementation plan

Baseline: `f53bf8f300db41a63ef2b917a98adb998f07713f`. Initial worktree clean.
No formal optimization, checkpoint creation, mesh generation, manuscript edit or
production R1/2DGS/M2M3 change is authorized in this stage.

1. Audit the three models and every registered RGB. Join the FFT re-numbering log
   to raw SHA256, not equal stems. Validate crop foreground pixel identity and
   distorted/undistorted image IDs, poses, camera parameters and point tracks.
2. Freeze the complete Phase20D runtime snapshot. Preserve old candidate precision
   and use original candidates only when their image hash matches the actual view.
   New GPU inference uses the same checkpoint/processor and P6 recipe, with full
   score and box precision retained. Two-frame smoke precedes each full scene.
3. Score one cache with unchanged production scorer and selector under base/r1.
   Save all score decompositions and selected mask identities, including empties.
4. Build official-camera inverse maps with COLMAP half-pixel centers. Compare
   RGB with native COLMAP undistortion and actual historical images; validate
   sparse observation conversion and projected 3D tracks. Mask uses nearest
   neighbor, outside valid domain zero. Never resize masks to a camera grid.
5. Finish WanNianQing1 first: candidates, ranking, remapping, matched datasets and
   real loader plus one no-optimizer forward render. Then CaoMei2 and HongZhang.
   Reconstruct versioned A only from verified historical masks; otherwise create
   an explicitly named raw-RGB control, never call it recovered ForeSplat.
6. Freeze sorted registered-name every-eighth validation split, explicit training
   defaults, resolution=2, black background, seed=0, 30000 iterations. All groups
   share camera bundle and the same inherited target sparse points, with no new
   treatment-specific filtering. Pre-composited black RGB is
   the treatment, because current default loader does not apply stored alpha.
7. Run focused and prior CPU tests, source protection, GPU load checks and delivery
   audit. Publish small scripts/manifests/reports with managed-data recovery,
   explicit distinct training count, normal commit/push, then stop.

Tests: hash joins; reordered names; half-pixel map identity/invalid background;
strict binary masks; frozen config differences; cache score/order restoration;
production ranking replay; split/group/camera equality; no missing loader views;
RGB composite and alpha identity; no optimizer training or checkpoint artifacts.

Risks: corrupted original files, incomplete legacy A masks, camera/RGB identity
ambiguity, GPU extension ABI, zero candidate and no B/C differences. Never skip
views or substitute scenes; retain real partial outputs and report exact blockers.

External retrieval: Bocha returned HTTP403 quota exhausted. Context7 resolved
`/colmap/colmap` and fetched official documentation; official warp source was
also fetched directly. Experiment conclusions come from local measured assets.

## Evidence-driven corrections

- FFT renumbering disproves the earlier equal-stem raw/camera join. CaoMei2's
  registered crop0146 is raw0170. TEST32 raw0146 maps to crop0129, which is NOT
  registered. Reliable historic candidates in the actual CaoMei2 viewset: 0, not 1.
- Raw HongZhang originals include 12 unreadable files, unlike the earlier audit
  which inspected headers only. The readable registered raw COLMAP input copies
  recover these files into managed data, preserving corrupted originals.
- Four damaged CaoMei2 masks/crops cannot prove a full historical A. Secondary
  readable COLMAP input crops prove coordinate identity via exact raw nonblack
  pixels. Do not infer an exact legacy alpha from guessed black-pixel thresholds.
- Several old SAM masks/crops and incomplete RGBA prevent exact historical A
  reproduction unless lossless alternate evidence can be recovered. An explicitly
  named raw-RGB control is permitted; it is never called recovered ForeSplat.

## Completion checkpoints

- Initial audit, complete frozen V1 snapshot, exact raw/FFT/camera joins and recovery
  complete: all 532 registered views retained, 12 HongZhang original RGBs restored
  only in managed copies, four CaoMei2 coordinate identities independently proven.
- WanNianQing1 first complete closed loop PASS before remaining scene GPU inference;
  CaoMei2 and HongZhang use the same validated protocol with individual smoke tests.
- Full candidates: 928 instances, 922 newly inferred on 527 views, six frozen
  historical instances on five views. No zero detections, missing masks or fallback.
- True train switches: CaoMei2 raw0139; WanNianQing1 raw0037/raw0038/raw0165;
  HongZhang none. HongZhang actual consumed train RGB is identical B/C: eight future
  unique runs, not nine. No future reconstruction score drives this decision.
- All three official domain mappings, all 1596 dataset views, real loader/CUDA
  no-optimizer forwards PASS. Every Phase20D proposed registered-name split retained.
- Historical A cannot be identity-restored: readable alternate legacy RGBA is tested
  and its alpha conflicts with original masks/registered crops. All A groups are
  explicitly versioned raw-RGB controls. No exact historical reproduction claim.
- Late damaged-input photometric audit exposes a conflicting historical derived
  RGB for WanNianQing1 0136. Actual native CPU SIFT against original distorted
  COLMAP observations confirms raw0136 belongs to image_id133, while adjacent
  wrong-frame controls fail. No camera is re-registered and old RGB is rejected.
- Fifteen gates per scene pass; 7914 original assets unchanged and all managed
  artifacts hashed. Final CPU checks: 34 E0 + 76 D-Final/C-FIX + 107 D legacy = 217
  passed, no final skips. Normal Git publication is the last step; no training follows.
