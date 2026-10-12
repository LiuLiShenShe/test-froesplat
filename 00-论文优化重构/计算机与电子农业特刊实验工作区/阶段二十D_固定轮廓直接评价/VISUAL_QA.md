# Direct visual inspection and numeric adjudication

Date: 2026-10-12. The executing assistant directly opened the local image files
using its native image-viewing capability. No new external vision call was used.
This is machine-assisted presentation QA, **not human approval or GT certification**.

## Directly inspected files

- `figures/comparison_CaoMei2_0146.png`: four aligned panels are readable. R0 covers
  a small region above the large foreground plant; R1 closely follows the rasterized
  target silhouette. The original geometry ambiguity is explicitly labelled.
- `figures/comparison_WanNianQing1_0227.png`: R0 includes a small white region at
  the upper-left image edge, rather than an empty mask. R1 follows the large target
  silhouette. The original geometry ambiguity is explicitly labelled.
- `figures/comparison_WanNianQing1_0223.png`: both unchanged predictions include
  a white upper-right region. They are non-empty and fail to overlap the rasterized
  target. This genuine remaining failure is retained, not hidden.
- `figures/paired_and_raster_sensitivity.png`: all 32 frame labels and zero paired
  differences remain visible. Only the two genuine switched cases show positive
  F1 differences; the secondary plot separates geometry warnings from other frames.

All 32 comparison plates, including every recorded difficulty category and the
remaining failure, are available in `figures/DIFFICULTY_GALLERY.html`. Coverage and
image hashes are checked programmatically; inspecting the four named files does
not claim separate visual certification of every contour in all 32 images.

## Earlier external-description errors

Earlier Agnes responses are preserved in `logs/AGNES_VISUAL_QA_v1.json` and its
stdout log. Their source images were the first figure version, retained locally in
ignored `reproduction/figures_v1/`. Some descriptions incorrectly called small
predictions empty or black and miscounted plot labels. These descriptions are
untrusted presentation comments, never input to rasterization, scores, statistics,
difficulty labels or method selection.

Full-resolution decoded mask areas establish the actual facts:

| Frame | R0 foreground pixels | R1 foreground pixels | Empty? |
|---|---:|---:|---|
| CaoMei2_0146 | 281476 | 1452435 | No / No |
| WanNianQing1_0227 | 25695 | 6028221 | No / No |
| WanNianQing1_0223 | 142322 | 142322 | No / No |

The final plates explicitly display these areas. A non-empty mask can have F1=0
when it has no intersection with the target. Scientific values come exclusively
from the original metric implementation and actual full-resolution pixels.

The user-authorized global instruction change is limited to the vision section of
`/home/test/.codex/AGENTS.md`: use native vision first; call Agnes only if native
vision is unavailable. Its credential-containing contents and local backup are
not copied into this repository.
