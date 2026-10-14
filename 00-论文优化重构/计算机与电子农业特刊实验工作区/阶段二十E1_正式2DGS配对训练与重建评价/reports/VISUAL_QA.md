# Direct native visual QA (not human sign-off)

The agent directly opened the real local PNG figures with the available native
image viewer. No external Agnes vision/image-generation service was used, and
no AI-generated reconstruction image is substituted for a renderer output.
All views/crops use the predeclared deterministic selection, not favourable cases.

## WanNianQing1 — closed loop checked 2026-10-13

Viewed `figures/WanNianQing1_heldout.png`, `WanNianQing1_fixed_crop.png` and
`WanNianQing1_training.png`. Held-out views are 0000, 0107 and 0244.
The Original/A columns include the table, paper, blue cube, bicycle/curtain;
B/C intentionally show the potted plant on black. Same viewpoint and dimensions
are visibly consistent. The original pot remains foreground: this is not a
verified plant-shoot-only representation. B and C are visually very similar in
these predeclared views. Their pot base and leaf edges show subtle appearance
differences, but no clear robust improvement of physical holes/floating geometry
is established. Leaf texture is smoother than raw in all models.

The fixed crop includes table/cube in the first two views, demonstrating why it
cannot be called a pure plant foreground ROI. Lower full raw RGB scores in B/C
are expected from intentionally removed background and are retained. A's model
contains many more Gaussians; the count curve freezes after the native
densification endpoint. The training loss/time figure is readable and explicitly
warns that native training losses use different reference images.

No NaN, blank RGB rendering or missing panel is visible. This check establishes image
integrity/layout and descriptive observations, not independent geometric truth.

The subsequently inspected fixed-scale alpha plate adds an important negative
finding: B/C have low RGB background energy but alpha is NOT consistently clean.
At 0000, B shows broad diffuse opaque/gray lobes below the pot; C has a narrower
downward lobe. At 0107 and 0244 both B/C alpha panels are largely white over the
background despite their black RGB. Thus black-looking backgrounds do not prove
removal of opaque black Gaussians, and C cannot be declared uniformly better in
opacity contamination. The alpha plate uses the same 0..255 range without
adaptive enhancement. This is descriptive Gaussian-opacity evidence, not mesh GT.

## CaoMei2 — direct native viewing 2026-10-13

Viewed the real heldout, fixed_crop and alpha plates for crop_0095, crop_0160 and
crop_0208. Camera/subject placement is aligned across columns. A reconstructs
raw room/table appearance but shows blur/stretching in the distant room/table.
B/C suppress visible room RGB while keeping the pot and plant. At crop_0095,
C has a conspicuous pale triangular fragment extending right of the pot, not a
clear clean-up versus B; the fixed crop retains this negative case. B/C differ
little at crop_0160/0208 at figure scale. No claim of a uniformly improved leaf
boundary or filled geometric hole is justified.

All alpha plates are mostly opaque even outside the potted-plant appearance;
the low outside-mask RGB energy should not be mistaken for no background
Gaussians. C's mean outside-proxy alpha increases, and its three primary raw RGB
metrics and R0-proxy foreground PSNR all regress. This is reported, not used to
change data/configs or replace the single changed training raw0139.

## HongZhang — direct native viewing 2026-10-13

Viewed heldout, fixed_crop and alpha plates at 0000, 0105 and 0244. The B/C
columns are explicitly labelled as the same model and are visually identical.
The potted plant stays at the same camera/position as raw/A. B/C suppress room
RGB while the flower/pot/leaf appearance remains; the close-up leaf texture is
smoother than raw. Alpha still has large opaque/diffuse regions outside the
visible potted plant, including a wide low-image lobe in 0000/0105 and nearly
all-white alpha in 0244. Thus model sharing is normal and the same RGB-versus-
opacity caveat holds. No R1-versus-R0 gain is inferred from this identity control.

## Delivery visual status

All three predeclared scene image plates, fixed crops and fixed-scale alpha plates
were directly inspected, with negative fragments/opacity remnants retained.
Figure selection does not depend on C improving. Full native RGB/alpha and float
renders are indexed externally. The source-level figure audit has no failures;
PNG previews are not final journal-submission TIFF bundles. No mesh geometry or
physical dimensions are depicted or measured.
