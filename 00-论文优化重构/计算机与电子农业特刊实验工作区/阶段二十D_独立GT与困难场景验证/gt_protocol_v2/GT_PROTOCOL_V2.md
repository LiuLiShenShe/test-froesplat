# ForeSplat visible-region GT protocol V2

Version: `FORESPLAT_VISIBLE_GT_V2`;2026-10-12; based on6092e2ea/4d8dcd52.
This is an explicitly versioned protocol correction, **not** a silent V1 Gate relaxation,
algorithm update, re-annotation request or completed independent evaluation.

## 1. Human-visible regions, not amodal reconstruction

| label | V2 meaning and requirement |
| --- | --- |
| potted_plant | Every frame: visible foreground belonging to the single target potted-plant instance; hidden pot/plant parts are not included |
| plant | Actual visible target plant tissue; required when plant_visible=YES, absent when plant_visible=NO |
| pot | Required only if actual pot_visibility is VISIBLE or PARTIALLY_VISIBLE; no polygon required/allowed for an entirely unseen pot |
| blue_cube | Required only when a human confirms it is visible/present in this RGB |

`pot_visibility` ∈ VISIBLE, PARTIALLY_VISIBLE, NOT_VISIBLE, UNVERIFIED.
Missing pot label **never** sets NOT_VISIBLE. Existing pot polygon count is an objective
fact, not a visibility judgment. All current unspecified frame values start UNVERIFIED.
`plant_visible` and `blue_cube_present` are explicit human YES/NO decisions. A nonempty
visible target must include at least a plant or pot component. Identity belongs to the
same real target throughout; do not add neighbors or hidden regions from adjacent views.

Main GT stays **potted_clean = potted_plant AND NOT blue_cube**. No SAM3/R0/R1 initialization,
mask prediction, neighbor-frame completion, convex hull, interpolation or guessed hidden
pot geometry is allowed. Existing historical metrics and source JSONs remain unchanged.

The user has explicitly confirmed the real visibility explanation for missing components.
That is recorded at protocol level, not extrapolated to frame IDs or reviewer signatures
that were not provided. Old completion “15 missing labels” is a V1-requirement finding,
not proof of15 actual annotation mistakes; it is superseded for future V2 validation.

## 2. Existing linestrip audit and permitted conversion

`LINESTRIP_CONVERSION_AUDIT.csv` indexes every source shape by frozen frame/source JSON
hash, zero-based shape index, full-shape SHA256 and vertex-list SHA256. It reports:
vertex count, exact first/last coordinates, endpoint distance/image-diagonal ratio,
open edge length statistics, signed hypothetical closed area, open/closed self-crossing
edge pairs, finite/in-grid vertices and source dimensions. These are diagnostics, not
human proof of a complete boundary. No automatic gap-distance approval threshold exists.

Actual reviewer must confirm each linestrip is a complete intended region exterior,
has no obvious missing boundary, and may be implicitly closed last-to-first. Human
confirmation includes actual code, human attestation, time, record source and reason.
Uncertain/open paths stay HUMAN_REVIEW_REQUIRED. Numeric self-intersections or illegal
coordinates are reported, never automatically untangled, rounded, clipped or repaired.

Conversion in a **new directory only** changes `shape_type: linestrip → polygon`.
Original points, order, labels and all other shape fields are copied unchanged. The
polygon rasterizer supplies its ordinary implicit closing edge; no endpoint is appended
or moved. Original JSON/source_hash permanently retained; derived JSON/derived_hash and
per-shape conversion rule/reason/human confirmation are indexed. No partial-frame shape
deletion is allowed; any uncertain/invalid shape stops that frame's conversion.

Display-only overlay previews may be scaled and RGB encoded for review; they are not
GT and never enter a metric. Colored paths are original vertices, circles mark endpoints,
and magenta dashes are the **unapproved proposed closing edge**. No mask is filled.

## 3. Deterministic plant derivation, narrowly authorized

Missing plant can equal the existing human potted_plant **only when** actual records
explicitly confirm all of:

1. `pot_visibility=NOT_VISIBLE`, `plant_visible=YES`;
2. `potted_only_plant_confirmed=YES`;
3. `plant_equal_potted_derivation_confirmed=YES`;
4. actual consenting human, human declaration, timestamp and traceable source;
5. rule `PLANT_EQUALS_HUMAN_POTTED_PLANT_WHEN_CONFIRMED_PLANT_ONLY`.

Each existing potted_plant polygon is copied point-for-point, only the copied label
becomes plant; its union mask must be exactly equal to the human potted_plant union.
This is recorded human-authorized format derivation, not AI annotation. A present plant
polygon is retained; it is not replaced or “improved”. If pot is visible/partially visible,
plant cannot be copied from the whole instance. Seek the already-completed true plant
region or require only that actual missing visible component from a human, not32 redraws.

## 4. Separate V2 human Gate

V1 `scripts/validate_annotations.py` and FIX rasterizer are preserved byte-for-byte.
V2 entry is `gt_protocol_v2/scripts/validate_gt_v2.py`. Full32 TEST identity and all71
original candidate/source RGB hashes use the existing frozen input verifier. Original
human JSON hashes are bound to the immutable completion inventory at6092e2ea.

V2 checks valid confirmed polygons, same coordinate grid, nonempty potted_clean, plant/
pot containment in the target, actual visibility/component agreement, human-authorized
plant derivation, exact regenerated-document equality and final derived SHA256 approval.
Any unrecorded vertex/shape/label change, source mismatch, missing frame or stale final
hash blocks the Gate. NOT_VISIBLE relaxes **only an impossible visible pot requirement**;
it does not relax provenance, identity, independent human review or difficulty evidence.

Annotator and first reviewer may be the same real person; second reviewer must differ
from both. Real names may be replaced with user-approved traceable codes; private identity
mapping stays private. Both actual review timestamps include timezone, target description
is frame-specific, and all manual/visible-component/difficulty confirmations are explicit.
Every D1/D3/D6/D11/D12 decision is preserved; labels derive from that frame's five columns.
Evidence JSON covers exactly each actual label (or `none`). AI cannot sign, invent evidence,
infer completion from `checked`, file mtime/Git date, or create APPROVED status.

Existing actual batch review can be entered once in HUMAN_BATCH_RECORD.json, with exact
source-scope inventory hash, frame list, record source and final derived hashes. It only
supplies missing shared codes/dates; it cannot approve frame-specific identity, visibility,
outlines, derivation or evidence. Conflicting existing records block rather than overwrite.

## 5. Readiness is not segmentation success

Only full32 valid derived annotations + genuine complete approvals yields
`READY_FOR_PHASE20D_EVALUATION=YES`. The V2 Gate does not execute prediction, Oracle,
accuracy, GPU/3D training or method lock, even when ready. This phase stops after protocol
work. R1 formula/weights/risk/selection/candidates and potted_clean metric definition do
not change. User inputs still missing are listed without requesting wholesale re-annotation.

Source: current user's explicit visibility instruction, frozen local human annotations,
completion source hashes and unchanged input/human Gate. No online source asserts GT status.
