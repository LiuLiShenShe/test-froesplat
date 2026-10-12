# Phase20D-Completion execution plan — 2026-10-12

Baseline: `22073ca5`, clean `main`. No algorithm, GT contour, production, manuscript,
2DGS or M2M3-GO edits are permitted. Existing human work is retained.

1. **Locate and inspect existing annotations.** Read the 32 LabelMe files next to
   `annotation_package/rgb/`, the frozen manifest, both review tables and the original
   gate. Validate shape metadata, coordinates, labels and image identity without
   converting linestrips or manufacturing missing plant/pot/cube labels.
2. **Archive objective facts without signatures.** Create non-destructive local
   links in `annotation_package/annotations/`; record exact source hashes. Produce
   an `apply_patch` patch that fills only empty objective/derived fields in the two
   tables. Preserve all five manual YES/NO columns and every existing human field.
   Produce a batch registration sheet and a frame-specific supplement sheet, not
   new annotations or invented human visual evidence.
3. **Execute the unchanged GT gate.** Call the existing `scripts/validate_annotations.py`
   with a fresh completion log. Reuse its geometry checks; no alternate approval path.
   If blocked, describe every geometry and archival blocker, leave metric cells empty,
   and stop all GT-dependent runs. If passed, continue frozen CPU smoke/full runs and
   metrics in a later step of this same phase; no early method lock.
4. **Verify and publish safe artifacts.** Add synthetic tests for format rejection,
   non-destructive archival, derivation/preservation, coverage and forbidden approval.
   Check all 32 annotations and 71 cached candidate identities, production protection,
   both table baselines and `git diff --check`. Commit/push only public small files.

## Dependencies and risk

The actual located exports use `linestrip`, not the gate's required `polygon`.
Some component labels and all human provenance/evidence fields are absent. The user's
statement that annotation/review was performed is recorded, but is not converted into
unknown signatures, timestamps or per-frame judgments. Finding already completed work
is not a request to redraw it: request its final compatible export and existing review
records, or let the real reviewers supply missing archival facts. A format correction
or label addition is never applied by this agent.

## Verification and scope

Tests run in seconds; hash and gate checks take tens of seconds. No estimate or command
claims an accuracy result before the gate. The completion report distinguishes found
annotations, valid geometry, archived human review and independently measured accuracy.
Phase20E receives a preparation-only camera-domain/training/quality plan; nothing runs.

## Actual checkpoint

Steps1–3 reached the correct stop boundary:32 exports found and preserved, both tables'
objective fields/derived labels archived, original gate blocked with0 verified frames.
GT-dependent Step4 execution is not started; this is not an accuracy PASS. Public
preparation checks passed (76 tests,375 source hashes,32 source/backup hashes), leaving
only the normal safe-material submission step in this blocked completion. The research
objective of independent evaluation remains incomplete until actual human records and
compatible final exports are available.
