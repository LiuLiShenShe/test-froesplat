# Phase20D-GT-Protocol-FIX plan

Input baseline: `6092e2ea` (includes `4d8dcd52`); protocol date2026-10-12.
Scope: versioned visible-region GT protocol, human review preparation and gated format
conversion only. No production/ranking, source GT, prior Gate, metrics or paper edits.

1. **Audit existing human exports.** Recheck32 frozen RGB/source JSON hashes and71
   original candidates using the original input verifier. Compute each shape's vertex
   validity, endpoint gap and self-intersections (open path and proposed closed outline).
   Generate true RGB/open-outline review overlays, without filled or approved regions.
2. **Prepare V2 review records.** Preserve existing D decisions and human records.
   Missing pot means UNVERIFIED visibility, not NOT_VISIBLE or automatic annotation
   error. Record the user's protocol-level explanation without assigning unknown frame
   identities. Prepare32 visibility rows,75 shape rows, shared reviewer provenance and
   existing difficulty evidence; isolate reviewer identity mappings privately.
3. **Implement deterministic conversion and V2 Gate.** Convert only individually
   human-confirmed closed outlines in a new output directory. Preserve all coordinates;
   allow implicit polygon closing only. Copy potted_plant to plant only on explicitly
   confirmed plant-only/NOT_VISIBLE-pot frames; preserve and hash the source/derived
   lineage. Check full32 identity, valid visible masks and real independent approval.
4. **Test and report the actual state.** Cover both legal visibility paths and forbidden
   guesses/automatic signatures/point changes/hash tampering. Generate a versioned Gate
   report, concise missing-record list, protocol diff and Methods suggestion (standalone,
   not a manuscript edit). No independent evaluation or Phase20E work even after a PASS.

## Verification

All32/75 rows and review overlays must have indexed source hashes. Old JSONs and
byte-exact backups, previous reports/metrics/Gate, SAM3/R1/loader and2DGS remain unchanged.
Synthetic tests never write into real annotations. Output paths use exclusive creation;
reruns use new run directories. No AI visual judgment or generated image is human approval.
Audit/code tests take seconds to minutes; generating107 JPEG previews and hashing sources
may take a few minutes. Missing human records stop conversion/evaluation, not audit work.

## Completion checkpoint

All four implementation steps completed. Actual results:32 frozen sources/71 candidates
verified unchanged,75 shapes audited,107 full/frame overlays and21 crossing-detail crops,
versioned V2 scripts and review forms delivered,140 tests passed. V2 readiness is **NO**:
frame-specific human records are not archived;17 shapes/13 frames need local geometry
inspection. Conversion produced0 derived GT, no metric or Phase20E work executed.
