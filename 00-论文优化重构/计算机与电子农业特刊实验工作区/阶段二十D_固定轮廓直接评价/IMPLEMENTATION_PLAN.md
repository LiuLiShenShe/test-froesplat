# Phase20D-Final implementation plan

Date: 2026-10-12. Starting commit: d3b474b9; branch main; initially clean.

1. Inventory and hash the existing Phase20D files, including ignored local V2 work,
   original JSON/RGB/candidates, frozen source and historical results. Never overwrite them.
2. Freeze the contour-only interpretation below before examining TEST32 metrics;
   test invalid/collinear/bow-tie vertices, union, cube subtraction and source immutability.
3. Run the unchanged production CLI with strict metadata, first the two named smoke
   cases under both policies, then the complete frozen32 input set. Save real commands,
   parameters, scores, selection identities, output hashes and logs.
4. Import the Phase20B/20C metric implementation. Evaluate both rasterizers, raw
   selected and final masks, all71 candidates for the F1 oracle; compute complete
   paired, sequence, subset and overlapping difficulty statistics without exclusions.
5. Produce evidence-grounded comparison plates, sensitivity and method decisions.
   Audit full-view RGB/COLMAP/candidate availability for three Phase20E plants; plan only.
6. Test numeric reconstruction, manifests, production configuration identity, hashes,
   source protection and delivery. Explicitly stage only this directory, commit and push.

Risks: self-intersections are retained; component-only warnings are not necessarily
main-target ambiguity. Archived SAM scores have six-decimal precision only. Empty or
invalid derived GT is a technical failure, never a zero placeholder. R1 may regress.
The unchanged original human-signature Gate remains BLOCKED independently of this
user-authorized fixed-contour evaluation. TEST32 does not provide full-view3D masks.

No new inference, tuning, human signatures, Gate rewrite, paper edit or 3D training.
