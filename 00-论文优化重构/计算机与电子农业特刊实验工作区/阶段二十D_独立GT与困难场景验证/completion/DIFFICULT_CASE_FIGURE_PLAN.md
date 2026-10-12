# Difficult-case figures — BLOCKED

Recorded YES decisions exist (D1=12, D3=16, D6=15, D11=16, D12=12), but frame-specific
visual evidence and independent review are not approved. Each category is currently
**NOT COVERED by verified evidence**, not “absent from all real images”. No replacement
image or invented difficulty statement is generated.

After approval, Figure A selects actual RGB from reviewer-confirmed D1/D3/D6/D11/D12
frames, labels the frame ID and each true category, and retains original RGB hashes.
Figure B uses `RGB | Human GT potted_clean | R0 | R1` on the same source grid, with
candidate IDs, F1/IoU and source hashes. Include both historical switches and any
regression, not favorable-only examples; unknown target identity cannot be labeled
correct by AI. Multi-label cases can appear under several descriptive strata.

Generate with a deterministic raster-layout script from actual source pixels and
verified metric tables only; preserve script plus input/output path/hash manifest.
Do not create GT, AI-generated examples, synthetic effect panels or alternate masks.
Development DouBanLv2_0039–0041 panels remain explicitly separate DEV mechanism
evidence, never part of the TEST32 quantitative figures. Unsupported categories remain
NOT COVERED. Current generated figure files: **none**; see figures/README.md.
