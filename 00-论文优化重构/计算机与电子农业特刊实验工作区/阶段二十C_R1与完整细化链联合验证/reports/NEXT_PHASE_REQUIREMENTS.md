# NEXT_PHASE_REQUIREMENTS

Phase 20D must remain blocked pending remediation rather than being treated as a completed independent validation phase.

- Restore/match the three missing DouBanLv2 COLMAP image observations and rerun GT10 V2/V3 with clean output directories.
- Close candidate-cache score provenance: preserve `sam_score`, candidate box, and instance ordering, or explicitly rerun SAM3 inference under the frozen P6 configuration.
- Repeat the four-group comparison on an independent dataset with verified `potted_clean` GT; do not use Easy21/GT10 as the independent set.
- Keep A6/A7 OFF and keep all R1 weights and thresholds frozen.
- Re-run the Phase 20C test and hash checks after remediation; do not overwrite the current partial evidence.
