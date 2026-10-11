# R1 Production Replay

- New production runs: V0/V1 on Easy21 (21 frames) and GT10 (10 frames); V2/V3 on Easy21 (21 frames).
- Historical production replay comparison: `104/104` available frame/group selections match the Phase 17C expected instance after normalizing zero-padded instance IDs.
- Candidate source: frozen P6 raw instance masks. No candidate was regenerated and no R1 weight was changed.
- Cache caveat: the current reuse loader does not ingest historical SAM3 score metadata, so the production run records `SAM3原始分数=0.0` for reused candidates. Selection agreement is verified on this cache, but byte-for-byte scoring equivalence is not certified.

## Hard-case rescue

| Frame | Selected candidate | New V1 F1 | Oracle regret | Replay match |
| --- | --- | --- | --- | --- |
| 0039 | DouBanLv2_0039_01 | 0.9867 | 0.0000 | 1 |
| 0040 | DouBanLv2_0040_00 | 0.9882 | 0.0000 | 1 |
| 0041 | DouBanLv2_0041_03 | 0.9865 | 0.0000 | 1 |

R1 production selection validation is PASS for the observed frozen-cache replay, with the score-metadata caveat above.
