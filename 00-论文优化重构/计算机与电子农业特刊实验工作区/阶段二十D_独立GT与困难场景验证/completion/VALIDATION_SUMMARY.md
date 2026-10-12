# Actual validation summary

- Actual original Gate: exit2,32 annotations found,0 verified, evaluation_allowed=false.
  Same result reproduced through the unchanged preflight; no predictions or metrics.
- Synthetic/archive/Gate/strict metadata tests: **76 passed in1.35s** on the final
  invocation; source log is logs/TEST_RESULTS_FINAL_02.txt. They are code tests, not
 32-frame accuracy measurements. The previous successful invocation is also retained.
- First test command named a nonexistent historical test file; no tests ran. Its
  logs/TEST_RESULTS.txt is retained. The corrected existing suite passed; no unrelated
  warning/configuration was changed (pytest_asyncio emitted a pre-existing warning).
- Protection check:375 immutable source hashes,32 original annotation hashes,71 frozen
  candidates,160 manual difficulty cells, all32frame/eight-sequence coverage,9 protected
  source/Gate/2DGS entry files unchanged, plus32 byte-exact local backup hashes. Final
  output is logs/FINAL_PROTECTION_CHECK_03.json. An intermediate recheck refused to
  overwrite an existing code-hash inventory; the output is now named per invocation.
- All51 blocked metric rows (32frames,8sequences,6summary rows,5types) retain blank
  numeric cells; raw-selection/Oracle/current-run IDs are not invented from historicalIDs.
- Independent CPU evaluation test: **BLOCKED_GT_GATE**. Metric recompute consistency:
  **BLOCKED_GT_GATE**. These are not PASS or FAIL research results.
- `git diff --check`: no errors in the actual validation invocation. Public publication
  outcome is separately recorded; no forced push, large cache or GT polygon upload.

Source: actual completion logs and deterministic original-source/Gate checks.
