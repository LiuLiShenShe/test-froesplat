# Task Plan: Experiment 2 Prompt Robustness Extension

## Goal
Define a low-cost extension for experiment 2 that shows prompt bias across multiple plant morphologies, not only `KongQueZhuYu` GT5.

## Phases
- [x] Phase 1: Inspect current experiment 2 scope and tables
- [x] Phase 2: Identify reusable multi-sample P1-P5 candidate masks and GT
- [x] Phase 3: Decide sample roles and required outputs
- [x] Phase 4: Update the experiment plan or deliver a concise recommendation

## Key Questions
1. Can existing four-sample representative outputs support P1-P5 prompt robustness without rerunning SAM3?
2. Which samples best represent P2 success, P3/P5 failure, and P4 empty-mask failure?
3. What minimum tables and figures are enough for paper writing?

## Decisions Made
- Reuse existing 2D masks and GT first; do not require 2DGS for this extension.
- Use the existing four-sample representative set (`KongQueZhuYu`, `DouBanLv1`, `ChangShouHua2`, `CaoMei1`) for a 100-mask P1-P5 robustness table.
- Use `DouBanLv1`/`ChangShouHua2` as P2-success and P3/P5/P4-failure evidence; use `CaoMei1` as low-canopy boundary-sensitivity evidence.
- Added a fixed script and generated three CSV drafts for summary, per-sample, and per-frame prompt robustness results.

## Errors Encountered
- Full-workspace `rg` over prompt terms produced excessive output; narrow future searches to experiment 2 and representative-set folders.

## Status
**Complete** - experiment 2 now has a concrete four-sample 2D prompt-robustness extension plan and generated table drafts.
