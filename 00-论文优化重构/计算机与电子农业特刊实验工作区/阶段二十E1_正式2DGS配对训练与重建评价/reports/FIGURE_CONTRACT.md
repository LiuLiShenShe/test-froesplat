# Figure contract — result-independent design

Backend: Python/matplotlib, consistent with the supplied production Python workflow.
Evidence: real trained-model held-out RGB/alpha and observed loss/count/time CSV.
Purpose: show the paired effects (including negative results), not assert geometric truth.

- Figure 1: Original RGB | A | B | C, all same camera and display policy.
- Figure 2: same predeclared views and fixed normalized central rectangle for every
  group, not an automatically selected R1-favouring region. This is a descriptive
  spatial ROI, not a claim that the whole rectangle is plant foreground.
- Figure 3: native RGB loss, Gaussian count and observed wall-clock by iteration.
  Losses against different training RGB cannot establish shared-reference gains.

Views are indices 0, floor(N/2), N-1 of sorted frozen validation view names; this
rule was recorded before any final model/evaluation existed. No brightness,
contrast, gamma or local enhancement is applied. Native PNG and float tensors
are preserved externally. Export previews do not replace full-resolution outputs.
HongZhang reuses C in the B panel with explicit B=C labelling.

Statistics: 3 scenes, 1 fixed seed each, 8 distinct training runs; validation views
are correlated observations, not independent biological replications. No tests,
confidence intervals or significance labels are added. Source CSV/index links
are the quantitative figure evidence. Inspection by the agent is not human QA.
