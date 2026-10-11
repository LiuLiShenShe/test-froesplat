# Final foreground method decision

RECOMMENDED METHOD: V1

The conservative multi-metric decision rule was written in `IMPLEMENTATION_PLAN.md`
before new GPU results. It requires both datasets to preserve identities and avoid
sample-equal regressions in F1, IoU, Boundary-F1, HD95 and Leakage. Observed V4
failed conditions: Easy21:final_hd95, Easy21:final_leakage, GT10:final_hd95, GT10:final_leakage.
No GT-dependent runtime switch, retuned weight or post-hoc threshold is introduced.

1. **Ranking:** R1 still rescues all three hard frames, Easy21 churn remains zero.
   Real archive-score replay plus rounding-bound selection certificate support this;
   unsaved confidence digits are not pretended to be available.
2. **SPNP:** real V4/V5 runs isolate its increment. It can improve overlap/boundary
   while worsening leakage/HD95 or some frames. These gains and losses are shown
   together in `SPNP_ONLY_ABLATION_REPORT.md`; a small F1 gain is not a universal win.
3. **GCFC:** missing GT10 registration plus historical coordinate mismatch and mostly
   negative Easy21 F1 delta do not establish stable benefit. It is OPTIONAL, not default.
4. **Residual:** no actual saved-mask change in historical Easy21; necessity unproven.
5. **Dependencies:** V1 needs no unverified geometric observations or auxiliary module.
   It fixes ranking at frozen R1 with SPNP/Residual/GCFC OFF. Existing modules stay
   available as optional/historical ablations; no CLI default is silently rewritten.
6. **Cost:** V4 needs loaded SAM3 and extra prompt inference. Simpler cached V1 avoids
   that dependency. This decision is DEV-supported, not independently generalized yet.

Recommended frozen foreground recipe: P6, per_instance, single, ranking_policy=r1,
fixed original score weights and threshold, SPNP=OFF, Residual=OFF, GCFC=OFF,
A6/A7=OFF, verified metadata or an explicitly new SAM3 candidate set. Independent
TEST32 must confirm the frozen method without tuning. Historical stages and paper
正文 are unchanged; no 2DGS/M2M3-GO training is launched.

Sources: predeclared rule, `DECISION_CHECK.json`, paired/sample-equal metrics,
geometry audit and archive-score/identity certificates in this directory.
