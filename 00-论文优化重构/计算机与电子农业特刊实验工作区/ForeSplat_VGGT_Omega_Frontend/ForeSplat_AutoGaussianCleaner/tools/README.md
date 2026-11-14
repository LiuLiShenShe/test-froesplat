# Historical AutoCleaner V3 audit tools

The Python scripts in this directory analyze the earlier AutoCleaner V3
experiments, whose variants were named B/C. Their reports do not establish a
selected method for the new GGGS-only C0–C5 comparison. In particular,
`finalize_reports.py` must not be used to generate the current C5 decision.

The current comparison uses the root-level `finalize_main_comparison.py`,
`collect_comparison.py`, `analyze_adaptive.py` and
`summarize_cross_plant_comparison.py`. Actual completed models and evaluations
are under `experiments/GGGS_Cleaner_Comparison/`.

`prepare_development_decision.py` records the completed, directly reviewed
development evidence. `verify_development_decision.py` binds those exact models
and evidence before and after independent testing. The root default source and
configuration are promoted only after the actual development selection.
