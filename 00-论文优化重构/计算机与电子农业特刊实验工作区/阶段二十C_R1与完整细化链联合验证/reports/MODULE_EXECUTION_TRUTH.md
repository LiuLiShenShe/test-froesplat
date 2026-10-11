# Module Execution Truth

The production entrypoint calls each stage function for every available frame. `accepted_or_changed` is not inferred from a flag alone: it requires a nonzero mask delta between adjacent saved stage masks. A module enabled with no delta is reported as `NO_CHANGE`; missing geometry observations are not treated as successful GCFC applications.

| Dataset | Group | Module | Calls | Enabled-frame calls | Accepted/changed | Changed frames | Mean changed ratio |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Easy21 | V0 | spnp | 21 | 0 | 0 | 0 | 0.000000 |
| Easy21 | V0 | residual | 21 | 0 | 0 | 0 | 0.000000 |
| Easy21 | V0 | gcfc | 21 | 0 | 0 | 0 | 0.000000 |
| Easy21 | V1 | spnp | 21 | 0 | 0 | 0 | 0.000000 |
| Easy21 | V1 | residual | 21 | 0 | 0 | 0 | 0.000000 |
| Easy21 | V1 | gcfc | 21 | 0 | 0 | 0 | 0.000000 |
| Easy21 | V2 | spnp | 21 | 21 | 21 | 21 | 0.002734 |
| Easy21 | V2 | residual | 21 | 21 | 0 | 0 | 0.000000 |
| Easy21 | V2 | gcfc | 21 | 21 | 21 | 21 | 0.000594 |
| Easy21 | V3 | spnp | 21 | 21 | 21 | 21 | 0.002734 |
| Easy21 | V3 | residual | 21 | 21 | 0 | 0 | 0.000000 |
| Easy21 | V3 | gcfc | 21 | 21 | 21 | 21 | 0.000594 |
| GT10 | V0 | spnp | 10 | 0 | 0 | 0 | 0.000000 |
| GT10 | V0 | residual | 10 | 0 | 0 | 0 | 0.000000 |
| GT10 | V0 | gcfc | 10 | 0 | 0 | 0 | 0.000000 |
| GT10 | V1 | spnp | 10 | 0 | 0 | 0 | 0.000000 |
| GT10 | V1 | residual | 10 | 0 | 0 | 0 | 0.000000 |
| GT10 | V1 | gcfc | 10 | 0 | 0 | 0 | 0.000000 |
| GT10 | V2 | spnp | 0 | 0 | 0 | 0 | — |
| GT10 | V2 | residual | 0 | 0 | 0 | 0 | — |
| GT10 | V2 | gcfc | 0 | 0 | 0 | 0 | — |
| GT10 | V3 | spnp | 0 | 0 | 0 | 0 | — |
| GT10 | V3 | residual | 0 | 0 | 0 | 0 | — |
| GT10 | V3 | gcfc | 0 | 0 | 0 | 0 | — |

Raw per-frame evidence is in `metrics/module_execution.csv`; production flags and backend reasons are in each run's `运行日志.json`, `提示词选择.csv`, and `几何修正提示.csv`.
