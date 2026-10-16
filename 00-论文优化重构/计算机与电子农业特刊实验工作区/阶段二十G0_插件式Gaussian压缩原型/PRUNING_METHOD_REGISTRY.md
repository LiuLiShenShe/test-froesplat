# Pruning Method Registry

| Method | Status | Score | Source | Recovery |
|---|---|---|---|---|
| `random` | IMPLEMENTED_BASELINE | PCG64 uniform score, seed recorded | `scoring/__init__.py` | none |
| `opacity` | IMPLEMENTED_BASELINE | stored opacity logit; sigmoid is monotone | `scoring/__init__.py` | none |
| `external` | IMPLEMENTED_WITH_PROVENANCE | caller-provided finite vector bound to source and score-file SHA256 | `scoring/__init__.py` | none |
| `PUP_2DGS_ADAPTED` | PUP_ADAPTATION_BLOCKED | no score emitted | `scoring/__init__.py` | not applicable |
| `LIGHTGAUSSIAN_2DGS_ADAPTED` | LIGHTGAUSSIAN_ADAPTATION_BLOCKED | no score emitted | `scoring/__init__.py` | not applicable |

The generic selector is independent of scoring: higher finite scores are retained, equal scores prefer the lower original index, and output rows are written in ascending original index. Random and opacity are baselines, not algorithm contributions. The blocked names are intentionally not aliases for heuristics.
