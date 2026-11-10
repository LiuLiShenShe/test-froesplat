# QGS forward parameter audit documentation

On2026-11-09, Context7 resolved PyTorch to `/pytorch/pytorch`, including version2.5.1. The first version-specific query returned unrelated layer-normalization benchmark snippets; those snippets were not used as vector-normalization documentation. A third and final command for this question queried `/websites/pytorch` and returned the current [official vector-normalization reference](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.normalize.html).

The current documentation defines vector normalization as `v/max(norm(v),eps)` with default `p=2`, `dim=1`, `eps=1e-12`. We separately inspected the actual installed PyTorch2.5.1+cu121 `torch.nn.functional.normalize` source and executed a CPU zero-quaternion probe: it returns four zeros, not an identity quaternion. Actual CPU `exp` and `tanh` on the probe retain float32.

The read-only checkpoint audit decodes the exact QGS scale equation from the [pinned official Gaussian model](https://github.com/will-zzy/QGS/blob/74d05c945e99fcaef7afe5a8831903be71ad9b55/scene/gaussian_model.py): `exp(raw_scale0..2)*tanh(raw_scale3..5)`. It distinguishes nonfinite raw parameters from additional nonfinite decoded parameters and reports exact zero scales/quaternions. CPU transcendental calculations are not certified bitwise equal to CUDA. Raw model parameters, renderer and optimizer are unchanged.

Bocha search returnedHTTP403; no reason or quota diagnosis is inferred from that response. The Context7/current-documentation result and inspected installed source are the evidence used here.
