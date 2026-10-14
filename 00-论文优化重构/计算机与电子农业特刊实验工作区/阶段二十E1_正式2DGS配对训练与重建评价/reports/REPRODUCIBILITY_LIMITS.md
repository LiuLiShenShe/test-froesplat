# Seed reproducibility versus bitwise deterministic training

All final groups use native `safe_state` seed 0 and the same full frozen config.
The compulsory same-config A restart after a logging failure already shows that
the unchanged native CUDA pipeline is not bitwise replay-identical: at iteration
7000 the preserved failed trace has 307212 Gaussians, versus 302627 in the retry
before its first successful logging checkpoint. Both traces/config hashes remain
available; these are observations, not independent final-model repeats.

Changing a backend to deterministic algorithms or replacing kernels would break
the frozen algorithm/environment agreement, so it is not done. Small B/C effects
cannot be claimed to exceed ordinary stochastic/numerical replay variability on
this evidence alone. A next-stage prospective repeated-run/multiseed protocol is
needed, even if all paired rendering metrics move favourably.

`reports/NATIVE_CUDA_REPLAY_DIAGNOSTIC.csv` derives the preserved same-config
pre-failure/retry comparison directly from the traces. It is separate from all
full-model metrics and cannot be used as a second 30000-iteration replication.
