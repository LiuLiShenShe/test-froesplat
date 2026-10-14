# Preserved failure and logging-only remediation

The first WanNianQing1 A attempt actually optimized to iteration 7000, then the
native TensorBoard depth image logger failed: Matplotlib 3.10.8 removed
`FigureCanvasAgg.tostring_rgb`. This occurs in logging, not optimizer/loss math.
The controller had already started B before the environment-wide fault was
diagnosed. It was stopped without interfering with B's existing optimizer;
that attempt is allowed to finish/fail normally and will be preserved too.

No package downgrade, source rewrite or optimizer parameter changes are made.
The E1 observer restores only the removed logger API using
`np.asarray(canvas.buffer_rgba())[..., :3].tobytes()`, exactly the prior RGB
buffer semantics. A unit test compares RGB bytes to reordered native ARGB bytes.
All subsequent A/B/C final runs receive this identical compatibility adapter.

Failed logs, runtime/exit code, all partial outputs and SHA256 are retained in
`logs/failed_attempts/` and external `phase20e1_failed_attempts/`. No intermediate
metrics or failed wall time enter complete-model summaries. Native checkpoints
do not preserve all RNG/camera-stack state, so no semantically unverifiable
resume is attempted: the retry starts at seed 0 with the exact frozen config.
The failed-attempt index records the real additional cost, not an estimated one.

Current official documentation was retrieved using Context7 on 2026-10-13:
`/websites/matplotlib_stable`, API changes in 3.10.0 and Agg `buffer_rgba` API.
It confirms removal of `tostring_rgb` and access through the RGBA renderer buffer
after `draw()`. The production logger already calls `draw()` before extraction.
Bocha general search was attempted separately and returned HTTP 403.
