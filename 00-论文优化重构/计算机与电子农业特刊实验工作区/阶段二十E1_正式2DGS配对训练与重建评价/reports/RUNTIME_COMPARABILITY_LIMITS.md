# Runtime comparability limitations

The exact same physical GPU1, production Python/CUDA extensions and frozen
algorithm settings are used. No CPU/GPU clock, fan, driver or competing job is
changed. The machine is shared: during WanNianQing1 B's startup at 11:14 on
2026-10-13, Linux reported approximately 78/84/73 load averages and nearly full
swap; the loader process waited in `folio_wait_bit_common` (disk/page-cache I/O),
with GPU1 idle. Training later proceeds normally.

Actual wall-clock includes frozen-input wrapper checks, loader/camera decode and
native 7000/30000 validation/image logging, so startup I/O contention can dominate
the difference between groups. Do not advertise that difference as a causal
algorithm-only speedup. GPU iteration timing is separately retained per iteration
and GPU utilization/temperature/device memory are sampled at 2-second intervals.
RTX A6000 temperature is observed near 88°C during sustained use; this is reported,
not corrected by silently changing a per-group resource policy.

CUDA allocated peak is exact for the training process's allocator; CUDA reserved
peak includes caching. `nvidia-smi` samples device-used memory and are a sampled
lower bound, not a continuously measured exact peak nor identical to allocated
tensor storage. Report both rather than replacing one with the other.

At approximately 12:04 on 2026-10-13, GPU1 has three other processes (PIDs
522686, 522687 and 522688), each using about 4236 MiB, alongside the C training
process PID504319. These are external workloads, not extra ForeSplat runs.
No process is killed or reconfigured. A read-only per-process memory audit is
started for the remainder of C and all subsequent runs. C's earlier audit start
is clearly recorded by its first timestamp; absence of earlier samples is not
claimed as proof of idle hardware. Device occupancy is therefore not attributed
entirely to this experiment. Exact allocator peaks still measure this training
process independently; actual wall-clock comparisons remain confounded.
