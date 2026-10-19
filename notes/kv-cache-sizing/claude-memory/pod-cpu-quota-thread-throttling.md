---
name: pod-cpu-quota-thread-throttling
description: rundemc-2-spyre-dev has a 32-CPU cgroup quota but 144 visible cores; oversized OpenMP pools get CFS-throttled and wreck Spyre decode benchmarks
metadata:
  node_type: memory
  type: project
  originSessionId: 25ed7aba-f712-45b8-bbba-a1af17c08830
  modified: 2026-10-12T20:33:19.183Z
---

Measured 2026-10-12 on `rundemc-2-spyre-dev`: `cpu.max` = 32 CPUs, 144 visible, pod env sets `OMP_NUM_THREADS=144` and `SPYRE_NUM_CPUS=32`. Idle-spinning OpenMP threads burn the quota and the process gets CFS-throttled (~58 ms device-idle stalls per decode step).

- In-process runs (`distributed_executor_backend="external_launcher"`, as in the Kineto profiling recipe) keep torch at 72 threads because `configure_threading` only edits `os.environ` after torch is initialized: Qwen3-0.6B B=4 decode 37.5 tok/s vs 105.7 at `OMP_NUM_THREADS=16` (2.8x); granite-3.3-8b 14.9 vs 27.3 (1.8x).
- Normal multiprocess `LLM`/`vllm bench` gets 32 threads: 90.2 tok/s; with `SPYRE_NUM_CPUS=16` 104.8 (+16%), zero throttling.

**Why:** any perf/profiling number from this pod is untrustworthy unless thread count is below the quota; check `/sys/fs/cgroup/cpu.stat` nr_throttled.
**How to apply:** set `OMP_NUM_THREADS=16` (in-process) or `SPYRE_NUM_CPUS=16` (multiproc) before benchmarking here; mention it when comparing numbers. Scripts/results in `~/mrv2-host-profile/`. Related: [[mrv2-host-profile-findings]].
