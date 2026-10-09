# MRV2 warmup measurements (#906)

Raw data behind the "Warmup measurements" section of `../mrv2-migration-906.html`.
Measured 2026-10-09 on an x86 Spyre host (Xeon 6736P, TP1), spyre-inference main
`881a59d` (vLLM 0.28.0, V1 runner), torch-spyre `812af57`.

- `meas.py`: in-process harness. `MEAS_MODE=B|D|A` selects the warmup variant (B = today,
  D = dummies run attention like MRV2, A = one request per dummy). It records time, attention
  calls, Dynamo graphs, Inductor FX-cache hits/misses/bypasses and Spyre backend compiles per phase.
- `run_all.sh`: the six granite-3.3-8b runs. `run_seed.sh`: the `PYTHONHASHSEED=0` kernel-cache
  check on micro-g3.3. Both use absolute paths from the original job; edit `M` and `WT` to rerun.
- `res-*.json`: one result per run. `summary.md` / `summary.json`: one row per run.

Run one configuration at a time: the Spyre device can't be shared.
