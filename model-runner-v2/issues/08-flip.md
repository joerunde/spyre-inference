# [MRV2] Make V2 the default and remove the V1 runner

Part of #906. Plan: [the flip PR](https://joerunde.github.io/spyre-inference/model-runner-v2/mrv2-plan-906.html#flip)

## Summary

Switch every run to the V2 runner and delete the V1 runner. This is where V2 gets full CI coverage: the existing end-to-end, attention, pooling, distributed, quality and upstream suites all start exercising it.

## Scope (WP-FLIP)

- [ ] Point the platform's default `worker_cls` at the V2 worker. Drop the `VLLM_USE_V2_MODEL_RUNNER=0` pin and the V1-only rejections
- [ ] Fold `SpyreWorkerV2` into `spyre_worker.py`, so one worker sits outside the package, as `gpu_worker.py` does upstream
- [ ] Delete `spyre_model_runner.py`, the V1 sampler (`v1/sample/` and its `csrc/` sampling kernels, unless kernel step 3a reused them), `SpyreCpuGpuBuffer`, the slot-mapping shim and the V1-internal tests listed in the report
- [ ] Move the shared helpers: `_SpyreModelWrapper` to `v1/worker/model_wrapper.py`, the block-compile helpers to `v1/worker/block_compile.py`. Every `# shared with V1; moves at the flip` comment in `spyre/` is resolved
- [ ] Fold back any subclass in `spyre/` that only existed to avoid editing a shared module
- [ ] `v1/worker/spyre/` keeps its name and its MRV2-only docstrings
- [ ] Note in the release notes that seeded sampling output changes, because MRV2 seeds per position (plan decision 3). Local references are all greedy, so none need regenerating
- [ ] Update the docs, CLAUDE.md (worker modules, compile notes) and the `upgrade-vllm` skill

## Acceptance criteria

- [ ] The full CI suite passes on hardware with V2 as the only runner
- [ ] No module in `spyre_inference/` imports `vllm.v1.worker.gpu_model_runner` or other V1-runner internals
- [ ] `grep -r "moves at the flip" spyre_inference/` returns nothing
- [ ] triton-cpu is no longer in `pyproject.toml` or `uv.lock` (dropped at kernel step 2), so every architecture runs the same kernels

## Out of scope

New features on V2.

## Dependencies

- Depends on: end-to-end bring-up.
- Contingency: if upstream removes V1 before bring-up is green, hold the pin at 0.31.x or vendor V1 for one release.

Size: 1 week.
