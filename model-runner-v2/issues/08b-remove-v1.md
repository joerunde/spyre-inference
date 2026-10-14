# [MRV2] Remove the V1 runner and the fallback flag

Filed as https://github.com/torch-spyre/spyre-inference/issues/1245.

Part of #906. Plan: [the flip](https://joerunde.github.io/spyre-inference/model-runner-v2/mrv2-plan-906.html#flip)

## Summary

Stage 2 of the flip. After the validation period on V2-by-default (the default-V2 issue, #1244), delete the V1 runner, the fallback flag and everything that only existed to keep V1 and V2 apart. This is cleanup: V2 behaviour doesn't change.

It has a hard deadline. The fallback only works while our vLLM pin ships the V1 runner, so this lands before, or together with, the vLLM bump that removes V1 (targeted for v0.32, unconfirmed). If that bump arrives first, hold the pin until this is in. Upstream activity suggests the removal may slip by a release; the plan doesn't rely on that.

## Scope (WP-FLIP-2)

- [ ] Remove `SPYRE_MODEL_RUNNER` (the gate and the V2 settings become unconditional), the V1 smoke test, the `VLLM_USE_V2_MODEL_RUNNER=0` pin and the V1-only rejections
- [ ] Fold `SpyreWorkerV2` into `spyre_worker.py`, so one worker sits outside the package, as `gpu_worker.py` does upstream
- [ ] Delete `spyre_model_runner.py`, the V1 sampler (`v1/sample/` and its `csrc/` sampling kernels, unless kernel step 3a reused them), `SpyreCpuGpuBuffer`, the slot-mapping shim and the V1-internal tests listed in the report
- [ ] Move the shared helpers: `_SpyreModelWrapper` to `v1/worker/model_wrapper.py` (unless the startup issue's extract-only PR already moved it), and the remaining block-compile helpers (`_repeated_block_lists`, `_compile_granularity`, ...) into `v1/worker/block_compile.py`, which that PR created. Every `# shared with V1; moves at the flip` comment in `spyre/` is resolved. The helpers extracted in Wave 1 ([shared helpers](https://joerunde.github.io/spyre-inference/model-runner-v2/mrv2-plan-906.html#shared)) already live outside the runner and don't move
- [ ] Fold back any subclass in `spyre/` that only existed to avoid editing a shared module
- [ ] `v1/worker/spyre/` keeps its name and its MRV2-only docstrings
- [ ] Update the docs, CLAUDE.md (worker modules, compile notes) and the `upgrade-vllm` skill

## Acceptance criteria

- [ ] The full CI suite passes on hardware with V2 as the only runner
- [ ] No module in `spyre_inference/` imports `vllm.v1.worker.gpu_model_runner` or other V1-runner internals
- [ ] `grep -r "moves at the flip" spyre_inference/` returns nothing
- [ ] triton-cpu is no longer in `pyproject.toml` or `uv.lock` (dropped in the upstream-kernels issue, #1238), so every architecture runs the same kernels
- [ ] Lands before the vLLM bump that removes the V1 runner

## Out of scope

New features on V2.

## Dependencies

- Depends on: the default-V2 issue (#1244) and its validation period.
- Deadline: the vLLM bump that removes V1.
- Contingency: if upstream removes V1 before bring-up is green, hold the pin at 0.31.x or vendor V1 for one release.

