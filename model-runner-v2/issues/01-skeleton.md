# [MRV2] Package skeleton and V2 worker

Part of #906. Plan: [package layout](https://joerunde.github.io/spyre-inference/model-runner-v2/mrv2-plan-906.html#layout) · [naming](https://joerunde.github.io/spyre-inference/model-runner-v2/mrv2-plan-906.html#naming) · [test plan](https://joerunde.github.io/spyre-inference/model-runner-v2/mrv2-plan-906.html#tests)

## Summary

Create the `spyre_inference/v1/worker/spyre/` package (Model Runner V2 only) with every file from the plan's layout, real class and function signatures, and `NotImplementedError` bodies. This PR settles the interfaces the other issues build against, so they can be worked in parallel without editing each other's files. Nothing outside the package imports it, and the platform keeps selecting `TorchSpyreWorker`.

## Scope (WP0)

- [ ] `spyre/__init__.py`: the MRV2-only package docstring from the plan, and nothing else
- [ ] Every module file from the plan's layout, each with an `MRV2:` docstring naming the upstream class or file it extends
- [ ] `model_runner.py`: `SpyreModelRunnerV2(GPUModelRunner)` on `device=cpu`, written in full as a coordinator that delegates to the component modules
- [ ] `worker.py`: `SpyreWorkerV2(TorchSpyreWorker)`. `init_device` builds the V2 runner and calls the device shims, the gate and the kernel binding. `execute_model` calls `runner.pool()` for pooling tasks. Logs `Using Spyre Model Runner V2`
- [ ] The cross-issue interfaces: `SpyreBucketManager.from_runner` / `.bucket`, `EncoderPlan`, `kv_cache.init` / `attn_groups.init` and the runner attributes they set, `load.load_and_wrap`, `kernels.bind` / `DEFAULT_PATH_KERNELS`, `warmup.plan` / `warmup.run`
- [ ] `tests/mrv2/conftest.py`: an autouse guard that makes `torch.compile` and `vllm.LLM.__init__` raise, plus the time-budget hook
- [ ] `tests/mrv2/fakes.py`: tiny fake `InputBatch`, scheduler output, `KVCacheConfig`, attention groups, and a toy model
- [ ] `tests/mrv2/test_layout.py`: no module outside the package imports it; components don't import `model_runner.py`; every module has the `MRV2:` docstring marker
- [ ] `tests/mrv2/test_imports.py`: every module imports against the pinned vLLM; importing changes nothing (`torch.cuda` is untouched)
- [ ] `tests/mrv2/test_worker.py`: with the engine flag False and a pooling task, `execute_model` still returns pooling output
- [ ] CLAUDE.md: a line under "Key modules" saying `v1/worker/spyre/` is MRV2-only and not wired up yet

## Acceptance criteria

- [ ] `uv run pytest tests/mrv2` passes in well under the 20 s budget, on `main` (vLLM 0.28) and on a checkout of the #945 branch (0.31)
- [ ] Overrides take extra upstream kwargs through `**kwargs`, so the same code runs on 0.28 and 0.31
- [ ] No file outside `spyre_inference/v1/worker/spyre/`, `tests/mrv2/` and CLAUDE.md changes
- [ ] Every interface listed above has a signature and a short docstring that another issue can implement against without changing it

## Out of scope

Any component behaviour, kernel work, and changes to `platform.py`.

## Dependencies

- Depends on: nothing. Starts on vLLM 0.28.
- Blocks: every other MRV2 issue.

Size: 2–3 days.
