# [MRV2] Package skeleton and V2 worker

Part of #906. Plan: [package layout](https://joerunde.github.io/spyre-inference/model-runner-v2/mrv2-plan-906.html#layout) · [naming](https://joerunde.github.io/spyre-inference/model-runner-v2/mrv2-plan-906.html#naming) · [test plan](https://joerunde.github.io/spyre-inference/model-runner-v2/mrv2-plan-906.html#tests)

## Summary

Create the `spyre_inference/v1/worker/spyre/` package (Model Runner V2 only) with every file from the plan's layout, real class and function signatures, and `NotImplementedError` bodies. This PR settles the interfaces the other issues build against, so they can be worked in parallel without editing each other's files. Nothing outside the package imports it, and the platform keeps selecting `TorchSpyreWorker`.

## Scope (WP0)

- [ ] `spyre/__init__.py`: the MRV2-only package docstring from the plan, and nothing else
- [ ] Every module file from the plan's layout, each with an `MRV2:` docstring naming the upstream class or file it extends
- [ ] `model_runner.py`: `SpyreModelRunnerV2(GPUModelRunner)` on `device=cpu`, written in full as a coordinator that delegates to the component modules
- [ ] `worker.py`: `SpyreWorkerV2(TorchSpyreWorker)`. `init_device` is copied from `TorchSpyreWorker` rather than calling `super()`, which would build a V1 runner and install its process-wide slot-mapping shim. A test fails when `TorchSpyreWorker.init_device` changes, so the copy doesn't drift. It builds the V2 runner and calls the device shims and the kernel binding (the Triton gate runs earlier, at plugin load). `execute_model` calls `runner.pool()` for pooling tasks. Logs `Using Spyre Model Runner V2`
- [ ] The full list of runner attributes MRV2's `initialize_kv_cache` sets and the step path reads (`block_tables`, `pcp_manager`, `ubatch_runner`, `cudagraph_manager`, `kv_connector`, `kernel_block_sizes`, the cudagraph-mode resolution, the KV-zeroing metadata, and `model_state.initialize_kv_cache` at 0.32), with a decision on how each gets set. Likely: `kv_cache.init` calls `super().initialize_kv_cache` with only the allocation and attention-group creation replaced
- [ ] The cross-issue interfaces: `SpyreBucketManager.from_runner` / `.bucket`, `EncoderPlan`, `kv_cache.init` / `attn_groups.init` and which of the attributes above they own, `load.load_and_wrap`, `kernels.bind` / `DEFAULT_PATH_KERNELS`, `warmup.plan` / `warmup.run`
- [ ] `tests/mrv2/conftest.py`: an autouse guard that makes `torch.compile` and `vllm.LLM.__init__` raise, plus the time-budget hook. Tests marked `triton_cpu` (they JIT-compile Triton on a cold cache) are exempt from the budget
- [ ] `tests/mrv2/fakes.py`: tiny fake `InputBatch`, scheduler output, `KVCacheConfig`, attention groups, and a toy model
- [ ] `tests/mrv2/test_layout.py`: no module outside the package imports it, except the opt-in switch's env-gated imports in `spyre_inference/__init__.py` and `platform.py` (added in the platform-glue issue); components don't import `model_runner.py`; every module has the `MRV2:` docstring marker
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
