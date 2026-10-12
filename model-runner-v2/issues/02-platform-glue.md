# [MRV2] Platform glue: device shims, Triton gate, kernel binding

Part of #906. Plan: [kernel strategy](https://joerunde.github.io/spyre-inference/model-runner-v2/mrv2-plan-906.html#kernels) · [starting before 0.31](https://joerunde.github.io/spyre-inference/model-runner-v2/mrv2-plan-906.html#pre031) · [hardware during Wave 1](https://joerunde.github.io/spyre-inference/model-runner-v2/mrv2-plan-906.html#hwtests)

## Summary

The pieces that let MRV2 run on our platform at all. MRV2 calls `torch.cuda` streams and events every step, refuses to start without Triton, and launches about 20 Triton kernels on its default path, the sampler included. Non-Triton replacements for those kernels don't exist upstream yet. This issue adds process-wide CUDA placeholders, a scoped bypass of the Triton gate, and a single module that binds kernels, following the plan's three kernel steps:

1. Run the Triton kernels on x86 through triton-cpu while we wait, shipped as a temporary x86-only locked dependency.
2. Swap to upstream's replacements as-is once they exist, on every architecture, and drop triton-cpu again so x86 validates the same kernels Power and Z run.
3. Measure them on Power and Z, then improve them here or upstream if needed.

The shims and the gate come first, because manual hardware testing of the other issues needs them.

## Scope (WP-A)

- [ ] `device_shims.py`: `torch.cuda` Stream/Event placeholders. `install()` is idempotent and reversible and runs only from `SpyreWorkerV2.init_device`. `spyre_fence()` replaces the CUDA event wait on pooling output
- [ ] `triton_gate.py`: a scoped bypass of `HAS_TRITON` / `_validate_v2_model_runner` (the report's Triton-gate option B), applied only when `SPYRE_MODEL_RUNNER=v2`. It has to take effect **before `vllm.triton_utils` is first imported** in every process (API server, engine core, each TP worker): that module binds real `triton`/`tl` or placeholders once, at import, and MRV2 decorates its kernels at import, so flipping `HAS_TRITON` later (for example in `init_device`) leaves placeholder kernels. On x86 it also overrides vLLM's `cpu` version-string and driver-count checks, so triton-cpu is accepted there
- [ ] The compile filter: the gate also drops `"stock torch.compile"` from `_get_v2_model_runner_unsupported_features` (v0.31 `config/vllm.py:3057`), so compiled V2 can run on 0.28/0.31 without vllm#59079. The startup issue verifies that compiled V2 works this way
- [ ] The opt-in switch `SPYRE_MODEL_RUNNER=v1|v2` (default `v1`): in `register()` it applies the gate; in `check_and_update_config` it sets `worker_cls` to `SpyreWorkerV2`, sets `VLLM_USE_V2_MODEL_RUNNER=1` and skips the V1 pin from #945. Both branches are env-gated. `spyre_inference/__init__.py` and `platform.py` are also touched by #945, so this lands after #945 or rebases with it. Until the compile filter is verified, manual V2 runs use `--enforce-eager`
- [ ] **Kernel step 1:** a small, separate PR that replaces the `"triton; sys_platform == 'never'"` override with an x86-only entry for the triton-cpu v3.8.0 wheel (as vLLM's `requirements/cpu.txt` does, `platform_machine == "x86_64"`) and regenerates `uv.lock`. Not cherry-picked to release-0.5
- [ ] `tests/mrv2/test_kernel_coverage.py`: lists every default-path `@triton.jit` kernel. Before step 2 it records them as Triton-bound; after step 2 it fails when any has no replacement bound
- [ ] **Kernel step 2** (after #945 and upstream): `kernels.bind()` binds upstream's replacements as-is, however they arrive: registered by upstream at our pin, copied verbatim into `_vendored/` with the source SHA, or compiled into `_spyre_C` if they're C++ in vLLM's `_C`. In the same change, drop triton-cpu: restore the `"triton; sys_platform == 'never'"` override, regenerate `uv.lock`, and remove `triton_gate.py`'s x86 Triton-acceptance path
- [ ] **Kernel step 3:** measure per-step host time for input preparation and sampling on Power and Z against today's V1 runner (a large-vocabulary decoder at max batch, and a pooling model). Record the results and the decision: no change, (a) our own improvement through `kernels.py`, or (b) a contribution upstream

## Acceptance criteria

- [ ] Importing the package leaves `torch.cuda` and `HAS_TRITON` untouched. After `install()` and the matching undo, `torch.cuda` is back to its original state (unit test)
- [ ] In a fresh process with `SPYRE_MODEL_RUNNER=v2`, `vllm.triton_utils.triton` is the real module and an MRV2 kernel is a real `JITFunction`; without it, both are placeholders (subprocess tests). Checked in the API server, engine core and TP worker processes by hand
- [ ] Light hardware test: install the shims, do a small Spyre copy, and run `spyre_fence()` (skips on CPU hosts)
- [ ] The lock resolves unchanged on ppc64le and s390x
- [ ] With triton-cpu installed on x86, V1 still sees `HAS_TRITON = False` (a test in a fresh process), including when `CUDA_VISIBLE_DEVICES` is set but empty, where vLLM's own check leaves Triton on. The existing x86 CI stays green
- [ ] Manual check on an x86 Spyre host with `SPYRE_MODEL_RUNNER=v2 --enforce-eager`: `SpyreWorkerV2.init_device` completes and MRV2's Triton kernels run unmodified
- [ ] After step 2: every default-path kernel has a non-Triton binding, triton-cpu is gone from `pyproject.toml` and `uv.lock`, and the coverage test and `tests/mrv2` pass on x86 with the same replacement kernels Power and Z use
- [ ] After step 3: measurements on Power and Z are posted on this issue, with the follow-up decision
- [ ] The only dependency change is the temporary x86-only triton-cpu entry, removed again at step 2

## Out of scope

Writing replacement kernels before step 3 shows a need. Upstream asks are tracked separately (see the upstream issue).

## Dependencies

- Depends on: skeleton. Step 2 also needs #945 (the dispatcher) and upstream's replacement kernels.
- Blocks: manual hardware testing of every other issue; end-to-end bring-up on Power and Z.

Version: shims, gate and step 1 on 0.28; step 2 after #945. The step-1 PR and #945 both touch `uv.lock`, so whichever lands second regenerates it.

Size: 3–4 days, plus waiting on upstream for steps 2 and 3.
