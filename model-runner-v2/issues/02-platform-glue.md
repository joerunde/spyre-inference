# [MRV2] Platform glue: device shims, Triton gate, opt-in switch

Part of #906. Plan: [kernel strategy](https://joerunde.github.io/spyre-inference/model-runner-v2/mrv2-plan-906.html#kernels) · [the opt-in switch](https://joerunde.github.io/spyre-inference/model-runner-v2/mrv2-plan-906.html#switch) · [starting before 0.31](https://joerunde.github.io/spyre-inference/model-runner-v2/mrv2-plan-906.html#pre031)

## Summary

The pieces that let MRV2 run on our platform at all. MRV2 calls `torch.cuda` streams and events every step, refuses to start without Triton, and launches Triton kernels on its default path, the sampler included. Non-Triton replacements for those kernels don't exist upstream yet, so for now MRV2 runs its Triton kernels unmodified on x86 through triton-cpu (plan kernel step 1). This issue adds process-wide CUDA placeholders, a scoped bypass of the Triton gate, the opt-in switch, and the x86-only triton-cpu dependency. It comes first, because manual hardware testing of the other issues needs it.

Swapping to upstream's replacement kernels once they exist, and measuring them on Power and Z, is a follow-up (the upstream-kernels issue).

## Scope (WP-A)

- [ ] `device_shims.py`: `torch.cuda` Stream/Event placeholders. `install()` is idempotent and reversible and runs only from `SpyreWorkerV2.init_device`. `spyre_fence()` replaces the CUDA event wait on pooling output
- [ ] `triton_gate.py`: a scoped bypass of `HAS_TRITON` / `_validate_v2_model_runner` (the report's Triton-gate option B), applied only when `SPYRE_MODEL_RUNNER=v2`. It has to take effect **before `vllm.triton_utils` is first imported** in every process (API server, engine core, each TP worker): that module binds real `triton`/`tl` or placeholders once, at import, and MRV2 decorates its kernels at import, so flipping `HAS_TRITON` later (for example in `init_device`) leaves placeholder kernels. On x86 it also overrides vLLM's `cpu` version-string and driver-count checks, so triton-cpu is accepted there
- [ ] The compile filter: the gate also drops `"stock torch.compile"` from `_get_v2_model_runner_unsupported_features` (v0.31 `config/vllm.py:3057`), so compiled V2 can run on 0.28/0.31 without vllm#59079. The startup issue verifies that compiled V2 works this way
- [ ] The opt-in switch `SPYRE_MODEL_RUNNER=v1|v2` (default `v1`): in `register()` it applies the gate; in `check_and_update_config` it sets `worker_cls` to `SpyreWorkerV2`, sets `VLLM_USE_V2_MODEL_RUNNER=1` and skips the V1 pin from #945. Both branches are env-gated. `spyre_inference/__init__.py` and `platform.py` are also touched by #945; this lands first and #945 rebases onto it. Until the compile filter is verified, manual V2 runs use `--enforce-eager`
- [ ] The x86-only triton-cpu dependency, as a small separate PR: replace the `"triton; sys_platform == 'never'"` override with an x86-only entry for the triton-cpu v3.8.0 wheel (as vLLM's `requirements/cpu.txt` does, `platform_machine == "x86_64"`) and regenerate `uv.lock`. Not cherry-picked to release-0.5
- [ ] `kernels.py` stays a stub that binds nothing: on x86 MRV2's Triton kernels run as they are

## Acceptance criteria

- [ ] Importing the package leaves `torch.cuda` and `HAS_TRITON` untouched. After `install()` and the matching undo, `torch.cuda` is back to its original state (unit test)
- [ ] In a fresh process with `SPYRE_MODEL_RUNNER=v2`, `vllm.triton_utils.triton` is the real module and an MRV2 kernel is a real `JITFunction`; without it, both are placeholders (subprocess tests). Checked in the API server, engine core and TP worker processes by hand
- [ ] Light hardware test: install the shims, do a small Spyre copy, and run `spyre_fence()` (skips on CPU hosts)
- [ ] The lock resolves unchanged on ppc64le and s390x
- [ ] With triton-cpu installed on x86, V1 still sees `HAS_TRITON = False` (a test in a fresh process), including when `CUDA_VISIBLE_DEVICES` is set but empty, where vLLM's own check leaves Triton on. The existing x86 CI stays green
- [ ] Manual check on an x86 Spyre host with `SPYRE_MODEL_RUNNER=v2 --enforce-eager`: `SpyreWorkerV2.init_device` completes and MRV2's Triton kernels run unmodified
- [ ] The only dependency change is the x86-only triton-cpu entry

## Out of scope

Binding upstream's replacement kernels, dropping triton-cpu again, and measuring on Power and Z (the upstream-kernels issue). Upstream asks (the upstream issue).

## Dependencies

- Depends on: skeleton.
- Blocks: manual hardware testing of every other issue; end-to-end bring-up on x86.

Version: all on 0.28. The triton-cpu PR and #945 both touch `uv.lock`, so whichever lands second regenerates it.
