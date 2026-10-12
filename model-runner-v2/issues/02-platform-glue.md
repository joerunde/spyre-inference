# [MRV2] Platform glue: device shims, Triton gate, kernel binding

Part of #906. Plan: [kernel strategy](https://joerunde.github.io/spyre-inference/model-runner-v2/mrv2-plan-906.html#kernels) · [starting before 0.31](https://joerunde.github.io/spyre-inference/model-runner-v2/mrv2-plan-906.html#pre031) · [hardware during Wave 1](https://joerunde.github.io/spyre-inference/model-runner-v2/mrv2-plan-906.html#hwtests)

## Summary

The pieces that let MRV2 run on our platform at all. MRV2 calls `torch.cuda` streams and events every step, refuses to start without Triton, and launches about 20 Triton kernels on its default path. This issue adds process-wide CUDA placeholders, a scoped bypass of the Triton gate, and a single module that binds kernel implementations. We don't write kernels: they come from upstream's CPU MRV2 work (vllm#55847 thread), vendored until they reach our pin. The shims and the gate come first, because manual hardware testing of the other issues needs them.

## Scope (WP-A)

- [ ] `device_shims.py`: `torch.cuda` Stream/Event placeholders. `install()` is idempotent and reversible and runs only from `SpyreWorkerV2.init_device`. `spyre_fence()` replaces the CUDA event wait on pooling output
- [ ] `triton_gate.py`: a scoped bypass of `HAS_TRITON` / `_validate_v2_model_runner` (the report's Triton-gate option B), turned on only by the V2 worker path. A dev-only switch for vLLM's `cpu` version-string check, for local triton-cpu runs on x86
- [ ] `kernels.py` (after #945 and upstream): `bind()` registers upstream fallbacks, or vendored copies, through the v0.31 dispatcher. `DEFAULT_PATH_KERNELS` lists the default-path kernels
- [ ] `_vendored/`: upstream kernel files copied verbatim with the source SHA in each header, only if upstream hasn't reached our pin
- [ ] `tests/mrv2/test_kernel_coverage.py`: fails when any default-path `@triton.jit` kernel has no binding

## Acceptance criteria

- [ ] Importing the package leaves `torch.cuda` and `HAS_TRITON` untouched. After `install()` / `enable()` and the matching undo, both are back to their original state (unit tests)
- [ ] Light hardware test: install the shims, do a small Spyre copy, and run `spyre_fence()` (skips on CPU hosts)
- [ ] Manual check on a Spyre host with overrides: `SpyreWorkerV2.init_device` completes and constructs the runner
- [ ] After #945: `kernels.bind()` covers every entry in `DEFAULT_PATH_KERNELS`, and the coverage test passes
- [ ] No dependency changes: Triton stays banned in `pyproject.toml`

## Out of scope

Writing our own kernels. Upstream asks are tracked separately (see the upstream issue).

## Dependencies

- Depends on: skeleton. `kernels.py` also needs #945 (the dispatcher) and the upstream CPU MRV2 kernel PR.
- Blocks: manual hardware testing of every other issue; end-to-end bring-up.

Version: shims and gate on 0.28; `kernels.py` after #945.

Size: 3–4 days, plus waiting on upstream.
