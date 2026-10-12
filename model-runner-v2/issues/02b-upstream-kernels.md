# [MRV2] Bind upstream's non-Triton kernels and measure them on Power and Z

Part of #906. Follow-up to the platform-glue issue. Plan: [kernel strategy](https://joerunde.github.io/spyre-inference/model-runner-v2/mrv2-plan-906.html#kernels)

## Summary

Until upstream ships non-Triton replacements for MRV2's kernels, V2 runs only on x86, through triton-cpu (the platform-glue issue). An Arm CPU maintainer announced a PR on the vllm#55847 thread (2026-10-09) that moves all non-Triton CPU backends to MRV2. This issue starts once those replacements exist. It binds them as-is on every architecture, drops triton-cpu again so x86 validates the same kernels Power and Z run, then measures them on Power and Z to decide whether any improvement is needed (plan kernel steps 2 and 3).

## Scope (WP-A, kernel steps 2 and 3)

- [ ] `kernels.py`: `bind()` binds upstream's replacements as-is, however they arrive: registered by upstream at our pin (registering them ourselves if they're keyed on `is_cpu()`), copied verbatim into `_vendored/` with the source SHA if we need them before our pin has them, or compiled into `_spyre_C` if they're C++ in vLLM's `_C`. `DEFAULT_PATH_KERNELS` lists the default-path kernels
- [ ] `tests/mrv2/test_kernel_coverage.py`: fails when any default-path `@triton.jit` kernel has no replacement bound
- [ ] Drop triton-cpu in the same change: restore the `"triton; sys_platform == 'never'"` override, regenerate `uv.lock`, and remove `triton_gate.py`'s x86 Triton-acceptance path
- [ ] Measure per-step host time for input preparation and sampling on Power and Z against today's V1 runner on the same models (a large-vocabulary decoder at max batch, and a pooling model). Post the results here with the decision: no change, (a) our own improvement through `kernels.py` or a component subclass, or (b) a contribution upstream (preferred)

## Asks for the upstream replacement PR

Raise these on the upstream PR when it's posted, and link them here:

- [ ] **A path that works without `_C`.** We build vLLM with `VLLM_TARGET_DEVICE=empty`. A pure-torch path that stays registered when `_C` is missing saves us from compiling upstream's C++ into `_spyre_C`
- [ ] **Registration usable from an out-of-tree platform.** Our platform's `is_cpu()` is False, so replacements registered only for `CpuPlatform` or behind `is_cpu()` don't reach us
- [ ] **Sampler kernels in scope:** gumbel/temperature, logprobs, penalties, min_p, logit bias, bad words, not only input preparation
- [ ] **Testing on s390x and ppc64le.** Offer IBM Z and Power hardware; s390x is big-endian, and no Triton CI covers it

## Acceptance criteria

- [ ] Every default-path kernel has a non-Triton binding, and the coverage test passes
- [ ] triton-cpu is gone from `pyproject.toml` and `uv.lock`, and `tests/mrv2` passes on x86 with the same replacement kernels Power and Z use
- [ ] V2 runs on Power and Z (end-to-end bring-up repeats there)
- [ ] Measurements on Power and Z are posted on this issue, with the follow-up decision and, for (a) or (b), a linked issue or PR
- [ ] The asks above are raised on the upstream PR and linked

## Out of scope

Writing replacement kernels before the measurements show a need. Other upstream workarounds (the upstream-workarounds issue).

## Dependencies

- Depends on: the platform-glue issue; #945 (the v0.31 kernel dispatcher); upstream's replacement kernels.
- Blocks: end-to-end bring-up on Power and Z; making V2 the default (Power and Z need it).
