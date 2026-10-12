# [MRV2] Track upstream asks for Triton-free MRV2

Part of #906. Plan: [kernel strategy](https://joerunde.github.io/spyre-inference/model-runner-v2/mrv2-plan-906.html#kernels) · [warmup policy](https://joerunde.github.io/spyre-inference/model-runner-v2/mrv2-plan-906.html#warmup) · report: [upstream threads](https://joerunde.github.io/spyre-inference/model-runner-v2/mrv2-migration-906.html)

## Summary

A tracking issue for the vLLM changes that would shrink or remove our workarounds. The main one is the upstream PR moving all non-Triton CPU backends to MRV2 (announced on the vllm#55847 thread on 2026-10-09). Each item links to the upstream thread where it's raised. This issue doesn't hold code.

## Asks

- [ ] **Pure-torch fallbacks without `_C`.** We build vLLM with `VLLM_TARGET_DEVICE=empty`. Every C++ kernel in the CPU MRV2 PR needs a pure-torch fallback that stays registered when `_C` is missing
- [ ] **Gate on a platform capability, not `is_cpu()`.** Our platform is OOT, so `is_cpu()` is False. The fallback registration and the Triton-gate exemption should work for an OOT platform (vllm#57684)
- [ ] **Sampler kernels in scope.** Gumbel/temperature, logprobs, penalties, min_p, logit bias, bad words, not only input prep
- [ ] **s390x and ppc64le testing.** Offer IBM Z and Power hardware, since s390x is big-endian and no Triton CI covers it
- [ ] **A warmup flag that skips attention without implying `is_profile`,** so warmup dummies go back through `dispatch()` and bucket padding
- [ ] **Optional:** a platform-provided component factory (vllm#52719 / #53895) and a per-platform KV allocation hook, which would remove several of our subclasses

## Acceptance criteria

- [ ] Each ask is raised on an upstream thread and linked here
- [ ] When an ask lands, the matching workaround in `spyre/` is removed or scheduled (vendored kernels deleted, `triton_gate.py` retired, the warmup plan routed through `dispatch()`)

## Dependencies

None. Start now: the CPU MRV2 PR is expected to be posted soon.
