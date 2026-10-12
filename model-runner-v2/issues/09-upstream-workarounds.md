# [MRV2] Track upstream workarounds

Filed as https://github.com/torch-spyre/spyre-inference/issues/1246.

Part of #906. Plan: [package layout](https://joerunde.github.io/spyre-inference/model-runner-v2/mrv2-plan-906.html#layout) · [the opt-in switch](https://joerunde.github.io/spyre-inference/model-runner-v2/mrv2-plan-906.html#switch) · [warmup policy](https://joerunde.github.io/spyre-inference/model-runner-v2/mrv2-plan-906.html#warmup) · report: [upstream threads](https://joerunde.github.io/spyre-inference/model-runner-v2/mrv2-migration-906.html#upstream)

## Summary

The V2 runner carries several workarounds for things vLLM's MRV2 doesn't support for an out-of-tree, Triton-free platform yet. This issue tracks each one, the upstream change that would let us remove it, and where that change is being discussed. It doesn't hold code. Replacement kernels and their performance are tracked separately, in the upstream-kernels issue (#1238).

## Workarounds

| Workaround in `v1/worker/spyre/` | Why we need it | Upstream change that removes it | Thread |
|---|---|---|---|
| `triton_gate.py`: bypass `HAS_TRITON` and `_validate_v2_model_runner` at plugin load | MRV2 refuses to start without Triton, and the check is keyed on Triton being installed, not on what the platform provides | A platform capability for running MRV2 without Triton, usable by an out-of-tree platform rather than keyed on `is_cpu()` | vllm#57684 |
| `triton_gate.py` compile filter: drop `"stock torch.compile"` from the unsupported list | Our default compile mode is rejected for MRV2 on 0.28 and 0.31 | vllm#59079 (merged on main) reaching our pin | vllm#59079 |
| `device_shims.py`: process-wide `torch.cuda` Stream/Event placeholders | MRV2 creates CUDA streams and events every step | Stream/event APIs that aren't tied to CUDA | to raise |
| `warmup.py`: pick bucketed shapes ourselves, because `is_profile=True` skips `dispatch()` | MRV2 only allows `skip_attn` together with `is_profile` | A warmup flag that skips attention without implying `is_profile` | to raise |
| `pool/output.py`: synchronous output instead of `AsyncPoolingOutput` | MRV2's pooling output path uses CUDA events and a non-blocking D2H copy | A pooling output path that isn't tied to CUDA events | to raise |
| `attn_groups.py`: split merged attention groups by sliding window | `FullAttentionSpec.merge` collapses windows, so hybrid models get one window on every layer | Include the sliding window in `init_attn_backend`'s group key | to raise |
| `kv_cache.py`: our own per-layer allocator inside `initialize_kv_cache` | MRV2's allocator wants one shared backing tensor, which torch-spyre can't re-view | A per-platform KV allocation hook | to raise |
| Several component subclasses (bucket manager, pooling runner, model-state mixin) | MRV2 builds its components itself | A platform-provided component factory | vllm#52719 / #53895, RFC vllm#51212 |

## Acceptance criteria

- [ ] Each workaround with an upstream change has a linked thread; "to raise" entries are raised and linked
- [ ] When an upstream change reaches our pin, the matching workaround is removed, or a follow-up is filed to remove it
- [ ] New workarounds added during bring-up are added to the table

## Dependencies

None. Update it as the other MRV2 issues land.
