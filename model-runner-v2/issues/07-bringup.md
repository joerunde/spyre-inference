# [MRV2] End-to-end bring-up on hardware

Part of #906. Plan: [hardware bring-up](https://joerunde.github.io/spyre-inference/model-runner-v2/mrv2-plan-906.html#bringup) · [warmup policy](https://joerunde.github.io/spyre-inference/model-runner-v2/mrv2-plan-906.html#warmup)

## Summary

The first time the whole V2 runner runs. Add an opt-in switch so a V2 run can be started without local patches, then bring the runner up model family by model family, from the smallest surface to the largest. Fixes go back into the module that owns the failing code, not into the runner. The existing V1 path stays the default throughout.

## Scope (WP-INT)

- [ ] Run everything with the opt-in switch `SPYRE_MODEL_RUNNER=v2` (added in the platform-glue issue)
- [ ] On the V2 path of `check_and_update_config`, reject MRV2 features we don't support: spec decode (MRV2 builds `self.speculator`, so V1's `hasattr(self, "drafter")` guard always passes), batch-sharded sampling (`compute_logits_local` bypasses the wrapper's H2D copy), `cudagraph_mm_encoder` (gemma-4 declares encoder CUDA-graph support) and PCP > 1
- [ ] Bring-up runs first on x86 with triton-cpu (kernel step 1), then repeats on x86, Power and Z once upstream's replacement kernels are bound and triton-cpu is dropped (step 2)
- [ ] Bring-up in this order, each with `-W error::torch_spyre.ops.fallbacks.FallbackWarning`:
  - [ ] Decoder, eager, TP1, greedy, head-major KV (a small granite)
  - [ ] Top-k/top-p and logprobs
  - [ ] Compiled: bucketing, per-block compile, attention recording, with `SPYRE_COMPILE_GUARD=error` (needs the gate's compile filter on 0.31, or vLLM 0.32)
  - [ ] Token-major KV, sliding-window hybrids (gemma), batched decode
  - [ ] Pooling and encoder-only models, rectangle and ragged plans
  - [ ] Multimodal, then TP2 and FP8
- [ ] Promote fakes that two or more test modules use into `tests/mrv2/fakes.py`

## Acceptance criteria

- [ ] Greedy decoder output is token-exact against V1 on the same model and settings
- [ ] Pooling output matches V1 by cosine similarity, at the existing references' thresholds
- [ ] Compiled runs record zero new graphs after warmup (`SPYRE_COMPILE_GUARD=error`), and cold warmup time is level with V1's measured baseline: 724 s for granite-3.3-8b-instruct, TP1, `max_model_len=2048`, `max_num_seqs=32` on x86 (harness in `docs/investigations/mrv2-warmup-measurements/` on `mrv2-investigation-906`)
- [ ] Each rejected feature fails at startup with a clear message (unit tests of `check_and_update_config`)
- [ ] No `FallbackWarning` on any step above
- [ ] With the switch off, V1 behaviour is unchanged (existing CI is green)
- [ ] The criteria above hold on x86 with triton-cpu, and then on x86, Power and Z on the replacement kernels once step 2 lands

## Out of scope

Making V2 the default and deleting V1 (the two flip issues, 08a and 08b).

## Dependencies

- Depends on: all Wave 1 issues. Compiled steps need the gate's compile filter on 0.31, or vLLM 0.32.
- Blocks: the flip (08a, then 08b).
