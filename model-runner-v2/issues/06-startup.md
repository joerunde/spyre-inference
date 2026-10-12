# [MRV2] Startup: load, compile, warmup, multimodal

Part of #906. Plan: [package layout](https://joerunde.github.io/spyre-inference/model-runner-v2/mrv2-plan-906.html#layout) · [warmup policy](https://joerunde.github.io/spyre-inference/model-runner-v2/mrv2-plan-906.html#warmup) · [version targeting](https://joerunde.github.io/spyre-inference/model-runner-v2/mrv2-plan-906.html#versions)

## Summary

Everything that happens once, at startup. MRV2's `load_model` builds the model state, sampler, pooling runner and encoder runner, each holding a reference to `self.model`. Our wrapper has to be what all of them hold, or pixel values stay on CPU and embeddings skip bucketing. From 0.32, MRV2 also compiles the whole model, which would nest over our per-block compiles, so that has to be suppressed. MRV2's default warmup dummies add attention compiles; the measured policy is to skip attention in decoder dummies. Multimodal is here because its main risk is the same load ordering, plus a staging buffer that has to live on Spyre.

## Scope

**WP-G: load and compile** (after 0.31; compiled runs on 0.31 rely on the gate's compile filter, and suppression is checked again at 0.32)
- [ ] `load.py`: `load_and_wrap(runner)`. Wrap the loader for the head-pad hooks, load on CPU, run the fix-ups, `.to(spyre)`, wrap, then point `model_state.model`, `encoder_runner.model` and `pooling_runner.model` at the wrapper
- [ ] `compile.py`: suppress MRV2's whole-model stock compile; run per-block compiles through the shared helpers
- [ ] Verify compiled V2 on 0.31 with the gate's compile filter (platform-glue issue). If it works, compiled bring-up and the flip don't wait for 0.32; if not, report what #59079 adds that we need
- [ ] Carry the V1 LoRA guard from `load_model` into `load.py`

**WP-H: warmup** (`plan()` now, `run()` after 0.31)
- [ ] `warmup.py`: `plan(runner)` returns the dummy calls. Decoder dummies use `_dummy_run(N, skip_attn=True, is_profile=True)` with already-bucketed `num_tokens` and `num_reqs = min(N, max_num_seqs)`, because `is_profile` bypasses `dispatch()`. Decoder dummies also publish null KV slots (`publish_null_slots`, as V1's `_dummy_run` does at `spyre_model_runner.py:1370`): with `skip_attn`, MRV2 builds no slot mappings, and the KV write inside the compiled block graph needs a target. Pooling dummies keep attention on
- [ ] `run(runner)` executes the plan, then the separate attention-recording pass over `attn_groups`, then the sampler dummy

**WP-I: multimodal** (after 0.31)
- [ ] `mm/encoder_runner.py`: replace `EncoderRunner.inputs_embeds` with a Spyre-resident staging buffer. Check that the wrapper is installed before `init_model_state`
- [ ] `ModelState.prepare_inputs_embeds` hook (renamed from `get_mm_embeddings` in 0.31)

## Acceptance criteria

- [ ] After `load_and_wrap`, `runner.model` is the wrapper and every holder points at it (unit test with a 2-block toy module)
- [ ] With `torch.compile` patched to a recorder: one call per repeated block, and no whole-model compile (unit test; re-checked on 0.32)
- [ ] `warmup.plan()` test: every `num_tokens` is a bucket value, `num_reqs` is `min(num_tokens, max_num_seqs)` and not 1, decoder dummies skip attention and publish null slots, and pooling dummies keep attention
- [ ] `is_text_generation_model(wrapper)` returns True (unit test; open question 8 in the report)
- [ ] Light hardware test: stage a small embed into the Spyre buffer and check the copy count (`buf[:n] = x` against torch-spyre's self-copy skip)
- [ ] Manual check on a Spyre host with `SPYRE_MODEL_RUNNER=v2 --enforce-eager`: the V2 worker loads and warms a small decoder

## Out of scope

Compiled end-to-end runs and the zero-compiles-after-warmup check, which happen at bring-up. Attention-cache misses across restarts, which don't depend on MRV2 (a known risk; caching work is under way separately).

## Dependencies

- Depends on: skeleton; #945. Warmup reads the bucket manager (step inputs), `attn_groups` (attention/KV) and `EncoderPlan` through the skeleton interfaces. Compiled runs need the gate's compile filter on 0.31, or vLLM 0.32.
- Blocks: end-to-end bring-up.

Size: 10–14 days. WP-G, WP-H and WP-I can go to different people.
