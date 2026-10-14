# [MRV2] Step inputs: bucket padding and model state

Filed as https://github.com/torch-spyre/spyre-inference/issues/1239.

Part of #906. Plan: [package layout](https://joerunde.github.io/spyre-inference/model-runner-v2/mrv2-plan-906.html#layout) · [test plan](https://joerunde.github.io/spyre-inference/model-runner-v2/mrv2-plan-906.html#tests) · [starting before 0.31](https://joerunde.github.io/spyre-inference/model-runner-v2/mrv2-plan-906.html#pre031) · [shared helpers](https://joerunde.github.io/spyre-inference/model-runner-v2/mrv2-plan-906.html#shared)

## Summary

Everything that decides what goes into each forward step. MRV2 has one padding decision, `CudaGraphManager.dispatch()`; without an override it pads nothing, and every distinct token count recompiles every block. MRV2 builds model inputs and attention metadata in a `ModelState` helper instead of private runner methods. This issue adds our bucket manager and moves our V1 input preparation (grid expansion, `token_type_ids`, the RoBERTa position offset, the encoder plan) onto the MRV2 model-state seams. The model state builds the encoder grid from the padded row count, so the two halves are built together.

## Scope

**Extract first: one V1 extract-only PR, cherry-picked to release-0.5**

Each extract-only PR moves the logic out of the V1 method into a shared function that the V1 method calls, with no behaviour change; it lands before the MRV2 PR that imports it, adds no V2 code, and is checked on hardware (the encoder and pooling tests skip on CPU-only hosts). Rules: [shared helpers](https://joerunde.github.io/spyre-inference/model-runner-v2/mrv2-plan-906.html#shared).

- [ ] Encoder plan selection into `spyre_encoder_attn.py`, next to `build_encoder_plan`: one plan per encoder metadata with that group's causality (from `_build_attention_metadata`), `forced_rect_plan(extent, width, dtype, device, causal)` (from `_forced_encoder_plan`), the mixed-causality `NotImplementedError` guard, and `EncoderDispatchStats` (from `_record_encoder_dispatch` and the five `spyre_encoder_*` counters)
- [ ] `expand_body_to_grid(...)` from the grid part of `_preprocess` (ids, positions, embeds, `token_type_ids`, the CLIP align mask) into a new `v1/worker/encoder_grid.py`
- [ ] `create_shape_bucketer(vllm_config)` and `bucket_pad(bucketer, n)` (with V1's pre-warmup gate) into `spyre_shape_bucketer.py`
- [ ] `test_encoder_inline_dispatch.py` and `test_encoder_dense_grid.py` call the shared functions instead of runner methods on a fake `self`

About 125 lines move. #1256 currently copies the encoder plan, and the copy has drifted from V1 in three ways (no causal mask on the forced rectangle, one plan for every group, no mixed-causality guard). It rebases onto this PR and imports the helpers instead.

**WP-B: bucketing**
- [ ] `bucketing.py`: `SpyreBucketManager(ModelCudaGraphManager)`. `dispatch()` returns `BatchExecutionDescriptor(cg_mode=NONE, num_tokens=bucket(n), num_reqs=…)` using `SpyreShapeBucketer`. Replaces V1's `_determine_batch_execution_and_padding` and `_spyre_bucket_batch_descriptor`, through the shared `create_shape_bucketer` and `bucket_pad`

**WP-D: model states**
- [ ] `model_states/__init__.py`: `resolve_spyre_model_state()` puts the Spyre mixin over whichever upstream class MRV2 resolves to
- [ ] `model_states/base.py`: `SpyreModelStateMixin` (the embeds hook moves to the startup issue, #1242)
- [ ] `model_states/default.py`: decoder `prepare_inputs` / `prepare_attn`
- [ ] `model_states/encoder_only.py`: grid expansion, `token_type_ids`, RoBERTa position offset, `prepare_attn` from the plan
- [ ] `model_states/encoder_plan.py`: `EncoderPlan` (rectangle vs ragged, grid, rows). This state moves off the V1 runner. The plan builder, forced plan and dispatch counters are the shared helpers above, not copies

## Acceptance criteria

- [ ] `dispatch()` pads every token count to a bucket from `SpyreShapeBucketer` and always returns `cg_mode=NONE` (unit test over a range of token counts)
- [ ] `test_encoder_dense_grid.py` and `test_encoder_inline_dispatch.py`, ported to the MRV2 `prepare_inputs` / `prepare_attn` signatures with fake input batches, pass in `tests/mrv2/model_states/`
- [ ] Decoder `prepare_attn` produces metadata with every field `SpyreAttentionMetadataBuilder.build` reads (unit test against the builder's field list)
- [ ] The encoder plan is published as `EncoderPlan` only; the pooling and warmup code reads it through that interface
- [ ] `tests/mrv2` passes on `main` and on the #945 branch
- [ ] The extract-only PR leaves V1 unchanged: the on-device encoder and pooling tests pass on `main` and on the release-0.5 backport
- [ ] No copy of V1's plan selection, forced plan or dispatch counters remains in `spyre/`; MRV2 imports the shared helpers
- [ ] With a causal and a bidirectional encoder group, each group gets its own plan with its own mask, the forced warmup rectangle carries the group's causality, and mixing them on the traced rectangle raises (unit test)

## Out of scope

The multimodal embeds hook (`get_mm_embeddings` is renamed to `prepare_inputs_embeds` in 0.31; it goes in the startup issue, #1242). Running the encoder or a model.

## Dependencies

- Depends on: skeleton (#1236); the extract-only PR above, for WP-D and for the bucketer helpers in WP-B.
- Blocks: the pooling half of step outputs (reads `EncoderPlan`), warmup (enumerates buckets), end-to-end bring-up (#1243).

Version: all on 0.28.

WP-B and WP-D can be done in parallel.

