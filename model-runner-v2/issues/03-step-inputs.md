# [MRV2] Step inputs: bucket padding and model state

Part of #906. Plan: [package layout](https://joerunde.github.io/spyre-inference/model-runner-v2/mrv2-plan-906.html#layout) · [test plan](https://joerunde.github.io/spyre-inference/model-runner-v2/mrv2-plan-906.html#tests) · [starting before 0.31](https://joerunde.github.io/spyre-inference/model-runner-v2/mrv2-plan-906.html#pre031)

## Summary

Everything that decides what goes into each forward step. MRV2 has one padding decision, `CudaGraphManager.dispatch()`; without an override it pads nothing, and every distinct token count recompiles every block. MRV2 builds model inputs and attention metadata in a `ModelState` helper instead of private runner methods. This issue adds our bucket manager and moves our V1 input preparation (grid expansion, `token_type_ids`, the RoBERTa position offset, the encoder plan) onto the MRV2 model-state seams. The model state builds the encoder grid from the padded row count, so the two halves are built together.

## Scope

**WP-B: bucketing**
- [ ] `bucketing.py`: `SpyreBucketManager(ModelCudaGraphManager)`. `dispatch()` returns `BatchExecutionDescriptor(cg_mode=NONE, num_tokens=bucket(n), num_reqs=…)` using `SpyreShapeBucketer`. Replaces V1's `_determine_batch_execution_and_padding` and `_spyre_bucket_batch_descriptor`

**WP-D: model states**
- [ ] `model_states/__init__.py`: `resolve_spyre_model_state()` puts the Spyre mixin over whichever upstream class MRV2 resolves to
- [ ] `model_states/base.py`: `SpyreModelStateMixin` (the embeds hook moves to the startup issue)
- [ ] `model_states/default.py`: decoder `prepare_inputs` / `prepare_attn`
- [ ] `model_states/encoder_only.py`: grid expansion, `token_type_ids`, RoBERTa position offset, `prepare_attn` from the plan
- [ ] `model_states/encoder_plan.py`: `EncoderPlan` (rectangle vs ragged, grid, rows), the forced plan, dispatch counters. This state moves off the V1 runner

## Acceptance criteria

- [ ] `dispatch()` pads every token count to a bucket from `SpyreShapeBucketer` and always returns `cg_mode=NONE` (unit test over a range of token counts)
- [ ] `test_encoder_dense_grid.py` and `test_encoder_inline_dispatch.py`, ported to the MRV2 `prepare_inputs` / `prepare_attn` signatures with fake input batches, pass in `tests/mrv2/model_states/`
- [ ] Decoder `prepare_attn` produces metadata with every field `SpyreAttentionMetadataBuilder.build` reads (unit test against the builder's field list)
- [ ] The encoder plan is published as `EncoderPlan` only; the pooling and warmup code reads it through that interface
- [ ] `tests/mrv2` passes on `main` and on the #945 branch

## Out of scope

The multimodal embeds hook (`get_mm_embeddings` is renamed to `prepare_inputs_embeds` in 0.31; it goes in the startup issue). Running the encoder or a model.

## Dependencies

- Depends on: skeleton.
- Blocks: the pooling half of step outputs (reads `EncoderPlan`), warmup (enumerates buckets), end-to-end bring-up.

Version: all on 0.28.

WP-B and WP-D can go to different people.
