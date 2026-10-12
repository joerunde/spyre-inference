# [MRV2] Step outputs: sampling and pooling

Part of #906. Plan: [package layout](https://joerunde.github.io/spyre-inference/model-runner-v2/mrv2-plan-906.html#layout) · [decisions](https://joerunde.github.io/spyre-inference/model-runner-v2/mrv2-plan-906.html#decisions) · [hardware during Wave 1](https://joerunde.github.io/spyre-inference/model-runner-v2/mrv2-plan-906.html#hwtests)

## Summary

Turning the forward's output into host results. MRV2's sampler skips `TopKTopPSampler`, which our sort-free sampler replaces today, and its torch fallback takes the sorting path. Its pooling path slices hidden states on the device, builds the cursor on the device, and returns output through CUDA events and a non-blocking D2H copy. This issue ports our top-k/top-p onto an MRV2 `Sampler` subclass, and our pooling onto a `PoolingRunner` subclass with synchronous, Spyre-safe output.

## Scope

**WP-E: sampler**
- [ ] `sample/topk_topp.py` (now): sort-free top-k/top-p and log-space draw, ported from `SpyreTopKTopPSampler.forward_native`
- [ ] `sample/sampler.py` (after 0.31, because upstream `sampler.py` changes a lot by then): `SpyreSampler(Sampler)` wired in through `ModelState.custom_sampler`. Everything else goes through the kernel seam
- [ ] RNG semantics as decided (plan decision 3: adopt MRV2's per-position seeding)

**WP-F: pooling**
- [ ] `pool/pooling_runner.py`: `SpyrePoolingRunner(PoolingRunner)` with a host cursor, no dim-0 slice on Spyre, grid-direct CLS/MEAN, unpadding, and the FP32-head CPU branch. Reads the encoder plan only through `EncoderPlan`
- [ ] `pool/output.py`: synchronous pooler output with a Spyre-safe D2H copy, replacing `AsyncPoolingOutput`. Its own copy of `copy_pooler_output_to_cpu` (the V1 one imports a private V1 symbol)

## Acceptance criteria

- [ ] `test_spyre_topk_sampler.py`, ported to `tests/mrv2/sample/` with CPU logits of [4, 32], passes
- [ ] After 0.31: `custom_sampler` returns `SpyreSampler`, and a unit test shows top-k/top-p goes through our implementation, not upstream's sorting fallback
- [ ] `test_spyre_pooler_select_rows.py`, ported to `tests/mrv2/pool/`, passes. Output comes back synchronously, and `finished_mask` comes from the cursor
- [ ] Light hardware test: pool a small [4, 64] Spyre tensor through `pool/output.py` and match the CPU result (skips on CPU hosts)
- [ ] `tests/mrv2` passes on `main` and on the #945 branch (except the `SpyreSampler` subclass, which lands after 0.31)

## Out of scope

Sampler kernels (gumbel, temperature, logprobs, penalties): they come through the platform-glue issue's kernel seam. Structured outputs and prompt logprobs, which are handled at bring-up.

## Dependencies

- Depends on: skeleton. Pooling also reads `EncoderPlan` from the step-inputs issue (through the skeleton interface, so the two can be worked at the same time). `SpyreSampler` needs #945.
- Blocks: end-to-end bring-up (pooling and sampling steps).

Size: 6–9 days. WP-E and WP-F can go to different people.
