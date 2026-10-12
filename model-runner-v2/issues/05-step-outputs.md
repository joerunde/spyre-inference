# [MRV2] Step outputs: sampling and pooling

Part of #906. Plan: [kernel strategy](https://joerunde.github.io/spyre-inference/model-runner-v2/mrv2-plan-906.html#kernels) · [decisions](https://joerunde.github.io/spyre-inference/model-runner-v2/mrv2-plan-906.html#decisions) · [hardware during Wave 1](https://joerunde.github.io/spyre-inference/model-runner-v2/mrv2-plan-906.html#hwtests)

## Summary

Turning the forward's output into host results. For sampling, we use MRV2's `Sampler` as-is: its kernels run through the x86-only locked triton-cpu until upstream's replacements exist, then through those (plan kernel steps 1 and 2). Our V1 sampler pieces (sort-free top-k/top-p, the vendored V1 CPU sampling kernels from #1111) are not ported up front; they're candidates if Power and Z measurements call for them (kernel step 3). The sampling work here is integration checks. Pooling is the larger half: MRV2's pooling path slices hidden states on the device, builds the cursor on the device, and returns output through CUDA events and a non-blocking D2H copy, so it needs a `PoolingRunner` subclass with synchronous, Spyre-safe output.

## Scope

**WP-E: sampler integration** (after #945; x86 with triton-cpu)
- [ ] Confirm MRV2's `Sampler` works unmodified for greedy, top-k/top-p and logprobs
- [ ] Measure MRV2's top-k/top-p cost on x86 against V1: it calls torch's `apply_top_k_top_p` with `allow_cpu_sync=current_platform.is_cpu()`, False for us, so it sorts the full vocabulary whatever the kernel source. Post the numbers; V1's sort-free path is a kernel step 3a candidate
- [ ] The warmup sampler dummy covers the logits row widths serving uses
- [ ] Structured outputs on the xgrammar CPU path (MRV2's `StructuredOutputsWorker` creates a `torch.cuda.Stream` and uses a Triton bitmask kernel)
- [ ] Prompt logprobs: MRV2 uses 1024-row chunks, which miss our lm_head row buckets. Add a bucket or reject prompt logprobs
- [ ] Seeded sampling follows MRV2's per-position seeding (plan decision 3). Local references are all greedy, so none need regenerating

**WP-F: pooling**
- [ ] `pool/pooling_runner.py`: `SpyrePoolingRunner(PoolingRunner)` with a host cursor, no dim-0 slice on Spyre, grid-direct CLS/MEAN, unpadding, and the FP32-head CPU branch. Reads the encoder plan only through `EncoderPlan`
- [ ] `pool/output.py`: synchronous pooler output with a Spyre-safe D2H copy, replacing `AsyncPoolingOutput`. Its own copy of `copy_pooler_output_to_cpu` (the V1 one imports a private V1 symbol)

## Acceptance criteria

- [ ] A small test runs MRV2's `Sampler` on [4, 32] CPU logits (greedy and top-k): through triton-cpu on x86 until kernel step 2, skipping elsewhere, then through the replacement kernels on every architecture. Marked `triton_cpu` (exempt from the time budget, since Triton JIT-compiles on a cold cache)
- [ ] Manual check on x86: greedy output from MRV2's sampler is token-exact against V1 on the same logits; top-k/top-p and logprobs run without errors
- [ ] The prompt-logprobs decision is made and implemented
- [ ] `test_spyre_pooler_select_rows.py`, ported to `tests/mrv2/pool/`, passes. Output comes back synchronously, and `finished_mask` comes from the cursor
- [ ] Light hardware test: pool a small [4, 64] Spyre tensor through `pool/output.py` and match the CPU result (skips on CPU hosts)
- [ ] `tests/mrv2` passes on `main` and on the #945 branch

## Out of scope

Sampler kernels and their performance on Power and Z: they come through the platform-glue issue (kernel steps 2 and 3).

## Dependencies

- Depends on: skeleton. Pooling also reads `EncoderPlan` from the step-inputs issue (through the skeleton interface, so the two can be worked at the same time). Sampler checks need #945 and triton-cpu on x86.
- Blocks: end-to-end bring-up (pooling and sampling steps).

Size: 5–7 days. WP-E and WP-F can go to different people.
