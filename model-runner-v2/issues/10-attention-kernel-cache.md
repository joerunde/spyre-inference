# Attention kernels miss the persistent kernel cache across restarts

Not part of #906: this doesn't depend on MRV2, and fixing it helps the V1 runner today. Found during the MRV2 warmup measurements. Report: [warmup measurements](https://joerunde.github.io/spyre-inference/model-runner-v2/mrv2-migration-906.html#warmup)

## Summary

Attention compiles dominate warmup, and none of them survive a restart. Measured on one x86 Spyre host (granite-3.3-8b-instruct, TP1, `max_model_len=2048`, `max_num_seqs=32`, 40 attention variants; harness in `docs/investigations/mrv2-warmup-measurements/` on `mrv2-investigation-906`):

- Recording the 40 attention variants is 64% of a 724 s cold warmup (about 11.6 s per variant), and 97% of a warm restart with the kernel cache on.
- **Inductor's FX graph cache:** all 40 attention graphs bypass it with `Can't cache HigherOrderOperator: scan`. The scan comes from torch-spyre's `for_each_tile`, which our attention kernels use by default (`SPYRE_ATTN_FOR_EACH_TILE=1`, `v1/attention/ops/tile_loop.py`).
- **torch-spyre's persistent kernel cache (`SPYRE_KERNEL_CACHE=1`):** block kernels hit 100% on a restart, but 37 of 40 attention kernels miss. Their cache keys differ between processes. Pinning `PYTHONHASHSEED=0` didn't help (on a 4-layer model, 36 of 40 still missed). The key covers the serialised kernel specs, affine strides and baked pool/slice offsets (`torch_spyre/execution/kernel_cache.py:296-375`), so something in the attention kernels' specs is process-dependent.
- Turning the kernel cache on costs nothing cold (741 s vs 724 s, within noise) and saves about 230 s of block compiles on every restart.

If attention kernels hit the cache across restarts, a warm restart's warmup would fall from about 510 s to roughly 15–20 s. That's a projection from these runs, not a measurement.

## Scope

- [ ] File a torch-spyre issue for the cross-process attention cache misses. Attach two cache directories from back-to-back runs and enable `kernel_cache` DEBUG logging to diff the hash ingredients
- [ ] Measure `SPYRE_ATTN_FOR_EACH_TILE=0` (Python loops instead of `scan`): whether attention graphs then hit the FX graph cache, and what it costs cold
- [ ] Track getting `scan` supported by Inductor's FX graph cache (PyTorch / torch-spyre)
- [ ] Decide whether to recommend `SPYRE_KERNEL_CACHE=1` with a persistent `TORCHINDUCTOR_CACHE_DIR` volume (one writer at a time) for deployments now

## Acceptance criteria

- [ ] The torch-spyre issue is filed and linked here
- [ ] On a warm restart with the kernel cache on, attention kernels hit the cache (measured with the same harness), or the remaining blocker is identified upstream
- [ ] The deployment recommendation is documented

## Dependencies

None. Can start now, independently of the MRV2 work. Repeating the measurements on Power and Z would give the numbers that matter for availability SLAs.
