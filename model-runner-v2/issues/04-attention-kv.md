# [MRV2] Attention groups and KV cache

Part of #906. Plan: [package layout](https://joerunde.github.io/spyre-inference/model-runner-v2/mrv2-plan-906.html#layout) · [starting before 0.31](https://joerunde.github.io/spyre-inference/model-runner-v2/mrv2-plan-906.html#pre031) · [hardware during Wave 1](https://joerunde.github.io/spyre-inference/model-runner-v2/mrv2-plan-906.html#hwtests)

## Summary

MRV2 folds attention-group setup, metadata-builder creation and KV allocation into one `initialize_kv_cache` method. Our V1 overrides for those steps are never called under MRV2. If they're dropped, sliding-window hybrids silently get one window on every layer. MRV2's allocator also wants one shared backing tensor that torch-spyre can't re-view, and its block-table and slot-mapping kernels are Triton with pointer tables. This issue re-homes the window split, the per-layer Spyre allocator (head-major and token-major), and the block-table math.

## Scope (WP-C)

**Now (vLLM 0.28)**
- [ ] `attn_groups.py`: split merged full-attention groups by sliding window, then `create_metadata_builders` on the split groups. Ported from `_split_attn_groups_by_layer_window`
- [ ] `block_table.py`: `SpyreBlockTables(BlockTables)` with method-level overrides for `gather_block_tables` / `compute_slot_mappings` / `apply_staged_writes`, ported from `_compute_slot_mapping_impl`, plus `idx_mapping`, `-1 → PAD` and multi-group handling. Drop it if upstream's CPU `BlockTables` covers this by the time it lands

**After #945 (vLLM 0.31)**
- [ ] `kv_cache.py`: per-layer `SpyrePagedKVCache` in head-major and token-major layouts on the new `KVCacheTensor` schema, starting from #945's V1 port; bind to layers
- [ ] Declare `supported_kv_cache_layouts()` for both backends (head-major and token-major)

## Acceptance criteria

- [ ] A fake hybrid config with two window sizes gets one builder group per window, and the last `install()` wins (port of `test_attn_group_window_split.py`)
- [ ] `SpyreBlockTables` slot mappings match V1's `_compute_slot_mapping_impl` on the same inputs, including PAD and multiple groups
- [ ] On CPU: both layouts allocate the right shapes for 2 layers × 4 blocks × head size 64, and each layer gets its own tensor bound to the right layer
- [ ] Light hardware test: the same allocation on Spyre in both layouts (skips on CPU hosts)
- [ ] `tests/mrv2` passes on `main` and on the #945 branch until `kv_cache.py` lands, then on the 0.31 pin

## Out of scope

The attention kernels, the attention bucketer and the head-major store, which MRV2 doesn't change. Attention recording during warmup goes in the startup issue.

## Dependencies

- Depends on: skeleton. `kv_cache.py` also needs #945.
- Blocks: warmup's attention recording; end-to-end bring-up.

Size: 4–6 days.
