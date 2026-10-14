# [MRV2] Attention groups and KV cache

Filed as https://github.com/torch-spyre/spyre-inference/issues/1240.

Part of #906. Plan: [package layout](https://joerunde.github.io/spyre-inference/model-runner-v2/mrv2-plan-906.html#layout) · [starting before 0.31](https://joerunde.github.io/spyre-inference/model-runner-v2/mrv2-plan-906.html#pre031) · [hardware during Wave 1](https://joerunde.github.io/spyre-inference/model-runner-v2/mrv2-plan-906.html#hwtests) · [shared helpers](https://joerunde.github.io/spyre-inference/model-runner-v2/mrv2-plan-906.html#shared)

## Summary

MRV2 folds attention-group setup, metadata-builder creation and KV allocation into one `initialize_kv_cache` method. Our V1 overrides for those steps are never called under MRV2. If they're dropped, sliding-window hybrids silently get one window on every layer. MRV2's allocator also wants one shared backing tensor that torch-spyre can't re-view, and its block-table and slot-mapping kernels are Triton with pointer tables. This issue re-homes the window split, the per-layer Spyre allocator (head-major and token-major), and the block-table math.

## Scope (WP-C)

**Extract first: V1 extract-only PRs**

Each extract-only PR moves the logic out of the V1 method into a shared function that the V1 method calls, with no behaviour change; it lands before the MRV2 PR that imports it, adds no V2 code, and is checked on hardware (the encoder and pooling tests skip on CPU-only hosts). Rules: [shared helpers](https://joerunde.github.io/spyre-inference/model-runner-v2/mrv2-plan-906.html#shared).

- [ ] Now, cherry-picked to release-0.5: `split_attn_groups_by_window(attn_groups, layers)` from `_split_attn_groups_by_layer_window`, into a new `v1/worker/kv_cache_setup.py`. `test_attn_group_window_split.py` calls it directly instead of a `__new__` runner
- [ ] After #945, `main` only (release-0.5 has the old `KVCacheTensor` schema): `allocate_spyre_kv_caches(kv_cache_config, static_ctx, shared_layers, device)` from `initialize_kv_cache_tensors`, into `kv_cache_setup.py`. Each runner keeps its own binding

About 50 lines move.

**Now (vLLM 0.28)**
- [ ] `attn_groups.py`: split merged full-attention groups by sliding window, then `create_metadata_builders` on the split groups, through the shared `split_attn_groups_by_window`
- [ ] `block_table.py`: `SpyreBlockTables(BlockTables)` with method-level overrides for `gather_block_tables` / `compute_slot_mappings` / `apply_staged_writes`, ported from `_compute_slot_mapping_impl`, plus `idx_mapping`, `-1 → PAD` and multi-group handling. Drop it if upstream's CPU `BlockTables` covers this by the time it lands

**After #945 (vLLM 0.31)**
- [ ] `kv_cache.py`: per-layer `SpyrePagedKVCache` in head-major and token-major layouts on the new `KVCacheTensor` schema, allocated through the shared `allocate_spyre_kv_caches`; bind to layers. `kv_cache.init` sets every runner attribute the skeleton's list assigns to it (likely by calling `super().initialize_kv_cache` with only the allocation and attention-group creation replaced), so `block_tables`, `kv_connector` and the rest still exist
- [ ] Declare `supported_kv_cache_layouts()` for both backends (head-major and token-major)

## Acceptance criteria

- [ ] A fake hybrid config with two window sizes gets one builder group per window, and the last `install()` wins (port of `test_attn_group_window_split.py`)
- [ ] `SpyreBlockTables` slot mappings match V1's `_compute_slot_mapping_impl` on the same inputs, including PAD and multiple groups
- [ ] On CPU: both layouts allocate the right shapes for 2 layers × 4 blocks × head size 64, and each layer gets its own tensor bound to the right layer
- [ ] Light hardware test: the same allocation on Spyre in both layouts (skips on CPU hosts)
- [ ] After `initialize_kv_cache`, every attribute on the skeleton's list is set (unit test)
- [ ] `tests/mrv2` passes on `main` and on the #945 branch until `kv_cache.py` lands, then on the 0.31 pin
- [ ] The extract-only PRs leave V1 unchanged: on-device attention and KV tests pass on `main`, and on the release-0.5 backport for the window split

## Out of scope

The attention kernels, the attention bucketer and the head-major store, which MRV2 doesn't change. Attention recording during warmup goes in the startup issue (#1242).

## Dependencies

- Depends on: skeleton (#1236); the extract-only PRs above. `kv_cache.py` also needs #945.
- Blocks: warmup's attention recording; end-to-end bring-up (#1243).

