# MRV2 bring-up log (fork-only spike, `mrv2-full`)

Fast, isolated spike of the full #906 plan on vLLM 0.31.0 (#945), to surface
blockers early. Not for review. Opt-in: `SPYRE_MODEL_RUNNER=v2`.

## Branch contents

- `main` + #945 (vLLM 0.31.0) + `mrv2-wpa-platform-glue` (WP-A) + #1256 (step inputs).
- Kernel step 1: triton-cpu 3.8.0 x86-only in `pyproject.toml` / `uv.lock`.
- Platform: #945's V1 pin (`VLLM_USE_V2_MODEL_RUNNER=0` + V1-only rejections) now
  applies only under `SPYRE_MODEL_RUNNER=v1`; the merge had left it running first.
- Extract-only shared helpers (decision 6), V1 calls them unchanged:
  - `v1/worker/block_compile.py`: `prepare_loader`, `fix_loaded_model`,
    `model_has_spyre_fp8`, `compile_blocks`, `compile_for_spyre`.
  - `v1/worker/kv_cache_setup.py`: `split_attn_groups_by_window`,
    `allocate_spyre_kv_caches`.
  - `spyre_encoder_attn.plan_encoder_step`: per-metadata plan with its own
    causality, the forced warmup rectangle with its causal mask, and the
    mixed-causality guard. Replaces #1256's drifted copy (the three drifts the
    #906 comment lists).
  - `v1/pool/encoder_grid.py`: `unpad_encoder_grid`, `rect_pooling_method`,
    `pool_rows_for` (the CLS/MEAN cursor rewrite), `warm_pooler_row_widths`,
    `warm_token_pool_widths`.

## How the V2 runner hooks upstream

Upstream `GPUModelRunner` methods run as-is; only module-level calls inside
`vllm.v1.worker.gpu.model_runner` are swapped for the duration of a call:

| Upstream call | Swapped for | Where |
|---|---|---|
| `get_model_loader` | loader whose `load_model` does host load, fix-ups, `.to(spyre)`, mm patches, per-block compile, `_SpyreModelWrapper` | `load.py` |
| `init_model_state` | `resolve_spyre_model_state` (Spyre mixin, `spyre_device`) | `load.py` |
| `init_attn_backend` | upstream + `split_attn_groups_by_window` | `attn_groups.py` |
| `init_kv_cache` | per-layer `SpyrePagedKVCache` on Spyre, returns `{}` | `kv_cache.py` |
| `PoolingRunner` | `SpyrePoolingRunner` | `load.py` |

Everything else MRV2 builds (block tables, input prep, sampler, prompt
logprobs, structured outputs) runs upstream's Triton kernels on host tensors
through triton-cpu.

## Results (x86 Spyre host, TP1, `micro-g3.3-8b-instruct-1b`, 7 prompts x 20 tokens)

| Step | Result |
|---|---|
| Eager, greedy | **7/7 token-exact vs V1**, no `FallbackWarning` |
| Eager, greedy + logprobs(3) + prompt_logprobs(2) | 7/7 token-exact, sample logprobs identical (max diff 0.0); prompt logprobs same keys, max diff 0.047 (see findings) |
| Eager, seeded top-k/top-p | runs; output differs from V1 by design (MRV2 per-position seeding, decision 3) |

## Findings / blockers

1. **`is_text_generation_model` on the wrapper (plan question 8): answered, no.**
   `ModelState.get_supported_generation_tasks` does isinstance checks on
   `self.model`, which is the `_SpyreModelWrapper`; the engine then refuses
   `generate`. Fixed in `SpyreModelStateMixin` by unwrapping for the check.
2. **No upstream hook for KV allocation / attention-group construction.**
   `initialize_kv_cache` calls module-level `init_attn_backend` / `init_kv_cache`;
   we swap them. 0.32's `ModelState.initialize_kv_cache` may give a seam.
   Upstream also filters the returned caches by `cache.device`, which a
   `SpyrePagedKVCache` tuple lacks, so we return `{}`; the KV connector and
   prefix-cache COW block copies (`copy_kv_cache_blocks_inplace`) therefore see
   no caches. COW copies are a latent gap for partial prefix-cache hits.
3. **Prompt logprobs drift up to 0.047 vs V1** (same top-k keys). MRV2 computes
   prompt logits in its own chunking, so the lm_head matmul on Spyre runs at
   different row counts than V1's; fp16 accumulation differs. Not a blocker; worth
   a reference check against CPU.
