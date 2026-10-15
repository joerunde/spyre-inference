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
| **Compiled** (STOCK_TORCH_COMPILE via the gate's compile filter on 0.31), greedy, `SPYRE_COMPILE_GUARD=warn` | **7/7 token-exact vs compiled V1, zero compiles after warmup**, decode wall time equal to V1 (1.2s). 4 body buckets + 100 attention graphs recorded, as V1 |
| gemma-3-1b-it compiled (sliding-window hybrid, 22x512 + 4 full layers) | 7/7 token-exact vs V1, zero post-warmup compiles |
| gemma-3-1b-it compiled, + two ~1.2k-token prompts (chunked prefill, past the 512 window), 9 prompts at max_num_seqs=4 | 9/9 token-exact vs V1, zero post-warmup compiles |
| Token-major KV (`SPYRE_ATTN_KV_LAYOUT=token_major`, per-sequence decode loop), compiled | 7/7 token-exact vs V1, zero post-warmup compiles |
| **TP2** compiled | 7/7 token-exact vs V1, zero post-warmup compiles |
| Structured outputs (choice + JSON schema, xgrammar), eager | identical to V1; JSON parses |
| Async scheduling | on by default for decoders on both runners (multiproc executor); every decoder row above ran with it |

Pooling (compiled unless noted, `SPYRE_COMPILE_GUARD=warn`, all zero post-warmup compiles):

| Model | Result vs V1 |
|---|---|
| granite-embedding-125m-english (CLS), eager | cosine 1.0 on 6/6 |
| granite-embedding-125m-english (CLS), compiled, rectangle (64, 4) | cosine 1.0 on 6/6 |
| all-MiniLM-L6-v2 (MEAN, head 32 padded to 64), max_model_len 256, 8 seqs, rectangles (64,32),(128,16),(256,8) | cosine 1.0 on 6/6 |
| bge-reranker-large (score) | identical scores (max diff 0.0) |

Startup times compare poorly between runners here: whichever runs second on a model
hits a warm Inductor cache.

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
4. **Attention recording must run under `torch.inference_mode()`.** MRV2's
   `execute_model` is decorated with it, and grad mode is a Dynamo guard on every
   recorded kernel; recording outside it recompiled every attention variant on
   the first real step (found via `TORCH_LOGS=recompiles`: `GLOBAL_STATE changed:
   grad_mode`). Fixed in `warmup.py`.
5. **Split attention groups need their own metadata builders.** MRV2's
   `init_attn_backend` creates builders in its phase 3, before our window split;
   V1 split before builders existed. Gemma-3 failed recording until
   `attn_groups.init` builds them for the new groups.
6. **Multimodal staging buffer.** Upstream `EncoderRunner.inputs_embeds` lives on
   the runner device (host); `get_inputs_embeds` then copies Spyre embeddings into
   a host buffer and the wrapper never moves float inputs back. `SpyreEncoderRunner`
   allocates it on Spyre and hands it to the wrapper (V1's self-copy trick), patched
   in via `interface.EncoderRunner` during model-state construction.
7. **Pooling hidden states vs `logits_indices`.** MRV2's `_dummy_run` ends with
   `hidden_states[input_batch.logits_indices]` (host index); with the wrapper keeping
   pooling outputs on Spyre that is a Spyre gather by a host index. Warmup turns
   `keep_outputs_on_device` off around the dummy, as V1 did.
8. **Partial prefix-cache hit (same block-aligned prompt twice) differs at the last
   token on both runners identically** (V1 and V2 give the same `[..., 32, 34]` vs a
   cold `[..., 32, 35]`). Not V2-specific; likely the cached path recomputing one token
   through a different kernel. Worth a separate look, since V1's
   `copy_kv_cache_blocks_inplace` would also choke on `SpyrePagedKVCache` tuples if a
   CoW copy were ever emitted.
3. **Prompt logprobs drift up to 0.047 vs V1** (same top-k keys). MRV2 computes
   prompt logits in its own chunking, so the lm_head matmul on Spyre runs at
   different row counts than V1's; fp16 accumulation differs. Not a blocker; worth
   a reference check against CPU.
