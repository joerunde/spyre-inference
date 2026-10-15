# MRV2 bring-up log (fork-only spike, `mrv2-full`)

A fast, isolated spike of the full #906 plan on vLLM 0.31.0 (#945), to surface
blockers early. Not for review. Opt in with `SPYRE_MODEL_RUNNER=v2`; V1 stays the default.

## TL;DR

On x86 with triton-cpu, the V2 runner matches V1 across every configuration tried:
decoder eager and compiled, TP1 and TP2, head-major and token-major KV, a
sliding-window hybrid with long chunked prompts, structured outputs, pooling (CLS,
MEAN, score) on the rectangle path, and CLIP text and image embeddings. It is
token-exact or cosine 1.0, with zero compiles after warmup. Compiled V2 runs on 0.31
with the gate's compile filter, so nothing here waits for 0.32. The one performance
regression found (top-k/top-p, −26%) is fixed by binding V1's sort-free host version
through `kernels.py` (now +3%).

The real blockers are not on x86:

- **Power/Z cannot run V2 at all** until upstream's non-Triton MRV2 kernels exist
  (kernel step 2). `kernels.bind_or_raise()` now fails at worker start there,
  rather than at the first kernel launch.
- **No upstream seam** for KV allocation or attention-group construction. We swap
  module-level calls inside upstream's `initialize_kv_cache` (see *Hooks*). Each bump
  needs `tests/mrv2` plus a bring-up rerun.

## Branch contents

- `main` + #945 (vLLM 0.31.0) + `mrv2-wpa-platform-glue` (WP-A) + #1256 (step inputs).
- Kernel step 1: triton-cpu 3.8.0, x86-only, in `pyproject.toml` / `uv.lock`. The relock
  also fixed stale `xgrammar` / `oss-harmony` pins that the #945 merge had left.
- Platform: #945's V1 pin (`VLLM_USE_V2_MODEL_RUNNER=0` plus the V1-only rejections)
  now applies only under `SPYRE_MODEL_RUNNER=v1`; the merge had left it running ahead
  of the switch. V2 rejects spec decode, batch-sharded sampling and PCP>1, and forces
  `cudagraph_mm_encoder` off.
- Extract-only shared helpers (decision 6). V1 calls them with no behaviour change, and
  the existing V1 tests pass (650 in `tests/runtime`, `tests/pool`, `test_fp8_linear`):
  - `v1/worker/block_compile.py`: `prepare_loader`, `fix_loaded_model`,
    `model_has_spyre_fp8`, `compile_blocks`, `compile_for_spyre`.
  - `v1/worker/kv_cache_setup.py`: `split_attn_groups_by_window`,
    `allocate_spyre_kv_caches`.
  - `spyre_encoder_attn.plan_encoder_step`: one plan per metadata with that group's own
    causality, the forced warmup rectangle with its causal mask, and the
    mixed-causality guard. Replaces #1256's drifted copy (the three drifts the #906
    comment lists).
  - `v1/pool/encoder_grid.py`: `unpad_encoder_grid`, `rect_pooling_method`,
    `pool_rows_for` (the CLS/MEAN cursor rewrite), `warm_pooler_row_widths`,
    `warm_token_pool_widths`.
- V2 package filled in: `load`, `kv_cache`, `attn_groups`, `bucketing` (warmup gate),
  `warmup` (plan/run, pooling, multimodal), `kernels` (default-path list, coverage
  test, top-k/top-p override), `compile` (whole-model-compile tripwire),
  `pool/{pooling_runner,output}`, `mm/encoder_runner`.
- Not done: #1245 (V1 removal, out of scope by request); `block_table.py` stays a stub
  (see *Hooks*); upstream PR comments (decision 5); Power/Z (no hardware path, see
  above).

## Hooks: how the V2 runner reaches into upstream

Upstream `GPUModelRunner` methods run as-is. Only module-level names that upstream
looks up are swapped, and only for the duration of the call that uses them:

| Upstream name | Swapped for | Where |
|---|---|---|
| `model_runner.get_model_loader` | a loader whose `load_model` does the host load, fix-ups, `.to(spyre)`, mm patches and per-block compile, and returns `_SpyreModelWrapper` | `load.py` |
| `model_runner.init_model_state` | `resolve_spyre_model_state` (Spyre mixin, `spyre_device`) | `load.py` |
| `model_runner.PoolingRunner` | `SpyrePoolingRunner` | `load.py` |
| `model_states.interface.EncoderRunner` | `SpyreEncoderRunner` (Spyre-resident `inputs_embeds`) | `model_states/base.py` |
| `model_runner.init_attn_backend` | upstream, plus the window split, plus builders for the new groups | `attn_groups.py` |
| `model_runner.init_kv_cache` | per-layer `SpyrePagedKVCache` on Spyre; returns `{}` | `kv_cache.py` |
| `sample.{sampler,states}.apply_top_k_top_p` | sort-free host top-k/top-p (bound once, in `kernels.bind()`) | `kernels.py` |

Everything else MRV2 builds runs upstream's Triton kernels on host tensors through
triton-cpu: block tables and slot mappings, input prep, staged writes, the sampler,
prompt logprobs and structured outputs. So `block_table.py` is unnecessary at kernel
step 1.

## Results

x86 Spyre host. Unless noted, the model is `micro-g3.3-8b-instruct-1b`, with 7 prompts
of 20 tokens, greedy, async scheduling on (the default), and `SPYRE_COMPILE_GUARD=warn`.
Every compiled row had **zero compiles after warmup**.

| Configuration | V2 vs V1 |
|---|---|
| Eager | 7/7 token-exact; no `FallbackWarning` |
| Eager, logprobs(3) + prompt_logprobs(2) | 7/7 token-exact; sample logprobs identical; prompt logprobs have the same keys, max diff 0.047 (finding 9) |
| Eager, seeded top-k/top-p | runs; differs from V1 by design (MRV2's per-position seeding, decision 3) |
| **Compiled** (0.31, the gate's compile filter) | **7/7 token-exact**; decode wall time equal (1.2s); same 4 body buckets and 100 recorded attention graphs |
| gemma-3-1b-it compiled (22 layers at window 512 + 4 full) | 7/7 token-exact |
| gemma-3-1b-it compiled, plus two ~1.2k-token prompts (chunked prefill past the window), 9 prompts at 4 seqs | 9/9 token-exact |
| Token-major KV (per-sequence decode loop), compiled | 7/7 token-exact |
| **TP2** compiled | 7/7 token-exact |
| **granite-3.3-8b-instruct TP4** compiled, plus two ~1.2k-token prompts | **9/9 token-exact**; 1000 attention graphs per rank; generate 8.5s vs V1 10.1s |
| FP8 (granite-4.1-3b-fp8) compiled, 800 attention graphs | 7/7 token-exact; decode equally slow on both (146s, the existing FP8 path) |
| Structured outputs (choice + JSON schema, xgrammar), eager | identical; the JSON parses |
| Partial prefix-cache hit (same 2-block prompt twice) | identical to V1, including V1's own last-token difference (finding 8) |

Pooling (all with zero post-warmup compiles):

| Model | V2 vs V1 |
|---|---|
| granite-embedding-125m-english (CLS), eager and compiled (rectangle (64, 4)) | cosine 1.0, 6/6 |
| all-MiniLM-L6-v2 (MEAN, head 32 padded to 64), 256 len, 8 seqs, 3 rectangles | cosine 1.0, 6/6 |
| bge-reranker-large (score) | identical scores |
| CLIP ViT-B/32: `tests/e2e/test_clip_embeddings.py` run with `SPYRE_MODEL_RUNNER=v2` | **7/7 pass** against HF (text and image, eager and compiled, batched) |
| `tests/e2e/test_encoder_models.py` run with `SPYRE_MODEL_RUNNER=v2` (6 embedding models, MEAN multi-seq, LAST, rerankers incl. the long (1024, 2) rectangle, token classify) | **21/21 pass** against stored references |

Decode throughput: compiled, 32 concurrent requests, 64 tokens each, `ignore_eos`.

| | V1 | V2 (as upstream) | V2 + sort-free top-k/top-p |
|---|---|---|---|
| greedy | 555 tok/s | 579 tok/s (+4%) | 567 tok/s (+2%) |
| top-k=50, top-p=0.9 | 533 tok/s | **397 tok/s (−26%)** | 550 tok/s (+3%) |

Startup times are not comparable between runners in this log: whichever runner goes
second on a model hits a warm Inductor cache.

## Findings

1. **Power/Z: V2 cannot run there until kernel step 2.** MRV2's default path has 35
   `@triton.jit` functions (`kernels.DEFAULT_PATH_KERNELS`; `test_kernel_coverage.py`
   diffs that list against the pinned vLLM). Without Triton they are placeholders,
   so `bind_or_raise()` now fails at worker start.
2. **MRV2's top-k/top-p cost 26% decode throughput at 32 seqs.** With the gate open,
   upstream takes `apply_top_k_top_p_triton` on triton-cpu; without Triton it would take
   the full-vocabulary sort, because `is_cpu()` is False for our OOT platform. Binding
   V1's sort-free version (`apply_top_k_top_p_sort_free`, with `apply_top_k_only`
   for k-only) recovers it. This is kernel step 3a reached early. It belongs in the
   upstream ask (gate on capability, not `is_cpu()`).
3. **No upstream seam for KV allocation or attention-group construction.**
   `initialize_kv_cache` calls module-level `init_attn_backend` / `init_kv_cache`, and
   we swap both. Upstream also filters its returned caches by `cache.device`, which a
   `SpyrePagedKVCache` tuple lacks, so we return `{}`. The KV connector and
   prefix-cache CoW copies (`copy_kv_cache_blocks_inplace`) therefore see no caches
   (finding 8). 0.32's `ModelState.initialize_kv_cache` may be the seam to ask for.
4. **`is_text_generation_model` / `is_pooling_model` on the wrapper (plan question
   8): no.** These are isinstance checks, so the engine refused `generate` and
   `PoolingRunner` refused every task. The mixin, `SpyrePoolingRunner` and the runner's
   `get_supported_tasks` unwrap for the check.
5. **Attention recording must run under `torch.inference_mode()`.** MRV2's
   `execute_model` is decorated with it, and grad mode is a Dynamo guard on every
   recorded kernel. Recording outside it recompiled every attention variant on the
   first real step (`TORCH_LOGS=recompiles`: `GLOBAL_STATE changed: grad_mode`), and
   cost 25x decode time on the smoke run.
6. **Split attention groups need their own metadata builders.** MRV2 builds builders
   inside `init_attn_backend`, before our window split; V1 splits before builders exist.
   Gemma-3 failed recording until `attn_groups.init` builds them for the new groups.
7. **Multimodal staging buffer.** Upstream's `EncoderRunner.inputs_embeds` lives on the
   runner device (the host). `get_inputs_embeds` copies Spyre embeddings into it, and the
   wrapper never moves float inputs back to Spyre. `SpyreEncoderRunner` allocates it on
   Spyre and hands it to the wrapper, so V1's self-copy trick still applies.
   Multimodal *pooling* stays on the packed path, since #1256 never expands
   `inputs_embeds` onto the grid; CLIP passes regardless.
8. **Partial prefix-cache hit: shared with V1, not V2-specific.** The same block-aligned
   prompt, sent twice, differs at its last greedy token on both runners identically.
   Probably the cached path recomputes one token through a different kernel. V1's
   `copy_kv_cache_blocks_inplace` call would also fail on `SpyrePagedKVCache` tuples
   if a CoW copy were ever emitted.
9. **Prompt logprobs drift up to 0.047 against V1** (same top-k keys). MRV2 computes
   prompt logits in its own chunks, so the lm_head matmul on Spyre runs at different
   row counts; fp16 accumulation differs. Not a blocker; worth a CPU reference check.
10. **Pooling hidden states and `logits_indices`.** MRV2's `_dummy_run` ends with
    `hidden_states[logits_indices]` (a host index). Warmup turns the wrapper's
    `keep_outputs_on_device` off around the dummy run, as V1 did.
11. **`tests/mrv2` time budget is flaky.** The first test collected absorbs lazy-import
    cost and goes over 1s (pre-existing on the merged branch).

## Reproducing

```bash
uv sync --group dev --no-install-package torch-spyre --inexact   # picks up triton-cpu
SPYRE_MODEL_RUNNER=v2 python -c "from vllm import LLM; print(LLM('ibm-ai-platform/micro-g3.3-8b-instruct-1b', max_model_len=1024).generate('Hello')[0].outputs[0].text)"
SPYRE_MODEL_RUNNER=v2 uv run --no-sync pytest tests/e2e/test_clip_embeddings.py   # any e2e suite
uv run --no-sync pytest tests/mrv2
```
