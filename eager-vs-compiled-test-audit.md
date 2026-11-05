# Eager vs compiled test audit

Audit of `tests/` on main at `36c8e43` (2026-11-05), assuming #1504 is merged. The goal is to find
tests that run eager when the code's default production path is compiled (`STOCK_TORCH_COMPILE`,
head-major attention), so that redundant eager coverage can be removed and compiled-only paths get
coverage. Related: #1399 (remove token-major and non-for-each-tile kernels), #1503 and #1504 (the
CI runner-usage cuts), and the
[CI runner usage summary](https://joerunde.github.io/spyre-inference/ci_runner_usage/summary/index.html).

Line numbers are as of `36c8e43`. I checked the conftest skip, the eager-only custom-op tests, the
encoder `block_size` axis and the TP eager tests against the code by hand. The attention
parametrization counts come from reading the decorators, not from collection, so they are
approximate.

## How tests choose eager or compiled

- Local tests: no fixture forces eager. The session config and `default_vllm_config` (in
  `tests/plugin/spyre_testing_plugin/pytest_plugin.py`) build a `VllmConfig` without
  `enforce_eager`, so `apply_config_platform_defaults` sets `STOCK_TORCH_COMPILE`.
- A bare call to a `@maybe_compile` kernel therefore runs the production "compile outermost" path.
  A bare call to any other op (linear `apply`, rope `forward_oot`, `SiluAndMul`) really is eager.
- Upstream tests: `_force_eager_vllm_runner` (`pytest_plugin.py:942`) defaults every upstream
  `vllm_runner` to `enforce_eager=True`.
- Caveat: `test_parallel_lm_head.py:540` monkeypatches `get_cached_compilation_config` to STOCK,
  which suggests the cached mode can be stale. Check this before relying on what a bare call does.

## 1. Compiled-by-default paths with no device coverage (gaps)

### 1a. Head-major batched decode is skipped everywhere

`tests/conftest.py:33-48` skips every item whose id contains `batched_decode` / `batcheddecode`, or
that uses the `enable_batched_decode` / `batched_decode_calls` fixtures. The skip reason is "will be
re-enabled once batched decode is working with for_each_tile".

#1009 (2026-09-24) made it work for head-major. `SpyreHeadMajorAttentionImpl` returns
`_tiled_batched_decode_supported() = True` (`spyre_head_major_attn.py:267`), so batched decode is
the default production decode path, and it is not tested directly.

Skipped head-major tests:

- `test_head_major_batched_decode_correctness` (`test_spyre_head_major_attn.py:1715`, 4 runs,
  including the only batched soft-cap case)
- `test_head_major_warmup_records_a_batched_decode_variant` (`:1747`)
- `test_head_major_batched_decode_uses_plain_page_ids` (`:1582`, CPU, skipped by name only)

Batched decode is reached today only by tests whose names happen to dodge the filter: sinks
`multi_seq_decode` (`:1048`), grouped decode (`:923`) and core `decode_batch`.

**Fix:** limit the skip to token-major. These tests have not run for about six weeks, so expect some
breakage.

### 1b. Decoder custom ops are only tested eager on the card

In production these are traced into the block graph:

- `tests/custom_ops/test_mlp.py:40,107,183,241`: merged-column, QKV, row-parallel and replicated
  layers, called bare (`layer(x.to("spyre"))`). The only compiled case is the 3-D gate_up test in
  `test_linear.py:246`.
- `tests/custom_ops/test_rotary_embedding.py:151,200,234,258,294,433,531,618,685`: every
  `*_on_spyre` / `forward_oot` device test. The file never calls `torch.compile`.
- `tests/custom_ops/test_silu_and_mul.py:41` (`SiluAndMul`) and
  `tests/custom_ops/test_activation.py:33,47` (`NewGELU`).
- `tests/multimodal/test_utils.py:146` (`rope_rotate_matmul`): its callers (the pixtral and gemma4
  towers) are compiled per budget, but this function is tested only through `test_pixtral_vision.py:292` /
  `test_gemma4_mm.py:994`.

None of these files calls `spyre_available()`, so on a host without a card they error instead of
skipping.

E2E compiled runs cover these ops indirectly; at op level they are not covered.

### 1c. Upstream vLLM model tests all run eager

`_force_eager_vllm_runner` (`pytest_plugin.py:942`) defaults every upstream `vllm_runner` to eager,
to save CI time. That covers:

- `generation/test_common.py`: opt-125m, Llama-3.2-1B, Mistral-7B, Qwen2.5-0.5B (tied lm_head)
- `generation/test_granite.py`: micro-g3.3, granite-4.1-3b
- `pooling/test_embedding.py`: bge-base-en-v1.5, all-MiniLM-L12-v2

Local `tests/e2e/test_compile.py` covers only micro-g3.3 (plus its FP8 checkpoint), gemma-3-1b,
gemma-4-31B and gemma-4-moe-nano compiled. So OPT, Mistral, Qwen2.5, granite-4.1-3b, bge-base and
MiniLM never run on the default path.

**Decision needed:** add a few cheap compiled rows, or flip the default. This is a cost tradeoff.
GSM8K (server based) already runs compiled.

### 1d. Head-major attention has a thinner matrix than token-major

`test_spyre_head_major_attn.py` has about 37 compiled-on-Spyre runs that are not skipped: core 5,
fp8 10, query-width dispatch 1, grouped decode 4, block sizes 2, sliding window 2, soft cap 1,
sinks 6, MHA/MQA 2, plus `test_head_major_matches_token_major`, which goes away with #1399.

Cases covered only by token-major today, to port before #1399 deletes it:

- bf16
- head_size 64 outside the sinks tests
- block_size 256
- small or decode-only sliding windows (head-major only runs windows 64 and 256 on a mixed batch,
  `:989`)
- soft cap on decode only or batched decode (`:1011` runs one mixed case)
- chunked prefill as an explicit case (reached today only incidentally)
- the fused-output-store assertion
- the #284 head-size regression, `test_spyre_attn_decode_head_size` (`test_spyre_attn.py:477`),
  which runs eager only, 4 runs

### 1e. Smaller gaps

- RMSNorm traced inside an enclosing block graph. LayerNorm has this test (`test_layer_norm.py:159`);
  RMSNorm only has the compile-outermost case.
- `_roberta_classifier_head_kernel` (`spyre_pooler.py:572`) has no on-card test.
- The DFlash helpers (`qwen3_dflash.py:66-152`): the spec_decode tests replace `torch.compile` with
  an identity or a recorder (`test_dflash_lifecycle.py:134`, `test_dflash_feedback.py:47`), so
  nothing runs them compiled on the card.
- `test_spyre_attn_recorder.py` (CPU, about 37 tests): the `impl` fixture (`:76`) builds a token-major
  `SpyreAttentionImpl` with `USE_FOR_EACH_TILE=False`. Only `test_non_causal_draft_variant…`
  (`:205`) also runs head-major. These test compiled-mode logic, but against the layout and walk
  that #1399 removes, so they need retargeting to head-major on the tiled walk.

## 2. Redundant eager coverage (candidates to drop or convert)

### 2a. Distributed e2e (biggest runner-time wins)

- `tests/e2e/test_distributed_tp2.py::test_tp2_llm_generate_matches_tp1`: eager, `e2e_smoke`.
  Its compiled twin `test_tp2_compiled_llm_generate_matches_tp1[micro-g3.3]` runs the same model and
  assertion and is also `e2e_smoke`. **Drop the eager one** (two engine boots per smoke run). Move
  its `record_serving_capability` tag to the compiled test if needed (the compiled test already
  records one).
- `tests/e2e/test_distributed_tp4.py::test_tp4_llm_generate_matches_tp1`: eager only, and nothing
  tests TP4 compiled. **Switch it to compiled** (with `compile_sizes` pinned as in the TP2 test)
  rather than adding a second test.

### 2b. Other e2e

- `tests/e2e/test_prefix_caching.py::test_prefix_caching[eager]`: duplicates `[compiled]` (the smoke
  one). Drop it.
- `tests/e2e/test_transformers_backend.py::test_transformers_matches_native`: eager. The compiled
  Transformers backend is covered by `test_compile.py::test_transformers_backend_compile`. Make it
  compiled, or drop it if the reference-output check is enough.
- `tests/e2e/test_vllm_spyre_next.py::test_long_context_model_load`: 131k context, eager, checks
  only that the output is not empty. Compiled is the path that matters at long context (buckets,
  warmup cost). Switch it, or document why it stays eager.
- Keep as-is: `test_multimodal.py::test_two_image_prompt_produces_output` and
  `test_granite_vision.py::test_multi_tile_image_prompt_produces_output`. Their docstrings justify
  eager (the mask and patch logic run on CPU either way), and #1504 left them alone on purpose.

### 2c. Attention (mostly goes with #1399)

- `tests/attention/test_spyre_attn.py` is entirely token-major (`SpyreAttentionImpl`). Approximate
  counts:

  | Combination | Runs |
  |---|---|
  | cpu / eager (NONE) | ~50 |
  | spyre / eager (NONE) | ~55 |
  | spyre / compiled (STOCK) | ~60, plus 2 from `multi_seq_decode_sinks` |
  | cpu / compiled: collected, then self-skipped (`:129`) | ~38 |
  | batched-decode tests, all compiled, all skipped by conftest | 23 |

  The 23 skipped are `batched_decode_correctness` (5), `_soft_cap` (3), `_fallback` (4),
  `_sliding_window` (5) and `mixed_batch_batched_decode` (6), at `:1848`-`:2603`.

  The spyre/eager runs all have token-major compiled twins and exist only for
  `SPYRE_ATTN_KV_LAYOUT=token_major` + `--enforce-eager`, so they go with #1399. The cpu/eager runs
  are the only card-free attention CI: shrink them to a smoke set or replace them with head-major
  CPU-reference tests rather than deleting them outright.
- `tests/attention/test_spyre_encoder_attn.py::test_spyre_encoder_attn` (`:521`): 2 devices ×
  NONE/STOCK × 5 seq_lens × 2 head sizes × 3 block sizes = 120 runs, 60 of them NONE.
  - The `block_size` axis (64/128/256) does nothing for this backend: it has no KV cache and no
    block walk. Removing it cuts the test to a third.
  - Encoder eager is still reachable under `--enforce-eager` (no head-major constraint), so keep a
    few NONE cases and drop the rest.
  - Production traces the rectangular path into the block graph. That is covered by
    `test_encoder_inline_dispatch.py` (CPU) and `test_fast_and_ragged_paths_agree` (`:950`, STOCK).
- Tests that go away with #1399:
  - `test_spyre_attn_multi_seq_decode_sinks` (`:1885`, pins the untiled walk)
  - `test_head_major_matches_token_major` (`:1124`)
  - the `token_major` param of `test_platform_selects_backend_by_kv_layout` (`:648`)
  - `test_token_major_write_index…` (`:641`)
  - `test_reshape_and_cache_scatter` / `test_kv_cache_update_traced_by_caller` (`:1625`, `:1707`),
    the token-major store, whose twin is `test_head_major_scatter` (`:690`)
  - the token-major fp32 CPU-reference walks (`test_spyre_attn.py:2221`, `:2329`)

### 2d. Custom ops and models

- `tests/custom_ops/test_linear.py:245`: drop the `mode="eager"` param. The bug it guards
  (torch-spyre#4155) was compile-only, as its own docstring says.
- `tests/custom_ops/test_parallel_lm_head.py:274`: an eager matmul+slice primitive, superseded by
  `:514` (self-compiled vs eager on the card).
- `tests/models/test_pixtral_vision.py`: `:275` (eager vs upstream), `:292` (compiled vs upstream)
  and `:335` (compiled vs eager) form a triangle, so one is redundant. Drop `:335` (two runs
  including a compile) to keep explicit eager coverage, or `:275` (the cheapest) if eager towers
  need no direct test.
- `tests/custom_ops/test_vocab_parallel_embedding.py:179,200`: raw eager `F.embedding` probes,
  while production's `forward` is `@maybe_compile`. `:59` and `:84` already cover the layer
  compiled. Drop or convert.
- `tests/models/test_gemma4_mm.py:979`: eager, next to its compiled twin `:994`. Keep it only if
  gemma-4 vision under `--enforce-eager` is a supported mode.

## 3. Legitimately eager (leave alone)

- **CPU and host logic:**
  - runtime: `test_platform.py` (`:239`, `:536`-`:569`, `:921`), `test_compile_granularity.py`
    (`:495`, `:537`), `test_lazy_compile.py`, `test_warmup_multimodal_encoder.py`,
    `test_encoder_inline_dispatch.py:124`, mrv2 `test_bucketing` / `test_warmup` / `test_load`
  - custom ops: `test_blocked_transpose`, `test_head_pad`, `test_mlp_pad`, `test_bert_head_pad`,
    `test_fp8.py:54,66`, `test_vit_attn` (except `:97`), the `*_cpu` rotary tests
  - multimodal: `test_siglip`, `test_clip_layer_norm`
  - models: `test_mistral`, `test_roberta`, `test_gemma4.py`, most of `test_vision_encoder_graph`
  - all of `tests/spec_decode/`, and the pooler bucketing and sequence tests
- **Eager in production:** the BLIP-2 and granite-vision towers (`test_blip2.py:219,246`,
  `test_granite4_vision.py:307,334`; these towers are not marked compilable), CLIP merge glue
  (`test_clip.py:205`), and pooler code outside the graph (`test_spyre_pooler_select_rows.py`,
  `test_spyre_pooler_config.py`).
- **Probes:** the eager/compile pairs in `tests/probes/test_spyre_fallback_probes.py`, often strict
  xfail, which document torch-spyre limits. `tp_probe.py:403` sets `enforce_eager=True` for config
  only; its collectives are compiled explicitly.
- **Deliberate `enforce_eager` canaries:** `test_layer_norm.py:98`, `test_fp8_linear.py:564`, and
  `test_rms_norm.py:174` (a strict xfail that documents why `force=True` exists).
- **fp32 CPU-reference kernels on the Python walk:** `test_spyre_head_major_attn.py:566`, `:604`,
  `:1385`, `:1457`, `test_batched_decode_entry_local.py`, `runtime/test_tile_loop.py`. These stay as
  long as `walk_tiles` is kept as a CPU-drivable fallback.

## Suggested order

1. Re-enable head-major batched decode: limit the `tests/conftest.py` skip to token-major, then fix
   whatever has rotted.
2. Drop or convert the e2e eager twins: drop TP2 eager, switch TP4 to compiled, drop prefix-caching
   eager, convert the transformers-backend test. These save the most runner time.
3. Fill the head-major gaps (1d), then delete the token-major tests together with #1399 and retarget
   the recorder suite.
4. Remove the encoder `block_size` axis, cut its NONE cases to a smoke set, and convert or drop the
   custom-op eager twins (1b, 2d).
5. Decide on the upstream eager default (1c).
