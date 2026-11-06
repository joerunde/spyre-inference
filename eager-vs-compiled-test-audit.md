# Eager vs compiled test audit

Audit of `tests/` on main (originally `36c8e43`, rechecked against `a1ab32c`, 2026-11-06). Goal:
every path that is compiled by default has a compiled test, and no eager test duplicates a compiled
one. Token-major attention is out of scope: #1535 removes it.

Related: #1399, #1503, #1504 (merged), #1533 (batched-decode skip), #1535 (token-major removal),
[CI runner usage summary](https://joerunde.github.io/spyre-inference/ci_runner_usage/summary/index.html).

## Already handled

| Item | PR |
|---|---|
| Encoder, CLIP, vision e2e eager twins | #1504 (merged) |
| e2e tests covered elsewhere, padded head-dim eager param | #1503 (merged) |
| Batched-decode skip in `tests/conftest.py` (default decode path untested since #876; all 119 skipped tests pass on Spyre) | #1533 |
| Token-major attention tests, eager attention matrix, recorder retarget | #1535 |

## How tests pick a mode

- Local tests default to compiled: the session config and `default_vllm_config` leave
  `enforce_eager` unset, so the platform sets `STOCK_TORCH_COMPILE`.
- A bare call to a `@maybe_compile` kernel runs the compile-outermost path. A bare call to any other
  op (linear `apply`, rope `forward_oot`, `SiluAndMul`) really is eager.
- Upstream tests: `_force_eager_vllm_runner` (`tests/plugin/spyre_testing_plugin/pytest_plugin.py:942`)
  defaults every upstream `vllm_runner` to `enforce_eager=True`.

## 1. Compiled paths without a compiled test

| Path | Today | Proposed |
|---|---|---|
| Linear layers: merged-column, QKV, row-parallel, replicated (`custom_ops/test_mlp.py:40,107,183,241`) | eager on card; only 3-D gate_up is compiled (`test_linear.py:246`) | compile in the test (traced as in a block) |
| Rotary embedding, all on-device tests (`custom_ops/test_rotary_embedding.py`) | eager; the file never calls `torch.compile` | compile |
| `SiluAndMul` (`test_silu_and_mul.py:41`), `NewGELU` (`test_activation.py:33,47`) | eager | compile |
| `rope_rotate_matmul` (`multimodal/test_utils.py:146`) | eager; reached compiled only via the pixtral and gemma4 tower tests | compile |
| RMSNorm inside an enclosing graph | only the compile-outermost case (LayerNorm has the in-graph case, `test_layer_norm.py:159`) | add the in-graph case |
| `_roberta_classifier_head_kernel` (`spyre_pooler.py:572`) | no on-card test | add one |
| DFlash helpers (`qwen3_dflash.py:66-152`) | spec_decode tests replace `torch.compile` with identity or a recorder | add one compiled on-card test |
| TP4 (`e2e/test_distributed_tp4.py::test_tp4_llm_generate_matches_tp1`) | eager only; nothing tests TP4 compiled | switch to compiled (`compile_sizes` pinned as in TP2) |
| Granite Vision multi-tile (`e2e/test_granite_vision.py::test_multi_tile_image_prompt_produces_output`) | eager; the docstring says the patch runs on CPU, but #1437 made the tower compiled, so multi-tile budgets go untested compiled | re-evaluate: switch to compiled, or confirm the single-image compiled test covers it |
| Upstream model tests: opt-125m, Mistral-7B, Qwen2.5-0.5B, granite-4.1-3b, bge-base, MiniLM | all eager via the plugin; no compiled run anywhere | decide: a few compiled rows, or change the plugin default |

Notes:
- The custom-op files in rows 1-3 never call `spyre_available()`, so on a host without a card they
  error instead of skipping.
- The attention review guide in #1535 notes that a lone decode through batched decode fails to
  compile on Spyre. Three tests there pin `SPYRE_BATCHED_DECODE=0`. That is a default path that
  fails to compile and should be tracked on its own.

## 2. Eager tests that duplicate a compiled one

| Test | Compiled twin | Action |
|---|---|---|
| `e2e/test_distributed_tp2.py::test_tp2_llm_generate_matches_tp1` (`e2e_smoke`) | `test_tp2_compiled_llm_generate_matches_tp1[micro-g3.3]` (`e2e_smoke`) | drop (two engine boots per smoke run) |
| `e2e/test_prefix_caching.py::test_prefix_caching[eager]` | `[compiled]` (`e2e_smoke`) | drop |
| `e2e/test_transformers_backend.py::test_transformers_matches_native` | `test_compile.py::test_transformers_backend_compile` | switch to compiled, or drop |
| `e2e/test_vllm_spyre_next.py::test_long_context_model_load` (131k, checks only that output is non-empty) | none; compiled is the path that matters at long context | switch to compiled |
| `custom_ops/test_linear.py:245`, `mode="eager"` | `mode="compile"` (the bug was compile-only) | drop the param |
| `custom_ops/test_parallel_lm_head.py:274` | `:514` | drop |
| `custom_ops/test_vocab_parallel_embedding.py:179,200` (raw eager `F.embedding`) | `:59`, `:84` | drop |
| `models/test_pixtral_vision.py:335` (compiled vs eager) | `:275` eager vs upstream + `:292` compiled vs upstream | drop one of the three |
| `models/test_gemma4_mm.py:979` | `:994` | drop, unless gemma-4 vision under `--enforce-eager` is supported |
| `attention/test_spyre_encoder_attn.py::test_spyre_encoder_attn`, `block_size` axis (64/128/256) | n/a: the encoder backend has no KV cache, so the axis does nothing | drop the axis (cuts the test to a third) |
| same test, 60 NONE cases | STOCK cases | keep a few NONE cases (encoder eager is reachable under `--enforce-eager`) |

## 3. Legitimately eager (leave alone)

- CPU and host logic: platform and config tests, `test_lazy_compile.py`, `test_compile_granularity.py`,
  mrv2, spec_decode logic, padding shims, pooler bucketing.
- Eager in production: the BLIP-2 tower, CLIP merge glue, pooler code outside the graph.
- Card-side mask logic: `test_multimodal.py::test_two_image_prompt_produces_output` (the mask is
  built on CPU either way).
- Probes: the eager/compile pairs in `tests/probes/` that document torch-spyre limits.
- `enforce_eager` canaries: `test_layer_norm.py:98`, `test_fp8_linear.py:564`, and the strict xfail
  `test_rms_norm.py:174`.
