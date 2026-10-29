# MRV2 without Triton: vllm#61995 on Spyre (fork-only spike, `mrv2-no-triton`)

Companion to `mrv2-bringup.md` and `mrv2-upstream-outlook.md` (section 1a). Tried
2026-10-29 against fadara01's `mrv2-cpu` at `a76d7c41a2`, the rewrite in which
`NoTritonCPUModelRunner` is standalone: it no longer subclasses `GPUModelRunner`
and imports nothing from `vllm/v1/worker/gpu`.

## Update 2026-10-29: on vLLM 0.32 (branch `mrv2-no-triton-032`)

Rebuilt from `main` (vLLM 0.32.0, #1256, #1277) plus the open MRV2 PRs #1395
(startup extraction), #1366 (attention groups and KV cache), #1382 (pooling) and
#1354 (triton-cpu, hook-free gate), merged as-is. On top: the bring-up's load,
compile, warmup, kernels and multimodal staging (on main's shared helpers, not the
bring-up's `block_compile.py`), and the no-Triton runner. vLLM backport: branch
`mrv2-cpu-on-v0.32.0` on `joerunde/vllm`, one commit.

Same model and prompts as below, `micro-g3.3-8b-instruct-1b`.

| | V2, triton-cpu | V2, no Triton (#61995) |
|---|---|---|
| Eager vs V1 | 7/7 token-exact | 7/7 token-exact, logprobs identical |
| Compiled vs V1 | 6/7 (prompt 3 forks at token 15) | 6/7 (same prompt, same token) |
| Compiled, V2 vs V2 | 7/7, logprobs identical | (same) |
| Compiles after warmup | 0 | 0 body/attention; 4 first-use `cpu/logprob.py` |

The compiled fork is shared by both V2 runners and absent eager, so it is in the V2
compiled path, not in either runner's host code. It is new since 0.31 (0.31 V2
compiled was 7/7), and V1 0.31 vs V1 0.32 is 7/7, so it came in with main's changes
since `mrv2-full` was cut. The top-3 logprob ids agree on every step; the logprob
gap (0.047) is the same size as V1-eager vs V1-compiled. Not bisected.

Decode throughput, compiled, 32 requests × 64 tokens, `ignore_eos`, one run each:

| | V1 | V2, triton-cpu | V2, no Triton |
|---|---|---|---|
| greedy | 715 tok/s | 747 tok/s (+4%) | 705 tok/s (−1%) |
| top-k=50, top-p=0.9 | 698 tok/s | 691 tok/s (−1%) | 657 tok/s (−6%) |

All three moved up from 0.31 (V1 571 → 715), so compare within a row. The no-Triton
runner is now within noise of V1 on greedy; top-k/top-p still trails by about the
same 6–8% as on 0.31.

### What is left, by line count

New on this branch over main + the four PRs (`git diff 1c81100 HEAD`, docs excluded):
1,990 lines added, 87 removed.

- **C++ host kernels, 1,154 lines** (`csrc/cpu`, `cmake`): the PR's verbatim, plus
  the bindings. Goes away if vLLM ships them in a device-independent extension
  (outlook section 4, option C), or becomes a vendored copy we resync on bumps.
- **No-Triton runner, 116 lines** (`no_triton.py`): its own model state (no
  multimodal), the bucket padding override, top-k/top-p binding. Plus 16 lines in
  `triton_gate.py`/`worker.py` to pick it.
- **Shared bring-up still unmerged, about 590 lines** (load 137, warmup 174, kernels
  104, model_runner 50, compile 27, base 28, encoder_runner 21, platform 11): this is
  the `mrv2-full` work that #1395/#1366/#1382/#1354 don't cover, and both runners use it.
- **Tests, 123 lines.**

Gaps for full support on the no-Triton path:
- **Pooling and encoder-only.** `SpyrePoolingRunner` (#1382) and
  `SpyreEncoderOnlyModelState` subclass the GPU classes. The CPU ones are near-copies
  (diffed: `prompt_len` indexing, `_get_pooling_metadata` drops its device arg,
  `release_late_interaction_query_cache`), so this is a second base or a mixin,
  about 50 lines each.
- **Multimodal.** `SpyreEncoderRunner` subclasses the GPU `EncoderRunner`; same story.
- **Upstream asks that shrink the rest**: honour `batch_desc.num_tokens` in
  `prepare_inputs` (removes the padding override); gate the no-Triton runner on
  capability, not `is_cpu()` (removes the triton_gate extension); a KV-allocation hook
  (removes the `init_kv_cache` swap on both runners).

## TL;DR (0.31 spike)

With **triton uninstalled**, the Spyre V2 runner runs on top of
`NoTritonCPUModelRunner` and matches V1 on the decoder path: 7/7 greedy
token-exact, eager and compiled, TP1 and TP2, with identical logprobs at TP1.
Startup, warmup, attention recording, bucket padding and the sort-free top-k/top-p
all carry over. The kernels are the PR's C++ host ops, compiled into our own
extension. This is kernel step 2 (#906) for the decoder path, run on x86 because
that is where the hardware is; nothing here is x86-specific.

Scope: the default (decoder) model state only. Pooling, encoder-only and
multimodal raise `NotImplementedError` on this path (see *Not done*).

## How it is wired

- **vLLM: a backport of #61995 onto v0.31.0.** The PR is based on `main` at
  v0.33.0rc0+178, which needs torch 2.14, and torch-spyre is on 2.13. Instead:
  `vllm/v1/worker/cpu/` and the ten `_custom_ops` wrappers from the PR, over
  v0.31.0, with stubs for what 0.31 lacks. Branch `mrv2-cpu-on-v0.31.0` on
  `joerunde/vllm` (two commits), copied over the venv's `empty`-target vLLM:
  - `customize_attention_spec` → the backend's `customize_spec`;
    `map_kv_caches_to_kernel_blocks` → identity (0.31 has no kernel-block views).
  - `AttentionGroup.build_metadata` → `get_metadata_builder().build(...)`, as 0.31's
    GPU path does.
  - `compute_mm_prefix_ranges` → 0.31's `gpu/attn_utils` one (different signature).
  - Stock compile (vllm#59079), in-place `reload_weights`, `rswa_window`,
    `encoder_input_updates`, `num_retained_tokens` and `has_topp_only_rows`:
    dropped or `getattr` defaults.
  Against the PR's own base none of these are needed; on our pin they are
  mechanical.
- **Host kernels: `csrc/cpu/{block_table,input_batch,mrv2_sampling}.cpp`** (verbatim
  from the PR) plus `mrv2_bindings.cpp`. They are registered with
  `TORCH_LIBRARY_FRAGMENT(_C)`, so `vllm/_custom_ops.py` resolves
  `torch.ops._C.*_cpu` unchanged: decision B in the outlook doc's section 4. Only
  the portable `VEC` sampling path is built. `utils.hpp` keeps the four helpers the
  kernels use (`ISA`, `Counter`, `get_max_threads`, the dispatch macros), so the
  vendored `cpu_types` headers did not grow.
- **Runner: `SpyreNoTritonModelRunnerV2(SpyreRunnerMixin, NoTritonCPUModelRunner)`**
  (`spyre/no_triton.py`). `SpyreRunnerMixin` is what `SpyreModelRunnerV2` used to
  be, minus the cudagraph manager and the synchronous pool. `upstream_runner_cls`
  says which upstream module `load.py` / `kv_cache.py` patch. The hooks are the
  same names (`get_model_loader`, `init_model_state`, `init_attn_backend`,
  `init_kv_cache`); the CPU runner imports them by name too.
- **Worker:** `_bind_kernels_and_pick_runner()` picks by `HAS_TRITON`: the GPU runner
  with `kernels.bind_or_raise()`, or the no-Triton runner with `bind_kernels()` (the
  sort-free top-k/top-p into `cpu/sampler.py`).
- **Config:** `triton_gate.open()` also lets `_validate_v2_model_runner` through
  when Triton is absent and the installed vLLM has the no-Triton runner. Upstream
  gates this on `is_cpu()` and non-x86, which our OOT platform fails: the outlook's
  "capability, not `is_cpu()`" ask.

### Bucket padding is ours again

`NoTritonCPUModelRunner.prepare_inputs` sets `num_tokens_after_padding = num_tokens`
(gap 3 in the outlook). `SpyreNoTritonModelRunnerV2.prepare_inputs` pads the body
onto the warmed bucket after upstream's, leaving the tail as the GPU runner does:
zeroed ids and positions, `is_padding` set, and `PAD_SLOT_ID` slots
(`compute_slot_mapping_cpu` already fills past `query_start_loc[num_reqs]`). It is
the only override on the step path, and it is what keeps compiled V2 at zero
body recompiles. Upstream ask unchanged: honour `batch_desc.num_tokens`.

## Results

x86 Spyre host, `triton` uninstalled (`HAS_TRITON` False). Model
`micro-g3.3-8b-instruct-1b`, 7 prompts, greedy, 32 tokens, `max_num_seqs=4`,
`-W error::FallbackWarning`, against V1 on the same build.

| Configuration | V2 (no Triton) vs V1 |
|---|---|
| Eager | **7/7 token-exact**; logprobs(3) identical (max diff 0.0000) |
| Compiled | **7/7 token-exact**; logprobs identical; same 4 body buckets, 80 attention graphs; generate 1.7s vs 1.8s |
| Compiled, **TP2** | **7/7 token-exact** against V1 TP1; logprobs max diff 0.038 (TP reduction order) |
| Seeded top-k/top-p | runs; differs from V1 by design (MRV2's per-position seeding) |

Decode throughput, compiled, 32 requests × 64 tokens, `ignore_eos`:

| | V1 | V2 (no Triton) |
|---|---|---|
| greedy | 571 tok/s | 604 tok/s (+6%) |
| top-k=50, top-p=0.9 | 583 tok/s | 537 tok/s (−8%) |

The sort-free top-k/top-p is bound (`bind_kernels()`), so the −8% is not the
full-vocabulary sort the bring-up's −26% came from. Not profiled; candidates are
`CPUSampler`'s other per-step host work (the C++ Gumbel pass over the whole
vocabulary, which V1 restricts to the top-k survivors). One run each.

Zero body or attention compiles after warmup in the compiled runs. The guard reports
two upstream helpers (`cpu/logprob.py:compute_token_logprobs` and
`_compute_token_ranks`) that are `torch.compile`d and compile on first logprobs use
(twice each, a recompile on a dynamic shape). V1 has the same kind of first-use
compile for its own logprobs helper (`batched_count_greater_than`). Harmless at
`SPYRE_COMPILE_GUARD=warn`; at `error` they need warming up or the compile dropping.

Eager decode is slower on V2 (261s vs 196s for the same work). Every eager run
compiles its kernels lazily, and the guard reported different first-use compiles
on the two runners, so this is not a runner cost. The compiled runs are the
comparison that counts.

## Not done

- **Pooling / encoder-only, multimodal.** `SpyrePoolingRunner`, `SpyreEncoderRunner`
  and the encoder-only state subclass the GPU classes. The CPU ones (`cpu/pooling_runner.py`,
  `cpu/encoder_runner.py`, `cpu/model_states/encoder_only.py`) are near-copies, so
  each needs either a second base or a mixin. Mechanical, not attempted.
- **Power/Z.** Not tried (no hardware path). The kernels have no x86 code: `VEC`
  sampling, scalar block-table and input-batch ops. The only ISA flags are our
  existing ones in `cmake/cpu_extension.cmake`.
- **The PR's own base.** It needs torch 2.14, which needs a torch-spyre bump.
- **Upstream feedback.** Nothing posted to #61995.

## Reproducing

```bash
# 0.32: branch mrv2-no-triton-032 here, mrv2-cpu-on-v0.32.0 in vLLM (0.31: mrv2-no-triton-0.31 / -v0.31.0)
git clone -b mrv2-cpu-on-v0.32.0 https://github.com/joerunde/vllm ~/vllm-mrv2-cpu
uv sync --group dev --reinstall-package spyre-inference   # builds the host kernels
# V2 with triton-cpu runs right here; the steps below switch to the no-Triton runner.
SP=.venv/lib/python3.12/site-packages
cp -r ~/vllm-mrv2-cpu/vllm/v1/worker/cpu/. $SP/vllm/v1/worker/cpu/
cp ~/vllm-mrv2-cpu/vllm/_custom_ops.py $SP/vllm/_custom_ops.py
uv pip uninstall triton
SPYRE_MODEL_RUNNER=v2 uv run --no-sync python -c "from vllm import LLM; print(LLM('ibm-ai-platform/micro-g3.3-8b-instruct-1b', max_model_len=1024).generate('Hello')[0].outputs[0].text)"
uv run --no-sync pytest tests/mrv2
```

Any later `uv sync` reinstalls triton-cpu and the pinned vLLM. Use `--no-sync`.
