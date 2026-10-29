# MRV2: upstream outlook (fork-only spike, `mrv2-full`)

Companion to `mrv2-bringup.md`. It records what upstream vLLM is doing that affects
the Spyre V2 runner, as surveyed on 2026-10-15/16 against vLLM `main` and `v0.31.0`.

## 1. Power/Z: the state of non-Triton MRV2 kernels

**Update (`mrv2-kernel-patching`):** the V2 runner now binds a C++ host op over every
MRV2 Triton kernel it can launch (`kernels.HOST_KERNELS`; see `mrv2-bringup.md`, *Host
kernels*), so it needs no Triton anywhere. The survey below is kept for the exit
path: if upstream lands usable non-Triton kernels, point `bind()` at them and drop ours.

**Update 2026-10-16: fadara01's PR is open as vllm#61995** (see 1a below). It is the PR
to track. Earlier survey:

| PR | What | Status | Usable on Power/Z? |
|---|---|---|---|
| **vllm#61995** (fadara01) | MRV2 by default on CPUs without Triton (Arm, IBM, RISC-V) | Open 2026-10-16, not a draft, unreviewed | Close; three gaps, see 1a |
| vllm#55847 (ganeshr10) | Torch fallbacks for MRV2 kernels on CPU without Triton | Open, merge conflicts since 2026-09-15 | Torch only, so in principle |
| **vllm#60924** (LucasWilkinson) | MRV2 on CPU without Triton: `sampler_cls` / `block_tables_cls` seams on `GPUModelRunner`, `CPUSampler` (top-k/top-p over the top 1024 candidates + torch Gumbel), `CPUBlockTables`, fused `prepare_inputs` in `CPUModelRunner`, 14 per-kernel overrides (mostly numba) | Draft, opened 2026-10-09, merge conflicts, no reviews | **No, as written** (see below) |
| vllm#60206 (ganeshr10) | Collapse MRV2 staged writes and buffer pool on CPU | Open | Partial |

Problems with #60924 for us:

1. **numba has no Power/Z wheels.** On PyPI, numba 0.68 and llvmlite 0.50 ship only
   x86_64 and aarch64, and numba is not in our lock.
2. **Possible V1 regression on Power/Z.** Every worker calls
   `current_platform.register_triton_kernel_overrides()`. `TorchSpyrePlatform`
   inherits `CpuPlatform`'s version, which (with the PR) imports the numba modules
   whenever Triton is absent, and on Power/Z it always is. Unless we override that
   hook, this would fail at worker start for V1 as well. Read from the diff, not run.
3. **Gated on `is_cpu()`.** The "MRV2 requires Triton" exemption is
   `not current_platform.is_cpu()`, which our OOT platform fails. This is the plan's
   "gate on capability, not `is_cpu()`" ask.
4. **The Triton-free pieces live in `CPUModelRunner`.** We subclass `GPUModelRunner`,
   so we would set `sampler_cls` / `block_tables_cls` ourselves and take its
   `prepare_inputs`. The seams themselves are welcome: they would replace two of our
   swaps.

Takeaways:
- Kernel step 2 still depends on fadara01's torch/C++ PR or #55847.
- V1 removal is targeted for v0.32. If those slip, Power/Z have no runner, so a
  pure-torch fallback of our own through `kernels.py` (about 20 kernels that
  launch; the rest of the 35 are helpers inside them) is worth starting as a hedge.
- Points to raise upstream if wanted (none raised from this spike):
  - numba on ppc64le/s390x.
  - The override hook running for every platform when Triton is absent.
  - Capability gating instead of `is_cpu()`.
  - A nudge on fadara01's PR.

### 1a. vllm#61995 in detail (2026-10-16)

About +5.9k lines. Opened together with vllm#61992, which fixes CPU's slot-mapping
fallback after an upstream kernel signature change.

What it does:
- **Implementations, no numba:** pytorch (with `torch.compile`) where it is fast
  enough, C++ ops where it isn't (gumbel argmax, block-table writes, slot mapping,
  staged writes, M-RoPE, KV clearing), and Python/numpy for bookkeeping.
- **Seven new properties on `GPUModelRunner`:** `block_tables_cls`, `sampler_cls`,
  `prompt_logprobs_worker_cls`, `watermark_sampler_cls`, `structured_outputs_worker_cls`,
  `kv_block_zeroer_cls` and **`model_state_cls`**. `init_model_state(...,
  model_state_cls=)` replaces our need to swap `init_model_state`.
- **`NoTritonCPUModelRunner(CPUModelRunner)`** sets those properties and overrides
  `prepare_inputs` and the post-step updates. `CPUWorker` picks it when `HAS_TRITON`
  is false.
- **Scope:**
  - Supported: default, Mamba-hybrid, encoder-only and encoder-decoder states; TP up
    to 2; prefix caching; full sampling; logprobs; structured outputs.
  - Rejected: spec decode, prompt embeds, DCP/PCP, batch-sharded sampling.
- **Fixes both #60924 problems:** no numba, and nothing goes through
  `register_triton_kernel_overrides`.
- **The C++ sampling code** has a portable scalar `VEC` path besides Arm NEON, so it
  should build on ppc64le/s390x.

Gaps for Spyre as written:
1. **The C++ ops are in vLLM's `_C`.** There are 11 new `torch.ops._C.*_cpu` ops
   (`gumbel_sample_cpu`, `prepare_prefill_inputs_cpu`, `prepare_pos_seq_lens_cpu`,
   `combine_sampled_and_draft_tokens_cpu`, `compute_slot_mapping_cpu`,
   `gather_block_table_cpu`, `post_update_cpu`, `apply_logit_bias_cpu`,
   `get_prompt_logprobs_token_ids_cpu`, `prepare_mrope_positions_cpu`,
   `update_num_accepted_tokens_cpu`). We build vLLM with `VLLM_TARGET_DEVICE=empty`, so
   there is no `_C`. Options: build those sources into `_spyre_C` (as #1111 did for the
   V1 sampling kernels), pure-torch fallbacks, or a CPU-target vLLM build.
2. **Gated on `is_cpu()`.** Both "V2 by default without Triton" (`use_v2_model_runner`)
   and the Triton exemption (`_validate_v2_model_runner`) check
   `current_platform.is_cpu()`, which our OOT platform fails. Our Triton gate can work
   around it; upstream ask: gate on capability.
3. **`prepare_inputs` drops bucket padding.** It sets
   `num_tokens_after_padding=num_tokens` and `num_reqs_after_padding=num_reqs`, ignoring
   `batch_desc`. Compiled Spyre depends on body-bucket padding, so it can't be reused
   unchanged; upstream ask: honour `batch_desc.num_tokens` as the GPU path does.
4. **Separate CPU model-state classes** (`CPUDefaultModelState`, ...). Our mixin would
   layer over these. Mechanical.

fadara01 offered (on #55847) to ping Joe for testing on IBM hardware.

## 2. Runner base class: decision to stay on `GPUModelRunner`

Decided 2026-10-15: `SpyreModelRunnerV2` stays a subclass of upstream's MRV2
`GPUModelRunner`.

Evidence:
- **Upstream's own non-GPU MRV2 runners do the same.**
  - `vllm/v1/worker/cpu/model_runner.py` is a 16-line `GPUModelRunner` subclass. The
    CPU worker imports `cpu/shm.py`, which patches `torch.cuda.Stream`,
    `torch.cuda.Event`, `torch.Event`, `pin_memory`, `record_stream`,
    `torch.accelerator.{synchronize,empty_cache,get_memory_info}` and the UVA buffer
    class process-wide, without restoring them.
  - XPU aliases `torch.cuda.*` to `torch.xpu.*` the same way.
  - Our `device_shims` are narrower than either.
- **The CUDA calls live in the components, not just the runner.** The runner has
  three stream/event touchpoints. The rest are in `AsyncOutput`, `StepTimingCollector`,
  `StructuredOutputsWorker` (its own copy stream) and `EncoderRunner`. A runner built
  from these components would still meet `torch.cuda`, and Triton too.
- **Generic torch doesn't remove the need.** `torch.Stream("cpu")` and
  `torch.cpu.Stream()` work as no-ops, but `torch.Event("cpu").record()` raises "cpu
  backend doesn't support events". The fix has to be a platform hook.
- **Upstream's orchestration churns.** `gpu/model_runner.py` had 67 commits between
  v0.28.0 and v0.31.0 (about 5.5 weeks); `gpu/` had 138. A standalone runner would own
  that.
- **Upstream is adding OOT seams as runner properties.** The existing `pcp_manager_cls`
  is commented "Out-of-tree hardware runners can select a PCP manager class", and
  #60924 adds `sampler_cls` / `block_tables_cls`.

Alternatives rejected:
- **A standalone runner composed from MRV2 components.** It doesn't remove streams or
  Triton, and it means owning about 700–900 lines of orchestration against upstream
  churn.
- **Subclassing upstream's MRV2 `CPUModelRunner`.** Semantically close, but its
  process-wide `torch.accelerator` no-ops could silently drop real Spyre synchronizes.
  Revisit only if #60924 lands and this is ruled out.

Plan: replace our module-level swaps with overrides as upstream adds seams, and ask
for the missing ones (section 3).

## 3. Upstream `main` against v0.31.0 (surveyed 2026-10-15)

Files diffed: `gpu/model_runner.py`, `attn_utils.py`, `model_states/interface.py`,
`pool/pooling_runner.py`, `mm/encoder_runner.py`, `async_utils.py`,
`structured_outputs.py`, `block_table.py`, `buffer_utils.py`, `v1/worker/utils.py`,
`gpu_worker.py`, `platforms/{interface,cpu}.py`, `config/vllm.py`,
`triton_utils/importing.py`, `kv_cache_interface.py`, plus the 44 MRV2 commits since
the tag.

### The four hooks we want

| Hook | On `main` | Consequence |
|---|---|---|
| KV-cache allocation and binding | **Not added.** `init_kv_cache` gained `attn_groups=`; new `ModelState.initialize_kv_cache(kv_cache_config, block_tables)` runs *after* allocation; the `cache.device` filter remains | Our `kv_cache.py` swap keeps working (`**kwargs`); the KV connector / CoW gap remains |
| Model loader and model state | **Partly.** `load_model(..., model=...)` accepts a pre-built model; `init_model_state` is still a module-level name. New `Worker._make_model_runner()` for subclasses | Drop the `get_model_loader` swap; the model-state swap stays |
| `pooling_runner_cls` / `encoder_runner_cls` | **Not added.** `PoolingRunner` gained `release_late_interaction_query_cache`; `EncoderRunner.execute_mm_encoder` groups by modality | Swaps stay; our subclasses inherit the new behaviour |
| Streams/events via the platform | **Not added.** No stream/event hook in the platform interface | Shims stay |

### Breaking for compiled V2: stock compile on MRV2 has landed (vllm#59079)

Under `STOCK_TORCH_COMPILE`, `load_model` now ends with
`compile_model_with_stock_torch(self.model, ...)`, which calls
`model.compile(fullgraph=True, backend=vllm_backend)`.

- `self.model` is `_SpyreModelWrapper`, whose `__getattr__` forwards `.compile` to
  the inner model. Upstream would whole-model compile on top of our per-block graphs.
- `compile.check_no_whole_model_compile` only detects `@support_torch_compile`
  wrappers, not `nn.Module.compile` (`_compiled_call_impl`).
- "stock torch.compile" is no longer listed as unsupported by MRV2, so
  `triton_gate`'s compile filter becomes a harmless no-op. V2 still hard-requires
  `HAS_TRITON`, and V1 still exists on `main`.

Fix: a V2 subclass of the wrapper whose `compile()` is a logged no-op, plus a tripwire
that also checks `_compiled_call_impl`. Upstream ask: an opt-out property on the runner.

### Other changes that touch us

- **`Worker._make_model_runner()` / `_init_workspace_and_model_runner()`.** The latter
  also builds the workspace manager we currently skip. If our `init_device` calls it,
  `SpyreWorkerV2` only overrides `_make_model_runner`. The copied `init_device` and its
  drift test go away, and V1 and V2 share one `init_device`.
- **Churn our hooks already absorb:**
  - `BlockTables` no longer takes `kernel_block_sizes`.
  - Metadata is built through `AttentionGroup.build_metadata`, and `AttentionGroup`
    gained kernel-block fields; the first four positional fields our window split
    uses are unchanged.
  - `ExecuteModelState.dp_sync` became `dp_sync_state`, plus a new
    `num_spec_tokens_to_schedule`.
  - `idx_mapping` is now int64.
  - `NonUvaBuffer` copies are `non_blocking` (host to host for us).
  - `triton_utils/importing.py` uses `vllm_version_matches_substr("cpu")`; the gate
    hooks the module's result, so it is unaffected.
- **V1 issue for the bump:** `Worker.profile()` gained keyword-only `delay_iterations` /
  `max_iterations`; `TorchSpyreWorker.profile(is_start, profile_prefix)` would raise
  `TypeError` if they are passed.
- **No impact:**
  - `KVCacheTensor` is unchanged.
  - `CpuPlatform` gained `_check_aligned_block_size`, which only fires when alignment
    changes the block size; ours (a multiple of 64) shouldn't.

### Plan for the next release

1. Suppress the stock whole-model compile in V2 and fix the tripwire. Required.
2. Switch to `load_model(model=...)` and `_make_model_runner()`: two swaps and the
   `init_device` copy removed.
3. Accept the new `profile()` keyword arguments in `TorchSpyreWorker`.
4. Upstream asks, in priority order: a KV allocation and binding hook,
   `pooling_runner_cls` / `encoder_runner_cls`, a whole-model-compile opt-out, a
   stream/event platform hook, and capability gating instead of `is_cpu()`.

## 4. vLLM C++ kernels by category (for a "host kernels" build target proposal)

Surveyed 2026-10-16 from vLLM `main` (`csrc/` tree, `CMakeLists.txt`,
`cmake/cpu_extension.cmake`, `setup.py`, `csrc/cpu/torch_bindings.cpp`) plus vllm#61995.

How we consume vLLM's C++ today: copied sources for #1111's sampling kernels. Decision
(2026-10-16): **option B now** (compile the needed files straight from the pinned vLLM
source, registered with `TORCH_LIBRARY_FRAGMENT(_C)`, no copies), and **propose C
upstream** (a separate build target for the runner's host kernels).

Categories:
- **H, host runtime:** C++ host versions of the shared runners' (V1/MRV2) Triton
  kernels. They work on scheduler and request state, never on model tensors, and any
  platform without Triton on its host needs them.
- **I, host infrastructure:** device-independent, not compute.
- **C, CPU compute:** for running the model on the CPU.
- **G, CUDA/ROCm compute.**

| Source | Ops | Called from | Cat. | Deps |
|---|---|---|---|---|
| `cpu/sampling_kernels.cpp` | `fused_gumbel_argmax`, `greedy_argmax`; +`gumbel_sample_cpu` (#61995) | V1 CPU sampler; MRV2 `CPUSampler` | H | torch, `cpu_types.hpp` |
| `cpu/spec_decode_utils.cpp` | rejection sampler (×3), `expand`, `eagle_*` (×3), `copy_and_expand_{eagle,dflash}_inputs` | `vllm/utils/cpu_triton_utils.py` | H | torch |
| `cpu/utils.cpp` (mixed) | `compute_slot_mapping_kernel_impl` / `init_cpu_memory_env` | `cpu_triton_utils.py` / CPU worker (NUMA) | H / C | torch / NUMA |
| `cpu/block_table.cpp` (#61995) | `gather_block_table_cpu`, `compute_slot_mapping_cpu` | `CPUBlockTables` | H | torch, `cpu_types.hpp` |
| `cpu/input_batch.cpp` (#61995) | `prepare_prefill_inputs`, `prepare_pos_seq_lens`, `combine_sampled_and_draft_tokens`, `post_update`, `apply_logit_bias`, `get_prompt_logprobs_token_ids`, `prepare_mrope_positions` (`_cpu`) | `NoTritonCPUModelRunner`, logits processors | H | torch, `cpu_types.hpp` |
| `cpu/mamba_cpu.cpp` (#61995) | `update_num_accepted_tokens_cpu` | Mamba-hybrid state | H | torch |
| `spinloop.cpp`, `fs_io.cpp` | spin-wait; GIL-free file I/O | engine; `FileSystemTierManager` | I | none |
| `cpu/shm.cpp` | `shm_*` collectives | `CpuCommunicator` (CPU TP) | C | — |
| `cpu/activation*`, `layernorm`, `pos_encoding` | activations, RMS norm, RoPE | CPU custom ops | C | `cpu_types` |
| `cpu/cpu_attn*`, `mla_decode` | CPU attention, reshape-and-cache, MLA decode | CPU attention backend | C | ISA |
| `cpu/cpu_fused_moe*`, `dynamic_4bit_int_moe_cpu`, `cpu_wna16` | MoE, int4/int8 GEMM | CPU MoE / linear | C | — |
| `cpu/dnnl_*` | oneDNN matmul, scaled matmul, int8 quant | CPU linear | C | **oneDNN** (+ACL) |
| `cpu/mamba_kernels.hpp`, `sgl-kernels/{conv,fla}` | SSM, conv1d, GDN | CPU Mamba/GDN | C | — |
| `cpu/sgl-kernels/*` (~35 ops), `micro_gemm/*`, `cpu_isa`, fp8 quant | GEMM/MoE/attention/MLA/topk | CPU model layers | C | AVX-512/AMX, NEON |
| `quantization/` (94 files), `moe/`, `attention/`, cache, norm, activation, RoPE, `sampler.cu`, topk, custom all-reduce, `cumem_allocator`, `rocm/`, `cutlass_extensions/` | — | GPU | G | CUDA/ROCm |

Rationale for C (not Spyre-specific):
1. **H is the runner's host kernels, not model kernels.** They are the host
   implementations of the shared runners' Triton kernels, as device-agnostic as the
   runner. Users: CPU builds on Power/Z/Arm/RISC-V without Triton (the motivation for
   #61995), and any OOT platform built with `VLLM_TARGET_DEVICE=empty`.
2. **The dependency boundary is clean:** torch, `cpu_types.hpp` and `utils.hpp` only;
   no oneDNN/ACL/ISA compute libraries. Builds quickly, with no CPU-runtime coupling.
3. **Precedent:** `spinloop` and `fs_io_C` are already device-independent extensions
   built for every target, placed above the device branch in `CMakeLists.txt`. `empty`
   drops even those (`setup.py`: `if _no_device(): ext_modules = []`).
4. **Restructuring needed upstream:**
   - Split `utils.cpp` (H slot mapping vs CPU NUMA setup).
   - Move the H registrations out of the monolithic `torch_bindings.cpp` (1,072 lines)
     into their own `TORCH_LIBRARY_FRAGMENT(_C)` file, compiled into `_C` on CPU builds
     and into a standalone host extension elsewhere, so `torch.ops._C.*` resolves
     identically and nothing collides.
5. **Exclude C and G**, including host-side CPU-backend code such as `shm.cpp`. The
   rule is "the runner's host kernels".
