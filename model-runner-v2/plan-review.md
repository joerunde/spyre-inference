# Review of the MRV2 implementation plan and issue drafts

Reviewed: `mrv2-plan-906.html` and `issues/*.md` at gh-pages `365434c` (2026-10-12), against the investigation report, the
warmup measurements, vLLM `v0.28.0`/`v0.31.0`/`main@1bdd3fa2d`, and spyre-inference `main@a63dc01`.

Overall, the plan matches the investigation: the seam mapping, the Spyre impact, the #945 starting point, `SpyreWorkerV2`, the
measured warmup policy (option B) and taking MRV2's `Sampler` as-is all line up. Two items below would break the plan as written;
the rest are smaller gaps. This file is a list of findings for the plan's author to work through. It doesn't edit the plan.

## Blocking

### 1. The Triton gate can't take effect from `init_device` (kernel step 1 on x86)

`vllm/triton_utils/__init__.py` (v0.31, lines 11-25) chooses real `triton`/`tl` or placeholders **once, at first import**,
from `HAS_TRITON`. With triton-cpu installed and our vLLM version string lacking `cpu`, `HAS_TRITON` is False, so the placeholders
are bound as soon as anything imports `vllm.triton_utils`, long before any worker's `init_device`. MRV2's modules decorate their
kernels with `@triton.jit` at import, so they become placeholder functions. Flipping `HAS_TRITON` in `init_device` then passes the
config validation, but the first kernel launch fails.

- The gate has to be in effect before `vllm.triton_utils` is first imported, in every process (API front end, engine core, each
  TP worker). In practice that's the plugin's `register()`, keyed on V2 being selected (`--worker-cls`/env), so V1 runs are
  unaffected.
- This changes ground rule 1 ("the Triton-gate patch ... run[s] only from the V2 worker's `init_device`"; "importing the package
  does nothing") and touches drafts 01, 02 and 07.
- Draft 02's acceptance test "V1 still sees `HAS_TRITON = False`" should run in a fresh process.

### 2. Compiled bring-up waits on 0.32, but the flip's fallback needs a pin that still has V1

- Wave 2's compiled steps wait for vLLM 0.32 (#59079). Stage 1 of the flip needs a pin that still ships V1. 0.32 is the release
  targeted to remove V1. If both happen in 0.32, no single pin satisfies both.
- Facts today: #59079 is only in `v0.31.1rc0`, which is a main-line tag, not a 0.31.x patch release. There's no 0.32 yet and the V1
  runner is still on main, but `f8471164b` ("[MRV2] Detach weight-reload logic from MRV1") shows the removal being prepared.
- We may not need #59079. We compile per block ourselves and suppress MRV2's whole-model compile. What blocks compiled MRV2 on
  0.31 is the `"stock torch.compile"` entry in `_get_v2_model_runner_unsupported_features` (v0.31 `config/vllm.py:3057`), which also
  makes the forced-V2 validation raise. `triton_gate.py` could drop that one entry alongside `HAS_TRITON`. Worth checking in draft
  02's scope. If nothing else breaks (cudagraph mode already resolves to NONE for us via `support_static_graph_mode()`), compiled V2
  runs on 0.31, and Wave 2 and the flip stop depending on 0.32.
- Until then, every manual V2 run on 0.28/0.31 has to use `--enforce-eager`, or it fails that same validation. The plan's
  "manual testing with overrides" (Wave 1) should say so.

## Gaps

### 3. The `initialize_kv_cache` override doesn't list everything MRV2's method sets

MRV2's `initialize_kv_cache` (v0.31 `gpu/model_runner.py:578-771`) also builds `self.block_tables` (:666), `self.pcp_manager`
(:679), `self.ubatch_runner` (:686), `self.cudagraph_manager` (:720), `self.kv_connector` (:763-765), `self.kernel_block_sizes`,
the cudagraph-mode resolution and the KV-zeroing metadata (`_init_kv_zero_meta`, :773), plus `model_state.initialize_kv_cache` at
0.32. `_dummy_run` calls `self.kv_connector.set_disabled(...)` on every call (:844, :871). The `model_runner.py` sketch makes three
calls, and the WP0 interface lists only `attn_groups` and `kv_caches`. WP0 should list every runner attribute the step path reads,
and decide which ones `kv_cache.init` sets.

### 4. Warmup with `skip_attn` still needs `publish_null_slots`

V1's `_dummy_run` publishes null KV slots so the KV write inside the compiled block graph has a target
(`spyre_model_runner.py:1370` on main). With `skip_attn`, MRV2 builds no slot mappings, so without this the option B dummies differ
from V1 and from what was measured. Add it to WP-H (`warmup.py`) and to the `warmup.plan()` test.

### 5. No platform rejections for MRV2 features we don't support

None of these appear in the plan or the drafts:

- **Spec decode**: MRV2 builds `self.speculator` in `__init__` (`model_runner.py:269-274`), and V1's `hasattr(self, "drafter")`
  guard always passes under MRV2.
- **Batch-sharded sampling**: `compute_logits_local` bypasses the wrapper's H2D copy.
- **`cudagraph_mm_encoder`**: gemma-4 declares encoder CUDA-graph support.
- **PCP > 1.**
- **LoRA**: the V1 guard in `load_model` has to come across into `load.py`.

They fit in WP-INT's `check_and_update_config` switch, or in 08a when the V1-only rejections become conditional.

### 6. `SpyreWorkerV2(TorchSpyreWorker).init_device` is undecided

V1's `init_device` builds the V1 runner at the end (`spyre_worker.py:163`). Calling `super().init_device()` and then replacing
`self.model_runner` constructs a V1 runner first, with its side effects (the slot-mapping kernel shim is installed process-wide in
the V1 runner's `__init__`). Copying the method duplicates about 70 lines of env, dlopen and device setup until stage 2. Draft 01
should pick one. I'd copy it, with a test that diffs the two for drift.

### 7. MRV2's `Sampler` as-is costs top-k/top-p performance on x86 too

Top-k/top-p goes through torch's `apply_top_k_top_p` with `allow_cpu_sync=current_platform.is_cpu()`
(v0.31 `v1/sample/ops/topk_topp_sampler.py:459`), which is False for our OOT platform. That's the full-vocabulary sort
`SpyreTopKTopPSampler` exists to avoid, and it happens whatever the kernel source. Measure it in WP-E on x86 (draft 05), not only at
kernel step 3 on Power and Z.

### 8. Test budget vs. triton-cpu JIT

Tests that run MRV2 kernels through triton-cpu (the [4, 32] sampler test in draft 05) JIT-compile Triton on a cold cache, which
takes seconds per kernel. That breaks the 1 s per test and 20 s directory budget in CI. Exempt those tests from the budget, warm a
Triton cache in CI, or keep them out of the smoke shards.

## Consistency

### 9. The attention-cache finding has no issue

The plan says it's "tracked in the report", but the report isn't a tracker. It's the largest warmup lever measured: attention is
64% of a cold warmup, and 37 of 40 attention kernels miss torch-spyre's persistent kernel cache across processes even with
`SPYRE_KERNEL_CACHE=1`. Suggest a separate non-MRV2 issue, plus a torch-spyre issue for the cross-process misses (block kernels hit
100%; pinning `PYTHONHASHSEED` doesn't help).

### 10. Draft 07 should cite a warmup baseline

"Cold warmup level with V1" should cite a number. The measured x86 baseline is granite-3.3-8b-instruct, TP1, `max_model_len=2048`,
`max_num_seqs=32`: 724 s. The harness is in `docs/investigations/mrv2-warmup-measurements/` on `mrv2-investigation-906`.

### 11. Spot checks that hold

- The 0.28 seams exist (`ModelCudaGraphManager`, `BatchExecutionDescriptor`, `dispatch_cg_and_sync_dp`).
- The platform only fills `worker_cls` when it's `"auto"`, so `--worker-cls` works without code changes.
- #1111 merged on 2026-10-09.
- `_SlotMappingKernelShim` matches #945.
- The investigation report now marks its own-kernels decision, its rejection of triton-cpu and its EPIC list as superseded by the
  plan.

## Resolution (plan author, 2026-10-12)

All findings were checked against the code and accepted. Changes are in `mrv2-plan-906.html` and `issues/*.md`.

1. **Gate timing:** confirmed (`vllm/triton_utils/__init__.py` binds `triton`/`tl` at import). The gate now takes effect at plugin load, before `vllm.triton_utils` is imported, in every process. That needs an env-var opt-in, because `register()` runs before config parsing: `SPYRE_MODEL_RUNNER=v1|v2` (default `v1`; stage 1 of the flip flips it). The switch moves from Wave 2 into WP-A, with env-gated edits to `spyre_inference/__init__.py` and `platform.py`. Ground rule 1, `test_layout.py`'s boundary, drafts 01, 02, 06, 07, 08a and 08b updated. The `HAS_TRITON` tests run in fresh processes.
2. **0.32 vs. the V1 pin:** confirmed (`"stock torch.compile"` at v0.31 `config/vllm.py:3057`; #59079 is a small change). The gate also drops that entry, and WP-G verifies compiled V2 on 0.31. If it holds, nothing waits for 0.32; the flip section records the conflict and the fallback options if it doesn't. Manual runs use `--enforce-eager` until then.
3. **`initialize_kv_cache` attributes:** the sketch now makes one `kv_cache.init` call, a note lists every attribute MRV2's method sets, and WP0 owns the list and the decision (likely: wrap `super()` with only allocation and group creation replaced). Drafts 01 and 04 updated.
4. **`publish_null_slots`:** added to the warmup policy, WP-H, the plan test and draft 06.
5. **Platform rejections:** spec decode, batch-sharded sampling, `cudagraph_mm_encoder` and PCP > 1 are on the V2 branch of `check_and_update_config` (WP-INT, draft 07, applying on `v2` in 08a). The LoRA guard moves into `load.py` (WP-G, draft 06).
6. **`init_device`:** copied rather than `super()`, with a drift test (WP0, draft 01).
7. **Top-k/top-p sort:** recorded as a fact in the kernel strategy; WP-E measures it on x86 (draft 05).
8. **JIT vs. test budget:** triton-cpu tests carry a `triton_cpu` marker, are exempt from the budget, stay few, and CI keeps a warm `TRITON_CACHE_DIR`.
9. **Attention-cache finding:** new non-MRV2 draft `issues/10-attention-kernel-cache.md`, including filing the torch-spyre issue; the plan links it.
10. **Warmup baseline:** draft 07 and the plan's bring-up step cite 724 s with the configuration and harness location.
11. No action needed.
