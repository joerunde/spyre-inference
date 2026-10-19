# KV cache sizing on Spyre: investigation notes (2026-10-19)

Scratch notes, not for merge. Question: can we query free device memory and size the KV
cache dynamically instead of pinning it to `max_num_seqs × ceil(max_model_len / block_size) + 1`?

## TL;DR

- Today only `--num-gpu-blocks-override` controls the cache size. `--kv-cache-memory-bytes`
  and `--gpu-memory-utilization` are silently ignored: the platform still sets the batch-area
  override, and vLLM applies that after its own math.
- Each card has **96 GiB** of usable tensor memory. With granite-3.3-8b at TP=2, the model plus
  warmup takes about 8.2 GiB per rank, which leaves room for about **8,500 blocks**: about 130×
  the default, roughly 1.09M tokens.
- A bigger cache costs no measurable decode or prefill time. Init grows by about 100 s at
  8,500 blocks.
- The memory-query API exists. torch-spyre PR #5145 (merge `5a28a9dd`, fixes #4542) adds
  `torch.accelerator.get_memory_info("spyre")`. **main's pin `cf027821` already includes it**
  (16 commits past), so spyre-inference#901 is unblocked.

## Where sizing happens (main @ b685cbb, unchanged from the older branch)

- `spyre_inference/platform.py`, `check_and_update_config`:
  - Sets `os.environ.setdefault("VLLM_CPU_KVCACHE_SPACE", "4")` unless `kv_cache_memory_bytes`
    is set, so `CpuPlatform` fills `kv_cache_memory_bytes = 4 GiB`. This is a placeholder.
  - Sets `num_gpu_blocks_override = max_num_seqs * ceil(max_model_len / block_size) + 1`
    whenever the user didn't set it. This skips pooling models.
- `TorchSpyreWorker.determine_available_memory` returns `kv_cache_memory_bytes` as-is.
- vLLM's `get_kv_cache_configs` then replaces the computed block count with the override and
  logs `Overriding num_gpu_blocks=X with num_gpu_blocks_override=Y`.

## Measurements

Setup:
- Host: rundemc-2-spyre-dev, 2 cards.
- Model and engine: `/models/granite-3.3-8b-instruct`, TP=2, fp16, compiled (default),
  `max_model_len=2048`, `max_num_seqs=4`, block size 128. Head-major KV layout:
  40 layers × 2 (K, V) × 4 KV heads/rank × 128 × 128 × 2 B = **10 MiB per block per rank**.
- Runtime: `SPYRE_NUM_CPUS=16`, because the pod's CPU quota throttles larger thread counts.
- Stack: an **older** one, spyre-inference `7f4ecd1`, torch-spyre `87444370`, vLLM 0.28,
  system RPMs. Re-check the footprint numbers on main's stack.
- Workload: 4 × 128-token random prompts, 128 output tokens, greedy decoding.

| Setting | Blocks | Device memory per rank | Decode ms/step | Prefill (batch of 4) | Init time |
|---|---|---|---|---|---|
| default | 65 | 8.77 GiB | 122 | 0.87 s | 296 s |
| `kv_cache_memory_bytes=40GiB` | 65 (ignored) | 8.77 GiB | 125 | 0.87 s | 259 s |
| `gpu_memory_utilization=0.5` | 65 (ignored) | 8.77 GiB | 140 | 0.86 s | 261 s |
| `num_gpu_blocks_override=1025` | 1025 | 18.17 GiB | 136 | 0.95 s | 290 s |
| `num_gpu_blocks_override=4097` | 4097 | 48.23 GiB | 134 | 0.88 s | 334 s |
| `num_gpu_blocks_override=8000` | 8000 | 86.43 GiB | 134 | 0.91 s | 389 s |
| `num_gpu_blocks_override=8500` | 8500 | 91.32 GiB | 128 | — | 399 s |
| `num_gpu_blocks_override=8900` | out of memory in `allocate_pages` | needs about 95 GiB | — | — | — |
| `num_gpu_blocks_override=9200` | out of memory | — | — | — | — |

- **Model footprint:** device memory minus `blocks × 10 MiB` stays at 8.14–8.32 GiB per rank
  in every run. Warmup peaks only a few MB above steady state.
- **Ranks are symmetric:** both ranks' allocations agree to within a few KB.
- **Card ceiling:** a raw eager allocation loop (`probe_mem.py`) stops at exactly 96 GiB.
  The flex out-of-memory error reports `TotalCapacityBytes=103079215104` for domain 0.
- **Init-time noise:** init varied by about 40 s between runs of the same config, so treat the
  init column as rough.
- **Unexplained:** greedy output on random-token prompts matched the default in the 4097 and
  8500 runs but differed in 1025 and 8000. Likely fp16 differences from where pages land, but
  check this before relying on large caches.
- **Before this torch-spyre version:** `torch.accelerator.get_memory_info()` raised "not
  implemented". `memory_stats()`/`memory_allocated()` already reported correct numbers.

## Upstream state

- torch-spyre #4541: flex `FlexAllocator::getMemoryStats(MemoryType)`, from flex PR 1896.
  Closed.
- torch-spyre #4542 / PR #5145: `SpyreAllocator::getMemoryInfo` reports the tensor region,
  and `reserved_bytes` is now tracked too. Merged 2026-10-15 as `5a28a9dd`.
- The flex runtime headers also expose `DeviceMemoryInfo::getTotalSize()`, which on a
  physical-function (PF) device is a hardware query of LPDDR size, and per-region
  `MemoryRegion::free_size()`.
- spyre-inference #901 has the full design: implement `determine_available_memory` with
  `get_memory_info`, honor `gpu_memory_utilization`, keep the overrides as escape hatches, and
  drop the `VLLM_CPU_KVCACHE_SPACE` placeholder.

## Comments on #901

- **Don't hard-cap at the batch area.** Prefix caching is on by default, and blocks beyond
  `max_num_seqs × blocks/seq` are exactly what it uses. The attention buckets depend on
  `max_model_len`, not on the allocated block count, so a bigger cache adds no warmup work.
- **Where to measure:** warmup holds almost no transient memory, so measuring after warmup is
  fine.
- **TP:** take the minimum across ranks to be safe. In practice the ranks match.
- **Stopgap without the API:** `free ≈ 96 GiB − memory_allocated()`, with the 96 GiB
  hard-coded.

## Next steps

1. Build main's stack on a Spyre pod. The RPMs are now pinned by bundle (#1299):
   `scripts/install-pinned-rpms.sh --rebuild` and source `env.sh`.
2. At TP=2, confirm that `torch.accelerator.get_memory_info("spyre")` returns sane
   (free, total) values and that free + `memory_reserved()` is consistent.
3. Prototype #901 and re-run `sweep.sh`-style checks (default, `gpu_memory_utilization` 0.5
   and 0.9) to confirm the ceiling.

## Files

- `kv_probe.py`: engine probe at TP=2. Reports per-rank allocated bytes via `collective_rpc`
  plus prefill and decode timing.
  - Usage: `python kv_probe.py OUT.json [--blocks N] [--kv-bytes B] [--gmu F]`
  - Needs `VLLM_ALLOW_INSECURE_SERIALIZATION=1` for the `collective_rpc` closure.
- `probe_mem.py`: allocates 1 GiB at a time on one card until out of memory.
- `sweep.sh`: the sweep that produced the table above. Run it as `REPO=<checkout> bash sweep.sh`; output goes to `runs-new/`, or `$OUT` if set.
- `runs/*.json`: the per-run results.
- `runs/*.excerpt.log`: the relevant log lines from each run; the full logs were too large to
  keep.
- `claude-memory/`: copies of the session memory notes relevant here: the sizing findings, the pod CPU-quota caveat, and the RPM build caveat.
