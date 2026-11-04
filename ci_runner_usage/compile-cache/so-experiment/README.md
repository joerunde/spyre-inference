# CPU `.so`/`.o` reuse across a torch-spyre + RPM bump

Run on Nov 3–4, 2026, on one Spyre card in a dev pod. Every run went one at a time.

## Stacks

| | spyre-inference locks | torch-spyre | RPM bundle |
|---|---|---|---|
| yesterday | `658916f` (#1448) | `0.6.0.dev390+main.g2461a9e4` | `main_snap_2026_10_30_00` |
| today | `6348f4e` (#1483) | `0.6.0.dev417+main.gf8a09bb0` | `main_snap_2026_11_02_06` |

Torch was `2.13.0+cpu` on both. The two stacks are 27 torch-spyre commits apart.

## Setup

```bash
git worktree add --detach $WORKTREE 658916f
bash scripts/install-pinned-rpms.sh --prefix $SO/rpms-old              # yesterday's lock
source $SO/rpms-old/env.sh
export SEN_COMMON_HEADERS="$SENTIENT_BASE_INSTALL_DIR/runtime/include"
uv cache clean torch-spyre && uv sync --group dev --reinstall-package torch-spyre
# ... run A ...
git checkout 6348f4e -- pyproject.toml uv.lock spyre-rpms.lock
bash scripts/install-pinned-rpms.sh --prefix $SO/rpms-new
# re-source $SO/rpms-new/env.sh, rebuild as above, then run B and C
```

`run.sh` (one test) and `shard.sh` (smoke shard 2/11) give each run its own empty `TORCHINDUCTOR_CACHE_DIR`, with `SPYRE_KERNEL_CACHE=1`. For C they first copy only the `*.so`/`*.o` files from A's cache into it. They list the `.so`/`.o` files with their mtimes before and after the run (`file-lists/<run>.{pre,post}.tsv`), so a reused file shows up with an unchanged mtime.

## Results

**One CPU-compiled test** (`tests/attention/test_spyre_encoder_attn.py::test_spyre_encoder_attn[dtype(fp16)-block_size(64)-head_size(64)-GQA-batch_unaligned(3seqs)-compilation_STOCK-device_cpu]`):

| Run | Stack | Cache at start | `.so` built | pytest |
|---|---|---|---|---|
| A3 | yesterday | empty | 15 | 24.1 s |
| B, B2, B3 | today | empty | 15 | 20.3 / 20.3 / 20.5 s |
| C, C2, C3 | today | A3's `.so` only | 0 (15 reused) | 10.9 / 10.9 / 11.0 s |

**Smoke shard 2/11** (235 tests, listed in `smoke-shard-2-of-11.ids`):

| Run | Stack | Cache at start | `.so` built | Result | pytest |
|---|---|---|---|---|---|
| SA | yesterday | empty | 25 | 234 passed, 1 skipped | 879 s |
| SB | today | empty | 25 | 234 passed, 1 skipped | 847 s |
| SC | today | SA's `.so` only | 0 (25 reused) | 234 passed, 1 skipped | 843 s |

- **Reuse is correct across the bump.** Today's stack generated the same kernel paths as yesterday's (25 of 25), and every reused kernel passed.
- **No `.o` files were produced.** These kernels compile and link in one step.
- **Each kernel costs about 0.6 s to compile.** A full shard builds only 25 of them, 1.1 MB out of a 207 MB cache. Seeding saved 3.7 s on a 14-minute shard, which is noise.
- `A`/`A2` (`test_basic_llm_inference[micro-g3.3]` and `test_compiled_pooling_encoder_buckets`) built **no** CPU kernels. The compiled e2e model tests lower everything to Spyre.

## Not covered

- Only one of the 11 smoke shards ran. Other shards may build a different number of CPU kernels, but none is likely to reach minutes of compile time.
- Both stacks used the same venv path, so the include-path mismatch between our CI jobs and dispatched images, which would make every lookup miss, wasn't tested. Given these numbers it doesn't matter.
