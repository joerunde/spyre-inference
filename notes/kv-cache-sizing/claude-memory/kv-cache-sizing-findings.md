---
name: kv-cache-sizing-findings
description: "Measured Spyre KV-cache ceiling at TP2 (96 GiB tensor region/card, ~8500 blocks granite-8b); only --num-gpu-blocks-override works today; main's torch-spyre pin cf02782 already has get_memory_info"
metadata:
  node_type: memory
  type: project
  originSessionId: 73a06737-2570-4746-9dc6-140416302841
  modified: 2026-10-19T17:24:49.820Z
---

Measured 2026-10-19 on rundemc-2-spyre-dev, granite-3.3-8b TP2, max_model_len 2048, max_num_seqs 4, block 128 (10 MiB/block/rank):
- Each card's flex Tensor region is 96 GiB (`TotalCapacityBytes` 103079215104 in the OOM RAS message). Raw eager allocation tops out at exactly 96 GiB.
- Non-KV footprint after warmup is ~8.15-8.3 GiB/rank; warmup peak is only a few MB above steady state.
- `--num-gpu-blocks-override` 8500 works (91 GiB/rank). 8900 OOMs in `allocate_pages` while needing ~95 of 96 GiB.
- Decode ms/step and prefill are flat from 65 to 8500 blocks. Init grows by about 100 s at the top end.
- `--kv-cache-memory-bytes` and `--gpu-memory-utilization` are silently ignored. `check_and_update_config` always sets `num_gpu_blocks_override = seqs × blocks/seq + 1` unless the user set the override.

Path to dynamic sizing is spyre-inference#901, now unblocked. torch-spyre#4541/#4542 closed via PR 5145 (`get_memory_info`, merge 5a28a9dd). origin/main's pin `cf027821` (#1298) is 16 commits past it. The measurements above used the stale branch's stack (torch-spyre 87444370, vLLM 0.28), so re-check footprint and ceiling on main. KV sizing code on main is unchanged as of 2026-10-19.

**Why:** I once reported "176 commits behind" from a stale branch's pin; always check origin/main's pyproject before claiming a bump is needed.

**How to apply:** treat the 96 GiB ceiling as hardware fact; re-measure footprint only on main's stack. The old stack ran from the `mrv2-host-profile` worktree venv with the system RPMs; `~/spyre-libs` breaks that build. Scripts are in the job tmp dir and won't persist.
