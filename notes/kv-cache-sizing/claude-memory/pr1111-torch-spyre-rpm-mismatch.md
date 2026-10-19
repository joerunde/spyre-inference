---
name: pr1111-torch-spyre-rpm-mismatch
description: "torch-spyre pins on main won't build against this pod's ibm-flex RPMs; no on-device runs here"
metadata:
  node_type: memory
  type: project
  originSessionId: c4e5b844-78b1-40bb-9cb1-720e23f4848e
  modified: 2026-10-14T19:35:13.739Z
---

On the `rundemc-2-spyre-dev` pod, `uv sync` fails building torch-spyre:
- 2026-10-09, rev `3d43e671` (PR 1111's merge-base pin)
- 2026-10-14, main's pin `89545685`, even with `~/spyre-libs/env.sh` sourced

```
torch_spyre/csrc/spyre_stream.cpp:249:30: error: too many arguments to function 'flex::createDmaParams(...)'
torch_spyre/csrc/distributed/spyre_distributed.cpp:574:20: error: 'relinquish_work_schedule' is not a member of 'spyre_comms'
```

The installed `ibm-flex` / `spyre_comms` are older than these revs expect (the RPM-mismatch case in CLAUDE.md). An on-device run needs `scripts/install-pinned-rpms.sh --rebuild` plus its `env.sh` first.

**How to apply:**
- The main checkout has no `.venv`.
- For CPU-only validation, build one with `uv sync --group dev --no-install-package torch-spyre`. Tests needing torch_spyre then fail, about 85 in `tests/runtime`, `tests/attention` and `tests/pool`, so compare failure sets against the parent commit.
- Say plainly that hardware tests didn't run.

See [[pr1111-sampling-kernel-validation]].
