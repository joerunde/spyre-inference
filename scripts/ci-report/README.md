# CI test report

`ci_test_report.py` builds a per-attempt report of `test_each_commit` CI runs:
worker(s), pass/bus-error/other outcome, and installed Spyre RPM versions.

## What it produces

One row per **run attempt** (re-runs are separate rows with their own link):

| Column | Source |
| --- | --- |
| Run + link | `run_number`, links to `.../runs/<id>/attempts/<n>` |
| Att | `<attempt>/<max_attempt>` |
| Status | `Passed` / `Bus error` (`RAS::PCI::BusFence` in any job log) / `Other failure` / `In progress` |
| Worker(s) | `p1-worker-NN` from the `GHA_RUNNER_POD_NODE_NAME` log line (one per matrix job; the bus-error worker is **bold** in markdown) |
| Failed cfg | which matrix configs (Distributed spyre tests, etc.) failed |
| RPM columns | version of each RPM from the "Extract RPMs" step (`Extracting <pkg>-<ver>...`). Empty for runs before ~2026-07-14 when this logging was added. |

Outputs `ci_test_report.md` (human) and `ci_test_report.csv` (full versions, machine).

## Usage

```bash
# whole last month
python3 ci_test_report.py --since 2026-06-24 --out-dir ./out

# options
python3 ci_test_report.py \
  --since 2026-06-24 \
  --workflow test_each_commit \   # workflow name or file
  --cache-dir .ci-cache \         # cached gh responses + log zips (safe to delete)
  --out-dir ./out \
  --jobs 10 \                     # parallel gh workers
  --limit 20                      # cap runs (debugging)
```

Requires an authenticated `gh` CLI (`gh auth status`) with read access to
`torch-spyre/spyre-inference`.

## Notes

- **Caching**: every `gh api` response and log zip is cached under `--cache-dir`
  keyed by run id + attempt, so re-runs only fetch new attempts. Delete the
  cache dir (or a run's subdir) to force a refetch. Expired/absent logs are
  marked with a `.missing` sentinel and not refetched.
- **Granularity**: one row per attempt. Each attempt fans out to ~6 matrix
  jobs on *different* workers, so `Worker(s)` is the set across those jobs and
  `Status` is the worst outcome. Switch to per-job granularity by emitting a row
  per job in `process_run` if needed.
- **Bus error signature**: `RAS::PCI::BusFence` / `PCIe bus master fence`
  (see `BUS_FENCE_RE`). The RPM/worker regexes are also near the top of the
  script and easy to adjust.
- **RPM columns** follow `spyre-rpms.lock`; new packages are auto-appended.
