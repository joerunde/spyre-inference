#!/usr/bin/env python3
"""Build a per-attempt report of `test_each_commit` CI runs.

For every attempt of every run of a workflow (default: test_each_commit) in a
time window, this collects:
  - the run number + a link to that specific attempt
  - the physical worker(s) the attempt's jobs ran on (p1-worker-NN)
  - the outcome: Passed / Bus error (RAS::PCI::BusFence) / Other failure
  - the installed Spyre RPM versions (from the "Extract RPMs" step; only
    present in logs from ~2026-07-14 onward)

Data sources (all via `gh api`, so it uses your existing gh auth):
  - run list:      /actions/workflows/{id}/runs
  - per-attempt:   /actions/runs/{id}/attempts/{n}/jobs   (worker, conclusion)
  - per-attempt:   /actions/runs/{id}/attempts/{n}/logs   (worker, bus error, RPMs)

Everything is cached under --cache-dir so re-runs are fast and incremental:
delete the cache dir (or a specific run's subdir) to force a refetch.

Usage:
  python3 ci_test_report.py --since 2026-06-24 --out-dir ./out
  python3 ci_test_report.py --since 2026-06-24 --workflow test_each_commit \
      --cache-dir .ci-cache --jobs 8

Requires: gh (authenticated), python3.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import subprocess
import sys
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = "torch-spyre/spyre-inference"

# The physical k8s node the runner pod landed on, echoed by gather-runner-info.
WORKER_RE = re.compile(r"GHA_RUNNER_POD_NODE_NAME[ =]+(p1-worker-\d+)")
# PCIe bus-master fence hardware fault. Match the RAS name and the human string.
BUS_FENCE_RE = re.compile(r"RAS::PCI::BusFence|PCIe bus master fence", re.IGNORECASE)
# "Extracting ibm-deeptools-2.0.0-0.main.1+1520.0799224_216.el10.x86_64.rpm..."
# Capture <pkg> and <version> where version starts at the first "-<digit>".
RPM_RE = re.compile(r"Extracting\s+([a-z0-9][a-z0-9-]*?)-(\d[^ ]*?)\.(?:x86_64|noarch|aarch64)\.rpm")

# Order RPM columns by the families in spyre-rpms.lock (extend as needed).
RPM_COLUMN_ORDER = [
    "ibm-aiu-toolbox-e2e",
    "ibm-deeptools",
    "ibm-deeptools-devel",
    "ibm-flex",
    "ibm-flex-devel",
    "ibm-senlib-core",
    "ibm-senlib-dd2",
    "ibm-senlib-headers",
    "ibm-spyre-comms",
    "ibm-spyre-comms-devel",
    "ibm-spyre-comms-test",
]


def gh_api(path: str, paginate: bool = False) -> bytes:
    cmd = ["gh", "api"]
    if paginate:
        cmd += ["--paginate"]
    cmd.append(path)
    res = subprocess.run(cmd, capture_output=True)
    if res.returncode != 0:
        raise RuntimeError(f"gh api {path} failed: {res.stderr.decode(errors='replace')}")
    return res.stdout


def gh_api_json(path: str) -> dict:
    return json.loads(gh_api(path))


def resolve_workflow_id(name_or_file: str) -> int:
    data = gh_api_json(f"repos/{REPO}/actions/workflows?per_page=100")
    for wf in data["workflows"]:
        if name_or_file in (wf["name"], Path(wf["path"]).name, wf["path"]):
            return wf["id"]
    raise SystemExit(f"Workflow not found: {name_or_file}")


def list_runs(workflow_id: int, since: str) -> list[dict]:
    """All runs created on/after `since` (YYYY-MM-DD)."""
    out = gh_api(
        f"repos/{REPO}/actions/workflows/{workflow_id}/runs"
        f"?created=>={since}&per_page=100",
        paginate=True,
    )
    # --paginate concatenates JSON objects; split them.
    runs: list[dict] = []
    for obj in _iter_json_objects(out):
        runs.extend(obj.get("workflow_runs", []))
    # De-dup by run id (pagination can overlap).
    seen: dict[int, dict] = {}
    for r in runs:
        seen[r["id"]] = r
    return sorted(seen.values(), key=lambda r: r["run_number"])


def _iter_json_objects(raw: bytes):
    """Yield top-level JSON objects from a possibly-concatenated stream."""
    dec = json.JSONDecoder()
    text = raw.decode(errors="replace").strip()
    idx = 0
    n = len(text)
    while idx < n:
        while idx < n and text[idx].isspace():
            idx += 1
        if idx >= n:
            break
        obj, end = dec.raw_decode(text, idx)
        yield obj
        idx = end


def attempt_jobs(cache_dir: Path, run_id: int, attempt: int) -> list[dict]:
    cf = cache_dir / str(run_id) / f"attempt{attempt}_jobs.json"
    if cf.exists():
        return json.loads(cf.read_text())["jobs"]
    data = gh_api_json(
        f"repos/{REPO}/actions/runs/{run_id}/attempts/{attempt}/jobs?per_page=100"
    )
    cf.parent.mkdir(parents=True, exist_ok=True)
    cf.write_text(json.dumps(data))
    return data["jobs"]


def attempt_logs(cache_dir: Path, run_id: int, attempt: int) -> dict[str, str] | None:
    """Return {job_log_filename: text} for an attempt, or None if unavailable.

    The per-attempt logs endpoint returns a single zip containing one .txt per
    job that ran in that attempt (re-runs only include the re-run jobs).
    """
    cf = cache_dir / str(run_id) / f"attempt{attempt}_logs.zip"
    cf.parent.mkdir(parents=True, exist_ok=True)
    if not cf.exists():
        try:
            raw = gh_api(f"repos/{REPO}/actions/runs/{run_id}/attempts/{attempt}/logs")
        except RuntimeError as e:
            # 410 Gone (expired) or 404: record a marker so we don't refetch.
            (cf.with_suffix(".missing")).write_text(str(e))
            return None
        cf.write_bytes(raw)
    if cf.with_suffix(".missing").exists():
        return None
    try:
        zf = zipfile.ZipFile(io.BytesIO(cf.read_bytes()))
    except zipfile.BadZipFile:
        return None
    out: dict[str, str] = {}
    for name in zf.namelist():
        if name.endswith(".txt") and "/" not in name:  # top-level per-job logs
            out[name] = zf.read(name).decode(errors="replace")
    return out


def parse_log(text: str) -> dict:
    workers = sorted(set(WORKER_RE.findall(text)))
    bus = bool(BUS_FENCE_RE.search(text))
    rpms: dict[str, str] = {}
    for pkg, ver in RPM_RE.findall(text):
        # strip trailing ".elNN" arch-y noise is already excluded by the regex
        rpms[pkg] = ver
    return {"workers": workers, "bus_fence": bus, "rpms": rpms}


def classify(jobs: list[dict], any_bus: bool) -> str:
    concls = {j.get("conclusion") for j in jobs}
    if any_bus:
        return "Bus error"
    if concls <= {"success", "skipped", None}:
        # None => still running/queued; treat as passed-so-far only if completed
        if any(j.get("status") != "completed" for j in jobs):
            return "In progress"
        return "Passed"
    return "Other failure"


def short_ver(ver: str) -> str:
    """2.0.0-0.main.1+1520.0799224_216.el10 -> 1520.0799224 (build+hash)."""
    m = re.search(r"\+([0-9]+\.[0-9a-f]+)", ver)
    return m.group(1) if m else ver


def process_run(cache_dir: Path, run: dict) -> list[dict]:
    rows = []
    run_id = run["id"]
    max_attempt = run.get("run_attempt", 1) or 1
    for attempt in range(1, max_attempt + 1):
        try:
            jobs = attempt_jobs(cache_dir, run_id, attempt)
        except RuntimeError:
            jobs = []
        logs = attempt_logs(cache_dir, run_id, attempt)
        workers: set[str] = set()
        any_bus = False
        rpms: dict[str, str] = {}
        bus_workers: set[str] = set()
        failed_configs: list[str] = []
        if logs:
            for fname, text in logs.items():
                parsed = parse_log(text)
                workers.update(parsed["workers"])
                if parsed["bus_fence"]:
                    any_bus = True
                    bus_workers.update(parsed["workers"])
                rpms.update(parsed["rpms"])  # same lock across jobs in an attempt
        # Map runner_name -> worker isn't in API; workers come from logs only.
        for j in jobs:
            if j.get("conclusion") in ("failure", "cancelled", "timed_out"):
                # trim the "test / build and run tests (X)" prefix
                nm = j["name"]
                m = re.search(r"\(([^)]+)\)", nm)
                failed_configs.append(m.group(1) if m else nm)
        rows.append(
            {
                "run_number": run["run_number"],
                "attempt": attempt,
                "max_attempt": max_attempt,
                "created_at": run["created_at"],
                "event": run.get("event", ""),
                "branch": run.get("head_branch", ""),
                "url": f"{run['html_url']}/attempts/{attempt}",
                "workers": sorted(workers),
                "bus_workers": sorted(bus_workers),
                "status": classify(jobs, any_bus),
                "failed_configs": sorted(set(failed_configs)),
                "rpms": rpms,
                "logs_available": logs is not None,
            }
        )
    return rows


def write_csv(rows: list[dict], path: Path, rpm_cols: list[str]) -> None:
    import csv

    with path.open("w", newline="") as f:
        w = csv.writer(f)
        header = [
            "run_number", "attempt", "max_attempt", "created_at", "event",
            "branch", "status", "workers", "bus_workers", "failed_configs",
            "logs_available", "url",
        ] + rpm_cols
        w.writerow(header)
        for r in rows:
            w.writerow(
                [
                    r["run_number"], r["attempt"], r["max_attempt"], r["created_at"],
                    r["event"], r["branch"], r["status"],
                    ";".join(r["workers"]), ";".join(r["bus_workers"]),
                    ";".join(r["failed_configs"]), r["logs_available"], r["url"],
                ]
                + [r["rpms"].get(c, "") for c in rpm_cols]
            )


def write_markdown(rows: list[dict], path: Path, rpm_cols: list[str]) -> None:
    lines = []
    # Summary
    status_counts = Counter(r["status"] for r in rows)
    bus_by_worker = Counter()
    for r in rows:
        for w in r["bus_workers"]:
            bus_by_worker[w] += 1
    lines.append("# test_each_commit CI report\n")
    lines.append(f"Total attempts: **{len(rows)}**\n")
    lines.append("Outcomes: " + ", ".join(
        f"{k}: **{v}**" for k, v in sorted(status_counts.items())) + "\n")
    if bus_by_worker:
        lines.append("Bus errors by worker: " + ", ".join(
            f"{w}: **{c}**" for w, c in bus_by_worker.most_common()) + "\n")
    lines.append("")

    # Short RPM column headers (drop the ibm- prefix).
    short_cols = [c.replace("ibm-", "") for c in rpm_cols]
    header = (
        ["Run", "Att", "Created", "Status", "Worker(s)", "Failed cfg"]
        + short_cols
    )
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join("---" for _ in header) + " |")
    for r in sorted(rows, key=lambda x: (x["run_number"], x["attempt"])):
        worker_cell = ", ".join(r["workers"]) or "—"
        # highlight the bus-error worker
        if r["bus_workers"]:
            worker_cell = ", ".join(
                f"**{w}**" if w in r["bus_workers"] else w for w in r["workers"]
            ) or ", ".join(r["bus_workers"])
        run_cell = f"[{r['run_number']}]({r['url']})"
        att_cell = f"{r['attempt']}/{r['max_attempt']}"
        created = r["created_at"].replace("T", " ").replace("Z", "")
        failed = ", ".join(r["failed_configs"]) or "—"
        rpm_cells = [short_ver(r["rpms"].get(c, "")) or "—" for c in rpm_cols]
        row = (
            [run_cell, att_cell, created, r["status"], worker_cell, failed]
            + rpm_cells
        )
        lines.append("| " + " | ".join(row) + " |")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--since", required=True, help="YYYY-MM-DD (runs created on/after)")
    ap.add_argument("--workflow", default="test_each_commit",
                    help="workflow name or file (default: test_each_commit)")
    ap.add_argument("--cache-dir", default=".ci-cache", type=Path)
    ap.add_argument("--out-dir", default=".", type=Path)
    ap.add_argument("--jobs", type=int, default=8, help="parallel gh workers")
    ap.add_argument("--limit", type=int, default=0, help="cap runs (debug)")
    args = ap.parse_args()

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Resolving workflow '{args.workflow}'...", file=sys.stderr)
    wf_id = resolve_workflow_id(args.workflow)
    print(f"Listing runs since {args.since}...", file=sys.stderr)
    runs = list_runs(wf_id, args.since)
    if args.limit:
        runs = runs[-args.limit:]
    print(f"{len(runs)} runs. Fetching attempts/jobs/logs "
          f"(cached in {args.cache_dir})...", file=sys.stderr)

    all_rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(process_run, args.cache_dir, r): r for r in runs}
        done = 0
        for fut in as_completed(futs):
            all_rows.extend(fut.result())
            done += 1
            if done % 20 == 0 or done == len(runs):
                print(f"  {done}/{len(runs)} runs processed", file=sys.stderr)

    # RPM columns: known order first, then any extras discovered.
    discovered = set()
    for r in all_rows:
        discovered.update(r["rpms"].keys())
    rpm_cols = [c for c in RPM_COLUMN_ORDER if c in discovered]
    rpm_cols += sorted(discovered - set(rpm_cols))

    csv_path = args.out_dir / "ci_test_report.csv"
    md_path = args.out_dir / "ci_test_report.md"
    write_csv(all_rows, csv_path, rpm_cols)
    write_markdown(all_rows, md_path, rpm_cols)
    print(f"Wrote {csv_path} and {md_path} "
          f"({len(all_rows)} attempt rows)", file=sys.stderr)


if __name__ == "__main__":
    main()
