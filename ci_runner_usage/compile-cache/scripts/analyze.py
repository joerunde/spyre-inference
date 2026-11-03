"""Recompute every number on compile-cache/index.html from the files in ../data.

Usage (from a spyre-inference checkout, so the uv.lock-per-commit lookups resolve):

    python3 analyze.py --data ../data --repo /path/to/spyre-inference

Stdlib only. The --repo checkout must contain the spyre-inference commits the runs
tested (`git fetch origin`); runs whose commit is missing are left out of the
uv.lock-matched figures, as on the page.
"""

import argparse
import collections as C
import datetime as dt
import gzip
import hashlib
import json
import os
import re
import statistics as S
import subprocess

TEST_JOB = "build and run tests"
# Jobs over this ran on after a cancel (pre spyre-inference#1353) and are not real work.
ZOMBIE_H = 2.0
KERNEL_PATHS = ("torch_spyre/", "csrc/", "third_party/", "cmake/")
KERNEL_FILES = ("setup.py", "pyproject.toml", "CMakeLists.txt")
KERNEL_EXT = (".cpp", ".h", ".cc", ".c")


def ts(s):
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00"))


def job_h(j):
    return (ts(j["completed_at"]) - ts(j["started_at"])).total_seconds() / 3600


def test_jobs(r):
    return [j for j in r["jobs"] if TEST_JOB in j["name"] and j["started_at"] and j["completed_at"]]


def test_h(r):
    return sum(job_h(j) for j in test_jobs(r) if job_h(j) <= ZOMBIE_H)


def first_green_end(r):
    ends = [ts(j["completed_at"]) for j in test_jobs(r) if j["conclusion"] == "success"]
    return min(ends) if ends else None


def sha(version):
    return version.rsplit(".g", 1)[1] if version and ".g" in version else None


def kernel_relevant(files):
    return any(
        f.startswith(KERNEL_PATHS) or f.endswith(KERNEL_EXT) or f in KERNEL_FILES for f in files
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(os.path.dirname(__file__), "..", "data"))
    ap.add_argument("--repo", default=".")
    args = ap.parse_args()

    with gzip.open(os.path.join(args.data, "dispatch-runs.json.gz"), "rt") as fh:
        runs = json.load(fh)["runs"]
    with gzip.open(os.path.join(args.data, "test-each-commit-runs.json.gz"), "rt") as fh:
        tec = json.load(fh)["runs"]
    pr_of = json.load(open(os.path.join(args.data, "pr-url-hashes.json")))
    compares = json.load(open(os.path.join(args.data, "torch-spyre-compares.json")))

    locks = {}

    def lock(ref):
        if not ref or ref.startswith("("):
            return None
        if ref not in locks:
            p = subprocess.run(["git", "-C", args.repo, "show", f"{ref}:uv.lock"], capture_output=True)
            locks[ref] = hashlib.sha256(p.stdout).hexdigest() if p.returncode == 0 else None
        return locks[ref]

    for r in runs:
        url = pr_of.get(r["pr_url_hash"] or "")
        r["pr"] = url
        if url and "/torch-spyre/torch-spyre/" in url:
            r["src"] = "torch-spyre PR"
        elif url and "/spyre-inference/" in url:
            r["src"] = "spyre-inference PR"
        elif url and "/hf-adapters/" in url:
            r["src"] = "hf-adapters PR"
        elif r["pr_url_hash"] in (None, "(not specified)"):
            r["src"] = "no PR hash"
        else:
            r["src"] = "unmatched PR hash"

    print(f"== {len(runs)} dispatched runs")
    zombies = [j for r in runs for j in test_jobs(r) if job_h(j) > ZOMBIE_H]
    print(f"zombie test jobs (> {ZOMBIE_H} h): {len(zombies)}, {sum(map(job_h, zombies)):.0f} h; "
          f"all test-job h {sum(job_h(j) for r in runs for j in test_jobs(r)):.0f}")
    print("zombie jobs by start day:", sorted(C.Counter(j['started_at'][:10] for j in zombies).items()))
    by_src = C.defaultdict(list)
    for r in runs:
        by_src[r["src"]].append(r)
    for k, v in sorted(by_src.items()):
        print(f"  {k:20} runs {len(v):4}  test h {sum(map(test_h, v)):6.0f}")
    print("no torch-spyre version:", sum(not r["torch_spyre_version"] for r in runs))
    imgs = {r["image_labels"][0].rsplit("_r", 1)[0] for r in runs if r["image_labels"]}
    print("distinct image ids:", len(imgs))

    # -- histogram + within-PR classification (torch-spyre PRs)
    ts_runs = by_src["torch-spyre PR"]
    per_pr = C.defaultdict(list)
    for r in ts_runs:
        per_pr[r["pr"]].append(r)
    print(f"\n== {len(per_pr)} torch-spyre PRs, {len(ts_runs)} runs")
    print("bins (triggers: PRs, runs, test h, mean distinct commits):")
    bins = C.defaultdict(list)
    for v in per_pr.values():
        bins[len(v)].append(v)
    for n, prs in sorted(bins.items()):
        commits = S.mean(len({r["torch_spyre_version"] for r in v if r["torch_spyre_version"]}) for v in prs)
        print(f"  {n:2}: {len(prs):3} PRs  {sum(len(v) for v in prs):4} runs  "
              f"{sum(test_h(r) for v in prs for r in v):5.0f} h  {commits:.1f}")
    counts = [len(v) for v in per_pr.values()]
    print(f"median {S.median(counts)}, mean {S.mean(counts):.2f}, max {max(counts)}, repeats {sum(counts) - len(counts)}")
    gaps = [(ts(b["created_at"]) - ts(a["created_at"])).total_seconds() / 3600
            for v in per_pr.values() for a, b in zip(v, v[1:])]
    print(f"median gap {S.median(gaps):.1f} h, within 3 h {sum(g < 3 for g in gaps) / len(gaps):.0%}")

    cls_n, cls_h = C.Counter(), C.Counter()
    exact_n = exact_h = 0
    for v in per_pr.values():
        prev, prev_exact = [], C.defaultdict(list)
        for r in v:
            s, start = sha(r["torch_spyre_version"]), ts(r["created_at"])
            if not s:
                k = "unknown stack"
            elif not prev:
                k = "first run of PR"
            elif any(p == s and g and g <= start for p, g in prev):
                k = "same commit, earlier run warm"
            else:
                d = compares.get(f"{prev[-1][0]}...{s}", {})
                k = ("new commit, no kernel-relevant change"
                     if d.get("files") and not kernel_relevant(d["files"]) else "new commit, kernel-relevant change")
            cls_n[k] += 1
            cls_h[k] += test_h(r)
            if s:
                key = (s, lock(r["si_ref"]))
                if key[1] and any(g and g <= start for g in prev_exact[key]):
                    exact_n += 1
                    exact_h += test_h(r)
                prev_exact[key].append(first_green_end(r))
                prev.append((s, first_green_end(r)))
    for k in cls_n:
        print(f"  {k:40} {cls_n[k]:4} runs {cls_h[k]:6.0f} h")
    print(f"  same commit + same uv.lock, warm:        {exact_n:4} runs {exact_h:6.0f} h")

    # -- other dispatches: bursts on one torch-spyre main commit
    others = [r for r in runs if r["src"] != "torch-spyre PR"]
    seen_v, seen_k = C.defaultdict(list), C.defaultdict(list)
    hv = hk = nv = nk = 0
    for r in runs:
        v = r["torch_spyre_version"]
        if not v:
            continue
        start, mine = ts(r["created_at"]), r["src"] != "torch-spyre PR"
        k = (v, lock(r["si_ref"]))
        if mine and any(g and g <= start for g in seen_v[v]):
            nv += 1
            hv += test_h(r)
        if mine and k[1] and any(g and g <= start for g in seen_k[k]):
            nk += 1
            hk += test_h(r)
        seen_v[v].append(first_green_end(r))
        if k[1]:
            seen_k[k].append(first_green_end(r))
    vers = C.defaultdict(list)
    for r in others:
        if r["torch_spyre_version"]:
            vers[r["torch_spyre_version"]].append(ts(r["created_at"]))
    spans = [(max(t) - min(t)).total_seconds() / 3600 for t in vers.values() if len(t) > 1]
    print(f"\n== {len(others)} non-torch-spyre-PR dispatches, test h {sum(map(test_h, others)):.0f}")
    print(f"distinct versions {len(vers)}, used once {sum(len(t) == 1 for t in vers.values())}, "
          f"median span of a repeated version {S.median(spans):.1f} h")
    print(f"same version seen green earlier: {nv} runs {hv:.0f} h; with same uv.lock: {nk} runs {hk:.0f} h")
    print("distinct uv.lock contents:", len({h for h in locks.values() if h}))

    # -- exact-stack key churn (all runs)
    seen, written, read = C.defaultdict(list), C.Counter(), set()
    for r in runs:
        k = (r["torch_spyre_version"], lock(r["si_ref"]))
        if not all(k):
            continue
        if any(g and g <= ts(r["created_at"]) for g in seen[k]):
            read.add(k)
        if any(j["conclusion"] == "success" for j in test_jobs(r)):
            written[k] += 1
        seen[k].append(first_green_end(r))
    print(f"\n== exact-stack keys written {len(written)}, ever read {len(read & set(written))}, "
          f"single-run keys {sum(c == 1 for c in written.values())}")

    # -- warm vs cold shard minutes
    def shard_min(r, pat):
        out = {}
        for j in test_jobs(r):
            m = re.search(pat, j["name"])
            if m and j["conclusion"] == "success":
                out[m.group(1)] = job_h(j) * 60
        return out

    pats = {"smoke": r"Non-distributed spyre tests \(shard (\d+)/11\)", "dist": r"Distributed spyre tests \(shard (\d+)/3\)"}
    warm_runs = [r for r in tec if "restored" in (r["cache_line"] or "")]
    cold_runs = [r for r in runs if r["test_type"] == "integration" and r["created_at"] >= "2026-10-24"]
    print(f"\n== warm test_each_commit runs {len(warm_runs)} of {sum(bool(r['cache_line']) for r in tec)} with a cache line")
    tw = tc = 0
    for name, pat in pats.items():
        w, c = C.defaultdict(list), C.defaultdict(list)
        for r in warm_runs:
            for s, m in shard_min(r, pat).items():
                w[s].append(m)
        for r in cold_runs:
            for s, m in shard_min(r, pat).items():
                c[s].append(m)
        mw = {s: S.median(v) for s, v in w.items()}
        mc = {s: S.median(v) for s, v in c.items()}
        print(f"  {name}: warm {min(mw.values()):.1f}-{max(mw.values()):.1f}, cold {min(mc.values()):.1f}-{max(mc.values()):.1f} min/shard, "
              f"saved {1 - sum(mw.values()) / sum(mc.values()):.0%}")
        tw += sum(mw.values())
        tc += sum(mc.values())
    print(f"  run total: warm {tw:.0f}, cold {tc:.0f} runner-min")


if __name__ == "__main__":
    main()
