"""Collect the raw per-run data behind compile-cache/index.html. Needs an authenticated `gh`.

    python3 collect.py OUT_DIR 2026-10-19 2026-11-02   # dispatched integration runs
    python3 collect.py OUT_DIR --fill                 # retry runs with no torch-spyre version
    python3 collect.py OUT_DIR --tec                  # test_each_commit runs, Oct 24-31
    python3 collect.py OUT_DIR --hashes               # pr_url_hash -> PR URL
    python3 collect.py OUT_DIR --compares             # torch-spyre compare between a PR's consecutive commits
    python3 collect.py OUT_DIR --consolidate DATA_DIR # write the files analyze.py reads (../data)

Resumable: one JSON per run under OUT_DIR.
"""

import json, os, re, subprocess, sys, datetime as dt, io, zipfile
R = "torch-spyre/spyre-inference"
OUT = sys.argv[1]
START, END = (sys.argv[2], sys.argv[3]) if len(sys.argv) > 3 else ("", "")
os.makedirs(f"{OUT}/runs", exist_ok=True)

def api(path, raw=False):
    p = subprocess.run(["gh", "api", path] + (["--allow-escape-sequences"] if raw else []),
                       capture_output=True, check=False)
    if p.returncode:
        raise RuntimeError(p.stderr.decode()[:300])
    return p.stdout if raw else json.loads(p.stdout)

def list_runs():
    path = f"{OUT}/runlist.json"
    if os.path.exists(path):
        return json.load(open(path))
    runs = {}
    d = dt.datetime.fromisoformat(START)
    end = dt.datetime.fromisoformat(END)
    while d < end:
        nxt = d + dt.timedelta(hours=6)
        q = f"{d:%Y-%m-%dT%H:%M:%SZ}..{nxt:%Y-%m-%dT%H:%M:%SZ}"
        page = 1
        while True:
            r = api(f"repos/{R}/actions/workflows/integration-tests.yaml/runs?event=workflow_dispatch&per_page=100&page={page}&created={q}")
            for w in r["workflow_runs"]:
                runs[w["id"]] = {k: w[k] for k in ("id", "created_at", "updated_at", "run_started_at", "conclusion", "status", "run_attempt", "display_title", "head_sha")} | {"actor": w["actor"]["login"]}
            if len(r["workflow_runs"]) < 100:
                break
            page += 1
        d = nxt
        print("listed", q, len(runs), flush=True)
    out = sorted(runs.values(), key=lambda x: x["created_at"])
    json.dump(out, open(path, "w"))
    return out

def ts(s):
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00")) if s else None

def collect(run):
    path = f"{OUT}/runs/{run['id']}.json"
    if os.path.exists(path):
        return
    rec = dict(run)
    jobs = []
    page = 1
    while True:
        j = api(f"repos/{R}/actions/runs/{run['id']}/jobs?per_page=100&filter=latest&page={page}")
        jobs += j["jobs"]
        if len(jobs) >= j["total_count"] or not j["jobs"]:
            break
        page += 1
    labels = {l for job in jobs for l in job["labels"] if l.startswith("image")}
    rec["image_labels"] = sorted(labels)
    hours, hours_test = 0.0, 0.0
    for job in jobs:
        if "ubuntu-latest" in job["labels"] or not job["started_at"] or not job["completed_at"]:
            continue
        h = (ts(job["completed_at"]) - ts(job["started_at"])).total_seconds() / 3600
        hours += max(h, 0)
        if "build and run tests" in job["name"]:
            hours_test += max(h, 0)
    rec["runner_hours"] = hours
    rec["test_runner_hours"] = hours_test
    rec["jobs"] = [{"name": j["name"], "conclusion": j["conclusion"], "started_at": j["started_at"], "completed_at": j["completed_at"]} for j in jobs if "ubuntu-latest" not in j["labels"]]
    prov = [j for j in jobs if j["name"].startswith("Log upstream")]
    if prov:
        try:
            log = api(f"repos/{R}/actions/jobs/{prov[0]['id']}/logs", raw=True).decode("utf-8", "replace")
            log = re.sub(r"\x1b\[[0-9;]*m", "", log)
            for field, key in (("PR URL hash", "pr_url_hash"), ("Upstream repo", "triggering_repo"), ("Upstream commit", "triggering_sha"), ("spyre-inference ref", "si_ref"), ("torch-spyre ref", "torch_spyre_ref"), ("RPM names", "rpm_names"), ("Test type", "test_type")):
                m = re.search(rf'\| {re.escape(field)}\s*\| (.*?) \|', log)
                rec[key] = m.group(1).strip() if m else None
        except RuntimeError as e:
            rec["prov_error"] = str(e)
    tests = [j for j in jobs if "build and run tests" in j["name"] and j["completed_at"] and j["conclusion"] in ("success", "failure")]
    tests.sort(key=lambda j: j["conclusion"] != "success")
    if tests:
        try:
            log = api(f"repos/{R}/actions/jobs/{tests[0]['id']}/logs", raw=True).decode("utf-8", "replace")
            m = re.search(r"torch_spyre-([0-9][^-\s]*)-", log)
            rec["torch_spyre_version"] = m.group(1) if m else None
            m = re.search(r"compile cache: restored .*|compile cache MISS.*", log)
            rec["cache_line"] = m.group(0) if m else None
        except RuntimeError as e:
            rec["ts_error"] = str(e)
    json.dump(rec, open(path, "w"))


def fill_versions():
    """Retry runs whose first test job had no version line, trying up to four jobs."""
    import glob
    for f in glob.glob(f"{OUT}/runs/*.json"):
        d = json.load(open(f))
        if d.get("torch_spyre_version"):
            continue
        jobs = api(f"repos/{R}/actions/runs/{d['id']}/jobs?per_page=100&filter=latest")["jobs"]
        cands = [j for j in jobs if j["started_at"] and "ubuntu-latest" not in j["labels"]]
        cands.sort(key=lambda j: ("build and run tests" not in j["name"], j["conclusion"] != "cancelled"))
        for j in cands[:4]:
            try:
                m = re.search(r"torch_spyre-([0-9][^-\s]*)-", api(f"repos/{R}/actions/jobs/{j['id']}/logs", raw=True).decode("utf-8", "replace"))
            except RuntimeError:
                continue
            if m:
                d["torch_spyre_version"] = m.group(1)
                break
        json.dump(d, open(f, "w"))


def collect_tec():
    """test_each_commit runs, with the first smoke shard's cache restore line."""
    os.makedirs(f"{OUT}/tec", exist_ok=True)
    for d0 in ["2026-10-24", "2026-10-26", "2026-10-28", "2026-10-30"]:
        d1 = (dt.date.fromisoformat(d0) + dt.timedelta(days=1)).isoformat()
        for page in (1, 2):
            for w in api(f"repos/{R}/actions/workflows/test_each_commit.yaml/runs?per_page=100&page={page}&created={d0}..{d1}")["workflow_runs"]:
                path = f"{OUT}/tec/{w['id']}.json"
                if os.path.exists(path):
                    continue
                jobs = []
                for p in (1, 2):
                    jobs += api(f"repos/{R}/actions/runs/{w['id']}/jobs?per_page=100&filter=latest&page={p}")["jobs"]
                rec = {"id": w["id"], "event": w["event"], "created_at": w["created_at"],
                       "jobs": [{k: j[k] for k in ("id", "name", "conclusion", "started_at", "completed_at")} for j in jobs]}
                smoke = [j for j in jobs if "Non-distributed spyre tests (shard 1/" in j["name"] and j["conclusion"] == "success"]
                if smoke:
                    log = api(f"repos/{R}/actions/jobs/{smoke[0]['id']}/logs", raw=True).decode("utf-8", "replace")
                    m = re.search(r"compile cache: restored .*|compile cache MISS.*", log)
                    rec["cache_line"] = m.group(0) if m else None
                json.dump(rec, open(path, "w"))


def match_hashes(max_pr=12000):
    """pr_url_hash is sha256 of the PR URL; brute-force every org repo's PR numbers."""
    import glob, hashlib
    wanted = {json.load(open(f)).get("pr_url_hash") for f in glob.glob(f"{OUT}/runs/*.json")}
    repos = [r["full_name"] for r in api("orgs/torch-spyre/repos?per_page=100")]
    out = {}
    for repo in repos:
        for n in range(1, max_pr):
            url = f"https://github.com/{repo}/pull/{n}"
            if hashlib.sha256(url.encode()).hexdigest() in wanted:
                out[hashlib.sha256(url.encode()).hexdigest()] = url
    json.dump(out, open(f"{OUT}/pr-url-hashes.json", "w"), indent=0)


def compares():
    """Files changed between consecutive torch-spyre commits tested by the same PR."""
    import glob
    hashes = json.load(open(f"{OUT}/pr-url-hashes.json"))
    per = {}
    for f in sorted(glob.glob(f"{OUT}/runs/*.json")):
        d = json.load(open(f))
        url = hashes.get(d.get("pr_url_hash") or "")
        v = d.get("torch_spyre_version") or ""
        if url and "/torch-spyre/torch-spyre/" in url and ".g" in v:
            per.setdefault(url, []).append((d["created_at"], v.rsplit(".g", 1)[1]))
    out = {}
    for v in per.values():
        shas = [s for _, s in sorted(v)]
        for a, b in zip(shas, shas[1:]):
            if a != b and f"{a}...{b}" not in out:
                r = api(f"repos/torch-spyre/torch-spyre/compare/{a}...{b}")
                out[f"{a}...{b}"] = {"status": r.get("status"), "ahead_by": r.get("ahead_by"),
                                    "behind_by": r.get("behind_by"), "files": [x["filename"] for x in r.get("files", [])]}
    json.dump(out, open(f"{OUT}/torch-spyre-compares.json", "w"))


def consolidate(data_dir):
    import glob, gzip

    def slim(j):
        return {k: j.get(k) for k in ("name", "conclusion", "started_at", "completed_at")}

    keep = ("id", "created_at", "conclusion", "run_attempt", "actor", "image_labels", "pr_url_hash", "si_ref",
            "torch_spyre_ref", "rpm_names", "test_type", "torch_spyre_version", "cache_line")
    runs = sorted((json.load(open(f)) for f in glob.glob(f"{OUT}/runs/*.json")), key=lambda r: r["created_at"])
    with gzip.open(f"{data_dir}/dispatch-runs.json.gz", "wt") as fh:
        json.dump({"runs": [{k: r.get(k) for k in keep} | {"jobs": [slim(j) for j in r["jobs"]]} for r in runs]}, fh)
    tec = [json.load(open(f)) for f in sorted(glob.glob(f"{OUT}/tec/*.json"))]
    with gzip.open(f"{data_dir}/test-each-commit-runs.json.gz", "wt") as fh:
        json.dump({"runs": [{k: r.get(k) for k in ("id", "event", "created_at", "cache_line")}
                            | {"jobs": [slim(j) for j in r["jobs"] if "build and run tests" in j["name"]]} for r in tec]}, fh)
    for name in ("pr-url-hashes.json", "torch-spyre-compares.json"):
        os.replace(f"{OUT}/{name}", f"{data_dir}/{name}")


if __name__ == "__main__":
    mode = sys.argv[2] if len(sys.argv) > 2 else ""
    if mode == "--fill":
        fill_versions()
    elif mode == "--tec":
        collect_tec()
    elif mode == "--hashes":
        match_hashes()
    elif mode == "--compares":
        compares()
    elif mode == "--consolidate":
        consolidate(sys.argv[3])
    else:
        runs = [r for r in list_runs() if r["actor"] == "spyre-ci[bot]"]
        print(f"{len(runs)} runs", flush=True)
        from concurrent.futures import ThreadPoolExecutor

        def one(run):
            try:
                collect(run)
            except Exception as e:
                print(f"run {run['id']}: {e}", flush=True)

        with ThreadPoolExecutor(8) as ex:
            for i, _ in enumerate(ex.map(one, runs)):
                if i % 50 == 0:
                    print(f"{i}/{len(runs)}", flush=True)
        print("done", flush=True)
