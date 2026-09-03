# Copyright 2026 The Spyre-Inference Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Pin the `test-durations` artifact (uploaded by the `durations` job of an earlier
run) that this run's shard jobs weight their partition by. One id is resolved
here so every shard of a suite reads the SAME durations -- shards computing
different weights would drop or double-run tests.

Prefers the newest artifact on this PR's head branch (so a PR with heavy test
changes self-balances against its own last run) and falls back to the base
branch. Writes `durations_run_id=<id>` (empty when none found) to GITHUB_OUTPUT;
an empty pin makes the plugin fall back to its static heuristic weights.

Usage (called by _test_matrix.yaml's resolve_durations job):
    GH_TOKEN=... REPO=... HEAD_REF=... BASE_REF=... python3 resolve_durations_run.py
"""

import json
import os
import subprocess


def list_durations_artifacts(repo: str) -> list[dict]:
    result = subprocess.run(
        [
            "gh",
            "api",
            "--paginate",
            f"/repos/{repo}/actions/artifacts?name=test-durations&per_page=100",
            "--jq",
            ".artifacts[] | select(.expired == false) "
            "| {created_at: .created_at, branch: .workflow_run.head_branch, "
            "conclusion: .workflow_run.conclusion, run_id: .workflow_run.id}",
        ],
        capture_output=True,
        check=True,
        text=True,
    )
    return [json.loads(line) for line in result.stdout.splitlines() if line.strip()]


def pick_run_id(artifacts: list[dict], head_ref: str, base_ref: str) -> str:
    # Head branch first so a PR self-balances against its own last run; else base.
    # Within a branch prefer a passing run: the `durations` job is `if: always()`,
    # so a failed run uploads a partial, skewed durations file. Fall back to the
    # newest run when none has passed yet.
    for branch in (head_ref, base_ref):
        if not branch:
            continue
        matches = [a for a in artifacts if a["branch"] == branch]
        if not matches:
            continue
        passing = [a for a in matches if a.get("conclusion") == "success"]
        pool = passing or matches
        newest = max(pool, key=lambda a: a["created_at"])
        note = "passing" if passing else "most recent (no passing run yet)"
        print(f"Pinning durations to {note} run {newest['run_id']} (branch {branch}).")
        return str(newest["run_id"])
    print("No test-durations artifact found; shards will use heuristic weights.")
    return ""


def main() -> None:
    repo = os.environ["REPO"]
    head_ref = os.environ.get("HEAD_REF", "")
    base_ref = os.environ.get("BASE_REF", "")

    try:
        artifacts = list_durations_artifacts(repo)
    except subprocess.CalledProcessError as e:
        # Never block the run on a resolve failure: fall back to heuristic weights.
        print(f"Failed to list artifacts ({e}); shards will use heuristic weights.")
        artifacts = []

    run_id = pick_run_id(artifacts, head_ref, base_ref)
    with open(os.environ["GITHUB_OUTPUT"], "a") as f:
        f.write(f"durations_run_id={run_id}\n")


if __name__ == "__main__":
    main()
