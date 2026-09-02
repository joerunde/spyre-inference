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

"""Duration-weighted test sharding for CI fan-out.

Each suite (attn/smoke/upstream/dist) is split across N parallel jobs; a job
keeps only its shard's slice. Every shard job computes the same weighted greedy
longest-processing-time partition, so no cross-job coordination is needed and
the union of all shards is the full selection exactly once (guarded by
tests/test_sharding.py). Shards are weighted by recorded per-test runtime when a
durations file is pinned (see _load_durations), else by a static path heuristic.

pytest_plugin.py owns the hooks and delegates here: `add_shard_options` from
pytest_addoption, `apply_shards` from pytest_collection_modifyitems (called last,
after upstream markers are set), and `record_duration` / `write_durations` from
the per-test report hooks.
"""

from __future__ import annotations

import json
import os

import pytest

_DURATIONS_CACHE: dict[str, float] | None = None
_DURATIONS_LOADED = False


def add_shard_options(parser) -> None:
    """Register one --<suite>-shards / --<suite>-shard-id pair per CI suite."""
    group = parser.getgroup("spyre-test-sharding")
    for suite in ("attn", "smoke", "upstream", "dist"):
        group.addoption(
            f"--{suite}-shards",
            type=int,
            default=0,
            help=f"Partition the {suite} test selection into this many shards (0 = off).",
        )
        group.addoption(
            f"--{suite}-shard-id",
            type=int,
            default=0,
            help=f"0-based index of the shard to run when --{suite}-shards is set.",
        )


def apply_shards(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Run every suite applier. Each is a no-op unless its --<suite>-shards is set."""
    _apply_attention_shard(config, items)
    _apply_smoke_shard(config, items)
    _apply_upstream_shard(config, items)
    _apply_distributed_shard(config, items)


def _load_durations(config: pytest.Config) -> dict[str, float]:
    """Recorded per-test runtimes {nodeid: seconds}, or {} if unavailable.

    CI downloads a ``test-durations.json`` pinned to one prior run and points
    ``SPYRE_TEST_DURATIONS`` at it (see .github/actions/run-matrix-config); every
    shard job of a suite reads the same file, so the greedy partition they each
    compute is identical. Missing/malformed file -> {} -> the appliers fall back
    to their static path heuristic. Cached: pytest_collection_modifyitems runs the
    appliers back-to-back within one process.
    """
    global _DURATIONS_CACHE, _DURATIONS_LOADED
    if _DURATIONS_LOADED:
        return _DURATIONS_CACHE or {}
    _DURATIONS_LOADED = True

    path = os.environ.get("SPYRE_TEST_DURATIONS") or os.path.join(
        str(config.rootpath), ".test_durations"
    )
    try:
        with open(path) as f:
            raw = json.load(f)
        _DURATIONS_CACHE = {str(k): float(v) for k, v in raw.items() if float(v) >= 0}
    except (OSError, ValueError, TypeError):
        _DURATIONS_CACHE = {}
    return _DURATIONS_CACHE


def _apply_shard(
    config: pytest.Config,
    items: list[pytest.Item],
    *,
    num_shards: int,
    shard_id: int,
    select,
    weight,
    label: str,
    durations: dict[str, float] | None = None,
) -> None:
    """Keep only this shard's slice of the items ``select`` matches.

    Every shard job computes the same weighted greedy longest-processing-time
    partition and keeps its own slice, so shards need no cross-job coordination;
    items ``select`` rejects stay in every shard. Raises rather than silently
    drop if the partition ever fails to cover the selection exactly once (union
    property: tests/test_sharding.py).

    Weight (heavier = slower) is a test's recorded runtime from ``durations``
    (seconds, keyed by nodeid; see _load_durations). Static path heuristics can't
    tell a 400s parametrization from a 40s one in the same file, so measured time
    is the only proxy that balances. A test not in ``durations`` (brand new, or no
    durations file at all) falls back to ``weight(item)``, rescaled to the seconds
    scale via the mean of measured tests sharing its heuristic class so an unseen
    heavy test is packed as heavy, not as ~8s. When ``durations`` is empty every
    test uses ``weight(item)`` directly, reproducing the pre-durations partition.

    A ``skip``-marked item costs no runtime, so it is weighted 0 and packed for
    free — otherwise the many upstream tests that skip at setup (unsupported
    arch) would carry full weight and scatter the few that actually run.
    """
    if not num_shards or num_shards <= 1:
        return
    if not 0 <= shard_id < num_shards:
        raise pytest.UsageError(f"--{label}-shard-id must be in [0, {num_shards}); got {shard_id}")

    durations = durations or {}

    selected = [it for it in items if select(it)]
    if not selected:
        return

    # Mean measured runtime per heuristic class, so an unseen test inherits the
    # observed cost of tests the heuristic groups it with instead of a raw 1/8.
    class_totals: dict[int, list[float]] = {}
    for it in selected:
        secs = durations.get(it.nodeid)
        if secs is not None:
            class_totals.setdefault(weight(it), []).append(secs)
    class_mean = {cls: sum(v) / len(v) for cls, v in class_totals.items()}

    def effective_weight(item: pytest.Item) -> float:
        if item.get_closest_marker("skip") is not None:
            return 0.0
        secs = durations.get(item.nodeid)
        if secs is not None:
            return secs
        cls = weight(item)
        # No measured tests at all -> use the heuristic directly (old behavior).
        return class_mean.get(cls, float(cls))

    # Greedy longest-processing-time first over a stable order.
    selected.sort(key=lambda it: it.nodeid)
    selected.sort(key=effective_weight, reverse=True)
    loads = [0.0] * num_shards
    assigned: dict[str, int] = {}
    for item in selected:
        target = min(range(num_shards), key=lambda s: loads[s])
        assigned[item.nodeid] = target
        loads[target] += effective_weight(item)

    selected_nodeids = {it.nodeid for it in selected}
    if set(assigned) != selected_nodeids or not all(0 <= s < num_shards for s in assigned.values()):
        missed = selected_nodeids - set(assigned)
        raise pytest.UsageError(
            f"{label} sharding is unsound: {len(missed)} of {len(selected_nodeids)} selected "
            f"tests were not assigned to a valid shard (e.g. {sorted(missed)[:3]}). Refusing to "
            "run a partial suite."
        )

    kept, dropped = [], []
    for item in items:
        if item.nodeid in selected_nodeids and assigned[item.nodeid] != shard_id:
            dropped.append(item)
        else:
            kept.append(item)

    if dropped:
        config.hook.pytest_deselected(items=dropped)
        items[:] = kept


def _apply_attention_shard(config: pytest.Config, items: list[pytest.Item]) -> None:
    def select(item: pytest.Item) -> bool:
        return bool(item.get_closest_marker("attention")) and not item.get_closest_marker(
            "encoder_attention"
        )

    # Heavy = compiled kernel on device; those dominate wall time and HBM growth.
    def weight(item: pytest.Item) -> int:
        nid = item.nodeid
        return 8 if "device_spyre" in nid and "STOCK" in nid else 1

    _apply_shard(
        config,
        items,
        num_shards=config.getoption("--attn-shards"),
        shard_id=config.getoption("--attn-shard-id"),
        select=select,
        weight=weight,
        label="attn",
        durations=_load_durations(config),
    )


def _apply_smoke_shard(config: pytest.Config, items: list[pytest.Item]) -> None:
    # Smoke is dominated by the e2e model tests (incl. the compiled test_compile.py cases).
    def weight(item: pytest.Item) -> int:
        return 8 if "e2e" in item.nodeid else 1

    _apply_shard(
        config,
        items,
        num_shards=config.getoption("--smoke-shards"),
        shard_id=config.getoption("--smoke-shard-id"),
        select=lambda item: True,
        weight=weight,
        label="smoke",
        durations=_load_durations(config),
    )


def _apply_upstream_shard(config: pytest.Config, items: list[pytest.Item]) -> None:
    # models/ tests compile on device (heavy); match by path, not the `model` marker,
    # which is applied dynamically later in collection.
    def weight(item: pytest.Item) -> int:
        return 8 if "models/" in item.nodeid else 1

    _apply_shard(
        config,
        items,
        num_shards=config.getoption("--upstream-shards"),
        shard_id=config.getoption("--upstream-shard-id"),
        select=lambda item: True,
        weight=weight,
        label="upstream",
        durations=_load_durations(config),
    )


def _apply_distributed_shard(config: pytest.Config, items: list[pytest.Item]) -> None:
    # The multi-card (TP=2) suite: distributed tests, minus the probes, which the
    # matrix routes to their own 2-card job (Makefile: `distributed and not (upstream
    # or probe)`). Each spawns a TP=2 subprocess pair, so they're roughly uniform;
    # durations still separate the model-run cases from the light collective checks.
    def select(item: pytest.Item) -> bool:
        return (
            bool(item.get_closest_marker("distributed"))
            and not item.get_closest_marker("upstream")
            and not item.get_closest_marker("probe")
        )

    _apply_shard(
        config,
        items,
        num_shards=config.getoption("--dist-shards"),
        shard_id=config.getoption("--dist-shard-id"),
        select=select,
        weight=lambda item: 1,
        label="dist",
        durations=_load_durations(config),
    )


# Per-nodeid wall time this session, written out when SPYRE_TEST_DURATIONS_OUT is
# set (CI). A later run pins the merged file and feeds it back as
# SPYRE_TEST_DURATIONS so _apply_shard balances shards by measured runtime. Keyed
# by the exact item.nodeid _apply_shard sees -- the reason this records durations
# directly instead of parsing JUnit, whose testcases carry only a dotted classname
# that can't be inverted to a nodeid unambiguously.
_test_durations: dict[str, float] = {}
_ran_nodeids: set[str] = set()


def record_duration(report) -> None:
    if not os.environ.get("SPYRE_TEST_DURATIONS_OUT"):
        return
    # Sum setup+call+teardown so the recorded time matches the wall clock the
    # partition is balancing; mark actually-run tests so skips (setup-only,
    # ~0s) are dropped and fall to the class-mean estimate in the next run.
    _test_durations[report.nodeid] = _test_durations.get(report.nodeid, 0.0) + report.duration
    if report.when == "call":
        _ran_nodeids.add(report.nodeid)


def write_durations(log) -> None:
    out = os.environ.get("SPYRE_TEST_DURATIONS_OUT")
    if not out:
        return
    data = {nid: t for nid, t in _test_durations.items() if nid in _ran_nodeids}
    if not data:
        return
    try:
        with open(out, "w") as f:
            json.dump(data, f)
    except OSError as e:
        log(f"could not write test durations to {out}: {e}")
