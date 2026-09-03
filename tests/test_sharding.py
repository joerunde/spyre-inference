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

"""CPU-only meta-tests for the test-sharding partition (no hardware needed).

The shard jobs each keep only their slice of the suite, so the one invariant
that must never break is: running every shard id reproduces the full selection
exactly once (no test silently dropped, none run twice). This exercises
`_apply_shard` directly with fake items instead of real collection.
"""

import re
from pathlib import Path

import pytest
from spyre_testing_plugin.sharding import _apply_shard

_REPO_ROOT = Path(__file__).resolve().parents[1]


class _FakeItem:
    # Real Mark objects (not sentinels) so the plugin's condition-aware skip
    # check can run them through pytest's own evaluate_skip_marks.
    def __init__(self, nodeid: str, skip: bool = False, marks=()):
        self.nodeid = nodeid
        decorators = list(marks)
        if skip:
            decorators.append(pytest.mark.skip)
        self._marks = [d.mark for d in decorators]

    def iter_markers(self, name: str | None = None):
        return [m for m in self._marks if name is None or m.name == name]

    def get_closest_marker(self, name: str):
        for m in self._marks:
            if m.name == name:
                return m
        return None


class _FakeConfig:
    """Stands in for pytest.Config: _apply_shard only calls hook.pytest_deselected."""

    class _Hook:
        def pytest_deselected(self, items):
            pass

    hook = _Hook()


def _run_shard(master, *, num_shards, shard_id, select, weight, durations=None):
    items = list(master)
    _apply_shard(
        _FakeConfig(),
        items,
        num_shards=num_shards,
        shard_id=shard_id,
        select=select,
        weight=weight,
        label="test",
        durations=durations,
    )
    return {it.nodeid for it in items}


def _sample_items():
    # A skewed mix: heavy e2e tests plus many cheap per-op tests, like smoke.
    e2e = [_FakeItem(f"tests/e2e/test_models.py::test_x[{i}]") for i in range(6)]
    ops = [_FakeItem(f"tests/custom_ops/test_linear.py::test_y[{i}]") for i in range(40)]
    return e2e + ops


@pytest.mark.parametrize("num_shards", [2, 3, 4, 7])
def test_shards_partition_selection_exactly_once(num_shards):
    """Union of all shards == full selection, and shards are pairwise disjoint."""
    master = _sample_items()
    weight = lambda it: 8 if "e2e" in it.nodeid else 1  # noqa: E731
    slices = [
        _run_shard(master, num_shards=num_shards, shard_id=i, select=lambda it: True, weight=weight)
        for i in range(num_shards)
    ]

    all_ids = {it.nodeid for it in master}
    union = set().union(*slices)
    assert union == all_ids, "a test was dropped from every shard"

    for a in range(num_shards):
        for b in range(a + 1, num_shards):
            assert not (slices[a] & slices[b]), "a test ran in more than one shard"


def test_unselected_items_are_kept_in_every_shard():
    """Items `select` rejects (e.g. non-attention tests in an attention shard) stay everywhere."""
    attn = [_FakeItem(f"tests/attention/test_spyre_attn.py::test_a[{i}]") for i in range(10)]
    other = [_FakeItem(f"tests/custom_ops/test_linear.py::test_b[{i}]") for i in range(5)]
    master = attn + other
    other_ids = {it.nodeid for it in other}

    for shard_id in range(3):
        kept = _run_shard(
            master,
            num_shards=3,
            shard_id=shard_id,
            select=lambda it: "attention" in it.nodeid,
            weight=lambda it: 1,
        )
        assert other_ids <= kept, "an unselected item was dropped from a shard"


def test_weighting_balances_heavy_items_across_shards():
    """The heavy e2e items should not all pile into one shard."""
    master = _sample_items()
    weight = lambda it: 8 if "e2e" in it.nodeid else 1  # noqa: E731
    counts = [
        len(
            {
                nid
                for nid in _run_shard(
                    master, num_shards=3, shard_id=i, select=lambda it: True, weight=weight
                )
                if "e2e" in nid
            }
        )
        for i in range(3)
    ]
    # 6 e2e items across 3 shards: a balanced partition gives 2 each, never 4+ in one.
    assert max(counts) <= 3, f"heavy items co-located: {counts}"


def test_skip_marked_items_do_not_consume_shard_weight():
    """Skipped tests cost no runtime, so they must not pull real work off-balance.

    Upstream shards are dominated by tests that skip at setup (unsupported arch);
    weighting them would scatter the few that actually run.
    """
    real = [_FakeItem(f"tests/e2e/test_x.py::real[{i}]") for i in range(3)]
    skipped = [_FakeItem(f"tests/e2e/test_x.py::skip[{i}]", skip=True) for i in range(30)]
    master = real + skipped
    weight = lambda it: 8 if "e2e" in it.nodeid else 1  # noqa: E731
    real_ids = {it.nodeid for it in real}

    counts = [
        len(
            real_ids
            & _run_shard(master, num_shards=3, shard_id=i, select=lambda it: True, weight=weight)
        )
        for i in range(3)
    ]
    assert counts == [1, 1, 1], f"real items not balanced once skips are weightless: {counts}"


def test_single_shard_and_zero_are_noops():
    master = _sample_items()
    all_ids = {it.nodeid for it in master}
    for num_shards in (0, 1):
        assert (
            _run_shard(
                master,
                num_shards=num_shards,
                shard_id=0,
                select=lambda it: True,
                weight=lambda it: 1,
            )
            == all_ids
        )


def test_out_of_range_shard_id_raises():
    master = _sample_items()
    with pytest.raises(pytest.UsageError):
        _run_shard(master, num_shards=4, shard_id=4, select=lambda it: True, weight=lambda it: 1)


def test_durations_partition_still_covers_selection_exactly_once():
    """Union property must hold under duration weights, including unmeasured tests."""
    master = _sample_items()
    # Only some tests measured; the rest fall back to the heuristic class mean.
    durations = {it.nodeid: 300.0 for it in master if "e2e" in it.nodeid and "0" in it.nodeid}
    weight = lambda it: 8 if "e2e" in it.nodeid else 1  # noqa: E731

    slices = [
        _run_shard(
            master,
            num_shards=4,
            shard_id=i,
            select=lambda it: True,
            weight=weight,
            durations=durations,
        )
        for i in range(4)
    ]
    all_ids = {it.nodeid for it in master}
    assert set().union(*slices) == all_ids
    for a in range(4):
        for b in range(a + 1, 4):
            assert not (slices[a] & slices[b])


def test_durations_split_a_heavy_parametrization_off_its_cheap_siblings():
    """Two same-file tests with very different measured times land on different shards.

    This is the case no static heuristic catches: identical path/marker, 10x runtime.
    """
    items = [_FakeItem(f"tests/e2e/test_models.py::test_x[{i}]") for i in range(2)]
    durations = {items[0].nodeid: 400.0, items[1].nodeid: 40.0}
    filler = [_FakeItem(f"tests/e2e/test_models.py::test_x[fill{i}]") for i in range(4)]
    for f in filler:
        durations[f.nodeid] = 40.0
    master = items + filler

    heavy_shard = next(
        i
        for i in range(2)
        if items[0].nodeid
        in _run_shard(
            master,
            num_shards=2,
            shard_id=i,
            select=lambda it: True,
            weight=lambda it: 1,
            durations=durations,
        )
    )
    light_shard = next(
        i
        for i in range(2)
        if items[1].nodeid
        in _run_shard(
            master,
            num_shards=2,
            shard_id=i,
            select=lambda it: True,
            weight=lambda it: 1,
            durations=durations,
        )
    )
    assert heavy_shard != light_shard, "the 400s test should not share a shard with the 40s one"


def test_empty_durations_reproduces_heuristic_partition():
    """durations={} (no file) must give the exact same slices as the pre-durations code."""
    master = _sample_items()
    weight = lambda it: 8 if "e2e" in it.nodeid else 1  # noqa: E731
    for i in range(3):
        heuristic = _run_shard(
            master, num_shards=3, shard_id=i, select=lambda it: True, weight=weight
        )
        empty = _run_shard(
            master, num_shards=3, shard_id=i, select=lambda it: True, weight=weight, durations={}
        )
        assert heuristic == empty


def test_skip_marked_items_weightless_under_durations():
    """A skip marker beats a recorded duration: skipped tests still cost nothing."""
    real = [_FakeItem(f"tests/e2e/test_x.py::real[{i}]") for i in range(3)]
    skipped = [_FakeItem(f"tests/e2e/test_x.py::skip[{i}]", skip=True) for i in range(30)]
    master = real + skipped
    # Give the skipped tests huge recorded times; the skip marker must zero them.
    durations = {it.nodeid: 999.0 for it in skipped}
    durations.update({it.nodeid: 100.0 for it in real})
    real_ids = {it.nodeid for it in real}

    counts = [
        len(
            real_ids
            & _run_shard(
                master,
                num_shards=3,
                shard_id=i,
                select=lambda it: True,
                weight=lambda it: 1,
                durations=durations,
            )
        )
        for i in range(3)
    ]
    assert counts == [1, 1, 1], f"real items not balanced once skips are weightless: {counts}"


def test_triggered_skipif_is_weightless_but_false_skipif_keeps_weight():
    """A triggered skipif is zeroed like a skip; a skipif that will run keeps weight."""
    real = [_FakeItem(f"tests/e2e/test_x.py::real[{i}]") for i in range(3)]
    triggered = [
        _FakeItem(
            f"tests/e2e/test_x.py::skipped[{i}]",
            marks=[pytest.mark.skipif(True, reason="unsupported arch")],
        )
        for i in range(30)
    ]
    master = real + triggered
    weight = lambda it: 8 if "e2e" in it.nodeid else 1  # noqa: E731
    real_ids = {it.nodeid for it in real}
    counts = [
        len(
            real_ids
            & _run_shard(master, num_shards=3, shard_id=i, select=lambda it: True, weight=weight)
        )
        for i in range(3)
    ]
    assert counts == [1, 1, 1], f"triggered skipif not weightless: {counts}"

    # A false skipif runs, so it must keep weight: the union is still exact and
    # the two same-weight tests are balanced across the two shards.
    runs = [
        _FakeItem(
            f"tests/e2e/test_y.py::runs[{i}]",
            marks=[pytest.mark.skipif(False, reason="supported")],
        )
        for i in range(2)
    ]
    slices = [
        _run_shard(runs, num_shards=2, shard_id=i, select=lambda it: True, weight=lambda it: 1)
        for i in range(2)
    ]
    assert set().union(*slices) == {it.nodeid for it in runs}
    assert all(len(s) == 1 for s in slices), (
        "a runnable skipif=False test was weighted 0 and co-located"
    )


def test_unmeasured_heavy_class_scaled_onto_the_seconds_axis():
    """An unmeasured heavy-class test is scaled up, not left at raw float(cls)."""
    # Only cheap class-1 ops are measured (~40s); the class-8 e2e test is unseen.
    ops = [_FakeItem(f"tests/custom_ops/test_linear.py::op[{i}]") for i in range(6)]
    heavy = _FakeItem("tests/e2e/test_models.py::heavy[0]")
    master = ops + [heavy]
    durations = {it.nodeid: 40.0 for it in ops}
    weight = lambda it: 8 if "e2e" in it.nodeid else 1  # noqa: E731

    # secs_per_unit=40 makes the class-8 test weigh ~320s, so it packs alone;
    # a raw-8.0s fallback would have piled it onto a shard of measured 40s ops.
    heavy_shard = next(
        i
        for i in range(3)
        if heavy.nodeid
        in _run_shard(
            master,
            num_shards=3,
            shard_id=i,
            select=lambda it: True,
            weight=weight,
            durations=durations,
        )
    )
    others_on_heavy_shard = {
        nid
        for nid in _run_shard(
            master,
            num_shards=3,
            shard_id=heavy_shard,
            select=lambda it: True,
            weight=weight,
            durations=durations,
        )
        if nid != heavy.nodeid
    }
    assert not others_on_heavy_shard, (
        f"unmeasured heavy test not treated as heavy: {others_on_heavy_shard}"
    )


def _makefile_shard_counts() -> dict[str, int]:
    text = (_REPO_ROOT / "Makefile").read_text()
    counts = {}
    for suite, var in (
        ("smoke", "SMOKE_SHARDS"),
        ("attention", "ATTN_SHARDS"),
        ("upstream", "UPSTREAM_SHARDS"),
        ("distributed", "DIST_SHARDS"),
    ):
        m = re.search(rf"^{var}\s*\?=\s*(\d+)", text, re.MULTILINE)
        assert m, f"{var} not found in Makefile"
        counts[suite] = int(m.group(1))
    return counts


def _matrix_shard_ids() -> dict[str, list[int]]:
    text = (_REPO_ROOT / ".github/workflows/_test_matrix.yaml").read_text()
    return {
        suite: sorted({int(n) for n in re.findall(rf"test-{suite}-shard-(\d+)\b", text)})
        for suite in ("smoke", "attention", "upstream", "distributed")
    }


def test_matrix_shard_entries_match_makefile_counts():
    """Every declared shard 0..N-1 has exactly one matrix job.

    The Makefile ``*_SHARDS`` count and the matrix entries are two copies of the
    same number; if they drift, tests assigned to a shard id no job runs vanish
    from CI green, since _apply_shard checks each test landed in *a* valid shard,
    not that every shard is executed.
    """
    counts = _makefile_shard_counts()
    ids = _matrix_shard_ids()
    for suite, n in counts.items():
        assert ids[suite] == list(range(n)), (
            f"{suite}: matrix declares shard ids {ids[suite]} but the Makefile declares "
            f"{n} shards; every shard 0..{n - 1} needs exactly one matrix job or its "
            "tests never run in CI."
        )
