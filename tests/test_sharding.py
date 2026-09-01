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

import pytest
from spyre_testing_plugin.pytest_plugin import _apply_shard


class _FakeItem:
    def __init__(self, nodeid: str):
        self.nodeid = nodeid


class _FakeConfig:
    """Stands in for pytest.Config: _apply_shard only calls hook.pytest_deselected."""

    class _Hook:
        def pytest_deselected(self, items):
            pass

    hook = _Hook()


def _run_shard(master, *, num_shards, shard_id, select, weight):
    items = list(master)
    _apply_shard(
        _FakeConfig(),
        items,
        num_shards=num_shards,
        shard_id=shard_id,
        select=select,
        weight=weight,
        label="test",
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
