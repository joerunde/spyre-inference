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

"""SpyreEncoderOnlyModelState: prepare_attn picks the plan, prepare_inputs lays
the grid out for it.

The MRV2 port of tests/runtime/test_encoder_dense_grid.py's and
test_encoder_inline_dispatch.py's runner-side checks, through the
prepare_attn / prepare_inputs signatures on real InputBatches. The grid helpers
and the attention kernels themselves stay covered by those files.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from vllm.config.compilation import CUDAGraphMode

from spyre_inference.v1.attention.backends.spyre_attn import SpyreAttentionMetadataBuilder
from spyre_inference.v1.attention.backends.spyre_encoder_attn import (
    EncoderRectPlan,
    encoder_grid,
)
from spyre_inference.v1.worker.spyre.model_states.encoder_only import (
    SpyreEncoderOnlyModelState,
)
from spyre_inference.v1.worker.spyre_shape_bucketer import encoder_dense_row_indices

from .state_fakes import make_config, make_input_batch

# make_config()'s shapes: rectangles (64, 8) and (128, 4) over a 512-row body.
_BUDGET = 512
_ROBERTA = SimpleNamespace(
    architectures=["XLMRobertaModel"], position_embedding_type="absolute", pad_token_id=1
)


@pytest.fixture(autouse=True)
def _clean_grid():
    encoder_grid().publish(None)
    yield
    encoder_grid().publish(None)


class _EncoderBuilder:
    """Runs the real builder's encoder-only branch, which reads nothing else off it."""

    def build(self, common_prefix_len, common_attn_metadata, **kwargs):
        fake_self = SimpleNamespace(_encoder_only=True, block_size=64, num_kv_heads=1, num_heads=4)
        return SpyreAttentionMetadataBuilder.build(
            fake_self,  # ty: ignore[invalid-argument-type]
            common_prefix_len,
            common_attn_metadata,
        )


def _state(**config_kwargs) -> SpyreEncoderOnlyModelState:
    state = SpyreEncoderOnlyModelState(
        make_config(**config_kwargs),  # ty: ignore[invalid-argument-type]
        nn.Module(),
        None,
        torch.device("cpu"),
        torch.device("cpu"),
    )
    builder = _EncoderBuilder()
    group = SimpleNamespace(layer_names=["enc.0", "enc.1"], get_metadata_builder=lambda _: builder)
    state.encoder_attn_groups = [group]  # ty: ignore[invalid-assignment]
    return state


def _step(state, batch):
    """One MRV2 step's model-state calls, in execute_model's order."""
    attn_metadata = state.prepare_attn(
        batch,
        CUDAGraphMode.NONE,
        block_tables=(),
        slot_mappings=torch.empty(0, batch.num_tokens_after_padding, dtype=torch.int64),
        attn_groups=[],
        kv_cache_config=SimpleNamespace(kv_cache_groups=[]),
    )
    return attn_metadata, state.prepare_inputs(batch, req_states=None)


@pytest.mark.parametrize("query_lens", [[64, 64], [1, 64, 32], [10], [7, 1, 63, 64]])
def test_rectangle_lays_every_real_token_on_its_grid_row(query_lens):
    state = _state()
    batch = make_input_batch(query_lens, num_tokens_after_padding=_BUDGET)
    attn_metadata, inputs = _step(state, batch)

    plan = state.encoder_plan
    assert plan is not None and plan.is_rectangular
    assert plan.grid == (64, 8, query_lens)
    assert inputs["input_ids"].shape[0] == _BUDGET
    rows = encoder_dense_row_indices(query_lens, 64)
    assert torch.equal(inputs["input_ids"][rows], batch.input_ids[: sum(query_lens)])
    assert torch.equal(inputs["positions"][rows], batch.positions[: sum(query_lens)])
    # Every layer runs the same plan, and the traced forward reads its mask.
    assert {id(md.encoder_plan) for md in attn_metadata.values()} == {id(plan.attn_plan)}
    assert encoder_grid().mask is plan.attn_plan.mask


def test_real_pad_continues_positions_and_batch_pad_uses_the_pad_token():
    state = _state(hf_config=SimpleNamespace(pad_token_id=9))
    _, inputs = _step(state, make_input_batch([4, 2], num_tokens_after_padding=_BUDGET))
    ids, positions = inputs["input_ids"], inputs["positions"]
    assert ids[:4].tolist() == [1, 2, 3, 4]
    assert positions[:8].tolist() == list(range(8))
    assert ids[4:64].unique().tolist() == [9]
    assert ids[128:].unique().tolist() == [9]
    assert int(positions.max()) == 63


def test_rectangle_rows_follow_the_plan_not_the_body_order():
    """A longer request picks the 128-wide rectangle; the grid follows it."""
    state = _state()
    _, inputs = _step(state, make_input_batch([100, 5], num_tokens_after_padding=_BUDGET))
    assert state.encoder_plan.grid == (128, 4, [100, 5])
    assert inputs["input_ids"][128:133].tolist() == list(range(101, 106))


def test_token_types_follow_the_same_rows_as_the_ids():
    """Left packed, every sequence past the first pairs with another's tokens, and
    silently: the buffer still matches input_ids in shape."""
    query_lens = [4, 3, 2]
    state = _state()
    for i, n in enumerate(query_lens):
        state.token_type_ids[f"req_{i}"] = torch.full((n,), i % 2, dtype=torch.int32)
    _, inputs = _step(state, make_input_batch(query_lens, num_tokens_after_padding=_BUDGET))

    types = inputs["token_type_ids"]
    assert types.shape == inputs["input_ids"].shape
    assert types[:64].tolist() == [0] * 64
    assert types[64:67].tolist() == [1, 1, 1]
    assert types[67:128].sum() == 0, "sequence 1's pad rows must not carry its segment"
    assert types[128:130].tolist() == [0, 0]


def test_roberta_offset_lands_on_the_grid():
    state = _state(hf_config=_ROBERTA)
    _, inputs = _step(state, make_input_batch([3], num_tokens_after_padding=_BUDGET))
    assert inputs["positions"][:5].tolist() == [2, 3, 4, 5, 6]
    assert inputs["positions"][64:67].tolist() == [2, 3, 4]


def test_roberta_offset_on_the_packed_path():
    state = _state(hf_config=_ROBERTA)
    batch = make_input_batch([3, 2])  # not padded to the budget: ragged
    _, inputs = _step(state, batch)
    assert not state.encoder_plan.is_rectangular
    assert inputs["positions"].tolist() == [2, 3, 4, 2, 3]
    assert "input_ids" not in inputs


@pytest.mark.parametrize(
    ("query_lens", "padded", "force_ragged"),
    [
        ([3, 2], 5, False),  # no bucketer: the body is not the budget
        ([100] * 5, _BUDGET, False),  # too wide for the 128-long rectangle
        ([3, 2], _BUDGET, True),  # warmup forcing the ragged path
    ],
    ids=["unpadded", "too-wide", "forced"],
)
def test_ragged_steps_keep_the_body_packed(query_lens, padded, force_ragged):
    state = _state()
    state.force_ragged = force_ragged
    batch = make_input_batch(query_lens, num_tokens_after_padding=padded)
    attn_metadata, inputs = _step(state, batch)

    plan = state.encoder_plan
    assert plan is not None and not plan.is_rectangular and plan.grid is None
    assert inputs == {}
    assert encoder_grid().mask is None, "the ragged path must never be traced in"
    assert all(isinstance(md.encoder_plan, list) for md in attn_metadata.values())


def test_multimodal_stays_packed_until_its_embeds_hook_lands():
    state = _state()
    state.supports_mm_inputs = True
    state.encoder_cache = None  # ty: ignore[invalid-assignment]
    _step(state, make_input_batch([3, 2], num_tokens_after_padding=_BUDGET))
    assert not state.encoder_plan.is_rectangular


def test_a_batch_without_prepare_attn_does_not_inherit_the_last_grid():
    """MRV2's profile_run skips attention, so prepare_attn never sees its batch."""
    state = _state()
    _step(state, make_input_batch([3, 2], num_tokens_after_padding=_BUDGET))
    assert encoder_grid().mask is not None

    dummy = make_input_batch([_BUDGET // 8] * 8)
    inputs = state.prepare_inputs(dummy, req_states=None)  # ty: ignore[invalid-argument-type]
    assert inputs == {}
    assert state.encoder_plan is None
    assert encoder_grid().mask is None


class TestForcedWarmupPlan:
    """The plan warmup declares per rectangle, which the dummy batch cannot produce.

    Upstream's dummy batch splits its tokens evenly over max_num_seqs, so it only
    ever lands on the shortest rectangle.
    """

    @pytest.mark.parametrize(("extent", "width"), [(64, 8), (128, 4)])
    def test_forced_rectangle_is_an_identity_grid(self, extent, width):
        state = _state()
        state.forced_rect = (extent, width)
        dummy = make_input_batch([_BUDGET // 8] * 8)
        _, inputs = _step(state, dummy)

        plan = state.encoder_plan
        assert isinstance(plan.attn_plan, EncoderRectPlan)
        assert plan.grid == (extent, width, [extent] * width)
        # sum(query_lens) is the body, so the dummy's own buffers pass through.
        assert torch.equal(inputs["input_ids"], dummy.input_ids)
        assert torch.equal(inputs["positions"], dummy.positions)

    def test_forced_rectangle_must_fill_the_body(self):
        state = _state()
        state.forced_rect = (64, 4)
        with pytest.raises(AssertionError):
            _step(state, make_input_batch([_BUDGET // 8] * 8))

    def test_forced_rectangle_wins_over_forced_ragged(self):
        state = _state()
        state.forced_rect = (64, 8)
        state.force_ragged = True
        _step(state, make_input_batch([_BUDGET // 8] * 8))
        assert state.encoder_plan.is_rectangular


def test_eager_groups_ragged_requests_one_by_one():
    batch = make_input_batch([64, 64, 64], num_tokens_after_padding=192)
    compiled, eager = _state(), _state(compiled=False)
    _step(compiled, batch)
    _step(eager, batch)
    compiled_groups, eager_groups = compiled.encoder_plan.attn_plan, eager.encoder_plan.attn_plan
    assert isinstance(compiled_groups, list) and isinstance(eager_groups, list)
    assert [g.group for g in compiled_groups] == [2, 1]
    assert [g.group for g in eager_groups] == [1, 1, 1]


def test_dispatch_stats_count_each_path():
    state = _state()
    _step(state, make_input_batch([3, 2], num_tokens_after_padding=_BUDGET))
    _step(state, make_input_batch([3, 2]))
    stats = state.dispatch_stats
    assert (stats.rect_steps, stats.ragged_steps) == (1, 1)
    assert (stats.real_tokens, stats.body_rows, stats.seqs) == (10, _BUDGET + 5, 4)
