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

"""EncoderPlan's read interface (what pooling and warmup see), the forced
warmup plan, and the dispatch counters."""

import torch

from spyre_inference.v1.attention.backends.spyre_encoder_attn import (
    EncoderGroupPlan,
    EncoderRectPlan,
    encoder_key_pad_mask,
)
from spyre_inference.v1.worker.spyre.model_states import encoder_plan
from spyre_inference.v1.worker.spyre.model_states.encoder_plan import (
    EncoderDispatchStats,
    EncoderPlan,
)
from spyre_inference.v1.worker.spyre_shape_bucketer import encoder_dense_row_indices


def _rect(extent: int, width: int, query_lens: list[int]) -> EncoderPlan:
    kv_lens = query_lens + [1] * (width - len(query_lens))
    return EncoderPlan(
        attn_plan=EncoderRectPlan(
            extent=extent,
            width=width,
            mask=encoder_key_pad_mask(extent, kv_lens, torch.float16),
            query_lens=query_lens,
        ),
        body_rows=extent * width,
    )


def _group(starts: list[int], query_lens: list[int], extent: int) -> EncoderGroupPlan:
    return EncoderGroupPlan(
        starts=starts,
        query_lens=query_lens,
        extent=extent,
        row_table=torch.empty(0, dtype=torch.int64),
        mask=encoder_key_pad_mask(extent, query_lens, torch.float16),
    )


def test_rectangle_exposes_its_grid_and_lane_rows():
    plan = _rect(64, 4, [3, 1, 2])
    assert plan.is_rectangular
    assert plan.grid == (64, 4, [3, 1, 2])
    assert plan.rows == [(0, 3), (64, 65), (128, 130)]
    covered = [r for start, end in plan.rows for r in range(start, end)]
    assert covered == encoder_dense_row_indices([3, 1, 2], 64).tolist()


def test_ragged_plan_has_no_grid_or_rows():
    plan = EncoderPlan(
        attn_plan=[_group([0], [3], 64), _group([3, 203], [200, 150], 256)],
        body_rows=512,
    )
    assert not plan.is_rectangular
    assert plan.grid is None
    assert plan.rows == []


def test_stats_accumulate_and_log_periodically(monkeypatch):
    monkeypatch.setattr(encoder_plan, "_LOG_EVERY", 2)
    logged = []
    monkeypatch.setattr(encoder_plan.logger, "info", lambda *args: logged.append(args))
    stats = EncoderDispatchStats()
    stats.record(_rect(64, 4, [3, 1]))
    assert not logged
    stats.record(EncoderPlan(attn_plan=[_group([0], [5], 64)], body_rows=8))
    assert (stats.rect_steps, stats.ragged_steps, stats.seqs) == (1, 1, 3)
    assert (stats.real_tokens, stats.body_rows) == (9, 256 + 8)
    assert len(logged) == 1
