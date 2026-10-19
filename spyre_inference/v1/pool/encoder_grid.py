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

"""Pooling over the encoder grid, shared by both model runners.

``grid`` is ``(extent, width, query_lens)`` on the rectangular path, None on the
ragged one.
"""

import numpy as np
import torch
import torch.nn as nn
from vllm.config import ModelConfig
from vllm.model_executor.layers.pooler.abstract import Pooler
from vllm.model_executor.layers.pooler.seqwise.poolers import SequencePooler
from vllm.pooling_params import PoolingParams
from vllm.tasks import PoolingTask
from vllm.v1.pool.metadata import PoolingMetadata, PoolingStates

from spyre_inference.v1.pool.spyre_pooler import (
    SpyreCLSPool,
    SpyreDispatchPooler,
    SpyreMeanPool,
    SpyrePoolingCursor,
    select_rows,
)
from spyre_inference.v1.worker.spyre_shape_bucketer import encoder_cls_rows

EncoderGrid = tuple[int, int, list[int]]


def unpad_encoder_grid(hidden_states: torch.Tensor, grid: EncoderGrid | None) -> torch.Tensor:
    """Re-compact a rectangular-path grid to the packed order the poolers address.

    Except for eligible homogeneous Spyre CLS and MEAN batches, poolers need
    packed-order rows, including token-level poolers. Reporting padded lengths instead makes
    ``PoolingCursor.is_partial_prefill()`` true and ``SpyreCLSPool`` raise.

    The gather keeps its input's row count: sizing it to the real token count adds
    a ``torch.compile`` specialisation per distinct total, recompiling nearly every
    step once prompt lengths vary. The ragged path is already packed, and every
    pooler either gathers by row index or crops on the host, so its trailing pad
    needs no gather at all.
    """
    if grid is None:
        return hidden_states
    extent, width, query_lens = grid
    # Host Python, not tensor ops: every extra CPU aten op on the per-step path
    # lengthens the shared eager-op guard chain and costs more than the loop (#981).
    rows_list: list[int] = []
    for seq_idx, length in enumerate(query_lens):
        start = seq_idx * extent
        rows_list.extend(range(start, start + length))
    total = width * extent
    if len(rows_list) == total:
        return hidden_states
    rows_list = rows_list + [0] * (total - len(rows_list))
    return select_rows(hidden_states, torch.tensor(rows_list, dtype=torch.int64, device="cpu"))


def rect_pooling_method(
    pooler: nn.Module, pooling_metadata: PoolingMetadata, grid: EncoderGrid | None
) -> SpyreCLSPool | SpyreMeanPool | None:
    """This rectangular step's pooling method, when it can read the grid directly.

    CLS takes each lane's first row and MEAN reduces each lane, so both skip the
    unpad gather. Other poolers, including LAST and token-level ones, need packed
    rows. Mixed-task batches also take the unpad gather so one layout serves every
    task.
    """
    if grid is None:
        return None
    tasks = list(pooling_metadata.tasks)
    if len(set(tasks)) != 1:
        return None
    sub = pooler
    if isinstance(pooler, SpyreDispatchPooler):
        sub = pooler.poolers_by_task.get(tasks[0])
        if sub is None:
            return None
    pooling = sub.pooling if isinstance(sub, SequencePooler) else sub
    if not isinstance(pooling, SpyreCLSPool | SpyreMeanPool):
        return None
    cursor = pooling_metadata.get_pooling_cursor()
    counts = cursor.num_scheduled_tokens_cpu if cursor is not None else None
    if counts is None or int(counts.numel()) != len(grid[2]):
        return None
    return pooling


def pool_rows_for(
    hidden_states: torch.Tensor,
    pooler: nn.Module,
    pooling_metadata: PoolingMetadata,
    grid: EncoderGrid | None,
) -> torch.Tensor:
    """Point the cursor at the grid for a grid-direct CLS/MEAN step, or unpad the grid."""
    # ``first_token_indices_gpu`` stays a host tensor, like the rest of the cursor.
    rect_pooling = rect_pooling_method(pooler, pooling_metadata, grid)
    if grid is not None and isinstance(rect_pooling, SpyreCLSPool):
        extent, _width, query_lens = grid
        pooling_metadata.get_pooling_cursor().first_token_indices_gpu = torch.tensor(
            encoder_cls_rows(len(query_lens), extent), dtype=torch.int64
        )
        return hidden_states
    if grid is not None and isinstance(rect_pooling, SpyreMeanPool):
        cursor = pooling_metadata.get_pooling_cursor()
        pooling_metadata.pooling_cursor = SpyrePoolingCursor(
            **vars(cursor), spyre_grid_extent=grid[0]
        )
        return hidden_states
    # Not a crop: the row count stays the buffer's. On the rectangular path this
    # re-compacts the grid to the packed order the cursor addresses; on the ragged
    # path it is a no-op.
    return unpad_encoder_grid(hidden_states, grid)


def dummy_pooling_metadata(
    pooler: Pooler,
    model_config: ModelConfig,
    task: PoolingTask,
    num_tokens: int,
    max_num_reqs: int,
    device: torch.device,
) -> PoolingMetadata:
    """Upstream's dummy pooler metadata, with the cursor on the host as in a real step."""
    num_reqs = min(num_tokens, max_num_reqs)
    min_tokens_per_req = num_tokens // num_reqs
    num_scheduled_tokens_np = np.full(num_reqs, min_tokens_per_req)
    num_scheduled_tokens_np[-1] += num_tokens % num_reqs
    assert np.sum(num_scheduled_tokens_np) == num_tokens
    assert len(num_scheduled_tokens_np) == num_reqs

    req_num_tokens = num_tokens // num_reqs
    dummy_prompt_lens = torch.from_numpy(num_scheduled_tokens_np)
    dummy_token_ids = torch.zeros((num_reqs, req_num_tokens), dtype=torch.int32, device=device)

    dummy_pooling_params = PoolingParams(task=task)
    dummy_pooling_params.verify(model_config)
    to_update = pooler.get_pooling_updates(task)
    to_update.apply(dummy_pooling_params)

    dummy_metadata = PoolingMetadata(
        prompt_lens=dummy_prompt_lens,
        prompt_token_ids=dummy_token_ids,
        prompt_token_ids_cpu=dummy_token_ids.cpu(),
        pooling_params=[dummy_pooling_params] * num_reqs,
        pooling_states=[PoolingStates() for _ in range(num_reqs)],
    )
    dummy_metadata.build_pooling_cursor(
        num_scheduled_tokens_np,
        seq_lens_cpu=dummy_prompt_lens,
        device=torch.device("cpu"),
    )
    return dummy_metadata
