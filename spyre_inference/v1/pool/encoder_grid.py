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

"""Pooling over the encoder grid and pooler warmup, shared by both model runners.

``grid`` is ``(extent, width, query_lens)`` on the rectangular path, None on the
ragged one.
"""

from typing import cast

import torch
import torch.nn as nn
from vllm.model_executor.layers.pooler.seqwise.poolers import SequencePooler
from vllm.v1.pool.metadata import PoolingMetadata

from spyre_inference.custom_ops.utils import convert
from spyre_inference.v1.pool.spyre_pooler import (
    SpyreAllPool,
    SpyreClassifierLinear,
    SpyreCLSPool,
    SpyreDispatchPooler,
    SpyreEmbeddingPoolerHead,
    SpyreMeanPool,
    SpyrePoolingCursor,
    SpyreRobertaClassificationHead,
    SpyreTokenPooler,
    _iter_modules,
    _mean_pool_grid_reduce,
    _mean_pool_row_mask_mul,
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
    """Point the cursor at the grid for a grid-direct CLS/MEAN step, or unpad the grid.

    ``first_token_indices_gpu`` stays a host tensor, like the rest of the cursor.
    """
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


def warm_pooler_row_widths(
    pooler: nn.Module,
    hidden_states: torch.Tensor,
    rectangles: list[tuple[int, int]],
    max_num_seqs: int,
) -> None:
    """Compile the pooler's row gather at every row count serving can present.

    A dummy pooler run only ever pools ``min(num_tokens, max_num_seqs)`` rows, but
    serving pools one row per *request* -- any count from 1 up. That gather
    specializes on the exact index length, so every unseen count paid a full
    Inductor compile mid-request. The poolers round their row count to a power of
    two (``pad_row_count_to_bucket``), so this sweep is short.

    ``SpyreSequencePooler`` trims only after the classifier/embed head, so the head
    sees these same bucket widths and is warmed at each. ``SpyreMeanPool``'s
    rectangle reduce specializes on ``(extent, rows)`` and is warmed per declared
    rectangle.
    """
    rows = hidden_states.shape[0]
    # poolers_by_task is a plain dict; nn.Module.modules() doesn't descend into it.
    pooler_modules = list(_iter_modules(pooler))
    mean_pooling = any(isinstance(m, SpyreMeanPool) for m in pooler_modules)
    classifier_types = SpyreRobertaClassificationHead | SpyreClassifierLinear
    classifier_heads = [
        m for m in pooler_modules if isinstance(getattr(m, "classifier", None), classifier_types)
    ]
    embed_heads = [m for m in pooler_modules if isinstance(m, SpyreEmbeddingPoolerHead)]
    token_poolers = [m for m in pooler_modules if isinstance(m, SpyreTokenPooler)]
    # Up to the power of two at or above the limit, which is not itself always one:
    # 6 sequences round up to 8 rows, so stopping at the limit misses that width.
    limit = 1 << max(0, max_num_seqs - 1).bit_length()
    widths = [1 << i for i in range(limit.bit_length()) if 1 << i <= rows]
    # MEAN on a rectangle reduces its first min(bucket, lanes) lanes; a grid with a
    # non-power-of-two lane count hands the head that count instead of the bucket.
    grid_shapes: set[tuple[int, int]] = set()
    if mean_pooling:
        grid_shapes = {
            (extent, min(width, lanes))
            for extent, lanes in rectangles
            if extent * lanes == rows
            for width in widths
        }
    for width in sorted(set(widths) | {lanes for _, lanes in grid_shapes}):
        pooled = select_rows(hidden_states, torch.zeros(width, dtype=torch.int64))
        for head in classifier_heads:
            logits = head.classifier(pooled)
            if head.activation is not None:
                head.activation(logits)
        for head in embed_heads:
            embeddings = head.projector(pooled) if head.projector is not None else pooled
            if head.activation is not None:
                head.activation(embeddings)
    if grid_shapes:
        device = hidden_states.device
        mask = convert(torch.zeros(rows, 1, dtype=torch.float16), device)
        prod = _mean_pool_row_mask_mul(hidden_states, mask).clone()
        for extent, lanes in grid_shapes:
            lens = convert(torch.ones(lanes, 1, dtype=torch.float32), device)
            _mean_pool_grid_reduce(prod, lens, extent)
    for token_pooler in token_poolers:
        warm_token_pool_widths(cast(SpyreTokenPooler, token_pooler), hidden_states)


def warm_token_pool_widths(token_pooler: SpyreTokenPooler, hidden_states: torch.Tensor) -> None:
    """Compile ``SpyreAllPool``'s bucketed gather (and its head) at every declared length.

    ``SpyreAllPool`` rounds each request's real token count up to the nearest
    entry in ``len_ladder`` instead of tracking it exactly (see its docstring);
    nothing before this swept that same ladder, so the first request landing on
    an unseen length paid a full Inductor compile mid-request.
    """
    all_pool = token_pooler.pooling
    if not isinstance(all_pool, SpyreAllPool):
        return
    head = token_pooler.head
    rows = hidden_states.shape[0]
    for length in sorted(set(all_pool.len_ladder)):
        if length > rows:
            continue
        chunk = select_rows(hidden_states, torch.arange(length, dtype=torch.int64))
        if head is None:
            continue
        classifier = getattr(head, "classifier", None)
        logits = classifier(chunk) if classifier is not None else chunk
        projector = getattr(head, "projector", None)
        if projector is not None:
            logits = projector(logits)
        activation = getattr(head, "activation", None)
        if activation is not None:
            activation(logits)
