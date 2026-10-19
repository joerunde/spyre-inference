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

"""Pooler warmup sweeps, shared by both model runners."""

import torch
import torch.nn as nn

from spyre_inference.custom_ops.utils import convert
from spyre_inference.v1.pool.spyre_pooler import (
    SpyreAllPool,
    SpyreClassifierLinear,
    SpyreEmbeddingPoolerHead,
    SpyreMeanPool,
    SpyreRobertaClassificationHead,
    SpyreTokenPooler,
    _iter_modules,
    _mean_pool_grid_reduce,
    _mean_pool_row_mask_mul,
    select_rows,
)


@torch.inference_mode()
def warm_pooler_row_widths(
    pooler: nn.Module,
    hidden_states: torch.Tensor,
    rectangles: list[tuple[int, int]],
    max_num_seqs: int,
) -> None:
    """Compile the pooler's row gather at every row count serving can present.

    ``_dummy_pooler_run`` only ever pools ``min(num_tokens, max_num_seqs)``
    rows, but serving pools one row per *request* -- any count from 1 up. That
    gather specializes on the exact index length, so every unseen count paid a
    full Inductor compile mid-request; it was the whole residual warm-up gap
    once the attention kernels were covered. The poolers round their row count
    to a power of two (``pad_row_count_to_bucket``), so this sweep is short.

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
        warm_token_pool_widths(token_pooler, hidden_states)


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
