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

"""Lay a step's packed encoder body out as the rectangular path's dense grid."""

from __future__ import annotations

import torch

from spyre_inference.custom_ops.utils import convert
from spyre_inference.v1.pool.spyre_pooler import select_rows
from spyre_inference.v1.worker.spyre_shape_bucketer import (
    encoder_grid_source_rows,
    expand_packed_to_encoder_grid,
)


def expand_body_to_grid(
    input_ids: torch.Tensor | None,
    inputs_embeds: torch.Tensor | None,
    positions: torch.Tensor,
    grid: tuple[int, int, list[int]],
    *,
    pad_token_id: int,
    position_offset: int,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor]:
    """``(input_ids, inputs_embeds, positions)`` for the ``(extent, width, query_lens)`` grid.

    The `input_ids` path packs only integer tensors (tens of KB), avoiding a per-layer
    activation gather. Multimodal pooling with only `inputs_embeds` instead gathers the
    embeddings into the grid on the device, with an index built on the host, so they
    never round-trip through the host. Each result is on its input's device.
    """
    extent, width, query_lens = grid
    num_tokens = sum(query_lens)

    if input_ids is not None:
        ids, pos = expand_packed_to_encoder_grid(
            input_ids[:num_tokens].cpu(),
            positions[:num_tokens].cpu(),
            query_lens,
            width,
            extent,
            pad_token_id=pad_token_id,
            position_offset=position_offset,
        )
        assert ids.shape[0] == width * extent, (ids.shape[0], width * extent)
        return convert(ids, input_ids.device), inputs_embeds, convert(pos, positions.device)

    # A multimodal pooling model (e.g. CLIP) preprocesses straight to embeddings; expand
    # those instead of ids. Positions still need the same padding, so reuse that helper
    # with dummy ids (discarded).
    assert inputs_embeds is not None, "upstream must supply ids or embeds"
    _, pos = expand_packed_to_encoder_grid(
        torch.zeros(num_tokens, dtype=torch.int64),
        positions[:num_tokens].cpu(),
        query_lens,
        width,
        extent,
        position_offset=position_offset,
    )
    # Gathered from the whole body buffer, not a `[:num_tokens]` slice: the gather
    # specialises on its source's row count, which then stays fixed.
    grid_embeds = select_rows(
        inputs_embeds,
        torch.tensor(encoder_grid_source_rows(query_lens, width, extent), dtype=torch.int64),
    )
    assert grid_embeds.shape[0] == width * extent, (grid_embeds.shape[0], width * extent)
    return None, grid_embeds, convert(pos, positions.device)
