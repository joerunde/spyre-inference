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

"""MRV2: the per-step encoder plan, moved off the runner onto the model state.

V1 keeps this on TorchSpyreModelRunner (_encoder_grid, _forced_encoder_rect,
...); MRV2 expects per-step state on the model state instead. Set by
encoder_only.py, read by pooling and warmup through this module only.
"""

from dataclasses import dataclass

import torch
from vllm.logger import init_logger

from spyre_inference.custom_ops.utils import convert
from spyre_inference.v1.attention.backends.spyre_encoder_attn import (
    EncoderGroupPlan,
    EncoderRectPlan,
    encoder_key_pad_mask,
)

logger = init_logger(__name__)

_LOG_EVERY = 200


@dataclass(frozen=True)
class EncoderPlan:
    """The current step's encoder dispatch: a dense rectangle or ragged rows."""

    attn_plan: EncoderRectPlan | list[EncoderGroupPlan]
    # The padded token count.
    body_rows: int

    @property
    def is_rectangular(self) -> bool:
        return isinstance(self.attn_plan, EncoderRectPlan)

    @property
    def grid(self) -> tuple[int, int, list[int]] | None:
        """``(extent, width, query_lens)`` on the rectangular path; None on the ragged one."""
        plan = self.attn_plan
        if not isinstance(plan, EncoderRectPlan):
            return None
        return plan.extent, plan.width, plan.query_lens

    @property
    def rows(self) -> list[tuple[int, int]]:
        """Body row range of each request's real tokens, in batch order."""
        plan = self.attn_plan
        if isinstance(plan, EncoderRectPlan):
            return [
                (seq * plan.extent, seq * plan.extent + length)
                for seq, length in enumerate(plan.query_lens)
            ]
        # Groups are ordered by extent, not by request; starts restore batch order.
        return sorted(
            (start, start + length)
            for group in plan
            for start, length in zip(group.starts, group.query_lens, strict=True)
        )


def forced_rect_plan(
    extent: int, width: int, dtype: torch.dtype, device: torch.device
) -> EncoderRectPlan:
    """Warmup's rectangle, full-length so the grid expansion is an identity."""
    return EncoderRectPlan(
        extent=extent,
        width=width,
        mask=convert(encoder_key_pad_mask(extent, [extent] * width, dtype), device),
        query_lens=[extent] * width,
    )


@dataclass
class EncoderDispatchStats:
    """Path counts and body occupancy, logged periodically from the worker."""

    rect_steps: int = 0
    ragged_steps: int = 0
    real_tokens: int = 0
    body_rows: int = 0
    seqs: int = 0

    def record(self, plan: EncoderPlan) -> None:
        rows = plan.rows
        if plan.is_rectangular:
            self.rect_steps += 1
        else:
            self.ragged_steps += 1
        self.real_tokens += sum(end - start for start, end in rows)
        self.body_rows += plan.body_rows
        self.seqs += len(rows)
        steps = self.rect_steps + self.ragged_steps
        if steps % _LOG_EVERY:
            return
        logger.info(
            "Encoder dispatch over %d steps: %d fast, %d slow; mean %.2f seqs/step; "
            "body occupancy %.1f%% (%d real tokens in %d rows).",
            steps,
            self.rect_steps,
            self.ragged_steps,
            self.seqs / steps,
            100.0 * self.real_tokens / max(1, self.body_rows),
            self.real_tokens,
            self.body_rows,
        )
