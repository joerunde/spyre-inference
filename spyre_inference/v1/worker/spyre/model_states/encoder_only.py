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

"""MRV2: encoder-only model state (vllm/v1/worker/gpu/model_states/encoder_only.py:
EncoderOnlyModelState).

Dispatches through an EncoderPlan (rectangular grid vs. ragged rows)
instead of upstream's non-causal metadata path directly. Ports V1's
_build_attention_metadata (the plan) and _preprocess (grid expansion,
token_type_ids, the RoBERTa position offset).

MRV2 calls prepare_attn before prepare_inputs, so the plan is built first.
"""

from typing import Any

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.config.compilation import CompilationMode, CUDAGraphMode
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
from vllm.v1.worker.gpu.model_states.encoder_only import EncoderOnlyModelState
from vllm.v1.worker.gpu.states import RequestState
from vllm.v1.worker.utils import AttentionGroup

from spyre_inference.models.roberta import offset_host_positions, roberta_position_delta
from spyre_inference.v1.attention.backends.spyre_encoder_attn import (
    plan_encoder_step,
    publish_encoder_grid,
)
from spyre_inference.v1.worker.spyre.model_states.base import SpyreModelStateMixin
from spyre_inference.v1.worker.spyre.model_states.encoder_plan import (
    EncoderDispatchStats,
    EncoderPlan,
)
from spyre_inference.v1.worker.spyre_shape_bucketer import (
    encoder_group_width_caps,
    encoder_shape_tables,
    expand_packed_to_encoder_grid,
    expand_packed_token_types,
)


class SpyreEncoderOnlyModelState(SpyreModelStateMixin, EncoderOnlyModelState):
    """MRV2: EncoderOnlyModelState, dispatching through an EncoderPlan."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        model: nn.Module,
        encoder_cache: EncoderCache | None,
        device: torch.device,
        spyre_device: torch.device,
    ) -> None:
        super().__init__(vllm_config, model, encoder_cache, device, spyre_device)
        tables = encoder_shape_tables(vllm_config)
        self._rectangles = list(tables.rectangles)
        self._width_caps = encoder_group_width_caps(vllm_config)
        self._budget = tables.budget
        # Grouping only pays off through the compiled kernels.
        self._batched = vllm_config.compilation_config.mode is CompilationMode.STOCK_TORCH_COMPILE
        # The platform settles on float16; the config field is typed wider.
        self._mask_dtype = self.dtype if isinstance(self.dtype, torch.dtype) else torch.float16
        hf_config = getattr(self.model_config, "hf_config", None)
        self._position_offset = roberta_position_delta(hf_config)
        pad = getattr(hf_config, "pad_token_id", None)
        self._pad_token_id = pad if isinstance(pad, int) else 0

        # Warmup only; a forced rectangle wins over force_ragged.
        self.forced_rect: tuple[int, int] | None = None
        self.force_ragged = False

        self.encoder_plan: EncoderPlan | None = None
        # profile_run skips prepare_attn, so its batch must not inherit the last grid.
        self._planned_batch: InputBatch | None = None
        self.dispatch_stats = EncoderDispatchStats()

    def prepare_attn(
        self,
        input_batch: InputBatch,
        cudagraph_mode: CUDAGraphMode,
        block_tables: tuple[torch.Tensor, ...],
        slot_mappings: torch.Tensor,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        for_capture: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:
        # Not in the forward: the plan's D2H/H2D would become graph nodes there.
        attn_metadata = super().prepare_attn(
            input_batch,
            cudagraph_mode,
            block_tables,
            slot_mappings,
            attn_groups,
            kv_cache_config,
            for_capture,
            *args,
            **kwargs,
        )
        rows = input_batch.num_tokens_after_padding
        if self.forced_rect is not None:
            # The expansion is an identity only over a body exactly this size.
            assert self.forced_rect[0] * self.forced_rect[1] == rows, (self.forced_rect, rows)
        # A rectangle is the whole body buffer. Multimodal stays packed until #1242:
        # its body arrives as inputs_embeds, which prepare_inputs never sees.
        use_rectangles = rows == self._budget and not self.force_ragged
        use_rectangles = use_rectangles and not self.supports_mm_inputs
        attn_plan = plan_encoder_step(
            attn_metadata,
            rectangles=self._rectangles if use_rectangles else (),
            width_cap_for=self._width_caps,
            device=self.spyre_device,
            dtype=self._mask_dtype,
            batched=self._batched,
            forced_rect=self.forced_rect,
        )
        plan = None
        if attn_plan is not None:
            plan = EncoderPlan(attn_plan=attn_plan, body_rows=rows)
            self.dispatch_stats.record(plan)
        self.encoder_plan = plan
        self._planned_batch = input_batch
        return attn_metadata

    def prepare_inputs(
        self, input_batch: InputBatch, req_states: RequestState
    ) -> dict[str, torch.Tensor | None]:
        """Lay the packed body out as the rectangle's grid; offset RoBERTa positions."""
        model_inputs = super().prepare_inputs(input_batch, req_states)
        if self._planned_batch is not input_batch:
            self._set_plan(None, None)
        positions = model_inputs.get("positions")
        if positions is None:
            positions = input_batch.positions

        grid = self.encoder_plan.grid if self.encoder_plan is not None else None
        if grid is None:
            if self._position_offset:
                model_inputs["positions"] = offset_host_positions(positions, self._position_offset)
            return model_inputs

        extent, width, query_lens = grid
        num_tokens = sum(query_lens)
        ids, pos = expand_packed_to_encoder_grid(
            input_batch.input_ids[:num_tokens].cpu(),
            positions[:num_tokens].cpu(),
            query_lens,
            width,
            extent,
            pad_token_id=self._pad_token_id,
            position_offset=self._position_offset,
        )
        model_inputs["input_ids"] = ids.to(input_batch.input_ids.device)
        model_inputs["positions"] = pos.to(positions.device)
        token_types = model_inputs.get("token_type_ids")
        if token_types is not None:
            model_inputs["token_type_ids"] = expand_packed_token_types(
                token_types[:num_tokens].cpu(), query_lens, width, extent
            ).to(token_types.device)
        return model_inputs

    def _set_plan(self, plan: EncoderPlan | None, input_batch: InputBatch | None) -> None:
        self.encoder_plan = plan
        self._planned_batch = input_batch
        publish_encoder_grid(plan.attn_plan if plan is not None else None)
