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

"""MRV2: XPress on Spyre (vllm/v1/worker/gpu/spec_decode/xpress/speculator.py).

Upstream's propose() runs as-is: draft input prep, the draft's attention metadata and
slot mappings, and the target's rejection sampler stay upstream's. Only the seams
where the draft touches the device are overridden:

- the draft model moves to Spyre and compiles once upstream has loaded it;
- the context K/V is projected over the target's padded rows, as on V1, so each body
  bucket compiles once and no device tensor is narrowed (rows past the batch write
  the null block);
- the draft forward takes Spyre inputs, and the refinement is the Spyre model's
  compiled head (propose_block) instead of upstream's Triton Jacobi loop.
"""

from typing import TYPE_CHECKING, Any, cast

import torch
from vllm.forward_context import BatchDescriptor, set_forward_context
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.cudagraph_dispatcher import CUDAGraphMode
from vllm.v1.worker.gpu.spec_decode.xpress.speculator import XPressSpeculator

from spyre_inference.custom_ops.utils import convert
from spyre_inference.v1.attention import attn_layer
from spyre_inference.v1.worker.model_load import move_model_to_spyre

if TYPE_CHECKING:
    from spyre_inference.models.qwen3_dflash import SpyreQwen3XPressForCausalLM


class SpyreXPressSpeculator(XPressSpeculator):
    model: "SpyreQwen3XPressForCausalLM"

    def __init__(self, vllm_config, device: torch.device, spyre_device: torch.device) -> None:
        super().__init__(vllm_config, device)
        self.spyre_device = spyre_device
        # This step's combined target features: on Spyre, over the target's padded rows.
        self._context: torch.Tensor | None = None

    def load_draft_model(self, target_model, target_attn_layer_names):
        model = cast(
            "SpyreQwen3XPressForCausalLM",
            super().load_draft_model(target_model, target_attn_layer_names),
        )
        move_model_to_spyre(model, self.spyre_device)
        model.prepare_for_spyre()
        return model

    def set_attn(self, *args: Any, **kwargs: Any) -> None:
        super().set_attn(*args, **kwargs)
        if len(self.draft_kv_cache_group_ids) != 1:
            raise NotImplementedError("Spyre XPress expects one draft KV cache group.")

    def _store_hidden_states(
        self,
        num_tokens: int,
        last_hidden_states: torch.Tensor | None,
        aux_hidden_states: list[torch.Tensor] | None,
        num_graph_tokens: int = 0,
    ) -> None:
        assert aux_hidden_states, "XPress consumes the target's auxiliary hidden states"
        self._context = self.model.combine_hidden_states(torch.cat(aux_hidden_states, dim=-1))

    def _precompute_context_kv(self, start: int, end: int, dummy_run: bool = False) -> None:
        assert start == 0 and self._context is not None
        rows = self._context.shape[0]
        positions = self.context_positions[:rows].clone()
        positions[end:] = 0
        slots = None
        if not dummy_run:
            slots = self._context_slot_mappings[0][:rows].clone()
            slots[end:] = PAD_SLOT_ID
        self.model.precompute_and_store_context_kv(
            self._context, convert(positions, self.spyre_device), slots
        )

    @torch.inference_mode()
    def _run_model(
        self,
        num_tokens: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    ) -> torch.Tensor:
        if attn_metadata is None:
            # No metadata build publishes slots; point the traced KV write at null.
            attn_layer.publish_null_slots(num_tokens)
        with set_forward_context(
            attn_metadata,
            self.vllm_config,
            num_tokens=num_tokens,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            num_tokens_across_dp=num_tokens_across_dp,
            slot_mapping=slot_mappings,
            batch_descriptor=BatchDescriptor(num_tokens=num_tokens),
        ):
            return self.model(
                input_ids=convert(
                    self.input_buffers.input_ids[:num_tokens], self.spyre_device, torch.int64
                ),
                positions=convert(
                    self.input_buffers.positions[:num_tokens], self.spyre_device, torch.int64
                ),
            )

    def _jacobi_refine(self, num_reqs: int, head_hidden: torch.Tensor) -> None:
        assert head_hidden.shape[0] == num_reqs * self.num_query_per_req
        anchors = self.input_buffers.input_ids[self._anchor_idx[:num_reqs]].to(torch.int64)
        draft = self.model.propose_block(head_hidden, anchors)
        self.draft_tokens[:num_reqs] = torch.tensor(draft, dtype=self.draft_tokens.dtype)

    def propose(self, input_batch, *args: Any, **kwargs: Any) -> torch.Tensor:
        if kwargs.get("dummy_run") and kwargs.get("skip_attn_for_dummy_run"):
            return self._warm(input_batch, kwargs["aux_hidden_states"])
        return super().propose(input_batch, *args, **kwargs)

    def _warm(self, input_batch, aux_hidden_states: list[torch.Tensor]) -> torch.Tensor:
        """Upstream's profiling path on Spyre: the context projection at this body
        bucket, then one draft block per request, writing no KV."""
        num_reqs = input_batch.num_reqs
        self._store_hidden_states(input_batch.num_tokens, None, aux_hidden_states)
        self._precompute_context_kv(0, input_batch.num_tokens, dummy_run=True)
        block = self.num_query_per_req
        num_query = num_reqs * block
        ids = torch.full((num_query,), self.parallel_drafting_token_id, dtype=torch.int32)
        ids[::block] = 0
        self.input_buffers.input_ids[:num_query] = ids
        self.input_buffers.positions[:num_query] = torch.arange(block).repeat(num_reqs)
        self._generate_draft(num_reqs, num_query, None, None, None)
        return self.draft_tokens[:num_reqs]
