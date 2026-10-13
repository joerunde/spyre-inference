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

"""MRV2: resolves the Spyre ModelState
(vllm/v1/worker/gpu/model_states/__init__.py:init_model_state).

Mirrors upstream's dispatch, but to the Spyre mixin over whichever upstream
class it resolves to, rather than to the upstream class itself.
"""

import functools

import torch
import torch.nn as nn
from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.model_executor.layers.attention import Attention, CrossAttention
from vllm.v1.attention.backend import AttentionType
from vllm.v1.worker.gpu import model_states as upstream_model_states
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
from vllm.v1.worker.gpu.model_states.default import DefaultModelState
from vllm.v1.worker.gpu.model_states.encoder_only import EncoderOnlyModelState
from vllm.v1.worker.gpu.model_states.interface import ModelState

from spyre_inference.v1.worker.spyre.model_states.base import SpyreModelStateMixin
from spyre_inference.v1.worker.spyre.model_states.default import SpyreDefaultModelState
from spyre_inference.v1.worker.spyre.model_states.encoder_only import (
    SpyreEncoderOnlyModelState,
)

_SPYRE_STATES: dict[type[ModelState], type[ModelState]] = {
    DefaultModelState: SpyreDefaultModelState,
    EncoderOnlyModelState: SpyreEncoderOnlyModelState,
}


def _resolve_model_state_cls_v028(vllm_config: VllmConfig, model: nn.Module) -> type[ModelState]:
    """0.28's init_model_state dispatch; delete at the 0.31 bump (resolve_model_state_cls)."""
    if hasattr(model, "get_model_state_cls"):
        return model.get_model_state_cls()  # ty: ignore[call-non-callable]
    if any(isinstance(m, CrossAttention) for m in model.modules()):
        from vllm.v1.worker.gpu.model_states.encoder_decoder import EncoderDecoderModelState

        return EncoderDecoderModelState
    if any(
        layer.attn_type == AttentionType.ENCODER_ONLY
        for layer in get_layers_from_vllm_config(vllm_config, Attention).values()
    ):
        return EncoderOnlyModelState
    if vllm_config.model_config.is_hybrid or vllm_config.model_config.is_attention_free:
        from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState

        return MambaHybridModelState
    return DefaultModelState


_resolve_model_state_cls = getattr(
    upstream_model_states, "resolve_model_state_cls", _resolve_model_state_cls_v028
)


@functools.cache
def spyre_model_state_cls(upstream_cls: type[ModelState]) -> type[ModelState]:
    spyre_cls = _SPYRE_STATES.get(upstream_cls)
    if spyre_cls is None:
        spyre_cls = type(f"Spyre{upstream_cls.__name__}", (SpyreModelStateMixin, upstream_cls), {})
    return spyre_cls


def resolve_spyre_model_state(
    vllm_config: VllmConfig,
    model: nn.Module,
    encoder_cache: EncoderCache | None,
    device: torch.device,
    spyre_device: torch.device | None = None,
) -> ModelState:
    """Pick and construct the Spyre-mixed ModelState; ``device`` is the host."""
    cls = _resolve_model_state_cls(vllm_config, model)
    # 0.31's init_model_state rejects a state that would silently ignore this.
    if vllm_config.model_config.enable_prompt_embeds and not getattr(
        cls, "supports_prompt_embeds", True
    ):
        raise ValueError(f"--enable-prompt-embeds not supported with {cls.__name__}.")
    return spyre_model_state_cls(cls)(
        vllm_config,
        model,
        encoder_cache,
        device,
        spyre_device,  # ty: ignore[too-many-positional-arguments]
    )
