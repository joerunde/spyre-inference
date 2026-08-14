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

"""Native-path attention head padding to a stick-aligned head_dim.

A head_dim whose half is not a multiple of the 64-element fp16 stick (e.g.
head_size=64) cannot restickify after RoPE, so the KV write-back fails to lower
on Spyre. ``TorchSpyrePlatform._maybe_pad_head_dim`` overrides ``head_dim`` to a
128-multiple before the model is built (sizing QKV/o_proj/Attention/KV-cache/RoPE
at the padded width); the passes here fill the padded region on load and restore
the two things the width override would otherwise corrupt — the RoPE frequencies
and the attention scale.

Ported from hf-adapters' ``pad_attention_heads`` / ``PrecomputedRotaryEmbedding``
(interleaved RoPE-compatible padding for Q/K, end-padding for V/O, original-
frequency rotation cache), adapted to vLLM's fused/TP-sharded checkpoint stream.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.rotary_embedding import get_rope

logger = init_logger(__name__)

_ORIG_ATTR = "_spyre_orig_head_dim"


def head_padding_active(hf_config) -> bool:
    """True when the platform padded this model's head_dim for stick alignment."""
    return getattr(hf_config, _ORIG_ATTR, None) is not None


def _pad_qk_interleaved(w: torch.Tensor, n_heads: int, orig: int, padded: int) -> torch.Tensor:
    """Interleaved padding on the output dim (dim 0), RoPE half-split compatible.

    Per head: ``[first_half | zeros | second_half | zeros]`` so that the padded
    dims pair with zeros under the ``[2, D/2]`` RoPE reshape.
    """
    orig_half, padded_half = orig // 2, padded // 2
    w = w.view(n_heads, orig, *w.shape[1:])
    new = w.new_zeros(n_heads, padded, *w.shape[2:])
    new[:, :orig_half] = w[:, :orig_half]
    new[:, padded_half : padded_half + orig_half] = w[:, orig_half:orig]
    return new.reshape(n_heads * padded, *w.shape[2:])


def _pad_output_end(w: torch.Tensor, n_heads: int, orig: int, padded: int) -> torch.Tensor:
    """End-pad each head on the output dim (dim 0). Used for V (no RoPE)."""
    w = w.view(n_heads, orig, *w.shape[1:])
    new = w.new_zeros(n_heads, padded, *w.shape[2:])
    new[:, :orig] = w
    return new.reshape(n_heads * padded, *w.shape[2:])


def _pad_input_end(w: torch.Tensor, n_heads: int, orig: int, padded: int) -> torch.Tensor:
    """End-pad each head on the input dim (dim 1). Used for O."""
    hidden = w.shape[0]
    w = w.view(hidden, n_heads, orig)
    new = w.new_zeros(hidden, n_heads, padded)
    new[:, :, :orig] = w
    return new.reshape(hidden, n_heads * padded)


def _pad_weight(
    name: str, w: torch.Tensor, n_heads: int, n_kv_heads: int, orig: int, padded: int
) -> torch.Tensor:
    """Dispatch a single checkpoint tensor to the right padding by its name."""
    if name.endswith(("q_proj.weight", "q_proj.bias")):
        return _pad_qk_interleaved(w, n_heads, orig, padded)
    if name.endswith(("k_proj.weight", "k_proj.bias")):
        return _pad_qk_interleaved(w, n_kv_heads, orig, padded)
    if name.endswith(("v_proj.weight", "v_proj.bias")):
        return _pad_output_end(w, n_kv_heads, orig, padded)
    if name.endswith("o_proj.weight"):
        return _pad_input_end(w, n_heads, orig, padded)
    return w


def install_head_pad_weight_loader(model_loader, hf_config) -> None:
    """Wrap ``model_loader.get_all_weights`` to pad q/k/v/o head_dim 64->128.

    The transform runs on the raw ``(name, tensor)`` stream before vLLM's
    ``WeightsMapper`` and ``weight_loader`` (which ``.narrow`` and assert exact
    shapes against the now-128-wide params). Full unsharded tensors are padded
    per-head, so TP narrowing downstream still selects whole padded heads.
    """
    if not head_padding_active(hf_config):
        return
    if not hasattr(model_loader, "get_all_weights"):
        logger.warning(
            "Head padding active but %s has no get_all_weights; weights not padded.",
            type(model_loader).__name__,
        )
        return

    orig = getattr(hf_config, _ORIG_ATTR)
    padded = hf_config.head_dim
    n_heads = hf_config.num_attention_heads
    n_kv_heads = getattr(hf_config, "num_key_value_heads", None) or n_heads

    original_get_all_weights = model_loader.get_all_weights

    def padded_get_all_weights(model_config, model) -> Iterable[tuple[str, torch.Tensor]]:
        for name, weight in original_get_all_weights(model_config, model):
            yield name, _pad_weight(name, weight, n_heads, n_kv_heads, orig, padded)

    model_loader.get_all_weights = padded_get_all_weights


def fix_padded_attention_scale(model, hf_config) -> None:
    """Restore the attention scale to ``1/sqrt(orig_head_dim)`` for head_dim-derived scales.

    A model that computes ``scale = head_dim**-0.5`` (Llama, Mistral) picks up
    ``padded**-0.5`` from the widened head_dim, but the real dot product is still
    over the original dims (padded dims are zero), so it must be divided by
    ``sqrt(orig_head_dim)`` or softmax flattens. Models with a head_dim-independent
    scale (Granite's ``attention_multiplier``) were never corrupted by padding, so
    their scale must be left untouched — detected by comparing the built scale
    against the padded head_dim default.
    """
    if not head_padding_active(hf_config):
        return
    orig = getattr(hf_config, _ORIG_ATTR)
    padded_default = float(hf_config.head_dim**-0.5)
    orig_default = float(orig**-0.5)
    n = 0
    for module in model.modules():
        impl = getattr(module, "impl", None)
        if (
            impl is not None
            and hasattr(impl, "scale")
            and math.isclose(float(impl.scale), padded_default, rel_tol=1e-3)
        ):
            impl.scale = orig_default
            n += 1
    logger.info("Reset attention scale to 1/sqrt(%d) on %d head_dim-derived layers.", orig, n)


def fix_padded_rope(model, hf_config) -> None:
    """Inject the original-frequency cos/sin cache into each padded RoPE.

    ``get_rope(padded)`` built frequencies at the padded spacing (wrong); rebuild
    a reference rope at the original head_dim (reusing vLLM's rope-scaling dispatch
    for correct Llama3/YaRN frequencies) and swap its narrower cos_sin_cache in.
    ``SpyreRotaryEmbedding._get_rotation_cache`` then derives the real rotations
    from it and zero-pads the trailing dims (harmless — the matching x pair dims
    are zero from weight padding).
    """
    if not head_padding_active(hf_config):
        return
    orig = getattr(hf_config, _ORIG_ATTR)
    max_position = hf_config.max_position_embeddings
    rope_parameters = getattr(hf_config, "rope_parameters", None)

    seen: set[int] = set()
    n = 0
    for module in model.modules():
        if not hasattr(module, "_rope_key") or id(module) in seen:
            continue
        seen.add(id(module))
        ref = get_rope(
            orig,
            max_position=max_position,
            is_neox_style=module.is_neox_style,
            rope_parameters=rope_parameters,
            dtype=module.dtype,
        )
        module.cos_sin_cache = ref.cos_sin_cache.to(module.cos_sin_cache.dtype)
        module._rotation_cache = None
        module._device_rotation_cache = None
        n += 1
    logger.info("Injected original head_dim=%d RoPE frequencies into %d modules.", orig, n)
