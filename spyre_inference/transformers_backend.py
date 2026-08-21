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

"""Spyre adaptation of vLLM's Transformers backend.

Upstream's fusers replace HF's linear/norm/GLU modules with vLLM layers, which the Spyre
OOT registrations pick up on their own. Two things are left to HF's module code:

* RoPE — there is no RoPE fuser, so HF's ``rotary_emb`` survives and derives cos/sin
  inside the forward from int64 ``position_ids``, a cast torch-spyre cannot lower.
  Replaced here with a precomputed rotation cache and a matmul-only rotation.
* Models shipping both ``config.json`` and ``params.json`` parse into a bare
  ``PretrainedConfig``, which HF cannot build a model from.
"""

from __future__ import annotations

import sys
from collections.abc import Iterable
from typing import TYPE_CHECKING, cast

import torch
import torch.nn as nn
from transformers import AutoConfig
from transformers.configuration_utils import PretrainedConfig

from vllm.logger import init_logger
from vllm.model_executor.models.transformers import TransformersForCausalLM

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)

# 128-byte stick / 2 bytes per fp16 element. RoPE halves head_dim, so a head_dim that is
# not a multiple of twice this cannot be rotated in place (see _qk_expand_matrix).
FP16_ELEMS_PER_STICK = 64


def _build_rotation_cache(
    inv_freq: torch.Tensor,
    scaling: float,
    max_position: int,
    padded_head_dim: int | None,
    dtype: torch.dtype,
) -> torch.Tensor:
    """``[max_position, 2, 2, head_dim // 2]`` rotation matrices ``[[cos, -sin], [sin, cos]]``.

    Working from ``inv_freq``/``attention_scaling`` inherits whatever rope scaling the
    module being replaced had baked into them. *padded_head_dim* extends the cache with
    identity blocks, so a Q/K padded up to it passes its trailing dimensions through.
    """
    rope_half = inv_freq.shape[0]
    freqs = torch.outer(torch.arange(max_position, dtype=torch.float32), inv_freq)
    cos, sin = torch.cos(freqs) * scaling, torch.sin(freqs) * scaling
    rot = torch.stack([cos, -sin, sin, cos], dim=1).view(max_position, 2, 2, rope_half)

    if padded_head_dim is not None and padded_head_dim // 2 > rope_half:
        identity = torch.zeros(max_position, 2, 2, padded_head_dim // 2 - rope_half)
        identity[:, 0, 0, :] = 1.0
        identity[:, 1, 1, :] = 1.0
        rot = torch.cat([rot, identity], dim=-1)

    return rot.contiguous().to(dtype)


def _apply_rope_matmul(x: torch.Tensor, rot: torch.Tensor) -> torch.Tensor:
    """Rotate ``x`` ``[B, H, L, D]`` by ``rot`` ``[B, L, 2, 2, D // 2]``.

    Multiply-and-reduce rather than HF's ``rotate_half`` cat: Spyre cannot restickify the
    halves that slicing a stick-aligned head_dim produces.
    """
    b, h, seq, head_dim = x.shape
    pairs = x.transpose(1, 2).reshape(b, seq, h, 2, head_dim // 2)
    out = rot.unsqueeze(2).mul(pairs.unsqueeze(-3)).sum(4, keepdim=True).flatten(3)
    return out.transpose(1, 2)


class _SpyreRotaryEmbedding(nn.Module):
    """Drop-in for an HF rotary embedding, returning ``(rot, None)`` in place of
    ``(cos, sin)``; the patched ``apply_rotary_pos_emb`` ignores the second element.

    The cache covers ``max_position`` up front rather than growing on demand: sizing it
    from the batch's positions needs an ``.item()``, so a host sync per step and a
    data-dependent guard ``torch.compile`` cannot trace.
    """

    def __init__(
        self,
        original: nn.Module,
        max_position: int,
        padded_head_dim: int | None,
        dtype: torch.dtype,
    ):
        super().__init__()
        self._cpu_cache = _build_rotation_cache(
            original.get_buffer("inv_freq").to("cpu", torch.float32),
            float(getattr(original, "attention_scaling", 1.0)),
            max_position,
            padded_head_dim,
            dtype,
        )
        self._cache = self._cpu_cache

    def _apply(self, fn, recurse=True):
        # Prime the device cache when the model moves to Spyre, i.e. before compile, so only
        # the index_select is traced. The cache is not a buffer (it is built after weight
        # loading) and there are no children, so super() has nothing to do.
        self._cache = fn(self._cpu_cache)
        return self

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor):
        rot = self._cache.index_select(0, position_ids.flatten())
        return rot.view(*position_ids.shape, *self._cache.shape[1:]), None


def _qk_expand_matrix(orig_head_dim: int, padded_head_dim: int) -> torch.Tensor:
    """Expand matrix widening Q/K for the rotation.

    RoPE pairs element ``i`` with ``i + head_dim // 2``, so the pad lands mid-head rather
    than at the end: each half is expanded to the padded half separately.
    """
    half, padded_half = orig_head_dim // 2, padded_head_dim // 2
    m = torch.zeros(orig_head_dim, padded_head_dim)
    m[:half, :padded_half] = torch.eye(half, padded_half)
    m[half:, padded_half:] = torch.eye(half, padded_half)
    return m


def _make_spyre_apply_rotary(qk_expand: torch.Tensor | None):
    """Replacement for a modeling file's ``apply_rotary_pos_emb``.

    With *qk_expand*, Q/K are widened for the rotation and contracted back afterwards, so
    attention and the KV cache stay on the original head_dim.
    """
    qk_contract = qk_expand.t().contiguous() if qk_expand is not None else None
    cached: dict[torch.device, tuple[torch.Tensor, torch.Tensor]] = {}

    @torch.no_grad()
    def wrapper(q, k, cos, sin=None, *args, **kwargs):
        if qk_expand is not None:
            assert qk_contract is not None  # set together with qk_expand above
            device = q.device
            if device not in cached:
                cached[device] = (
                    qk_expand.to(device=device, dtype=q.dtype),
                    qk_contract.to(device=device, dtype=q.dtype),
                )
            expand, contract = cached[device]
            q, k = torch.matmul(q, expand), torch.matmul(k, expand)

        q, k = _apply_rope_matmul(q, cos), _apply_rope_matmul(k, cos)

        if qk_expand is not None:
            q, k = torch.matmul(q, contract), torch.matmul(k, contract)

        return q, k

    wrapper._spyre_patched = True
    return wrapper


class SpyreTransformersForCausalLM(TransformersForCausalLM):
    """Transformers backend with the Spyre RoPE replacement wired in."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        self._fix_generic_config(vllm_config)
        self._max_position = vllm_config.model_config.max_model_len
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        logger.debug("SpyreTransformersForCausalLM ready: %s", type(self.model).__name__)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        result = super().load_weights(weights)
        self._patch_rope()
        return result

    @staticmethod
    def _fix_generic_config(vllm_config: VllmConfig) -> None:
        """Re-resolve generic PretrainedConfig produced by vLLM's
        config parser for some models where both config.json and params.json exists
        and force HF-format weight loading."""
        hf_config = vllm_config.model_config.hf_config
        if type(hf_config) is not PretrainedConfig:
            return

        model_id = vllm_config.model_config.hf_config_path or vllm_config.model_config.model
        try:
            resolved = AutoConfig.from_pretrained(
                model_id,
                trust_remote_code=vllm_config.model_config.trust_remote_code,
                revision=vllm_config.model_config.revision,
            )
        except Exception:
            logger.warning("AutoConfig re-resolve failed for %s", model_id, exc_info=True)
            return

        skip = {"model_type", "_name_or_path", "transformers_version", "auto_map", "architectures"}
        for key, val in hf_config.to_dict().items():
            if key not in skip and val is not None:
                setattr(resolved, key, val)

        vllm_config.model_config.hf_config = resolved
        vllm_config.model_config.hf_text_config = resolved.get_text_config()
        if vllm_config.load_config.load_format in ("auto", "mistral"):
            vllm_config.load_config.load_format = "hf"
        logger.debug(
            "Re-resolved config: %s (model_type=%s), load_format=hf",
            type(resolved).__name__,
            resolved.model_type,
        )

    def _patch_rope(self):
        """Swap HF's rotary embedding and ``apply_rotary_pos_emb`` for the Spyre ones.

        Partial rotary dimensions (e.g. Phi-3) are unsupported: the rotated and
        pass-through halves would have to be permuted apart before widening.
        """
        cfg = self.model.config
        orig_head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads

        stick = 2 * FP16_ELEMS_PER_STICK
        stick_aligned = ((orig_head_dim + stick - 1) // stick) * stick
        padded_head_dim = stick_aligned if stick_aligned > orig_head_dim else None
        qk_expand = (
            _qk_expand_matrix(orig_head_dim, padded_head_dim)
            if padded_head_dim is not None
            else None
        )

        # The text backbone holding rotary_emb; multimodal models nest it one level
        # deeper, at model.model.language_model.
        inner = self.model.model if hasattr(self.model, "model") else self.model
        backbone = cast(nn.Module, getattr(inner, "language_model", inner))

        spyre_rope = _SpyreRotaryEmbedding(
            backbone.get_submodule("rotary_emb"),
            self._max_position,
            padded_head_dim,
            next(self.model.parameters()).dtype,
        )
        backbone.rotary_emb = spyre_rope

        patched_mods: set[int] = set()
        for name, module in self.model.named_modules():
            if module is spyre_rope:
                continue

            cls_name = module.__class__.__name__

            if cls_name.endswith("RotaryEmbedding"):
                parent_name, _, attr = name.rpartition(".")
                parent = self.model.get_submodule(parent_name) if parent_name else self.model
                setattr(parent, attr, spyre_rope)
                continue

            if "Attention" not in cls_name:
                continue

            if not hasattr(module, "rotary_emb"):
                module.rotary_emb = spyre_rope

            # apply_rotary_pos_emb is a module-level function in HF modeling files, so it
            # is patched once per modeling module rather than per layer.
            mod = sys.modules.get(type(module).__module__)
            if mod is None or id(mod) in patched_mods:
                continue
            orig = getattr(mod, "apply_rotary_pos_emb", None)
            if orig is None or getattr(orig, "_spyre_patched", False):
                continue
            mod.apply_rotary_pos_emb = _make_spyre_apply_rotary(qk_expand)
            patched_mods.add(id(mod))


# using_transformers_backend() compares _ModelInfo.architecture, which is model_cls.__name__,
# against "TransformersForCausalLM", so the subclass has to keep answering to that name.
SpyreTransformersForCausalLM.__name__ = "TransformersForCausalLM"
