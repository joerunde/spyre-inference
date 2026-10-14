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

"""Model load fix-ups and per-block compile, shared by both model runners."""

from typing import Any

import torch
import torch.nn as nn
from vllm.config import CompilationMode, VllmConfig
from vllm.config.model import ModelConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.attention import Attention
from vllm.model_executor.models.utils import PPMissingLayer

from spyre_inference.custom_ops.bert_head_pad import install_bert_head_pad
from spyre_inference.custom_ops.conv import SpyreConv2d
from spyre_inference.custom_ops.head_pad import (
    fix_padded_attention_scale,
    fix_padded_qk_norm_eps,
    fix_padded_rope,
    install_head_pad_weight_loader,
    install_padded_head_dim,
    verify_padded_head_dim,
    verify_padded_qk_norm_weights,
)
from spyre_inference.custom_ops.mlp_pad import (
    install_mlp_pad_weight_loader,
    verify_padded_intermediate_size,
)
from spyre_inference.v1.worker import compile_guard

logger = init_logger(__name__)


def prepare_loader(model_loader: Any, model_config: ModelConfig) -> None:
    """Pad attention weights (q/k/v/o, and QK-norm) to the stick-aligned head_dim
    as they stream in, when the platform overrode head_dim (e.g. head_size=64).
    Must run before load_model builds+loads the (now 128-wide) params."""
    text_config = model_config.hf_text_config
    install_padded_head_dim(model_config)
    install_bert_head_pad(model_config)
    install_head_pad_weight_loader(model_loader, text_config, model_config)
    install_mlp_pad_weight_loader(model_loader, text_config, model_config)


def fix_loaded_model(
    model: nn.Module, model_config: ModelConfig, spyre_device: torch.device
) -> None:
    """Undo the padding's side effects on the CPU-loaded model, then move it to Spyre."""
    # Restore original RoPE frequencies and attention scale, and compensate
    # QK-norm epsilon for the padded head_dim. All passes are scoped to the
    # padded text backbone (see custom_ops.text_backbone).
    text_config = model_config.hf_text_config
    verify_padded_head_dim(model, text_config, model_config)
    verify_padded_qk_norm_weights(model, text_config, model_config)
    verify_padded_intermediate_size(model, text_config, model_config)
    fix_padded_rope(model, text_config, model_config)
    fix_padded_attention_scale(model, text_config, model_config)
    fix_padded_qk_norm_eps(model, text_config, model_config)

    # Keep Attention module buffers (_k_scale, _v_scale, etc.) on CPU.
    # Note: This _apply cannot reside in SpyreAttentionImpl, as it is not
    # an nn.Module, but just the attention implementation.
    Attention._apply = lambda self, fn, recurse=True: self  # ty: ignore[invalid-assignment]

    # Move layer weights to Spyre device.
    model.to(device=spyre_device)
    for module in model.modules():
        if isinstance(module, SpyreConv2d):
            module.process_weights_after_loading()


def model_has_spyre_fp8(model: nn.Module) -> bool:
    """True if the model has Spyre FP8 linears.

    ``Fp8LinearMethod`` stores the kernel at ``quant_method.fp8_linear``.
    Granite ``compressed-tensors`` stores it on the scheme
    (``quant_method.scheme.fp8_linear`` / ``layer.scheme.fp8_linear``).
    Checkpoint FP8 is dequanted to CPU fp16 in
    ``process_weights_after_loading``; first Spyre forward caches ``qfp8wt``.
    """
    from spyre_inference.custom_ops.fp8_linear_kernel import SpyreFp8LinearKernel

    def _is_fp8_kernel(obj: object) -> bool:
        return isinstance(obj, SpyreFp8LinearKernel) or isinstance(
            getattr(obj, "fp8_linear", None), SpyreFp8LinearKernel
        )

    for module in model.modules():
        weight = getattr(module, "weight", None)
        if isinstance(weight, torch.Tensor) and weight.dtype == torch.float8_e4m3fn:
            return True
        quant_method = getattr(module, "quant_method", None)
        if _is_fp8_kernel(quant_method) or _is_fp8_kernel(getattr(quant_method, "scheme", None)):
            return True
        if _is_fp8_kernel(getattr(module, "scheme", None)):
            return True
    return False


def compile_blocks(model: nn.Module, fullgraph: bool = True) -> tuple[int, int]:
    """Compile every repeated block in place; returns (decoder blocks, tower blocks)."""
    from spyre_inference.v1.worker.spyre_model_runner import _repeated_block_lists

    # Only decoder blocks are counted first: the count decides the whole-model fallback,
    # which an opted-in tower must not pre-empt (it compiles beside either form).
    num_blocks = num_tower = 0
    # Models that re-register a slice of `layers` (e.g. gemma-4's self-/cross-decoder)
    # alias blocks across lists; recompiling one is harmless but would double the count.
    seen: set[int] = set()
    for blocks, marked in _repeated_block_lists(model):
        for block in blocks:
            if isinstance(block, PPMissingLayer) or id(block) in seen:
                continue
            seen.add(id(block))
            # In place: rebinding blocks[i] to the returned OptimizedModule reparents
            # the block under `_orig_mod`, renaming every parameter and breaking
            # reload_weights and save_sharded_state.
            block.compile(backend="inductor", fullgraph=fullgraph, dynamic=False)
            # Identical blocks share one `forward` code object -- that is why this
            # granularity shares a single artifact -- so this collapses to one
            # registration per block *class*, which is the right granularity for
            # the message too. A marked tower is watched like the decoder:
            # `_warmup_multimodal_encoder` compiles it at every image-batch width.
            compile_guard.watch(block, f"{type(block).__name__} (transformer block)")
            if marked:
                num_tower += 1
            else:
                num_blocks += 1
    return num_blocks, num_tower


def compile_for_spyre(model: nn.Module, vllm_config: VllmConfig) -> tuple[nn.Module, int]:
    """Install torch.compile wrappers; tracing happens on the first forward.

    Returns the (possibly whole-model-compiled) model and the compiled tower block
    count. `dynamic=False` is mandatory: the Spyre backend rejects SymInt shapes.
    FP8 apply is dynamo-disabled, so ``fullgraph=False`` when the model has Spyre FP8
    linears.
    """
    from spyre_inference.v1.worker.spyre_model_runner import (
        _block_sharing_defeated_by,
        _compile_granularity,
    )

    mode = vllm_config.compilation_config.mode
    if mode not in (CompilationMode.NONE, CompilationMode.STOCK_TORCH_COMPILE):
        raise ValueError(
            f"Unsupported compilation mode {mode} for Spyre. Only "
            f"CompilationMode.NONE and CompilationMode.STOCK_TORCH_COMPILE "
            f"are supported."
        )

    # Validated before the eager short-circuit so a typo is not silently ignored.
    # Per-block is the default: identical blocks share a ``forward`` code object,
    # so the backend compiles one block rather than a program that grows with depth.
    granularity = _compile_granularity()

    if vllm_config.model_config.enforce_eager or mode is CompilationMode.NONE:
        logger.info("Compilation disabled (enforce_eager=True)")
        return model, 0

    # FP8 apply is dynamo-disabled so the torch-spyre scaled_mm graph
    # stays isolated. fullgraph=True cannot graph-break.
    fullgraph = not model_has_spyre_fp8(model)
    model_name = type(model).__name__

    tower = 0
    if granularity == "block":
        defeated_by = _block_sharing_defeated_by()
        if defeated_by:
            logger.warning(
                "SPYRE_COMPILE_GRANULARITY=block will not share one artifact across "
                "blocks: %s. Expect compile time to grow with layer count; "
                "SPYRE_COMPILE_GRANULARITY=model may warm up faster.",
                defeated_by,
            )
        num_blocks, tower = compile_blocks(model, fullgraph=fullgraph)
        if num_blocks:
            logger.info(
                "Wrapped %d transformer blocks%s of %s for per-block compile on Spyre "
                "(fullgraph=%s).",
                num_blocks,
                f" and {tower} tower blocks" if tower else "",
                model_name,
                fullgraph,
            )
            return model, tower
        logger.warning(
            "Found no attention-bearing block ModuleList in %s; falling back to a "
            "whole-model graph. Models whose attention is not a vLLM Attention "
            "(MLA, encoder-only vision towers) take this path.",
            model_name,
        )

    compiled = torch.compile(model, backend="inductor", fullgraph=fullgraph, dynamic=False)
    compile_guard.watch(compiled, f"{model_name} (whole-model graph)")
    logger.info("Wrapped %s as a single graph for Spyre (fullgraph=%s).", model_name, fullgraph)
    return compiled, tower
