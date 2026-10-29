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

"""MRV2: attention-group construction (vllm/v1/worker/gpu/attn_utils.py:init_attn_backend).

Splits each group by per-layer sliding window after init_attn_backend
runs: FullAttentionSpec.merge collapses a hybrid decoder onto one window,
clamping its full-attention layers too. The split itself is the shared
split_attn_groups_by_window (kv_cache_setup.py), which V1 calls too.
"""

from collections.abc import Callable
from typing import Any

import torch
from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.model_executor.layers.attention.attention import Attention
from vllm.v1.kv_cache_interface import KVCacheConfig

from spyre_inference.v1.worker.kv_cache_setup import split_attn_groups_by_window


def init(
    upstream_init_attn_backend: Callable[..., Any],
    kv_cache_config: KVCacheConfig,
    vllm_config: VllmConfig,
    device: torch.device,
    *args: Any,
    **kwargs: Any,
) -> tuple[Any, ...]:
    """Upstream's init_attn_backend, then split each group by sliding window.

    Upstream builds each group's metadata builders inside init_attn_backend, so the
    groups the split creates get theirs here (V1 splits before its builders exist).
    The GPU runner's returns (groups, cudagraph support, kernel block sizes), the
    no-Triton CPU runner's (groups, kernel block sizes); the result keeps that shape.
    """
    result = upstream_init_attn_backend(kv_cache_config, vllm_config, device, *args, **kwargs)
    attn_groups, kernel_block_sizes = result[0], result[-1]
    split_attn_groups_by_window(attn_groups, get_layers_from_vllm_config(vllm_config, Attention))
    for kv_cache_group_id, groups in enumerate(attn_groups):
        for group in groups:
            if group.metadata_builders:
                continue
            group.create_metadata_builders(
                vllm_config=vllm_config,
                device=device,
                kernel_block_size=(
                    kernel_block_sizes[kv_cache_group_id]
                    if kv_cache_group_id < len(kernel_block_sizes)
                    else None
                ),
            )
    return result
