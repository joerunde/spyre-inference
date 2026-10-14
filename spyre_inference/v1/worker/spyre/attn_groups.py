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

from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.model_executor.layers.attention.attention import Attention
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.utils import AttentionGroup

from spyre_inference.v1.worker.kv_cache_setup import split_attn_groups_by_window


def init(
    upstream_init_attn_backend: Callable[..., Any],
    kv_cache_config: KVCacheConfig,
    vllm_config: VllmConfig,
    *args: Any,
    **kwargs: Any,
) -> tuple[list[list[AttentionGroup]], Any, list[int]]:
    """Upstream's init_attn_backend, then split each group by sliding window."""
    attn_groups, attn_cg_support, kernel_block_sizes = upstream_init_attn_backend(
        kv_cache_config, vllm_config, *args, **kwargs
    )
    split_attn_groups_by_window(attn_groups, get_layers_from_vllm_config(vllm_config, Attention))
    return attn_groups, attn_cg_support, kernel_block_sizes
