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

"""MRV2: SpyreModelRunnerV2.initialize_kv_cache body
(vllm/v1/worker/gpu/model_runner.py:GPUModelRunner.initialize_kv_cache).

Per-layer SpyrePagedKVCache (head-major or token-major), replacing only the
allocation and attention-group creation in upstream's initialize_kv_cache
(block_tables, pcp_manager, kv_connector, ... all stay upstream's). Waits
on #945: KVCacheTensor's shared_by/stride schema is rewritten at 0.29.
"""

from typing import TYPE_CHECKING, Any

from vllm.v1.kv_cache_interface import KVCacheConfig

if TYPE_CHECKING:
    from spyre_inference.v1.worker.spyre.model_runner import SpyreModelRunnerV2


def init(runner: "SpyreModelRunnerV2", kv_cache_config: KVCacheConfig, **kwargs: Any) -> None:
    """Allocate the KV cache and bind attn_groups.init(), replacing only
    those two parts of upstream's initialize_kv_cache."""
    raise NotImplementedError
