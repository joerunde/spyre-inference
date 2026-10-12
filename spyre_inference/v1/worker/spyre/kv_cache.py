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

Per-layer SpyrePagedKVCache (head-major or token-major). Upstream's
initialize_kv_cache has no override hook for its init_attn_backend /
BlockTables calls, so init() owns a full copy of the method body (like
worker.py's init_device), guarded by a drift test, with those two calls
swapped for attn_groups.init() / SpyreBlockTables. Waits on #945:
KVCacheTensor's shared_by/stride schema is rewritten at 0.29, so the copy
starts from the post-#945 body, not from the 0.28 one this skeleton pins.
"""

from typing import TYPE_CHECKING, Any

from vllm.v1.kv_cache_interface import KVCacheConfig

if TYPE_CHECKING:
    from spyre_inference.v1.worker.spyre.model_runner import SpyreModelRunnerV2


def init(runner: "SpyreModelRunnerV2", kv_cache_config: KVCacheConfig, **kwargs: Any) -> None:
    """Copy of GPUModelRunner.initialize_kv_cache, with init_attn_backend
    replaced by attn_groups.init() and BlockTables by SpyreBlockTables."""
    raise NotImplementedError
