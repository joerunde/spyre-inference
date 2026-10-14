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

Upstream's initialize_kv_cache runs as-is, so every runner attribute it sets
(block_tables, pcp_manager, ubatch_runner, kernel_block_sizes, kv_connector,
...) stays upstream's. Only its two module-level calls are swapped for the
call: init_attn_backend gains the sliding-window split (attn_groups.py), and
init_kv_cache allocates per-layer SpyrePagedKVCache pages on Spyre instead of
one raw tensor per KVCacheTensor. Upstream has no override hook for either
(0.32's ModelState.initialize_kv_cache is the closest), hence the patch.

The pages are kept on runner.spyre_kv_caches and returned to upstream as an
empty dict: upstream filters what it returns by ``cache.device``, which a
SpyrePagedKVCache tuple has no attribute for, and only the KV connector and
prefix-cache block copies read it, neither supported on Spyre.
"""

import contextlib
from collections.abc import Iterator
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any

from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu import model_runner as upstream_model_runner
from vllm.v1.worker.gpu.attn_utils import get_shared_kv_cache_layers
from vllm.v1.worker.utils import bind_kv_cache

from spyre_inference.v1.worker.kv_cache_setup import allocate_spyre_kv_caches
from spyre_inference.v1.worker.spyre import attn_groups

if TYPE_CHECKING:
    from spyre_inference.v1.worker.spyre.model_runner import SpyreModelRunnerV2


@contextlib.contextmanager
def _patched(runner: "SpyreModelRunnerV2") -> Iterator[None]:
    init_attn_backend = upstream_model_runner.init_attn_backend
    init_kv_cache = upstream_model_runner.init_kv_cache

    def spyre_init_attn_backend(*args: Any, **kwargs: Any) -> Any:
        return attn_groups.init(init_attn_backend, *args, **kwargs)

    def spyre_init_kv_cache(
        forward_context: dict[str, Any],
        kv_cache_config: KVCacheConfig,
        device: Any,
        kernel_block_sizes: list[int],
        vllm_config: Any,
        kv_cache_allocation_context: Any = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        with kv_cache_allocation_context or nullcontext():
            kv_caches = allocate_spyre_kv_caches(
                kv_cache_config,
                forward_context,
                get_shared_kv_cache_layers(vllm_config),
                runner.spyre_device,
            )
        bind_kv_cache(kv_caches, forward_context, [])  # ty: ignore[invalid-argument-type]
        runner.spyre_kv_caches = dict(kv_caches)
        return {}

    upstream_model_runner.init_attn_backend = spyre_init_attn_backend
    upstream_model_runner.init_kv_cache = spyre_init_kv_cache
    try:
        yield
    finally:
        upstream_model_runner.init_attn_backend = init_attn_backend
        upstream_model_runner.init_kv_cache = init_kv_cache


def init(runner: "SpyreModelRunnerV2", kv_cache_config: KVCacheConfig, **kwargs: Any) -> None:
    """Upstream's initialize_kv_cache with the Spyre attention groups and pages."""
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    with _patched(runner):
        GPUModelRunner.initialize_kv_cache(runner, kv_cache_config, **kwargs)
