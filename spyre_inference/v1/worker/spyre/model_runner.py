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

"""MRV2: Spyre model runner (vllm/v1/worker/gpu/model_runner.py:GPUModelRunner).

A thin coordinator: every method here delegates to the component module
that owns the concern.
"""

from typing import Any

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.tasks import SupportedTask
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.model_runner import GPUModelRunner

from spyre_inference.v1.attention.backends.spyre_attn import SpyrePagedKVCache
from spyre_inference.v1.worker.spyre import bucketing, kv_cache, load, warmup
from spyre_inference.v1.worker.spyre.model_states import resolve_spyre_model_state
from spyre_inference.v1.worker.spyre.pool import output as pool_output
from spyre_inference.v1.worker.spyre.pool.pooling_runner import SpyrePoolingRunner
from spyre_inference.v1.worker.spyre_shape_bucketer import SpyreShapeBucketer


class SpyreRunnerMixin:
    """What both upstream MRV2 runners need to run on Spyre with the runner on the host.

    ``upstream_runner_cls`` is the upstream class whose module the load and KV hooks
    patch. Defined before it in the MRO, so its __init__ runs on device=cpu.
    """

    upstream_runner_cls: type
    resolve_model_state = staticmethod(resolve_spyre_model_state)
    pooling_runner_cls: type = SpyrePoolingRunner

    def __init__(self, vllm_config: VllmConfig, device: torch.device) -> None:
        super().__init__(vllm_config, torch.device("cpu"))  # ty: ignore[too-many-positional-arguments]
        self.spyre_device = device
        dtype = self.model_config.dtype
        self.model_dtype = dtype if isinstance(dtype, torch.dtype) else torch.float16
        # Set by load.load_and_wrap.
        self.spyre_shape_bucketer: SpyreShapeBucketer | None = None
        self.pooling_on_spyre = False
        self.compiled_tower_blocks = 0
        # Set by kv_cache.init.
        self.spyre_kv_caches: dict[str, SpyrePagedKVCache] = {}

    def load_model(self, *args: Any, **kwargs: Any) -> None:
        load.load_and_wrap(self, *args, **kwargs)

    def get_model(self) -> nn.Module:
        return load.unwrap(self.model)

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        tasks: list[SupportedTask] = []
        if self.model_config.runner_type == "generate":
            tasks.extend(self.model_state.get_supported_generation_tasks())
        if self.is_pooling_model:
            tasks.extend(SpyrePoolingRunner.get_supported_tasks(self.model))
        return tuple(tasks)

    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig, **kwargs: Any) -> None:
        kv_cache.init(self, kv_cache_config, **kwargs)

    def warming_up_model(self) -> None:
        warmup.run(self)


class SpyreModelRunnerV2(SpyreRunnerMixin, GPUModelRunner):
    """MRV2: vllm/v1/worker/gpu/model_runner.py:GPUModelRunner, on device=cpu."""

    upstream_runner_cls = GPUModelRunner

    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig, **kwargs: Any) -> None:
        super().initialize_kv_cache(kv_cache_config, **kwargs)
        self.cudagraph_manager = bucketing.SpyreBucketManager.from_runner(self)

    def pool(self, *args: Any, **kwargs: Any) -> Any:
        return pool_output.pool_sync(self, *args, **kwargs)
