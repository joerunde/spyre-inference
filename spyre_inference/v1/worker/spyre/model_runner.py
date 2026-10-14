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
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.model_runner import GPUModelRunner

from spyre_inference.v1.attention.backends.spyre_attn import SpyrePagedKVCache
from spyre_inference.v1.worker.spyre import bucketing, kv_cache, load, warmup
from spyre_inference.v1.worker.spyre.pool import output as pool_output
from spyre_inference.v1.worker.spyre_shape_bucketer import SpyreShapeBucketer


class SpyreModelRunnerV2(GPUModelRunner):
    """MRV2: vllm/v1/worker/gpu/model_runner.py:GPUModelRunner, on device=cpu."""

    def __init__(self, vllm_config: VllmConfig, device: torch.device) -> None:
        super().__init__(vllm_config, torch.device("cpu"))
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

    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig, **kwargs: Any) -> None:
        kv_cache.init(self, kv_cache_config, **kwargs)
        self.cudagraph_manager = bucketing.SpyreBucketManager.from_runner(self)

    def pool(self, *args: Any, **kwargs: Any) -> Any:
        return pool_output.pool_sync(self, *args, **kwargs)

    def warming_up_model(self) -> None:
        warmup.run(self)
