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

"""MRV2: SpyreModelRunnerV2.load_model (vllm/v1/worker/gpu/model_runner.py:
GPUModelRunner.load_model).

Upstream's load_model runs as-is around a wrapped model loader: the loader
loads on the host, applies the shared fix-ups, moves the model to Spyre,
compiles per block and returns the _SpyreModelWrapper. Everything upstream
builds from ``self.model`` afterwards (model state, sampler, pooling runner)
therefore holds the wrapper. init_model_state and PoolingRunner are swapped
for their Spyre counterparts for the duration of the call.
"""

import contextlib
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

import torch.nn as nn
from vllm.logger import init_logger
from vllm.model_executor.model_loader.utils import get_model_architecture
from vllm.v1.worker.gpu import model_runner as upstream_model_runner

from spyre_inference.models import check_spyre_model
from spyre_inference.multimodal import apply_multimodal_patches
from spyre_inference.v1.pool import configure_pooling_for_spyre
from spyre_inference.v1.worker import block_compile
from spyre_inference.v1.worker.spyre import bucketing, compile
from spyre_inference.v1.worker.spyre.model_states import resolve_spyre_model_state
from spyre_inference.v1.worker.spyre.pool.pooling_runner import SpyrePoolingRunner
from spyre_inference.v1.worker.spyre_model_runner import (  # shared with V1; moves at the flip
    _SpyreModelWrapper,
)
from spyre_inference.v1.worker.spyre_shape_bucketer import (
    encoder_len_ladder,
    logits_row_buckets,
)

if TYPE_CHECKING:
    from spyre_inference.v1.worker.spyre.model_runner import SpyreModelRunnerV2

logger = init_logger(__name__)


class _SpyreModelLoader:
    """Wraps upstream's model loader so its load_model returns the Spyre wrapper."""

    def __init__(self, loader: Any, runner: "SpyreModelRunnerV2") -> None:
        self._loader = loader
        self._runner = runner

    def __getattr__(self, name: str) -> Any:
        return getattr(self._loader, name)

    def load_model(self, *, vllm_config: Any, model_config: Any) -> _SpyreModelWrapper:
        runner = self._runner
        if runner.lora_config:
            raise NotImplementedError("LoRA adapters are not yet implemented and tested for Spyre.")
        block_compile.prepare_loader(self._loader, model_config)
        model = self._loader.load_model(vllm_config=vllm_config, model_config=model_config)
        check_spyre_model(model, get_model_architecture(model_config)[1])
        block_compile.fix_loaded_model(model, model_config, runner.spyre_device)

        runner.pooling_on_spyre = False
        if model_config.runner_type == "pooling":
            runner.pooling_on_spyre = configure_pooling_for_spyre(
                model, runner.spyre_device, encoder_len_ladder(vllm_config)
            )
        logger.info("Spyre-native layer weights moved to %s", runner.spyre_device)

        # Patches instances/classes, so it runs after load and before compile wraps modules
        # in OptimizedModule and breaks traversal.
        apply_multimodal_patches(model, runner.spyre_device)
        compiled, runner.compiled_tower_blocks = block_compile.compile_for_spyre(
            model, vllm_config
        )

        bucketer = bucketing.shape_bucketer(vllm_config)
        runner.spyre_shape_bucketer = bucketer
        return _SpyreModelWrapper(
            compiled,
            runner.spyre_device,
            keep_outputs_on_device=runner.pooling_on_spyre,
            logits_row_buckets=(
                []
                if bucketer is None
                else logits_row_buckets(bucketer.bucket_sizes, runner.max_num_reqs)
            ),
            shape_bucketer=bucketer,
            model_dtype=runner.model_dtype,
        )


@contextlib.contextmanager
def _patched(runner: "SpyreModelRunnerV2") -> Iterator[None]:
    get_model_loader = upstream_model_runner.get_model_loader
    init_model_state = upstream_model_runner.init_model_state
    pooling_runner_cls = upstream_model_runner.PoolingRunner

    def spyre_model_loader(load_config: Any) -> _SpyreModelLoader:
        return _SpyreModelLoader(get_model_loader(load_config), runner)

    def spyre_init_model_state(vllm_config, model, encoder_cache, device):
        return resolve_spyre_model_state(
            vllm_config, model, encoder_cache, device, runner.spyre_device
        )

    upstream_model_runner.get_model_loader = spyre_model_loader
    upstream_model_runner.init_model_state = spyre_init_model_state
    upstream_model_runner.PoolingRunner = SpyrePoolingRunner
    try:
        yield
    finally:
        upstream_model_runner.get_model_loader = get_model_loader
        upstream_model_runner.init_model_state = init_model_state
        upstream_model_runner.PoolingRunner = pooling_runner_cls


def load_and_wrap(runner: "SpyreModelRunnerV2", *args: Any, **kwargs: Any) -> None:
    """Load weights, move the model to Spyre, wrap, and compile per block.

    Postcondition: ``runner.model`` is the wrapper, and every other holder of
    the model (``model_state.model``, ``encoder_runner.model``,
    ``pooling_runner.model``) points at that same wrapper.
    """
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    with _patched(runner):
        GPUModelRunner.load_model(runner, *args, **kwargs)
    runner.model_memory_usage = 0
    assert isinstance(runner.model, _SpyreModelWrapper)
    compile.check_no_whole_model_compile(runner)
    pooling_runner = runner.pooling_runner
    if pooling_runner is not None:
        assert isinstance(pooling_runner, SpyrePoolingRunner)
        pooling_runner.pooling_on_spyre = runner.pooling_on_spyre
        model_state = runner.model_state
        pooling_runner.encoder_plan_source = lambda: getattr(model_state, "encoder_plan", None)


def unwrap(model: Any) -> nn.Module:
    """The bare model under the wrapper and any whole-model OptimizedModule."""
    if isinstance(model, _SpyreModelWrapper):
        model = model._model
    return getattr(model, "_orig_mod", model)
