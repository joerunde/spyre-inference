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

"""MRV2: Spyre bucket padding via ModelCudaGraphManager.dispatch
(vllm/v1/worker/gpu/cudagraph_utils.py).

Spyre has no cudagraph capture, so dispatch() only picks the padded bucket
size and always reports CUDAGraphMode.NONE. Replaces V1's
_determine_batch_execution_and_padding / _spyre_bucket_batch_descriptor.

Warmup dummies never reach dispatch(): MRV2 runs them with is_profile=True,
which skips it, so warmup.plan() picks bucket sizes itself through bucket().
"""

from typing import TYPE_CHECKING

import torch
from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.logger import init_logger
from vllm.v1.worker.gpu.cudagraph_utils import (
    BatchExecutionDescriptor,
    ModelCudaGraphManager,
)

from spyre_inference.v1.worker.spyre_shape_bucketer import SpyreShapeBucketer

if TYPE_CHECKING:
    from spyre_inference.v1.worker.spyre.model_runner import SpyreModelRunnerV2

logger = init_logger(__name__)


def _shape_bucketer(vllm_config: VllmConfig) -> SpyreShapeBucketer | None:
    # Pooling pads in eager too: the encoder rectangle is exactly the token budget.
    if vllm_config.model_config.runner_type == "pooling":
        return SpyreShapeBucketer.for_pooling(vllm_config)
    if vllm_config.model_config.enforce_eager:
        logger.info("Bucket padding disabled (enforce_eager=True)")
        return None
    if not vllm_config.compilation_config.compile_sizes:
        logger.info("Bucket padding disabled (no compile_sizes configured)")
        return None
    return SpyreShapeBucketer(vllm_config)


class SpyreBucketManager(ModelCudaGraphManager):
    """MRV2: ModelCudaGraphManager, replacing capture/dispatch with bucket padding."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        decode_query_len: int,
        lora_capture_cases: list[int] | None = None,
    ) -> None:
        super().__init__(
            vllm_config,
            device,
            CUDAGraphMode.NONE,
            decode_query_len,
            lora_capture_cases=lora_capture_cases,
        )
        self.shape_bucketer = _shape_bucketer(vllm_config)

    @classmethod
    def from_runner(cls, runner: "SpyreModelRunnerV2") -> "SpyreBucketManager":
        """Build the bucket manager from an already-initialized runner."""
        return cls(
            runner.vllm_config,
            runner.device,
            runner.decode_query_len,
            lora_capture_cases=runner.lora_capture_cases,
        )

    def bucket(self, num_tokens: int) -> int:
        """Pad up to the next ``compile_sizes`` bucket; 0 stays 0 ("nothing to run")."""
        if self.shape_bucketer is None or num_tokens <= 0:
            return num_tokens
        return self.shape_bucketer.find_bucket(num_tokens) or num_tokens

    def dispatch(
        self,
        num_reqs: int,
        num_tokens: int,
        uniform_token_count: int | None,
        num_active_loras: int,
        max_query_len: int | None = None,
        num_ubatches: int = 1,
    ) -> BatchExecutionDescriptor:
        assert num_ubatches == 1, "Spyre has no DBO (0.31 microbatching)"
        return BatchExecutionDescriptor(
            cg_mode=CUDAGraphMode.NONE,
            num_tokens=self.bucket(num_tokens),
            num_reqs=num_reqs,
            num_active_loras=self._resolve_effective_loras(num_active_loras),
        )
