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

"""MRV2: pooling (vllm/v1/worker/gpu/pool/pooling_runner.py:PoolingRunner).

Host cursor instead of a GPU pooling cursor, no dim-0 slice (Spyre pads
differently than CUDA), grid-direct CLS/MEAN pooling, and unpadding before
the pooler sees hidden_states. The grid comes from the encoder model state's
EncoderPlan, read through ``encoder_plan_source``.
"""

from collections.abc import Callable

import numpy as np
import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.pooling_params import PoolingParams
from vllm.tasks import PoolingTask
from vllm.v1.outputs import PoolerOutput
from vllm.v1.pool.metadata import PoolingMetadata, PoolingStates
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.pool.pooling_runner import PoolingRunner
from vllm.v1.worker.gpu.states import RequestState

from spyre_inference.custom_ops.utils import convert
from spyre_inference.v1.pool.encoder_grid import pool_rows_for, unpad_encoder_grid
from spyre_inference.v1.worker.spyre.model_states.encoder_plan import EncoderPlan

_HOST = torch.device("cpu")


class SpyrePoolingRunner(PoolingRunner):
    """MRV2: PoolingRunner, pooling synchronously with a host cursor."""

    def __init__(self, model: nn.Module, vllm_config: VllmConfig) -> None:
        super().__init__(model, vllm_config)
        # Set by the runner once both the model state and the wrapper exist.
        self.pooling_on_spyre = False
        self.encoder_plan_source: Callable[[], EncoderPlan | None] = lambda: None

    @staticmethod
    def get_supported_tasks(model: nn.Module) -> list[PoolingTask]:
        # is_pooling_model is an isinstance check, which the _SpyreModelWrapper fails.
        bare = getattr(model, "_model", model)
        return PoolingRunner.get_supported_tasks(getattr(bare, "_orig_mod", bare))

    def _grid(self) -> tuple[int, int, list[int]] | None:
        plan = self.encoder_plan_source()
        return plan.grid if plan is not None else None

    def pool(
        self,
        hidden_states: torch.Tensor,
        input_batch: InputBatch,
        req_states: RequestState,
    ) -> tuple[PoolerOutput, list[bool]]:
        grid = self._grid()
        if not self.pooling_on_spyre:
            # FP32 linear heads keep the pooler on the host.
            hidden_states = unpad_encoder_grid(convert(hidden_states, "cpu"), grid)
            return super().pool(hidden_states, input_batch, req_states)

        # Build the cursor on the host: upstream's ``cumsum[1:] - 1`` for
        # last_token_indices is an offset-1 view that is not stick-aligned on Spyre,
        # so the ``*_gpu`` cursor fields are host tensors here, despite the name.
        # No ``[:num_tokens]`` crop either: Spyre dim-0 slices are unsafe, and each
        # pooling method gathers its own rows.
        pooling_metadata = self._get_pooling_metadata(input_batch, req_states, _HOST)
        num_reqs = input_batch.num_reqs
        seq_lens_cpu = input_batch.seq_lens_cpu_upper_bound[:num_reqs]
        pooling_metadata.build_pooling_cursor(
            input_batch.num_scheduled_tokens, seq_lens_cpu, device=_HOST
        )
        hidden_states = pool_rows_for(hidden_states, self.model.pooler, pooling_metadata, grid)
        pooler_output = self.model.pooler(hidden_states, pooling_metadata)
        finished_mask = pooling_metadata.get_pooling_cursor().get_finished_mask()
        pooler_output = self.late_interaction_runner.postprocess_pooler_output(
            raw_pooler_output=pooler_output,
            pooling_params=pooling_metadata.pooling_params,
            req_ids=input_batch.req_ids,
            finished_mask=finished_mask,
        )
        return pooler_output, finished_mask

    def _dummy_pooler_run_task(
        self, hidden_states: torch.Tensor, task: PoolingTask
    ) -> PoolerOutput:
        if not self.pooling_on_spyre:
            return super()._dummy_pooler_run_task(convert(hidden_states, "cpu"), task)
        # Upstream's, with the cursor on the host as in pool().
        num_tokens = hidden_states.shape[0]
        num_reqs = min(num_tokens, self.max_num_reqs)
        num_scheduled_tokens = np.full(num_reqs, num_tokens // num_reqs, dtype=np.int32)
        if num_extra := num_tokens % num_reqs:
            num_scheduled_tokens[-num_extra:] += 1
        prompt_lens = torch.from_numpy(num_scheduled_tokens)

        pooling_params = PoolingParams(task=task)
        pooling_params.verify(self.model_config)
        self.model.pooler.get_pooling_updates(task).apply(pooling_params)
        prompt_token_ids_cpu = None
        if pooling_params.requires_token_ids:
            prompt_token_ids_cpu = torch.zeros(
                (num_reqs, int(prompt_lens.max())), dtype=torch.int64
            )
        pooling_metadata = PoolingMetadata(
            prompt_lens=prompt_lens,
            prompt_token_ids=None,
            prompt_token_ids_cpu=prompt_token_ids_cpu,
            pooling_params=[pooling_params] * num_reqs,
            pooling_states=[PoolingStates() for _ in range(num_reqs)],
        )
        pooling_metadata.build_pooling_cursor(
            num_scheduled_tokens, seq_lens_cpu=prompt_lens, device=_HOST
        )
        return self.model.pooler(hidden_states, pooling_metadata)
