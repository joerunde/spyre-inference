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

"""MRV2 without Triton: the Spyre runner over vllm#61995's NoTritonCPUModelRunner.

Spike (#906 kernel step 2). Selected when vllm.triton_utils reports no Triton and
vllm.v1.worker.cpu.no_triton_model_runner exists, i.e. on a vLLM carrying #61995.
Decoder models only: the default model state, eager or compiled.

The upstream runner is standalone (it does not subclass GPUModelRunner), so the
hooks load.py and kv_cache.py install go into its own module, and the Spyre
model state mixes into its CPU DefaultModelState. Its prepare_inputs ignores bucket
padding, so prepare_inputs here pads the body onto the warmed bucket itself,
which GPUModelRunner gets from SpyreBucketManager.dispatch.
"""

from typing import Any

import torch
from vllm.v1.worker.cpu.model_states.default import DefaultModelState as CPUDefaultModelState
from vllm.v1.worker.cpu.no_triton_model_runner import NoTritonCPUModelRunner

from spyre_inference.v1.worker.spyre import kernels
from spyre_inference.v1.worker.spyre import model_runner as spyre_model_runner
from spyre_inference.v1.worker.spyre.load import unwrap


def bind_kernels() -> None:
    """The sort-free host top-k/top-p, as kernels.bind() binds for the GPU runner's
    sampler. Without it CPUSampler takes upstream's full-vocabulary sort: our OOT
    platform fails its is_cpu() check."""
    import vllm.v1.worker.cpu.sampler as cpu_sampler

    def apply_top_k_top_p(logits, k, p, has_topp_only_rows=True):
        return kernels.spyre_apply_top_k_top_p(logits, k, p)

    cpu_sampler.apply_top_k_top_p = apply_top_k_top_p


class SpyreCPUDefaultModelState(CPUDefaultModelState):
    """MRV2: vllm#61995's CPU DefaultModelState, aware of the Spyre device.

    No multimodal staging: the Spyre EncoderRunner subclasses the GPU one.
    """

    def __init__(self, vllm_config, model, encoder_cache, device, spyre_device) -> None:
        if encoder_cache is not None:
            raise NotImplementedError("Spyre's no-Triton V2 runner has no multimodal path yet.")
        super().__init__(vllm_config, model, encoder_cache, device)
        self.spyre_device = spyre_device

    def get_supported_generation_tasks(self):
        # isinstance checks, which the _SpyreModelWrapper fails.
        model = self.model
        self.model = unwrap(model)
        try:
            return super().get_supported_generation_tasks()
        finally:
            self.model = model


def resolve_cpu_model_state(vllm_config, model, encoder_cache, device, spyre_device):
    from vllm.v1.worker.cpu.model_states import resolve_model_state_cls

    # The wrapper forwards get_model_state_cls, which resolve_model_state_cls rejects.
    cls = resolve_model_state_cls(vllm_config, unwrap(model))
    if cls is not CPUDefaultModelState:
        raise NotImplementedError(
            f"Spyre's no-Triton V2 runner supports decoder models only, not {cls.__name__}."
        )
    return SpyreCPUDefaultModelState(vllm_config, model, encoder_cache, device, spyre_device)


class SpyreNoTritonModelRunnerV2(spyre_model_runner.SpyreRunnerMixin, NoTritonCPUModelRunner):
    """MRV2: vllm#61995's NoTritonCPUModelRunner, on device=cpu with the model on Spyre."""

    upstream_runner_cls = NoTritonCPUModelRunner
    resolve_model_state = staticmethod(resolve_cpu_model_state)

    def prepare_inputs(self, scheduler_output: Any, batch_req_state: Any) -> Any:
        input_batch = super().prepare_inputs(scheduler_output, batch_req_state)
        bucketer = self.spyre_shape_bucketer
        if bucketer is None or not bucketer.is_warmed_up:
            return input_batch
        num_tokens = input_batch.num_tokens
        padded = bucketer.find_bucket(num_tokens) or num_tokens
        if padded == num_tokens:
            return input_batch
        buffers = self.input_buffers
        # Same layout the GPU runner leaves in the padded tail: zeroed ids and
        # positions, flagged as padding, and PAD_SLOT_ID slots (compute_slot_mapping_cpu
        # fills everything past query_start_loc[num_reqs]).
        buffers.input_ids[num_tokens:padded].zero_()
        buffers.positions[num_tokens:padded].zero_()
        buffers.is_padding[:num_tokens].fill_(False)
        buffers.is_padding[num_tokens:padded].fill_(True)
        input_batch.num_tokens_after_padding = padded
        input_batch.input_ids = buffers.input_ids[:padded]
        input_batch.positions = buffers.positions[:padded]
        input_batch.is_padding = buffers.is_padding[:padded]
        return input_batch

    @torch.inference_mode()
    def pool(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("Spyre's no-Triton V2 runner supports decoder models only.")
