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

"""MRV2: mm-embed staging (vllm/v1/worker/gpu/mm/encoder_runner.py:EncoderRunner).

A Spyre-resident staging buffer for multimodal embeddings. Upstream allocates
``inputs_embeds`` on the runner's device, the host here, and get_inputs_embeds
copies the Spyre-resident result of embed_input_ids into it, leaving the model
a host float tensor the wrapper does not move. On Spyre, the wrapper stages
embed_input_ids' result in this same buffer at the bucket's shape and returns
its prefix view, so upstream's ``inputs_embeds[:n] = x`` is a self-copy
torch-spyre skips (see _SpyreModelWrapper.embed_input_ids). Needs the model
wrapper installed first, which load.load_and_wrap guarantees.

Multimodal kwargs stay on the host (``device``): the wrapper's embed_multimodal
moves the float ones to Spyre itself.
"""

from typing import Any

import torch
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
from vllm.v1.worker.gpu.mm.encoder_runner import EncoderRunner


class SpyreEncoderRunner(EncoderRunner):
    """MRV2: EncoderRunner, staging inputs_embeds on Spyre instead of the GPU."""

    def __init__(
        self,
        model: Any,
        max_num_tokens: int,
        hidden_size: int,
        encoder_cache: EncoderCache,
        dtype: torch.dtype,
        device: torch.device,
        *,
        spyre_device: torch.device,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model, max_num_tokens, hidden_size, encoder_cache, dtype, device, **kwargs
        )
        self.inputs_embeds = torch.zeros(
            max_num_tokens, hidden_size, dtype=dtype, device=spyre_device
        )
        # Bypasses _SpyreModelWrapper.__setattr__, which forwards to the model.
        object.__setattr__(model, "_inputs_embeds_buffer", self.inputs_embeds)
