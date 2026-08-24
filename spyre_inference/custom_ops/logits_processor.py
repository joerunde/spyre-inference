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

from vllm.model_executor.layers.logits_processor import LogitsProcessor

from .utils import convert


@LogitsProcessor.register_oot(name="LogitsProcessor")
class SpyreLogitsProcessor(LogitsProcessor):
    def _apply_head(self, lm_head, hidden_states, embedding_bias=None):
        """D2H the logits: the head is replicated so upstream runs no gather and
        they stay on Spyre, where the sampler's ``.to(torch.float32)`` would
        crash torch-spyre's ``copy_from_d2d``."""
        logits = super()._apply_head(lm_head, hidden_states, embedding_bias)
        if logits.device.type != "cpu":
            logits = convert(logits, device="cpu")
        return logits
