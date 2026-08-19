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

"""Spyre-specific SiluAndMul implementation"""

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMul


logger = init_logger(__name__)


@SiluAndMul.register_oot(name="SiluAndMul")
class SpyreSiluAndMul(SiluAndMul):
    """Out-of-tree (OOT) SiluAndMul implementation for IBM's Spyre device."""

    def forward_oot(self, x) -> torch.Tensor:
        """SwiGLU: silu(gate) * up, output shape [..., d]."""

        # forward_native slices the fused [..., 2*d] tensor on the last dim.
        # That indirect-access slice used to corrupt in eager mode (hence a
        # torch.compile workaround); torch-spyre now handles it directly.
        return self.forward_native(x)
