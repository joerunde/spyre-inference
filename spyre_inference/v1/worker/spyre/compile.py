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

"""MRV2: per-block compile, and the guard against MRV2 compiling the whole model.

On 0.31 MRV2's load_model compiles nothing itself: STOCK_TORCH_COMPILE on MRV2
arrives with vLLM 0.32 (vllm#59079), and the Triton gate only drops it from the
unsupported list. load.py compiles one block at a time through the shared
block_compile.compile_for_spyre, as V1 does. At 0.32 this module suppresses the
whole-model compile; until then check_no_whole_model_compile() is the tripwire.
"""

from typing import TYPE_CHECKING

from vllm.v1.worker.gpu.cudagraph_utils import has_compiled_submodule

if TYPE_CHECKING:
    from spyre_inference.v1.worker.spyre.model_runner import SpyreModelRunnerV2


def check_no_whole_model_compile(runner: "SpyreModelRunnerV2") -> None:
    """Fail if upstream wrapped the model in its own compile (it would trace the
    whole model behind our per-block graphs)."""
    if has_compiled_submodule(runner.get_model()):
        raise RuntimeError(
            "MRV2 compiled the model itself; Spyre compiles one block at a time "
            "(block_compile.compile_for_spyre). Suppress upstream's compile here."
        )
