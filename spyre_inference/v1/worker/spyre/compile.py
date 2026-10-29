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

"""MRV2: per-block compile, suppressing MRV2's whole-model stock compile.

Under STOCK_TORCH_COMPILE, 0.32's load_model ends with
``compile_model_with_stock_torch(self.model, ...)`` (vllm#59079), whose
``model.compile`` reaches the inner model through _SpyreModelWrapper.__getattr__
and would trace it whole, behind our per-block graphs. load.py compiles one block
at a time through the shared compile_for_spyre (v1/worker/compile_granularity.py),
as V1 does, and swaps upstream's call for a no-op for the duration of load_model.
check_no_whole_model_compile() is the tripwire if a later vLLM compiles some
other way.
"""

from typing import TYPE_CHECKING, Any

from vllm.logger import init_logger
from vllm.v1.worker.gpu.cudagraph_utils import has_compiled_submodule

if TYPE_CHECKING:
    from spyre_inference.v1.worker.spyre.model_runner import SpyreModelRunnerV2

logger = init_logger(__name__)


def skip_stock_compile(model: Any, vllm_config: Any) -> None:
    """Stands in for upstream's compile_model_with_stock_torch."""
    logger.info("Skipping MRV2's whole-model stock compile; Spyre compiles per block.")


def check_no_whole_model_compile(runner: "SpyreModelRunnerV2") -> None:
    """Fail if upstream wrapped the model in its own compile."""
    model = runner.get_model()
    # nn.Module.compile sets _compiled_call_impl; has_compiled_submodule only sees
    # @support_torch_compile wrappers.
    if has_compiled_submodule(model) or getattr(model, "_compiled_call_impl", None) is not None:
        raise RuntimeError(
            "MRV2 compiled the model itself; Spyre compiles one block at a time "
            "(compile_granularity.compile_for_spyre). Suppress upstream's compile here."
        )
