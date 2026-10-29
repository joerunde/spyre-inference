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

"""The Triton gate, checked in fresh interpreters: it acts once, at the first
import of vllm.triton_utils, so this process (whose plugin load has already
closed it) can't show either state."""

import importlib.util
import os
import platform
import subprocess
import sys
import textwrap

import pytest

from spyre_inference.v1.worker.spyre import triton_gate

_needs_triton_cpu = pytest.mark.skipif(
    platform.machine() != "x86_64" or importlib.util.find_spec("triton") is None,
    reason="triton-cpu is x86-only and not installed",
)

_PROBE = textwrap.dedent("""
    import vllm
    import vllm.config.vllm as config_module
    from vllm.config import VllmConfig
    from vllm.config.compilation import CompilationMode
    from vllm.triton_utils import HAS_TRITON, triton
    from vllm.v1.worker.gpu.block_table import _gather_block_tables_kernel

    config = VllmConfig()
    config.compilation_config.mode = CompilationMode.STOCK_TORCH_COMPILE
    print(HAS_TRITON)
    print(type(triton).__name__)
    print(type(_gather_block_tables_kernel).__name__)
    print("stock torch.compile" in config._get_v2_model_runner_unsupported_features())
    print(config_module.HAS_TRITON)
""")


def _probe(**env: str) -> list[str]:
    result = subprocess.run(
        [sys.executable, "-c", _PROBE],
        capture_output=True,
        text=True,
        env={**os.environ, **env},
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.split()[-5:]


@_needs_triton_cpu
@pytest.mark.subprocess_isolated
def test_v2_on_triton_binds_real_triton_and_accepts_stock_compile() -> None:
    assert _probe(SPYRE_MODEL_RUNNER="v2", SPYRE_MRV2_KERNELS="triton") == [
        "True",
        "module",
        "JITFunction",
        "False",
        "True",
    ]


@pytest.mark.subprocess_isolated
def test_v2_on_host_kernels_hides_triton_but_accepts_v2() -> None:
    """Triton stays a placeholder for every vLLM module, yet VllmConfig's
    "Model Runner V2 requires Triton" checks pass and stock compile is allowed."""
    assert _probe(SPYRE_MODEL_RUNNER="v2") == [
        "False",
        "TritonPlaceholder",
        "function",
        "False",
        "True",
    ]


@pytest.mark.subprocess_isolated
@pytest.mark.parametrize("cuda_visible_devices", [None, ""])
def test_v1_stays_triton_free(cuda_visible_devices: str | None) -> None:
    """An empty CUDA_VISIBLE_DEVICES is where vLLM's own check keeps Triton on;
    with triton-cpu installed, so does its single active CPU driver."""
    env = {"SPYRE_MODEL_RUNNER": "v1"}
    if cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    assert _probe(**env) == [
        "False",
        "TritonPlaceholder",
        "function",
        "True",
        "False",
    ]


def test_gate_cannot_switch_once_set() -> None:
    state = triton_gate._state
    by_state = {
        "closed": triton_gate.close,
        "open": triton_gate.open,
        "open_host": triton_gate.open_host,
    }
    same = by_state[state]
    other = by_state["open" if state != "open" else "closed"]
    same()
    with pytest.raises(RuntimeError, match=f"already {state}"):
        other()
