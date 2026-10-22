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

"""The Triton gate, in fresh interpreters: vllm.triton_utils binds HAS_TRITON once."""

import importlib.util
import os
import subprocess
import sys
import textwrap
import types

import pytest

from spyre_inference.v1.worker.spyre import triton_gate

_HAS_TRITON_CPU = triton_gate.has_triton_cpu()

# What runs before the first vllm.triton_utils import. The first two resolve the
# platform, and so run register(), inside that import.
_FIRST_IMPORTS = {
    "bare_import": "import vllm",
    "triton_utils_first": "",
    "vllm_config": "from vllm.config import VllmConfig",
    "llm": "from vllm import LLM",
    "current_platform": "from vllm.platforms import current_platform\ncurrent_platform.device_type",
}

_PROBE = """
{first}
from vllm.triton_utils import HAS_TRITON, triton
from vllm.v1.worker.gpu.block_table import _gather_block_tables_kernel

print(HAS_TRITON, type(triton).__name__, type(_gather_block_tables_kernel).__name__)
"""


def _run(script: str, **env: str) -> list[str]:
    # Without the VLLM_USE_V2_MODEL_RUNNER an earlier in-process VllmConfig pinned.
    base = {k: v for k, v in os.environ.items() if k != "VLLM_USE_V2_MODEL_RUNNER"}
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        capture_output=True,
        text=True,
        env={**base, **env},
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.split()


def _probe(first: str, **env: str) -> list[str]:
    return _run(_PROBE.format(first=first), **env)[-3:]


@pytest.mark.subprocess_isolated
@pytest.mark.parametrize("cuda_visible_devices", [None, ""])
@pytest.mark.parametrize("first", _FIRST_IMPORTS)
def test_v1_stays_triton_free(first: str, cuda_visible_devices: str | None) -> None:
    """An empty CUDA_VISIBLE_DEVICES is where vLLM's own check keeps Triton on."""
    env = {"SPYRE_MODEL_RUNNER": "v1"}
    if cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    assert _probe(_FIRST_IMPORTS[first], **env) == ["False", "TritonPlaceholder", "function"]


@pytest.mark.subprocess_isolated
@pytest.mark.parametrize("first", _FIRST_IMPORTS)
def test_v2_binds_real_triton_when_triton_cpu_is_installed(first: str) -> None:
    want = (
        ["True", "module", "JITFunction"]
        if _HAS_TRITON_CPU
        else ["False", "TritonPlaceholder", "function"]
    )
    assert _probe(_FIRST_IMPORTS[first], SPYRE_MODEL_RUNNER="v2") == want


@pytest.mark.subprocess_isolated
def test_v2_config_builds_without_triton() -> None:
    """Power and Z: vLLM's "Model Runner V2 requires Triton" must not refuse V2."""
    out = _run(
        """
        import importlib.util

        real = importlib.util.find_spec
        importlib.util.find_spec = lambda name, *a: None if name == "triton" else real(name, *a)

        from vllm.config import VllmConfig

        config = VllmConfig()
        print(config.use_v2_model_runner, config.parallel_config.worker_cls)
        """,
        SPYRE_MODEL_RUNNER="v2",
    )
    assert out[-2:] == ["True", "spyre_inference.v1.worker.spyre.worker.SpyreWorkerV2"]


def test_v2_validation_keeps_the_unsupported_feature_check() -> None:
    config = types.SimpleNamespace(
        _get_v2_model_runner_unsupported_features=lambda: ["pipeline parallelism"]
    )
    with pytest.raises(ValueError, match="does not yet support: pipeline parallelism"):
        triton_gate._validate_v2_without_triton(config)


def test_close_hides_triton_only_from_vllm_importing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, package=None: name)
    triton_gate.close()
    find_spec = importlib.util.find_spec

    lookup = "find_spec('triton'), find_spec('torch')"
    assert eval(lookup, {"__name__": triton_gate._IMPORTING, "find_spec": find_spec}) == (
        None,
        "torch",
    )
    assert eval(lookup, {"__name__": "other", "find_spec": find_spec}) == ("triton", "torch")


def test_patches_apply_once(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, package=None: name)
    triton_gate.close()
    once = importlib.util.find_spec
    triton_gate.close()
    assert importlib.util.find_spec is once


def test_patches_reach_a_half_imported_importing_module(monkeypatch: pytest.MonkeyPatch) -> None:
    half = types.ModuleType(triton_gate._IMPORTING)
    half.find_spec = importlib.util.find_spec
    monkeypatch.setitem(sys.modules, triton_gate._IMPORTING, half)
    monkeypatch.setattr(importlib.util, "find_spec", importlib.util.find_spec)

    triton_gate.close()

    assert half.find_spec is importlib.util.find_spec
