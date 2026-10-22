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

"""MRV2: set vllm.triton_utils.HAS_TRITON per runner by patching the two names
vllm/triton_utils/importing.py reads. register() can run inside that import (it
resolves the platform), so each patch is also applied to the half-executed module.
"""

import importlib.util
import platform
import sys
from collections.abc import Callable
from typing import Any

import vllm.platforms

_IMPORTING = "vllm.triton_utils.importing"


def _patch(module: Any, name: str, wrap: Callable[[Any], Any]) -> None:
    # Once: vLLM calls register() on every plugin probe.
    original = getattr(module, name)
    if getattr(original, "_spyre_triton_gate", False):
        return
    replacement = wrap(original)
    replacement._spyre_triton_gate = True
    setattr(module, name, replacement)
    importing = sys.modules.get(_IMPORTING)
    if importing is not None and hasattr(importing, name):
        setattr(importing, name, replacement)


def open() -> None:
    """V2: report a CPU build, so importing.py takes vLLM's own triton-cpu branch."""

    def wrap(matches: Callable[[str], bool]) -> Callable[[str], bool]:
        return lambda substr: substr == "cpu" or matches(substr)

    _patch(vllm.platforms, "vllm_version_matches_substr", wrap)


def close() -> None:
    """V1: hide triton from importing.py only; torch has already imported it, and
    triton-cpu's CPU driver would otherwise pass vLLM's check."""

    def wrap(find_spec: Callable[..., Any]) -> Callable[..., Any]:
        def hidden(name: str, package: str | None = None) -> Any:
            if name == "triton" and sys._getframe(1).f_globals.get("__name__") == _IMPORTING:
                return None
            return find_spec(name, package)

        return hidden

    _patch(importlib.util, "find_spec", wrap)


def has_triton_cpu() -> bool:
    if platform.machine() != "x86_64" or importlib.util.find_spec("triton") is None:
        return False
    try:
        return importlib.util.find_spec("triton.backends.cpu") is not None
    except ImportError:
        return False


def check(runner: str) -> None:
    """Raise unless HAS_TRITON matches runner. V2 without triton-cpu (Power, Z) gets past
    vLLM's "requires Triton" check, with placeholder kernels until kernels.py binds some."""
    from vllm.triton_utils import HAS_TRITON

    want = runner == "v2" and has_triton_cpu()
    if want != HAS_TRITON:
        raise ValueError(
            f"SPYRE_MODEL_RUNNER={runner} needs HAS_TRITON={want}, but vLLM resolved "
            f"{HAS_TRITON}: set SPYRE_MODEL_RUNNER before vLLM is imported, and the same "
            "in every process."
        )
    if runner == "v2" and not HAS_TRITON:
        from vllm.config import VllmConfig

        VllmConfig._validate_v2_model_runner = _validate_v2_without_triton  # ty: ignore[invalid-assignment]


def _validate_v2_without_triton(self: Any) -> None:
    # VllmConfig._validate_v2_model_runner minus its HAS_TRITON check.
    unsupported = self._get_v2_model_runner_unsupported_features()
    if unsupported:
        raise ValueError(f"Model Runner V2 does not yet support: {', '.join(unsupported)}")
