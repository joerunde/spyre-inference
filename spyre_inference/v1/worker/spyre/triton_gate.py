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

"""MRV2: scoped bypass of vllm.triton_utils.HAS_TRITON / VllmConfig._validate_v2_model_runner.

vllm.triton_utils binds real triton/tl or placeholders once, at first
import, from HAS_TRITON, and MRV2 decorates its kernels with @triton.jit at
that same import, so the gate has to act before vllm.triton_utils is first
imported in the process. spyre_inference.register() calls open() or close()
from platform resolution, which vLLM runs during `import vllm` in every
process, ahead of that import. Both install a meta-path hook that rewrites
the result of vllm/triton_utils/importing.py as it is executed.

open() (V2 on triton-cpu): on x86 HAS_TRITON is True whenever triton-cpu is
installed, bypassing vLLM's `cpu`-in-version-string and one-active-driver
checks, and VllmConfig stops listing "stock torch.compile" as unsupported by MRV2.
open_host() (V2 on the host kernels): vLLM never sees triton, as under close(),
but VllmConfig's two "Model Runner V2 requires Triton" checks pass, since
kernels.bind() replaces every kernel MRV2 launches.
close() (V1): vLLM never sees triton, so V1 stays Triton-free even when
triton-cpu is installed, whose CPU driver vLLM's stock check would accept.
"""

import contextlib
import importlib.abc
import importlib.machinery
import importlib.util
import platform
import sys
from collections.abc import Callable, Iterator
from types import ModuleType
from typing import Any

_IMPORTING = "vllm.triton_utils.importing"
_CONFIG = "vllm.config.vllm"

_state: str | None = None
_hooks: dict[str, Callable[[ModuleType], contextlib.AbstractContextManager[None]]] = {}


class _HookLoader(importlib.abc.Loader):
    def __init__(self, loader: importlib.abc.Loader, hook: Callable[..., Any]) -> None:
        self._loader = loader
        self._hook = hook

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> ModuleType | None:
        return self._loader.create_module(spec)

    def exec_module(self, module: ModuleType) -> None:
        with self._hook(module):
            self._loader.exec_module(module)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._loader, name)


class _HookFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path: Any, target: Any = None) -> Any:
        hook = _hooks.pop(fullname, None)
        if hook is None:
            return None
        for finder in sys.meta_path[sys.meta_path.index(self) + 1 :]:
            spec = finder.find_spec(fullname, path, target)
            if spec is not None:
                if spec.loader is not None:
                    spec.loader = _HookLoader(spec.loader, hook)
                return spec
        return None


def _has_triton_cpu() -> bool:
    if platform.machine() != "x86_64" or importlib.util.find_spec("triton") is None:
        return False
    return importlib.util.find_spec("triton.backends.cpu") is not None


@contextlib.contextmanager
def _accept_triton_cpu(module: ModuleType) -> Iterator[None]:
    yield
    module.HAS_TRITON = module.HAS_TRITON or _has_triton_cpu()


@contextlib.contextmanager
def _hide_triton(module: ModuleType) -> Iterator[None]:
    find_spec = importlib.util.find_spec
    importlib.util.find_spec = lambda name, package=None: (  # ty: ignore[invalid-assignment]
        None if name in ("triton", "pytorch-triton-xpu") else find_spec(name, package)
    )
    try:
        yield
    finally:
        importlib.util.find_spec = find_spec
        module.find_spec = find_spec


@contextlib.contextmanager
def _drop_compile_from_unsupported(module: ModuleType) -> Iterator[None]:
    yield
    original = module.VllmConfig._get_v2_model_runner_unsupported_features

    def unsupported(self: Any) -> list[str]:
        return [f for f in original(self) if f != "stock torch.compile"]

    module.VllmConfig._get_v2_model_runner_unsupported_features = unsupported


@contextlib.contextmanager
def _host_kernels_satisfy_v2(module: ModuleType) -> Iterator[None]:
    with _drop_compile_from_unsupported(module):
        yield
    # Only VllmConfig's V2 gating reads this module's HAS_TRITON; triton_utils keeps
    # its placeholder, so no other vLLM code takes a Triton path.
    module.HAS_TRITON = True


def _apply(
    state: str, hooks: dict[str, Callable[..., Any]], has_triton: Callable[[], bool]
) -> None:
    global _state
    if _state == state:
        return
    if _state is not None:
        raise RuntimeError(f"Triton gate is already {_state}; cannot switch it to {state}")
    importing = sys.modules.get(_IMPORTING)
    if importing is not None and has_triton() != importing.HAS_TRITON:
        raise RuntimeError(
            f"{_IMPORTING} was imported before the Spyre plugin loaded, so the Triton gate "
            f"cannot be {state} in this process"
        )
    _state = state
    for name, hook in hooks.items():
        if name in sys.modules:
            with hook(sys.modules[name]):
                pass
        else:
            _hooks[name] = hook
    if _hooks and not any(isinstance(f, _HookFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, _HookFinder())


def open() -> None:
    """Make HAS_TRITON / _validate_v2_model_runner accept triton-cpu on x86."""
    _apply(
        "open",
        {_IMPORTING: _accept_triton_cpu, _CONFIG: _drop_compile_from_unsupported},
        _has_triton_cpu,
    )


def open_host() -> None:
    """Hide triton from vLLM, but let VllmConfig accept the V2 runner."""
    _apply(
        "open_host",
        {_IMPORTING: _hide_triton, _CONFIG: _host_kernels_satisfy_v2},
        lambda: False,
    )


def close() -> None:
    """Keep vLLM Triton-free for the V1 runner, whatever triton is installed."""
    _apply("closed", {_IMPORTING: _hide_triton}, lambda: False)
