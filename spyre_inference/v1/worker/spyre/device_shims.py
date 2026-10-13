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

"""MRV2: process-wide torch.cuda Stream/Event/stream/current_stream placeholders.

MRV2's GPUModelRunner constructs real torch.cuda.Stream/Event objects
unconditionally (main_stream, the pooling copy_stream/copy_event), and
calls torch.cuda.current_stream() (main_stream) and enters
torch.cuda.stream(...) (async_utils, structured_outputs, spec_decode/utils)
— all of which raise on CPU-only torch. install() swaps in no-op
placeholders for all four, plus torch.cuda.set_stream, which async_utils'
own stream() context manager calls with the placeholder streams.

The runner's own tensors live on the host (device=cpu), so there is no CUDA
stream to order against; a Spyre-to-host copy that may still be in flight
is fenced with spyre_fence() instead.
"""

import contextlib
from typing import Any

import torch


class _Event:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def record(self, stream: Any = None) -> None:
        pass

    def wait(self, stream: Any = None) -> None:
        pass

    def synchronize(self) -> None:
        pass

    def query(self) -> bool:
        return True

    def elapsed_time(self, end_event: "_Event") -> float:
        return 0.0


class _Stream:
    def __init__(self, device: Any = None, *args: Any, **kwargs: Any) -> None:
        self.device = device

    def wait_stream(self, stream: "_Stream") -> None:
        pass

    def wait_event(self, event: _Event) -> None:
        pass

    def record_event(self, event: _Event | None = None) -> _Event:
        return event or _Event()

    def synchronize(self) -> None:
        pass

    def query(self) -> bool:
        return True


_current = _Stream()

_PLACEHOLDERS = {
    "Stream": _Stream,
    "Event": _Event,
    "stream": lambda stream=None: contextlib.nullcontext(stream),
    "current_stream": lambda device=None: _current,
    "set_stream": lambda stream: None,
}

_originals: dict[str, Any] = {}


def install() -> None:
    """Install the torch.cuda placeholders (Stream, Event, stream,
    current_stream, set_stream). Idempotent."""
    if _originals:
        return
    for name, placeholder in _PLACEHOLDERS.items():
        _originals[name] = getattr(torch.cuda, name)
        setattr(torch.cuda, name, placeholder)


def uninstall() -> None:
    """Undo install(), restoring the original torch.cuda attributes."""
    for name, original in _originals.items():
        setattr(torch.cuda, name, original)
    _originals.clear()


def is_installed() -> bool:
    """Whether install() has run and uninstall() hasn't undone it."""
    return bool(_originals)


def spyre_fence() -> None:
    """Replace the CUDA event wait on the pooling output path.

    Pooling's AsyncPoolingOutput copies results to host behind a
    torch.cuda.Event; on Spyre this synchronizes the device directly instead.
    """
    torch.spyre.synchronize()
