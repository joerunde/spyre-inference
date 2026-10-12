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

"""SpyreWorkerV2.init_device builds a SpyreModelRunnerV2 and never a
TorchSpyreModelRunner, and the copied init_device body hasn't drifted
from TorchSpyreWorker's.
"""

import difflib
import inspect
import re
import sys
import types
from unittest import mock

import pytest
import torch

from spyre_inference.v1.worker.spyre import device_shims, kernels
from spyre_inference.v1.worker.spyre import worker as spyre_worker_module
from spyre_inference.v1.worker.spyre.worker import SpyreWorkerV2
from spyre_inference.v1.worker.spyre_model_runner import TorchSpyreModelRunner
from spyre_inference.v1.worker.spyre_worker import TorchSpyreWorker


class _FakeSpyreDevice:
    """Stands in for torch.device("spyre"): this CPU-only test environment
    has no torch_spyre to register the real device type with torch."""

    type = "spyre"

    def __repr__(self) -> str:
        return "device(type='spyre')"


class _FakeRunner:
    """Stands in for SpyreModelRunnerV2: records what it was built with."""

    def __init__(self, vllm_config: object, device: object) -> None:
        self.vllm_config = vllm_config
        self.device = device


def _make_bare_worker() -> SpyreWorkerV2:
    """A SpyreWorkerV2 with only the attributes init_device reads, skipping
    Worker.__init__ (which needs a full engine, not something tests/mrv2
    may construct — ground rule 2)."""
    worker = SpyreWorkerV2.__new__(SpyreWorkerV2)
    worker.vllm_config = mock.Mock()
    worker.vllm_config.parallel_config.world_size = 1
    worker.rank = 0
    worker.local_rank = 0
    worker.distributed_init_method = "env://"
    worker.model_config = mock.Mock()
    worker.model_config.seed = 0
    return worker


@pytest.fixture
def patched_worker_internals(monkeypatch: pytest.MonkeyPatch) -> None:
    # `import torch_spyre` resolves against this fake rather than the real
    # package, which this CPU-only test environment doesn't have installed.
    fake_torch_spyre = types.ModuleType("torch_spyre")
    fake_torch_spyre._autoload = lambda: None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch_spyre", fake_torch_spyre)
    monkeypatch.setattr(
        torch, "spyre", types.SimpleNamespace(set_device=lambda rank: None), raising=False
    )

    # torch.device("spyre") raises without torch_spyre's real device registration.
    real_torch_device = torch.device

    def _fake_torch_device(device_str: str) -> object:
        if device_str == "spyre":
            return _FakeSpyreDevice()
        return real_torch_device(device_str)

    monkeypatch.setattr(torch, "device", _fake_torch_device)

    monkeypatch.setattr(spyre_worker_module, "_raise_dynamo_recompile_limits", lambda: None)
    monkeypatch.setattr(spyre_worker_module, "register_all", lambda: None)
    monkeypatch.setattr(spyre_worker_module, "register_models", lambda: None)
    monkeypatch.setattr(
        spyre_worker_module, "init_worker_distributed_environment", lambda *a, **k: None
    )
    monkeypatch.setattr(spyre_worker_module, "set_random_seed", lambda seed: None)
    monkeypatch.setattr(device_shims, "install", lambda: None)
    monkeypatch.setattr(kernels, "bind", lambda: set())
    monkeypatch.setattr(spyre_worker_module, "SpyreModelRunnerV2", _FakeRunner)


def test_init_device_builds_spyre_model_runner_v2(patched_worker_internals: None) -> None:
    worker = _make_bare_worker()
    worker.init_device()
    assert isinstance(worker.model_runner, _FakeRunner)
    assert not isinstance(worker.model_runner, TorchSpyreModelRunner)
    assert worker.model_runner.device.type == "spyre"


def test_init_device_calls_device_shims_and_kernels_before_building_the_runner(
    patched_worker_internals: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """GPUModelRunner.__init__ constructs a real torch.cuda.Stream, so
    device_shims.install() must run before the runner, not just at some
    point during init_device."""
    calls: list[str] = []
    monkeypatch.setattr(device_shims, "install", lambda: calls.append("device_shims"))
    monkeypatch.setattr(kernels, "bind", lambda: (calls.append("kernels"), set())[1])

    class _RecordingFakeRunner(_FakeRunner):
        def __init__(self, vllm_config: object, device: object) -> None:
            calls.append("runner")
            super().__init__(vllm_config, device)

    monkeypatch.setattr(spyre_worker_module, "SpyreModelRunnerV2", _RecordingFakeRunner)

    worker = _make_bare_worker()
    worker.init_device()

    assert calls == ["device_shims", "kernels", "runner"]


_TY_IGNORE_RE = re.compile(r"\s*#\s*ty:\s*ignore(\[[\w,-]+\])?\s*$")


def _normalized_body(source: str, *, strip_marker: str | None = None) -> list[str]:
    """init_device's statements, one per line, blank lines and the `def`
    line dropped. strip_marker drops any line carrying that comment too.
    A trailing `# ty: ignore[...]` is stripped from every line: V2 needs
    one where it builds a nominally mismatched runner type (worker.py);
    V1 doesn't, so comparing the suppression itself would be noise."""
    lines = source.splitlines()[1:]
    out = []
    for line in lines:
        if strip_marker and strip_marker in line:
            continue
        stripped = _TY_IGNORE_RE.sub("", line).strip()
        if stripped:
            out.append(stripped)
    return out


def test_init_device_has_not_drifted_from_v1() -> None:
    """The copy in worker.py must match TorchSpyreWorker's line for line,
    apart from the MRV2-only additions and the runner class it builds."""
    v1_body = _normalized_body(inspect.getsource(TorchSpyreWorker.init_device))
    v2_body = _normalized_body(
        inspect.getsource(SpyreWorkerV2.init_device), strip_marker="MRV2-only"
    )
    v2_body = [line.replace("SpyreModelRunnerV2", "TorchSpyreModelRunner") for line in v2_body]

    diff = list(difflib.unified_diff(v1_body, v2_body, lineterm=""))
    assert not diff, (
        "SpyreWorkerV2.init_device has drifted from TorchSpyreWorker.init_device "
        "(apart from the lines marked '# MRV2-only'):\n" + "\n".join(diff)
    )
