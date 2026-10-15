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

"""SPYRE_MODEL_RUNNER picks the worker and turns on vLLM's V2 runner flag."""

import os

import pytest
from vllm.config import VllmConfig

from spyre_inference import envs
from spyre_inference.platform import TorchSpyrePlatform
from spyre_inference.v1.worker.spyre import triton_gate


@pytest.fixture
def vllm_config(monkeypatch: pytest.MonkeyPatch) -> VllmConfig:
    """Built before the test sets SPYRE_MODEL_RUNNER, under this process's real gate."""
    config = VllmConfig()
    config.parallel_config.worker_cls = "auto"
    # setenv first, so teardown also removes a value the platform hook adds.
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "")
    monkeypatch.delenv("VLLM_USE_V2_MODEL_RUNNER")
    return config


def _set_runner(monkeypatch: pytest.MonkeyPatch, runner: str | None, gate: str | None) -> None:
    if runner is None:
        monkeypatch.delenv("SPYRE_MODEL_RUNNER", raising=False)
    else:
        monkeypatch.setenv("SPYRE_MODEL_RUNNER", runner)
    envs.clear_env_cache()
    monkeypatch.setattr(triton_gate, "_state", gate)


@pytest.mark.parametrize(
    ("runner", "worker_cls", "use_v2"),
    [
        (None, "spyre_inference.v1.worker.spyre_worker.TorchSpyreWorker", None),
        ("v1", "spyre_inference.v1.worker.spyre_worker.TorchSpyreWorker", None),
        ("v2", "spyre_inference.v1.worker.spyre.worker.SpyreWorkerV2", "1"),
    ],
)
def test_runner_switch(
    monkeypatch: pytest.MonkeyPatch,
    vllm_config: VllmConfig,
    runner: str | None,
    worker_cls: str,
    use_v2: str | None,
) -> None:
    _set_runner(monkeypatch, runner, "open" if runner == "v2" else "closed")

    TorchSpyrePlatform.check_and_update_config(vllm_config)

    assert vllm_config.parallel_config.worker_cls == worker_cls
    assert os.environ.get("VLLM_USE_V2_MODEL_RUNNER") == use_v2


def test_unknown_runner_is_rejected(
    monkeypatch: pytest.MonkeyPatch, vllm_config: VllmConfig
) -> None:
    _set_runner(monkeypatch, "v3", "closed")

    with pytest.raises(ValueError, match="SPYRE_MODEL_RUNNER='v3' is not v1 or v2"):
        TorchSpyrePlatform.check_and_update_config(vllm_config)


@pytest.mark.parametrize(("runner", "gate"), [("v2", "closed"), ("v1", "open"), ("v1", None)])
def test_gate_must_match_the_runner(
    monkeypatch: pytest.MonkeyPatch, vllm_config: VllmConfig, runner: str, gate: str | None
) -> None:
    """E.g. SPYRE_MODEL_RUNNER set after `import vllm`, or a gate error at plugin load."""
    _set_runner(monkeypatch, runner, gate)

    with pytest.raises(ValueError, match="needs the Triton gate"):
        TorchSpyrePlatform.check_and_update_config(vllm_config)


@pytest.mark.parametrize(("runner", "use_v2"), [("v2", "0"), ("v1", "1")])
def test_conflicting_vllm_use_v2_model_runner_is_rejected(
    monkeypatch: pytest.MonkeyPatch, vllm_config: VllmConfig, runner: str, use_v2: str
) -> None:
    _set_runner(monkeypatch, runner, "open" if runner == "v2" else "closed")
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", use_v2)

    with pytest.raises(ValueError, match=f"VLLM_USE_V2_MODEL_RUNNER={use_v2} conflicts"):
        TorchSpyrePlatform.check_and_update_config(vllm_config)
