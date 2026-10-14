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


@pytest.mark.parametrize(
    ("runner", "worker_cls", "use_v2"),
    [
        (None, "spyre_inference.v1.worker.spyre_worker.TorchSpyreWorker", "0"),
        ("v1", "spyre_inference.v1.worker.spyre_worker.TorchSpyreWorker", "0"),
        ("v2", "spyre_inference.v1.worker.spyre.worker.SpyreWorkerV2", "1"),
    ],
)
def test_runner_switch(
    monkeypatch: pytest.MonkeyPatch, runner: str | None, worker_cls: str, use_v2: str | None
) -> None:
    # Built before the switch is set: this process's Triton gate is closed, so a V2
    # VllmConfig would fail its own MRV2 validation here.
    vllm_config = VllmConfig()
    vllm_config.parallel_config.worker_cls = "auto"
    if runner is None:
        monkeypatch.delenv("SPYRE_MODEL_RUNNER", raising=False)
    else:
        monkeypatch.setenv("SPYRE_MODEL_RUNNER", runner)
    # setenv first, so teardown also removes a value the platform hook adds.
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "")
    monkeypatch.delenv("VLLM_USE_V2_MODEL_RUNNER")
    monkeypatch.setattr(envs, "_cache", {})

    TorchSpyrePlatform.check_and_update_config(vllm_config)

    assert vllm_config.parallel_config.worker_cls == worker_cls
    assert os.environ.get("VLLM_USE_V2_MODEL_RUNNER") == use_v2


def test_unknown_runner_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    vllm_config = VllmConfig()
    monkeypatch.setenv("SPYRE_MODEL_RUNNER", "v3")
    monkeypatch.setattr(envs, "_cache", {})

    with pytest.raises(ValueError, match="SPYRE_MODEL_RUNNER='v3' is not v1 or v2"):
        TorchSpyrePlatform.check_and_update_config(vllm_config)
