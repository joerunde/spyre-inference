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

"""The #1420 port's exit condition and error paths; tests/e2e covers serving."""

import inspect
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

from spyre_inference.v1.worker.spyre_model_runner import TorchSpyreModelRunner


@pytest.fixture()
def runner():
    # Request 0 is a seeded unfinished prefill, 1 a seeded decode.
    runner = object.__new__(TorchSpyreModelRunner)
    generators = {i: torch.Generator().manual_seed(37 + i) for i in range(2)}
    runner.input_batch = SimpleNamespace(
        num_reqs=2,
        generators=generators,
        sampling_metadata=SimpleNamespace(generators=generators),
    )
    runner.discard_request_mask = SimpleNamespace(np=np.array([True, False, False]))
    return runner


def test_upstream_rewind_still_needs_this_port():
    """Fails once vLLM restores CPU generators itself (vllm#62688): then drop the port."""
    assert "gen.set_offset(gen.get_offset() - 4)" in inspect.getsource(
        GPUModelRunner._bookkeeping_sync
    )
    assert "get_state" not in inspect.getsource(GPUModelRunner._sample)
    with pytest.raises(RuntimeError, match="does not use offset"):
        torch.Generator().set_offset(0)


def test_sample_restores_discarded_generator_on_error(runner, monkeypatch):
    generators = runner.input_batch.generators
    before = generators[0].get_state()

    def fail(self, *args):
        for gen in generators.values():
            torch.rand(1, generator=gen)
        raise RuntimeError("sampling failed")

    monkeypatch.setattr(GPUModelRunner, "_sample", fail)
    with pytest.raises(RuntimeError, match="sampling failed"):
        runner._sample(None, None)
    assert torch.equal(generators[0].get_state(), before)


def test_bookkeeping_restores_generator_mapping_on_error(runner, monkeypatch):
    generators = runner.input_batch.generators

    def fail(self, *args):
        assert self.input_batch.generators == {1: generators[1]}
        assert self.input_batch.sampling_metadata.generators is generators
        raise RuntimeError("bookkeeping failed")

    monkeypatch.setattr(GPUModelRunner, "_bookkeeping_sync", fail)
    with pytest.raises(RuntimeError, match="bookkeeping failed"):
        runner._bookkeeping_sync(None, None, None, None, 0)
    assert runner.input_batch.generators is generators
