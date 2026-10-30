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

"""A seeded request split across prefill chunks must not crash or shift its draws (#1420)."""

import inspect
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from vllm.v1.outputs import SamplerOutput
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

from spyre_inference.v1.worker.spyre_model_runner import TorchSpyreModelRunner


@pytest.fixture()
def runner():
    # Request 0 is a seeded unfinished prefill, 1 a seeded decode, 2 an unseeded prefill.
    runner = object.__new__(TorchSpyreModelRunner)
    generators = {i: torch.Generator().manual_seed(37 + i) for i in range(2)}
    runner.input_batch = SimpleNamespace(
        num_reqs=3,
        req_ids=["prefill", "decode", "unseeded_prefill"],
        req_id_to_index={"prefill": 0, "decode": 1, "unseeded_prefill": 2},
        generators=generators,
        sampling_metadata=SimpleNamespace(generators=generators),
        update_async_output_token_ids=MagicMock(),
        vocab_size=10,
        num_tokens_no_spec=np.array([2, 2, 2]),
        token_ids_cpu=np.zeros((3, 8), dtype=np.int64),
        is_token_ids=np.zeros((3, 8), dtype=bool),
    )
    runner.discard_request_mask = SimpleNamespace(np=np.array([True, False, True, True]))
    runner.use_async_scheduling = False
    runner.max_model_len = 8
    runner.requests = {
        req_id: SimpleNamespace(output_token_ids=[]) for req_id in runner.input_batch.req_ids
    }
    return runner


def test_upstream_rewind_still_needs_this_port():
    """Fails once vLLM stops offset-rewinding every generator: then drop the port (#1420)."""
    assert "gen.set_offset(gen.get_offset() - 4)" in inspect.getsource(
        GPUModelRunner._bookkeeping_sync
    )
    with pytest.raises(RuntimeError, match="does not use offset"):
        torch.Generator().set_offset(0)


@pytest.mark.parametrize("speculative", [False, True])
@pytest.mark.parametrize("raises", [False, True])
def test_discarded_prefill_does_not_advance_cpu_rng(runner, speculative, raises):
    generators = runner.input_batch.generators
    original_states = {i: gen.get_state() for i, gen in generators.items()}
    result = object()
    num_draws = 17 if speculative else 1

    def sample(*args, **kwargs):
        for gen in runner.input_batch.sampling_metadata.generators.values():
            torch.rand(num_draws, generator=gen)
        if raises:
            raise RuntimeError("sampling failed")
        return result

    runner.sampler = MagicMock(side_effect=sample)
    runner.rejection_sampler = MagicMock(side_effect=sample)
    runner._get_spec_decode_draft_probs = MagicMock(return_value=None)
    metadata = object() if speculative else None
    if raises:
        with pytest.raises(RuntimeError, match="sampling failed"):
            runner._sample(None, metadata)
    else:
        assert runner._sample(None, metadata) is result
    assert runner.sampler.call_count == int(not speculative)
    assert runner.rejection_sampler.call_count == int(speculative)
    assert torch.equal(generators[0].get_state(), original_states[0])
    assert not torch.equal(generators[1].get_state(), original_states[1])
    assert runner.input_batch.generators is generators
    assert runner.input_batch.sampling_metadata.generators is generators

    # Completing the prefill must start at the seed's first draw; live requests
    # continue from their next draw rather than repeating the previous one.
    runner.discard_request_mask.np[:3] = False
    for i, gen in generators.items():
        reference = torch.Generator()
        reference.set_state(original_states[i])
        if i == 1:
            torch.rand(num_draws, generator=reference)
        torch.testing.assert_close(
            torch.rand(num_draws, generator=gen),
            torch.rand(num_draws, generator=reference),
            rtol=0,
            atol=0,
        )


@pytest.mark.parametrize("speculative", [False, True])
def test_bookkeeping_discards_prefill_without_cuda_offsets(runner, speculative):
    generators = runner.input_batch.generators
    original_states = {i: gen.get_state() for i, gen in generators.items()}
    runner._to_list = lambda tokens: tokens.tolist()
    runner._get_prompt_logprobs_dict = MagicMock(return_value={})
    token_ids = [[5, 9, -1], [6, 7, -1], [7, 9, -1]] if speculative else [[5], [6], [7]]
    output = SamplerOutput(sampled_token_ids=torch.tensor(token_ids), logprobs_tensors=None)
    scheduler = SimpleNamespace(num_scheduled_tokens=dict.fromkeys(runner.input_batch.req_ids, 1))
    result = runner._bookkeeping_sync(scheduler, output, None, torch.empty(3, 1), 3)
    expected = [6, 7] if speculative else [6]
    assert result[3] == [[], expected, []]
    assert runner.requests["decode"].output_token_ids == expected
    assert runner.requests["prefill"].output_token_ids == []
    assert runner.requests["unseeded_prefill"].output_token_ids == []
    assert runner.input_batch.generators is generators
    assert runner.input_batch.sampling_metadata.generators is generators
    for i, gen in generators.items():
        assert torch.equal(gen.get_state(), original_states[i])


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
