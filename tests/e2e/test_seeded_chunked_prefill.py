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

"""A seeded request must not depend on how its prompt is split into prefill chunks (#1420)."""

import gc
import os

import pytest
from spyre_testing_plugin.pytest_plugin import spyre_hardware_present
from spyre_testing_plugin.vfio_reaper import wait_until_card_free

pytestmark = pytest.mark.uses_subprocess

_MODEL = "ibm-ai-platform/micro-g3.3-8b-instruct-1b"
# Three 128-token chunks, so two samples are discarded before the request decodes.
_PROMPT_TOKENS = list(range(1000, 1376))
_SEEDS = (0, 1, 2)


def _generate(max_num_batched_tokens):
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    engine = LLM(
        model=_MODEL,
        max_model_len=512,
        max_num_seqs=2,
        max_num_batched_tokens=max_num_batched_tokens,
    )
    try:
        prompt = TokensPrompt(prompt_token_ids=_PROMPT_TOKENS)
        outputs = []
        for seed in _SEEDS:
            params = SamplingParams(max_tokens=8, temperature=1.0, seed=seed, ignore_eos=True)
            output = engine.generate(prompt, params, use_tqdm=False)[0].outputs[0]
            outputs.append(list(output.token_ids))
        return outputs
    finally:
        engine.llm_engine.engine_core.shutdown(timeout=60)
        del engine
        gc.collect()
        wait_until_card_free(exclude_pids={os.getpid()}, timeout=60)


def test_seeded_request_matches_across_prefill_chunking(monkeypatch):
    if not spyre_hardware_present():
        pytest.skip("Requires a Spyre accelerator")
    monkeypatch.setenv("VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS", "36000")
    chunked = _generate(128)
    assert all(len(tokens) == 8 for tokens in chunked)
    assert len({tuple(tokens) for tokens in chunked}) == len(_SEEDS)
    assert chunked == _generate(512)
