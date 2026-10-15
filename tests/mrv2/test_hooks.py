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

"""The upstream names load.py and kv_cache.py swap are restored after the call,
including when it raises, so nothing leaks into later upstream use."""

from types import SimpleNamespace

import pytest
from vllm.v1.worker.gpu import model_runner as upstream

from spyre_inference.v1.worker.spyre import kv_cache, load

_NAMES = (
    "get_model_loader",
    "init_model_state",
    "PoolingRunner",
    "init_attn_backend",
    "init_kv_cache",
)


def _snapshot() -> dict[str, object]:
    return {name: getattr(upstream, name) for name in _NAMES}


@pytest.mark.parametrize("module", [load, kv_cache], ids=["load", "kv_cache"])
def test_swapped_names_are_restored_after_a_raise(module) -> None:
    before = _snapshot()
    with pytest.raises(RuntimeError, match="boom"), module._patched(SimpleNamespace()):
        assert _snapshot() != before
        raise RuntimeError("boom")
    assert _snapshot() == before
