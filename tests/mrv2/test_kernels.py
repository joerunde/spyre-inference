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

"""kernels.bind_or_raise() actually enforces the diff it claims to."""

import pytest

from spyre_inference.v1.worker.spyre import kernels


def test_bind_or_raise_names_a_kernel_bind_left_unbound(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(kernels, "DEFAULT_PATH_KERNELS", frozenset({"a", "b"}))
    monkeypatch.setattr(kernels, "bind", lambda: {"a"})

    with pytest.raises(RuntimeError, match=r"\['b'\]"):
        kernels.bind_or_raise()


def test_bind_or_raise_is_a_noop_once_everything_is_bound(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(kernels, "DEFAULT_PATH_KERNELS", frozenset({"a", "b"}))
    monkeypatch.setattr(kernels, "bind", lambda: {"a", "b"})

    kernels.bind_or_raise()


@pytest.mark.parametrize("use_k", [True, False])
@pytest.mark.parametrize("use_p", [True, False])
def test_sort_free_top_k_top_p_keeps_the_same_tokens(use_k: bool, use_p: bool) -> None:
    import torch
    from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p_pytorch

    torch.manual_seed(0)
    logits = torch.randn(4, 1000)
    k = torch.tensor([1, 5, 50, 1000]) if use_k else None
    p = torch.tensor([0.5, 0.9, 0.99, 1.0]) if use_p else None
    expected = apply_top_k_top_p_pytorch(logits.clone(), k, p)
    actual = kernels.spyre_apply_top_k_top_p(logits.clone(), k, p)
    assert torch.equal(torch.isfinite(actual), torch.isfinite(expected))
    assert torch.equal(actual[torch.isfinite(actual)], expected[torch.isfinite(expected)])
