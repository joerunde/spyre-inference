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

"""Guards for tests/mrv2: an autouse fixture makes torch.compile() and
vllm.LLM.__init__() raise, and a time budget (1s/test, ~20s/directory,
exempting tests marked ``triton_cpu`` or ``subprocess_isolated``) fails
the run if exceeded.

Only applies to items collected under tests/mrv2, so a bare `pytest` from
the repo root never has the budget hook constrain the rest of the suite.
"""

import pathlib

import pytest
import torch
import vllm

_MRV2_DIR = pathlib.Path(__file__).parent
_PER_TEST_BUDGET_S = 1.0
_TOTAL_BUDGET_S = 20.0

_total_duration = 0.0
_total_budget_reported = False


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "triton_cpu: JIT-compiles Triton on a cold cache (seconds per kernel); "
        "exempt from the tests/mrv2 time budget",
    )
    config.addinivalue_line(
        "markers",
        "subprocess_isolated: spawns a fresh interpreter so a before/after check "
        "isn't vacuous from an already-warm module cache; exempt from the budget",
    )


@pytest.fixture(autouse=True)
def _no_compile_no_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    """Raises only when a compiled function is actually called, not when
    torch.compile() decorates one at import time: upstream's own
    sample/ops/topk_topp_sampler.py applies @torch.compile(dynamic=True) at
    module scope, so a decoration-time raise would break importing it."""

    def _raise_on_call(*args: object, **kwargs: object) -> None:
        raise RuntimeError("tests/mrv2 must not call a torch.compile()-wrapped function")

    def _fake_compile(*args: object, **kwargs: object):
        if args and callable(args[0]):
            return _raise_on_call
        return lambda fn: _raise_on_call

    def _raise_llm_init(*args: object, **kwargs: object) -> None:
        raise RuntimeError("tests/mrv2 must not construct vllm.LLM(); use fakes instead")

    monkeypatch.setattr(torch, "compile", _fake_compile)
    monkeypatch.setattr(vllm.LLM, "__init__", _raise_llm_init)


def _is_mrv2_item(item: pytest.Item) -> bool:
    return _MRV2_DIR in item.path.parents


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo):
    """Fails the offending test's own report on a budget overrun, rather
    than only flipping the session exit code: a shard's JUnit XML must show
    which test went over, not every test green next to a red job."""
    outcome = yield
    report = outcome.get_result()
    if report.when != "call" or not _is_mrv2_item(item):
        return
    if "triton_cpu" in item.keywords or "subprocess_isolated" in item.keywords:
        return

    global _total_duration, _total_budget_reported
    _total_duration += report.duration

    if report.duration > _PER_TEST_BUDGET_S:
        report.outcome = "failed"
        report.longrepr = (
            f"{item.nodeid} took {report.duration:.2f}s "
            f"(tests/mrv2 budget is {_PER_TEST_BUDGET_S}s per test)"
        )
    elif _total_duration > _TOTAL_BUDGET_S and not _total_budget_reported:
        _total_budget_reported = True
        report.outcome = "failed"
        report.longrepr = (
            f"tests/mrv2 took {_total_duration:.2f}s total "
            f"(budget is {_TOTAL_BUDGET_S}s for the directory)"
        )
