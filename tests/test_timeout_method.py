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

"""`uses_subprocess` tests time out by signal, so a timeout fails one test instead of the run."""

import pytest_timeout
from spyre_testing_plugin.pytest_plugin import _signal_timeout_for_subprocess_tests

pytest_plugins = ["pytester"]


def _settings(pytester, source):
    pytester.makeini("[pytest]\ntimeout = 1800\ntimeout_method = thread\n")
    items = pytester.getitems(source)
    _signal_timeout_for_subprocess_tests(items)
    return {item.name: pytest_timeout._get_item_settings(item) for item in items}


def test_subprocess_tests_switch_to_signal_and_keep_their_timeout(pytester):
    settings = _settings(
        pytester,
        """
        import pytest

        def test_plain():
            pass

        @pytest.mark.uses_subprocess
        def test_default():
            pass

        @pytest.mark.uses_subprocess
        @pytest.mark.timeout(3600)
        def test_own_timeout():
            pass

        @pytest.mark.uses_subprocess
        @pytest.mark.timeout(timeout=60, func_only=True)
        def test_kwargs():
            pass

        @pytest.mark.uses_subprocess
        @pytest.mark.timeout(60, method="thread")
        def test_explicit_method():
            pass
        """,
    )
    got = {name: (s.timeout, s.method) for name, s in settings.items()}
    assert got == {
        "test_plain": (1800, "thread"),
        "test_default": (1800, "signal"),
        "test_own_timeout": (3600, "signal"),
        "test_kwargs": (60, "signal"),
        "test_explicit_method": (60, "thread"),
    }
    assert settings["test_kwargs"].func_only is True


def test_module_pytestmark_counts(pytester):
    settings = _settings(
        pytester,
        """
        import pytest

        pytestmark = pytest.mark.uses_subprocess

        def test_in_marked_module():
            pass
        """,
    )
    assert settings["test_in_marked_module"].method == "signal"


def test_signal_timeout_fails_only_that_test(pytester):
    """A timed-out subprocess test fails and the run continues to the next test."""
    pytester.makeini("[pytest]\ntimeout = 2\ntimeout_method = thread\n")
    pytester.makeconftest(
        """
        from spyre_testing_plugin.pytest_plugin import _signal_timeout_for_subprocess_tests

        def pytest_collection_modifyitems(items):
            _signal_timeout_for_subprocess_tests(items)
        """
    )
    pytester.makepyfile(
        """
        import time
        import pytest

        @pytest.mark.uses_subprocess
        def test_hangs():
            time.sleep(30)

        def test_after():
            pass
        """
    )
    result = pytester.runpytest_subprocess()
    result.assert_outcomes(passed=1, failed=1)
