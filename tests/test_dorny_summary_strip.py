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

"""CPU-only test for the dorny per-file strip (no hardware needed).

The strip is coupled to dorny/test-reporter's markdown shape, so pin the three
constructs it removes -- the `|Report|…|` table, the `## <file>.xml` headings,
their one-liner summaries -- against a fixture in dorny's real format, and pin
that the badge and failing suite/test detail survive.
"""

import importlib.util
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[1] / ".github" / "scripts" / "strip_dorny_summary.py"
_spec = importlib.util.spec_from_file_location("strip_dorny_summary", _SCRIPT)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
strip = _mod.strip

NBSP = " "

# Two passing shards + one failing shard, in dorny's layout (see upstream
# __tests__/__outputs__/python-xunit-pytest.md). Anchors are user-content-r{N}
# for files, user-content-r{N}s{M} for suites.
REPORT = f"""![Tests failed](https://img.shields.io/badge/tests-critical)
|Report|Passed|Failed|Skipped|Time|
|:---|---:|---:|---:|---:|
|[junit-test-smoke-shard-0.xml](#user-content-r0)|60 ✅||5 ⚪|12s|
|[junit-test-upstream-shard-5.xml](#user-content-r1)|88 ✅|2 ❌|10 ⚪|60s|
## ✅{NBSP}<a id="user-content-r0" href="#user-content-r0">junit-test-smoke-shard-0.xml</a>
**65** tests were completed in **12s** with **60** passed, **0** failed and **5** skipped.
## ❌{NBSP}<a id="user-content-r1" href="#user-content-r1">junit-test-upstream-shard-5.xml</a>
**100** tests were completed in **60s** with **88** passed, **2** failed and **10** skipped.
|Test suite|Passed|Failed|Skipped|Time|
|:---|---:|---:|---:|---:|
|[pytest](#user-content-r1s0)|88 ✅|2 ❌|10 ⚪|60s|
### ❌{NBSP}<a id="user-content-r1s0" href="#user-content-r1s0">pytest</a>
```
tests.entrypoints.test_chat
  ❌ test_chat_completion[case1]
	assert 500 == 200
```
"""


def test_strip_removes_per_file_clutter():
    out = strip(REPORT)

    # Per-shard xml names gone: no Report table, no `## <file>.xml` headings.
    assert "junit-test-smoke-shard-0.xml" not in out
    assert "junit-test-upstream-shard-5.xml" not in out
    assert "|Report|" not in out
    assert "tests were completed" not in out

    # Badge and the failing suite/test detail survive.
    assert out.startswith("![Tests failed]")
    assert "|Test suite|" in out
    assert "### ❌" in out
    assert "test_chat_completion[case1]" in out
    assert "assert 500 == 200" in out


def test_all_green_leaves_only_badge():
    badge = "![Tests passed](https://img.shields.io/badge/tests-success)"
    green = (
        f"{badge}\n"
        "|Report|Passed|Failed|Skipped|Time|\n"
        "|:---|---:|---:|---:|---:|\n"
        "|[junit-a.xml](#user-content-r0)|60 ✅|||12s|\n"
        f'## ✅{NBSP}<a id="user-content-r0" href="#user-content-r0">junit-a.xml</a>\n'
        "**60** tests were completed in **12s** with **60** passed, **0** failed.\n"
    )
    assert strip(green) == badge
