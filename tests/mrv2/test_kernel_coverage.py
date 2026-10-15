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

"""DEFAULT_PATH_KERNELS matches every @triton.jit in the pinned vLLM's default-path
modules, so a renamed or added kernel fails here on the bump, not mid-serving."""

import importlib.util
import re
from pathlib import Path

from spyre_inference.v1.worker.spyre import kernels

_JIT = re.compile(r"^@triton\.jit[^\n]*\n(?:@[^\n]*\n)*def ([A-Za-z_0-9]+)", re.M)


def _jit_names_in(module: str) -> set[str]:
    spec = importlib.util.find_spec(f"{kernels._PACKAGE}.{module}")
    assert spec is not None and spec.origin is not None, module
    return {f"{module}.{name}" for name in _JIT.findall(Path(spec.origin).read_text())}


def test_default_path_kernels_match_the_pinned_vllm() -> None:
    modules = {name.rpartition(".")[0] for name in kernels.DEFAULT_PATH_KERNELS}
    found = set().union(*(_jit_names_in(m) for m in modules))
    assert found == kernels.DEFAULT_PATH_KERNELS, (
        f"unlisted: {sorted(found - kernels.DEFAULT_PATH_KERNELS)}, "
        f"gone: {sorted(kernels.DEFAULT_PATH_KERNELS - found)}"
    )
