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

"""TEMPORARY (PR #1323 review): size the CPU C++ kernels in a restored compile cache
and time rebuilding them from their .cpp sources with Inductor's own compiler."""

import sys
import time
from pathlib import Path

from torch._inductor.codecache import CppCodeCache

root = Path(sys.argv[1])
sos = sorted(p for p in root.rglob("*.so") if "inductor-spyre" not in p.parts)
print(f"DIAG cpp: {len(sos)} .so files, {sum(p.stat().st_size for p in sos) / 1e6:.1f} MB")
total, rebuilt = 0.0, 0
for so in sos:
    cpp = so.with_suffix(".cpp")
    if not cpp.is_file():
        continue
    so.unlink()
    CppCodeCache.cache.clear()
    start = time.monotonic()
    try:
        CppCodeCache.load(cpp.read_text())
    except Exception as exc:  # noqa: BLE001
        print(f"DIAG cpp: rebuild of {cpp.name} failed: {type(exc).__name__}: {exc}")
        continue
    total += time.monotonic() - start
    rebuilt += 1
print(f"DIAG cpp: rebuilt {rebuilt} kernels in {total:.1f}s ({total / max(rebuilt, 1):.2f}s each)")
