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

"""Drop the per-file clutter from a dorny/test-reporter markdown report.

With ~35 sharded JUnit files, dorny emits one `|Report|…|` row and one
`## <file>.xml` heading + "N tests were completed…" one-liner per shard --
noise that buries the actual failures on the run Summary page. Strip those
three constructs; keep the badge and the per-suite/per-test failure detail.

Coupled to dorny's markdown shape (`__tests__/__outputs__/*.md` upstream): the
per-file table's first column is `Report` (the suite table's is `Test suite`),
and a file anchor is `<a id="…r{N}">` where a suite anchor is `…r{N}s{M}`.
"""

import re
import sys

# `## <icon> <a id="…r0" …>junit-foo.xml</a>` -- a file section header. The id
# ends in r<digits> with no trailing s<digits>, which is what separates it from
# the `### …r0s0` suite header we keep.
FILE_HEADING = re.compile(r'^##\s.*<a id="[^"]*r\d+"')
# `**10** tests were completed in **19ms** with …`, or dorny's empty fallback.
ONE_LINER = re.compile(r"^\*\*[\d,]+\*\* tests were completed\b|^No tests found\b")


def _is_report_table_header(line: str) -> bool:
    cells = [c.strip() for c in line.strip().strip("|").split("|")]
    return len(cells) >= 2 and cells[0] == "Report" and "Time" in cells


def strip(markdown: str) -> str:
    lines = markdown.splitlines()
    out: list[str] = []
    i, n = 0, len(lines)
    while i < n:
        line = lines[i]
        if _is_report_table_header(line):
            i += 1  # header
            while i < n and lines[i].lstrip().startswith("|"):
                i += 1  # alignment row + data rows
            continue
        if FILE_HEADING.match(line):
            i += 1
            while i < n and lines[i].strip() == "":
                i += 1
            if i < n and ONE_LINER.match(lines[i]):
                i += 1
            continue
        out.append(line)
        i += 1
    return "\n".join(out).strip("\n")


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit("usage: strip_dorny_summary.py <summary_file>")
    with open(sys.argv[1], encoding="utf-8") as f:
        result = strip(f.read())
    if result:
        print(result)


if __name__ == "__main__":
    main()
