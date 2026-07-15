#!/usr/bin/env python
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
"""Single-card stress worker. Invoked by stress_cards.py — not meant to be run directly.

Compiles and executes random matmuls on one Spyre device, verifying results
against a CPU reference to catch hardware errors.
"""

import argparse
import os
import sys

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import torch


def make_compiled_matmul():
    @torch.compile(backend="inductor", fullgraph=True, dynamic=False)
    def matmul(a, b):
        return a @ b

    return matmul


def run(iters: int, size: int):
    import torch_spyre

    torch_spyre._autoload()
    # torch.spyre.set_device(device_index)

    device = torch.device(f"spyre:0")
    atol = 1  # fp16 matmul on hardware has accumulation differences
    rtol = 5e-2

    for i in range(iters):
        m = size + (i * 7) % 64
        k = size + (i * 13) % 64
        n = size + (i * 11) % 64

        a_cpu = torch.randn(m, k, dtype=torch.float16)
        b_cpu = torch.randn(k, n, dtype=torch.float16)
        expected = a_cpu @ b_cpu

        a = a_cpu.to(device)
        b = b_cpu.to(device)

        compiled_matmul = make_compiled_matmul()
        result = compiled_matmul(a, b)
        result_cpu = result.cpu()

        try:
            torch.testing.assert_close(result_cpu, expected, atol=atol, rtol=rtol)
        except AssertionError as e:
            max_diff = (result_cpu - expected).abs().max().item()
            print(
                f"  MISMATCH iter {i}: shape=({m},{k})x({k},{n}) max_diff={max_diff:.6f}",
                file=sys.stderr,
            )
            raise SystemExit(1) from e

        print(f"  iter {i}: ({m},{k})x({k},{n}) OK")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--size", type=int, default=256)
    args = parser.parse_args()
    run(args.iters, args.size)


if __name__ == "__main__":
    main()
