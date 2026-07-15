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
"""Stress-test all Spyre cards by compiling random matmuls on each.

Discovers cards from AIU_WORLD_RANK_x environment variables and exercises each
one sequentially (Spyre is single-process-per-card). Each card gets a batch of
compiled matmuls with random inputs to flex the hardware and surface errors.

Usage:
    python scripts/stress_cards.py
    python scripts/stress_cards.py --iters 20 --size 2048
"""

import argparse
import os
import subprocess
import sys


def discover_cards() -> list[tuple[int, str]]:
    """Return sorted list of (index, pci_addr) from AIU_WORLD_RANK_x env vars."""
    cards = []
    for key, val in os.environ.items():
        if key.startswith("AIU_WORLD_RANK_"):
            idx = int(key.removeprefix("AIU_WORLD_RANK_"))
            cards.append((idx, val))
    cards.sort()
    return cards


def main():
    parser = argparse.ArgumentParser(description="Stress-test Spyre cards with compiled matmuls")
    parser.add_argument(
        "--iters", type=int, default=10, help="Number of matmuls to compile per card"
    )
    parser.add_argument("--size", type=int, default=256, help="Matrix dimension (NxN)")
    parser.add_argument(
        "--cards",
        type=str,
        default=None,
        help="Comma-separated card indices to test (default: all)",
    )
    args = parser.parse_args()

    cards = discover_cards()
    if not cards:
        print("ERROR: No AIU_WORLD_RANK_x environment variables found.", file=sys.stderr)
        sys.exit(1)

    if args.cards is not None:
        selected = {int(c) for c in args.cards.split(",")}
        cards = [(idx, addr) for idx, addr in cards if idx in selected]

    print(f"Discovered {len(cards)} card(s): {', '.join(f'{i}={a}' for i, a in cards)}")
    print(f"Running {args.iters} compiled matmuls (~{args.size}x{args.size}) per card\n")

    failures = []
    for idx, pci_addr in cards:
        print(f"--- Card {idx} ({pci_addr}) ---")
        env = os.environ.copy()
        env["RANK"] = "0"
        env["WORLD_SIZE"] = "1"
        env["LOCAL_RANK"] = "0"
        env["LOCAL_WORLD_SIZE"] = "1"
        env["SPYRE_DEVICES"] = str(idx)

        result = subprocess.run(
            [
                sys.executable,
                os.path.join(os.path.dirname(__file__), "_stress_one_card.py"),
                "--iters",
                str(args.iters),
                "--size",
                str(args.size),
            ],
            env=env,
            capture_output=True,
            text=True,
        )

        sys.stdout.write(result.stdout)
        if result.returncode != 0:
            print(f"FAIL: Card {idx} ({pci_addr}) exited with code {result.returncode}")
            if result.stderr:
                sys.stderr.write(result.stderr)
            failures.append((idx, pci_addr))
        else:
            print(f"PASS: Card {idx} ({pci_addr})\n")

    print("=" * 60)
    if failures:
        print(f"FAILED {len(failures)}/{len(cards)} card(s):")
        for idx, addr in failures:
            print(f"  Card {idx} ({addr})")
        sys.exit(1)
    else:
        print(f"ALL {len(cards)} card(s) PASSED")


if __name__ == "__main__":
    main()
