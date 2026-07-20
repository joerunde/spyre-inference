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

"""Diff two layer-bisection captures and report the first diverging submodule.

Usage:
    python diff.py ref.pt spyre.pt [--threshold 0.99]
"""

import argparse

import torch


def _sort_key(name: str):
    # Order by layer index when present so "first divergence" is meaningful.
    return [(0, int(p)) if p.isdigit() else (1, p) for p in name.split(".")]


def _cos(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    return (a @ b / (a.norm() * b.norm()).clamp_min(1e-12)).item()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("ref", help="reference capture (e.g. ref_hf.py output)")
    ap.add_argument("sut", help="system-under-test capture (e.g. capture_vllm.py output)")
    ap.add_argument("--threshold", type=float, default=0.99,
                    help="cosine below this flags divergence")
    args = ap.parse_args()

    ref = torch.load(args.ref, weights_only=False)
    sut = torch.load(args.sut, weights_only=False)
    ra, sa = ref["acts"], sut["acts"]
    names = sorted(set(ra) & set(sa), key=_sort_key)

    print(f"ref={args.ref} ({ref.get('framework')})  sut={args.sut} ({sut.get('framework')})")
    print(f"common modules: {len(names)}  (ref-only={len(set(ra) - set(sa))}, "
          f"sut-only={len(set(sa) - set(ra))})")
    print(f"{'module':<48} {'shape':<18} {'cos':>8} {'max_abs':>12} {'mean_abs':>12}")
    print("-" * 100)

    first_bad = None
    for n in names:
        a, b = ra[n], sa[n]
        if a.shape != b.shape:
            print(f"{n:<48} SHAPE MISMATCH ref={tuple(a.shape)} sut={tuple(b.shape)}")
            continue
        c = _cos(a, b)
        d = (a - b).abs()
        flag = ""
        if c < args.threshold and first_bad is None:
            first_bad = n
            flag = "  <<< FIRST DIVERGENCE"
        print(f"{n:<48} {str(tuple(a.shape)):<18} {c:>8.4f} "
              f"{d.max().item():>12.4g} {d.mean().item():>12.4g}{flag}")

    print("-" * 100)
    print(f"FIRST DIVERGENCE: {first_bad}")
    print("NOTE: the terminal decoder layer's reconstructed residual stream "
          "(out[0]+out[1]) can false-positive — trust submodule cos and generated text there.")


if __name__ == "__main__":
    main()
