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

"""System-under-test capture for layer-bisection: the model under vLLM.

Runs a single prefill and captures per-decoder-layer residual stream + submodule
outputs via collective_rpc forward hooks, keyed with canonical names (L{i},
L{i}.<submodule>) matching ref_hf.py. Diff with diff.py.

Requires VLLM_ALLOW_INSECURE_SERIALIZATION=1 (collective_rpc ships a callable).

Usage:
    VLLM_ALLOW_INSECURE_SERIALIZATION=1 HF_TOKEN=... \
        python capture_vllm.py --model google/gemma-3-1b-it --out spyre.pt
"""

import argparse

import torch
from vllm import LLM, SamplingParams


def _canon(name: str) -> str:
    if ".layers." in name:
        return "L" + name.split(".layers.")[1]
    return name


def _install_hooks(worker):
    root = worker.model_runner.model
    captured: dict = {}
    handles = []

    def make_hook(name, is_decoder_layer):
        def hook(mod, in_args, out):
            if name in captured:
                return  # first (prefill) call only
            if is_decoder_layer and isinstance(out, tuple) and len(out) >= 2 \
                    and isinstance(out[0], torch.Tensor) and isinstance(out[1], torch.Tensor):
                val = out[0] + out[1]  # residual stream (fused-add scheme)
            elif isinstance(out, torch.Tensor):
                val = out
            elif isinstance(out, (tuple, list)) and out and isinstance(out[0], torch.Tensor):
                val = out[0]
            else:
                return
            captured[name] = val.detach().to("cpu", torch.float32)
        return hook

    for name, mod in root.named_modules():
        if name == "":
            continue
        is_dl = type(mod).__name__.endswith("DecoderLayer")
        handles.append(mod.register_forward_hook(make_hook(_canon(name), is_dl)))

    root._bisect_captured = captured
    root._bisect_handles = handles
    return len(handles)


def _fetch(worker):
    return dict(worker.model_runner.model._bisect_captured)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="spyre", help="framework label stored in the capture")
    ap.add_argument("--max-model-len", type=int, default=256)
    ap.add_argument("--max-num-seqs", type=int, default=1)
    args = ap.parse_args()

    llm = LLM(model=args.model, max_model_len=args.max_model_len,
              max_num_seqs=args.max_num_seqs, enforce_eager=True)
    n = llm.collective_rpc(_install_hooks)[0]
    print(f"[{args.label}] installed {n} hooks")
    gen = llm.generate([args.prompt], SamplingParams(max_tokens=1, temperature=0))
    print(f"[{args.label}] GENERATED: {gen[0].outputs[0].text!r}")
    caps = llm.collective_rpc(_fetch)[0]
    torch.save({"framework": args.label, "model": args.model,
                "prompt": args.prompt, "acts": caps}, args.out)
    print(f"[{args.label}] saved {len(caps)} activations to {args.out}")


if __name__ == "__main__":
    main()
