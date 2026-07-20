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

"""Gold reference for layer-bisection: HuggingFace transformers on CPU.

Captures the per-layer residual stream (via output_hidden_states) plus per-layer
submodule outputs (via forward hooks), keyed with canonical names (L{i},
L{i}.<submodule>) so capture_vllm.py's Spyre run can be diffed against it.

Usage:
    HF_TOKEN=... python ref_hf.py --model google/gemma-3-1b-it --out ref.pt
"""

import argparse
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def canon(name: str) -> str:
    """model.layers.3.self_attn -> L3.self_attn (matches capture_vllm.py)."""
    if ".layers." in name:
        return "L" + name.split(".layers.")[1]
    return name


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="HF model id")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--out", required=True, help="output .pt path")
    ap.add_argument("--dtype", default="float32",
                    choices=["float32", "float16", "bfloat16"])
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN")
    tok = AutoTokenizer.from_pretrained(args.model, token=token)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, token=token, torch_dtype=getattr(torch, args.dtype)
    ).eval()

    input_ids = tok(args.prompt, return_tensors="pt").input_ids
    print("token_ids:", input_ids.tolist())

    acts: dict = {}
    handles = []

    def make_hook(cname):
        def hook(mod, in_args, out):
            if cname in acts:
                return
            if isinstance(out, (tuple, list)) and out and isinstance(out[0], torch.Tensor):
                val = out[0]
            elif isinstance(out, torch.Tensor):
                val = out
            else:
                return
            acts[cname] = val.detach().to(torch.float32).squeeze(0)  # drop batch
        return hook

    # Submodule outputs within each decoder layer (drill-down).
    for name, mod in model.named_modules():
        if ".layers." in name and name.count(".") >= 3:
            handles.append(mod.register_forward_hook(make_hook(canon(name))))

    with torch.no_grad():
        out = model(input_ids, output_hidden_states=True, use_cache=False)

    # hidden_states: (num_layers+1,); [0]=embed, [i+1]=residual stream after layer i.
    hs = out.hidden_states
    acts["embed"] = hs[0].detach().to(torch.float32).squeeze(0)
    for i in range(len(hs) - 1):
        acts[f"L{i}"] = hs[i + 1].detach().to(torch.float32).squeeze(0)

    torch.save({"framework": "hf_cpu", "model": args.model, "prompt": args.prompt,
                "token_ids": input_ids.tolist(), "acts": acts}, args.out)
    print(f"[hf] saved {len(acts)} activations to {args.out}")


if __name__ == "__main__":
    main()
