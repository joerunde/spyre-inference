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

"""Generate a tiny same-architecture gemma-4 MoE checkpoint for the Spyre tests.

No small gemma-4 MoE checkpoint on the Hub has a head size that is a multiple of 64
(Spyre's stick rule), so we build one: the real ``gemma-4-26B-A4B`` config shrunk only
in its size fields, leaving every architecture flag inherited verbatim (dual local/global
head_dim, the MoE block, the sliding/full layer mix, the per-layer-input embeddings, the
vision tower). The tests compare device-vs-CPU and TP1-vs-TP2, not output quality, so the
weights are controlled random init plus an optional short conditioning pass. Hosted as
joerunde/gemma-4-moe-nano.

    python tests/data/generate_micro_gemma4_moe.py --output-dir <dir> --train-steps 300
"""

from __future__ import annotations

import argparse

import torch

BASE_MODEL = "google/gemma-4-26B-A4B-it"

# Public-domain text (opening of Moby-Dick) for the optional conditioning pass; the
# goal is only to pull the output distribution off uniform, not linguistic quality.
_CORPUS = (
    "Call me Ishmael. Some years ago never mind how long precisely having little or no "
    "money in my purse and nothing particular to interest me on shore I thought I would "
    "sail about a little and see the watery part of the world. It is a way I have of "
    "driving off the spleen and regulating the circulation. Whenever I find myself growing "
    "grim about the mouth whenever it is a damp drizzly November in my soul whenever I find "
    "myself involuntarily pausing before coffin warehouses and bringing up the rear of "
    "every funeral I meet and especially whenever my hypos get such an upper hand of me "
    "that it requires a strong moral principle to prevent me from deliberately stepping "
    "into the street and methodically knocking people's hats off then I account it high "
    "time to get to sea as soon as I can. This is my substitute for pistol and ball. With "
    "a philosophical flourish Cato throws himself upon his sword I quietly take to the ship. "
    "There is nothing surprising in this. If they but knew it almost all men in their degree "
    "some time or other cherish very nearly the same feelings towards the ocean with me."
) * 4


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--base-model", default=BASE_MODEL, help="Full model whose config/tokenizer is inherited."
    )
    p.add_argument("--output-dir", required=True, help="Where to save the generated checkpoint.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--init-std", type=float, default=0.02, help="initializer_range for controlled init."
    )
    p.add_argument(
        "--text-layers",
        type=int,
        default=6,
        help="Keep >=6 so a full_attention layer (index 5) survives.",
    )
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--intermediate", type=int, default=128)
    p.add_argument("--moe-intermediate", type=int, default=64)
    p.add_argument(
        "--num-experts",
        type=int,
        default=128,
        help="Keep real count for fidelity; shrink only to speed iteration.",
    )
    p.add_argument("--vision-layers", type=int, default=2)
    p.add_argument(
        "--vision-heads", type=int, default=2, help="vision hidden = heads * vision head_dim (72)."
    )
    p.add_argument(
        "--train-steps", type=int, default=0, help="Optional light LM conditioning pass."
    )
    p.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    return p.parse_args()


def shrink_text_config(cfg, args: argparse.Namespace) -> None:
    """Replace cfg.text_config with a size-shrunk copy; architecture flags kept.

    Edits the plain dict and rebuilds via from_dict so the heterogeneous per-layer
    machinery re-derives cleanly. ``per_layer_config`` is a dict keyed by zero-padded
    layer index ('05','11',...) holding the full_attention layers' global-head_dim
    overrides; we keep only the keys below the new layer count.
    """
    tc = cfg.text_config
    d = tc.to_dict()
    n = args.text_layers
    types = d["layer_types"]
    if "full_attention" not in types[:n]:
        raise SystemExit(
            f"--text-layers={n} keeps no full_attention layer (first full at index "
            f"{types.index('full_attention')}); raise it."
        )
    d["num_hidden_layers"] = n
    d["layer_types"] = types[:n]
    d["per_layer_config"] = {
        k: v for k, v in (d.get("per_layer_config") or {}).items() if int(k) < n
    }
    if d.get("num_kv_shared_layers"):
        d["num_kv_shared_layers"] = min(d["num_kv_shared_layers"], n)
    d["hidden_size"] = args.hidden
    d["intermediate_size"] = args.intermediate
    d["moe_intermediate_size"] = args.moe_intermediate
    d["num_experts"] = args.num_experts
    if d.get("hidden_size_per_layer_input"):
        d["hidden_size_per_layer_input"] = min(d["hidden_size_per_layer_input"], args.hidden)
    d["initializer_range"] = args.init_std
    cfg.text_config = type(tc).from_dict(d)


def shrink_vision_config(vc, args: argparse.Namespace) -> None:
    head_dim = getattr(vc, "head_dim", None) or (vc.hidden_size // vc.num_attention_heads)
    vc.num_hidden_layers = args.vision_layers
    vc.num_attention_heads = args.vision_heads
    if getattr(vc, "num_key_value_heads", None):
        vc.num_key_value_heads = args.vision_heads
    vc.hidden_size = args.vision_heads * head_dim
    vc.intermediate_size = max(2 * vc.hidden_size, 64)
    vc.initializer_range = args.init_std


_PROBE_PROMPT = "The sailor looked out at the ocean and"


def _forward_stats(model, tokenizer) -> tuple[bool, float, float]:
    """Finite-ness, logit std, and entropy after a real text prefix (a random context
    reads as near-uniform even when conditioned, so only a real prefix shows it)."""
    model.eval()
    ids = tokenizer(_PROBE_PROMPT, return_tensors="pt").input_ids
    with torch.no_grad():
        logits = model(input_ids=ids).logits
    probs = torch.softmax(logits[0, -1].float(), dim=-1)
    entropy = -(probs * probs.clamp_min(1e-12).log()).sum().item()
    return bool(torch.isfinite(logits).all()), float(logits.float().std()), entropy


def light_lm_pass(model, tokenizer, steps: int, seed: int) -> None:
    """A short next-token pass over a bundled corpus, in fp32, to sharpen outputs."""
    ids = tokenizer(_CORPUS, return_tensors="pt").input_ids[0]
    seqlen = 64
    chunks = [ids[i : i + seqlen] for i in range(0, len(ids) - seqlen, seqlen)]
    if not chunks:
        raise SystemExit("corpus too short to form a training chunk")
    model.float().train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    g = torch.Generator().manual_seed(seed)
    for step in range(steps):
        batch = chunks[torch.randint(len(chunks), (1,), generator=g).item()].unsqueeze(0)
        loss = model(input_ids=batch, labels=batch).loss
        loss.backward()
        opt.step()
        opt.zero_grad()
        if step % 50 == 0 or step == steps - 1:
            print(f"  train step {step}: loss {loss.item():.3f}")
    model.eval()


def main() -> None:
    args = _parse_args()
    torch.manual_seed(args.seed)
    dtype = getattr(torch, args.dtype)

    from transformers import AutoConfig, AutoProcessor, AutoTokenizer

    cfg = AutoConfig.from_pretrained(args.base_model)
    shrink_text_config(cfg, args)
    shrink_vision_config(cfg.vision_config, args)

    from transformers import AutoModelForImageTextToText

    model = AutoModelForImageTextToText.from_config(cfg, dtype=torch.float32)
    model.init_weights()
    vocab = cfg.text_config.vocab_size
    max_entropy = torch.log(torch.tensor(float(vocab))).item()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"instantiated {type(model).__name__}: {n_params / 1e6:.1f}M params")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    finite, std, entropy = _forward_stats(model, tokenizer)
    print(
        f"after init:  finite={finite} std={std:.3f} entropy={entropy:.2f}/{max_entropy:.2f}"
    )
    if args.train_steps:
        light_lm_pass(model, tokenizer, args.train_steps, args.seed)
        finite, std, entropy = _forward_stats(model, tokenizer)
        print(
            f"after train: finite={finite} std={std:.3f} entropy={entropy:.2f}/{max_entropy:.2f}"
        )
    if not finite:
        raise SystemExit("non-finite logits; lower --init-std, or adjust --train-steps")

    model.to(dtype)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    try:
        AutoProcessor.from_pretrained(args.base_model).save_pretrained(args.output_dir)
    except Exception as e:
        print(f"(no processor saved: {e})")
    print(f"saved {n_params / 1e6:.1f}M-param checkpoint to {args.output_dir}")


if __name__ == "__main__":
    main()
