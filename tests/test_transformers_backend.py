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

"""Tests for the HuggingFace Transformers backend (model_impl='transformers').

Stands in for upstream's ``tests/models/transformers/test_backend.py``, which is
disabled in ``upstream_tests.yaml``: it compares against an HF CPU reference over 32
tokens of logprobs, which fp16 on Spyre is unlikely to satisfy. The native Spyre path
is the reference here instead.
"""

from __future__ import annotations

import pytest
import torch


def test_rope_frequencies_rebuilt_at_the_pre_pad_head_dim():
    """HF derives inv_freq from the widened head_dim, so the rebuild has to undo it."""
    from transformers import LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

    from spyre_inference.transformers_backend import _rope_at_original_head_dim

    orig, padded = 4, 128
    cfg = LlamaConfig(
        hidden_size=16,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_hidden_layers=1,
        head_dim=orig,
        max_position_embeddings=256,
    )
    expected = LlamaRotaryEmbedding(config=cfg).inv_freq.clone()
    assert expected.shape == (orig // 2,)

    cfg.head_dim = padded
    padded_rope = LlamaRotaryEmbedding(config=cfg)
    # What HF built off the padded config: too many frequencies, wrong spacing.
    assert padded_rope.inv_freq.shape == (padded // 2,)
    assert not torch.equal(padded_rope.inv_freq[: orig // 2], expected)

    rebuilt = _rope_at_original_head_dim(cfg, padded_rope, orig)

    assert torch.equal(rebuilt.inv_freq, expected)
    assert cfg.head_dim == padded, "the padded width must be restored for the model"


def test_padded_qk_logits_match_the_unpadded_reference():
    """Weight padding + rebuilt rotation + 1/sqrt(orig) scale must leave the logits
    unchanged versus stock HF at the native head_dim."""
    from transformers import LlamaConfig
    from transformers.models.llama.modeling_llama import (
        LlamaRotaryEmbedding,
        apply_rotary_pos_emb,
    )

    from spyre_inference.custom_ops.head_pad import _pad_weight
    from spyre_inference.transformers_backend import (
        _rope_at_original_head_dim,
        _spyre_apply_rotary,
        _SpyreRotaryEmbedding,
    )

    orig, padded = 4, 128
    n_heads, hidden, seq = 4, 16, 6
    torch.manual_seed(0)

    cfg = LlamaConfig(
        hidden_size=hidden,
        num_attention_heads=n_heads,
        num_key_value_heads=n_heads,
        num_hidden_layers=1,
        head_dim=orig,
        max_position_embeddings=64,
    )
    x = torch.randn(1, seq, hidden)
    position_ids = torch.arange(seq).unsqueeze(0)
    q_w, k_w = torch.randn(n_heads * orig, hidden), torch.randn(n_heads * orig, hidden)

    def heads(inputs, weight, head_dim):
        # [B, L, hidden] -> [B, H, L, head_dim], the layout RoPE and attention use.
        return (inputs @ weight.T).view(1, seq, n_heads, head_dim).transpose(1, 2)

    hf_rope = LlamaRotaryEmbedding(config=cfg)
    cos, sin = hf_rope(x, position_ids)
    q_ref, k_ref = apply_rotary_pos_emb(heads(x, q_w, orig), heads(x, k_w, orig), cos, sin)
    logits_ref = (q_ref @ k_ref.transpose(-1, -2)) * orig**-0.5

    cfg.head_dim = padded
    q_pad = heads(x, _pad_weight("q_proj.weight", q_w, n_heads, n_heads, orig, padded), padded)
    k_pad = heads(x, _pad_weight("k_proj.weight", k_w, n_heads, n_heads, orig, padded), padded)

    spyre_rope = _SpyreRotaryEmbedding(
        _rope_at_original_head_dim(cfg, hf_rope, orig),
        cfg.max_position_embeddings,
        padded,
        torch.float32,
    )
    rotation, _ = spyre_rope(x, position_ids)

    q_rot, k_rot = _spyre_apply_rotary(q_pad, k_pad, rotation)
    logits_pad = (q_rot @ k_rot.transpose(-1, -2)) * orig**-0.5

    torch.testing.assert_close(logits_pad, logits_ref, rtol=1e-5, atol=1e-5)

    half, padded_half = orig // 2, padded // 2
    assert torch.allclose(q_rot[..., :half], q_ref[..., :half], atol=1e-6)
    assert torch.allclose(
        q_rot[..., padded_half : padded_half + half], q_ref[..., half:], atol=1e-6
    )
    assert not q_rot[..., half:padded_half].any()
    assert not q_rot[..., padded_half + half :].any()


PROMPTS = [
    "Hello, my name is",
    "The capital of France is",
]

# The two paths are not bit-identical: they run different module code (HF's vs vLLM's)
# and round their rotation caches differently, so greedy sampling eventually tie-breaks
# apart. The failure mode being guarded against diverges from the first token or two.
MAX_TOKENS = 8


def _generate_greedy(model: str, model_impl: str) -> list[list[int]]:
    from vllm import LLM, SamplingParams
    from vllm.distributed import cleanup_dist_env_and_memory

    llm = LLM(
        model=model,
        dtype="float16",
        enforce_eager=True,
        max_model_len=128,
        max_num_seqs=2,
        model_impl=model_impl,
    )
    assert llm.llm_engine.model_config.using_transformers_backend() == (
        model_impl == "transformers"
    )
    outputs = llm.generate(PROMPTS, SamplingParams(max_tokens=MAX_TOKENS, temperature=0.0))
    token_ids = [list(o.outputs[0].token_ids) for o in outputs]

    del llm
    cleanup_dist_env_and_memory()
    return token_ids


@pytest.mark.uses_subprocess
@pytest.mark.parametrize(
    "model",
    [
        "ibm-ai-platform/micro-g3.3-8b-instruct-1b",
        # head_dim=64 -> padded; micro-g3.3 is 128 -> unpadded. Covers both branches.
        "meta-llama/Llama-3.2-1B-Instruct",
    ],
)
def test_transformers_matches_native(model: str) -> None:
    """The Transformers backend must generate what the native Spyre path does.

    Content, not just non-empty output: a broken RoPE or a norm falling back to an
    unsupported fp32 promotion still yields fluent text, just unrelated to the prompt.
    """
    transformers_ids = _generate_greedy(model, "transformers")
    native_ids = _generate_greedy(model, "vllm")

    assert transformers_ids == native_ids
