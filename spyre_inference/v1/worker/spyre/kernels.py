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

"""MRV2: the single kernel seam for MRV2's default-path Triton kernels.

bind() is the one place that picks each kernel's source: triton-cpu on x86
until upstream's non-Triton replacements exist, then those, then a local
override only if measurement on Power/Z calls for one. DEFAULT_PATH_KERNELS
is the full list a coverage test checks bind() against.

Step 1 (now): every kernel runs unmodified through triton-cpu, so bind() binds
nothing itself; it reports a kernel bound when vllm.triton_utils gave its module
the real ``@triton.jit``. Without Triton (Power/Z, or x86 without triton-cpu)
each one is a placeholder whose first launch fails mid-serving, so
bind_or_raise() fails at worker start instead.
"""

import importlib

import torch
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_only, apply_top_k_top_p_pytorch

from spyre_inference.v1.sample.topk_topp_sampler import apply_top_k_top_p_sort_free

_PACKAGE = "vllm.v1.worker.gpu"

# Every @triton.jit on MRV2's default step path (input prep, block tables, staged
# writes, sampler, prompt logprobs, structured outputs), as "<module>.<name>"
# relative to vllm/v1/worker/gpu. Spec decode, mamba, CP, prompt embeds, mrope and
# watermarking are off the default path.
DEFAULT_PATH_KERNELS: frozenset[str] = frozenset(
    {
        "block_table._gather_block_tables_kernel",
        "block_table._compute_slot_mappings_kernel",
        "buffer_utils._apply_write_kernel",
        "buffer_utils._load_ptr",
        "input_batch._prepare_prefill_inputs_kernel",
        "input_batch._prepare_pos_seq_lens_kernel",
        "input_batch._combine_sampled_and_draft_tokens_kernel",
        "input_batch._get_num_sampled_and_rejected_kernel",
        "input_batch._post_update_kernel",
        "input_batch._post_update_num_computed_tokens_kernel",
        "input_batch._expand_idx_mapping_kernel",
        "structured_outputs._apply_grammar_bitmask_kernel",
        "sample.bad_words._bad_words_kernel",
        "sample.gumbel._temperature_kernel",
        "sample.gumbel.tl_rand32",
        "sample.gumbel._murmur3_rotl32",
        "sample.gumbel._murmur3_mix",
        "sample.gumbel._murmur3_fmix32",
        "sample.gumbel.murmur3_hash32",
        "sample.gumbel.murmur3_uniform32",
        "sample.gumbel._uniform64_from_random53",
        "sample.gumbel.murmur3_uniform64",
        "sample.gumbel._log1p_neg_stable",
        "sample.gumbel.gumbel_noised_argmax",
        "sample.gumbel.gumbel_block_argmax",
        "sample.gumbel._gumbel_sample_kernel",
        "sample.logit_bias._bias_kernel",
        "sample.logprob._topk_log_softmax_kernel",
        "sample.logprob._ranks_kernel",
        "sample.logprob._fill_logprob_token_ids_kernel",
        "sample.min_p._min_p_kernel",
        "sample.output._compact_sampling_mask_kernel",
        "sample.penalties._penalties_kernel",
        "sample.penalties._bincount_kernel",
        "sample.prompt_logprob._prompt_logprobs_token_ids_kernel",
    }
)


# Host-side replacements bound over upstream's, at the modules that import them by name.
# MRV2's top-k/top-p is apply_top_k_top_p_triton once the gate opens Triton (or the
# full-vocab sort without it); the logits are already on the host, so V1's sort-free
# version is a candidate kernel step 3a reached early, measured on x86 bring-up.
_TOP_K_TOP_P_SITES = ("sample.sampler", "sample.states")


def spyre_apply_top_k_top_p(
    logits: torch.Tensor, k: torch.Tensor | None, p: torch.Tensor | None
) -> torch.Tensor:
    """apply_top_k_top_p without the full-vocab sort whenever top-k bounds it."""
    if k is None:
        return logits if p is None else apply_top_k_top_p_pytorch(logits, k, p)
    if p is None:
        return apply_top_k_only(logits, k)
    return apply_top_k_top_p_sort_free(logits, k, p)


def _bind_overrides() -> set[str]:
    for site in _TOP_K_TOP_P_SITES:
        module = importlib.import_module(f"{_PACKAGE}.{site}")
        module.apply_top_k_top_p = spyre_apply_top_k_top_p
    return {f"{site}.apply_top_k_top_p" for site in _TOP_K_TOP_P_SITES}


def _resolve(name: str) -> object:
    module_name, _, attr = name.rpartition(".")
    return getattr(importlib.import_module(f"{_PACKAGE}.{module_name}"), attr, None)


def bind() -> set[str]:
    """Bind every DEFAULT_PATH_KERNELS entry to its active source.

    Returns the kernel names actually bound, so callers (and the coverage
    test) can diff against DEFAULT_PATH_KERNELS.
    """
    from vllm.triton_utils import HAS_TRITON

    bound = _bind_overrides()
    if not HAS_TRITON:
        return bound
    from triton.runtime import JITFunction

    return bound | {
        name for name in DEFAULT_PATH_KERNELS if isinstance(_resolve(name), JITFunction)
    }


def bind_or_raise() -> None:
    """Call bind() and raise if any DEFAULT_PATH_KERNELS entry is still
    unbound, rather than letting it fall through to upstream's Triton path
    or fail at its first launch mid-serving."""
    missing = DEFAULT_PATH_KERNELS - bind()
    if missing:
        raise RuntimeError(
            f"kernels.bind() left {sorted(missing)} unbound. Spyre's V2 model runner "
            "runs MRV2's Triton kernels through triton-cpu until upstream's non-Triton "
            "replacements exist (#906 kernel step 2), so it needs x86 with triton-cpu; "
            "use SPYRE_MODEL_RUNNER=v1 elsewhere."
        )
