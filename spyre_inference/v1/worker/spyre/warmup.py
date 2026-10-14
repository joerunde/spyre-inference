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

"""MRV2: V2 warmup driver (vllm/v1/worker/gpu_worker.py:Worker.compile_or_warm_up_model,
TorchSpyreModelRunner.warming_up_model).

Decoder dummies skip attention and publish null KV slots; pooling dummies
keep attention on; a separate pass then records attention over attn_groups
and runs the sampler dummy. plan() is kept apart from run() so the
dummy-call list can be asserted without executing a model.

MRV2 only allows skip_attn together with is_profile, and is_profile skips
cudagraph_manager.dispatch() and builds the dummy batch unpadded, so plan()
picks already-bucketed token counts itself.
"""

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from vllm.config import CompilationMode
from vllm.logger import init_logger
from vllm.multimodal.encoder_budget import MultiModalBudget

from spyre_inference import envs
from spyre_inference.custom_ops.utils import convert
from spyre_inference.v1.attention import attn_layer
from spyre_inference.v1.attention.backends.spyre_attn import (
    SpyreAttentionImpl,
    SpyreAttentionMetadataBuilder,
    allocate_staging_buffers,
)
from spyre_inference.v1.attention.backends.spyre_encoder_attn import encoder_inline_active
from spyre_inference.v1.pool import select_rows
from spyre_inference.v1.pool.encoder_grid import warm_pooler_row_widths
from spyre_inference.v1.worker.spyre_model_runner import (  # shared with V1; moves at the flip
    _set_spyre_compilation_settings,
)
from spyre_inference.v1.worker.spyre_shape_bucketer import (
    encoder_group_shapes,
    encoder_rectangles,
    encoder_shape_tables,
    logits_row_buckets,
)

if TYPE_CHECKING:
    from spyre_inference.v1.worker.spyre.model_runner import SpyreModelRunnerV2

logger = init_logger(__name__)


@dataclass(frozen=True)
class DummyCall:
    """One _dummy_run() invocation in the warmup plan."""

    num_reqs: int
    num_tokens: int
    skip_attn: bool


def plan(runner: "SpyreModelRunnerV2") -> list[DummyCall]:
    """Build the ordered list of dummy calls warmup makes, without running any."""
    max_reqs = runner.max_num_reqs
    bucketer = runner.spyre_shape_bucketer
    is_pooling = runner.is_pooling_model
    if bucketer is None or (is_pooling and runner.model_config.enforce_eager):
        sizes = [min(max(16, max_reqs), runner.max_num_tokens)]
    else:
        # Largest first: Inductor's caches benefit from seeing the most complex shape first.
        sizes = sorted(bucketer.bucket_sizes, reverse=True)
    return [DummyCall(min(n, max_reqs), n, skip_attn=not is_pooling) for n in sizes]


def _record_attention_graphs(runner: "SpyreModelRunnerV2") -> None:
    if not envs.SPYRE_ATTN_RECORD:
        logger.info("Attention graph recording disabled (SPYRE_ATTN_RECORD=0)")
        return
    if runner.compilation_config.mode is CompilationMode.NONE:
        logger.info("Attention graph recording disabled (CompilationMode.NONE)")
        return
    builders: dict[str, SpyreAttentionMetadataBuilder] = {}
    for groups in runner.attn_groups:
        for group in groups:
            builder = group.metadata_builders[0] if group.metadata_builders else None
            if isinstance(builder, SpyreAttentionMetadataBuilder):
                builders.update(dict.fromkeys(group.layer_names, builder))
    static_ctx = runner.compilation_config.static_forward_context
    t0 = time.time()
    total = 0
    for layer_name, kv_cache in runner.spyre_kv_caches.items():
        layer = static_ctx.get(layer_name)
        impl = getattr(layer, "impl", None)
        if not isinstance(impl, SpyreAttentionImpl):
            continue
        builder = builders.get(layer_name)
        assert builder is not None, f"Layer {layer_name} has no Spyre metadata builder"
        total += impl.record_graphs(layer, kv_cache, builder)
    logger.info("Attention graph recording: %d graphs in %.3fs.", total, time.time() - t0)


def _dummy_run(runner: "SpyreModelRunnerV2", num_tokens: int, skip_attn: bool):
    """MRV2's _dummy_run, with pooling hidden states D2H'd for its CPU logits_indices
    gather and handed back on Spyre, where the pooler reads them."""
    if skip_attn:
        attn_layer.publish_null_slots(num_tokens)
    wrapper = runner.model
    keep = wrapper._keep_outputs_on_device
    object.__setattr__(wrapper, "_keep_outputs_on_device", False)
    try:
        hidden_states, sample_hidden_states = runner._dummy_run(
            num_tokens, skip_attn=skip_attn, is_profile=True
        )
    finally:
        object.__setattr__(wrapper, "_keep_outputs_on_device", keep)
    if keep and hidden_states is not None and hidden_states.numel() > 0:
        hidden_states = convert(hidden_states, runner.spyre_device)
    return hidden_states, sample_hidden_states


def _warm_pooling(runner: "SpyreModelRunnerV2", hidden_states: torch.Tensor) -> None:
    assert runner.pooling_runner is not None
    runner.pooling_runner.dummy_pooler_run(hidden_states)
    if not runner.pooling_on_spyre:
        return
    rectangles = encoder_rectangles(runner.vllm_config)
    warm_pooler_row_widths(
        runner.model.pooler, hidden_states, rectangles, runner.scheduler_config.max_num_seqs
    )
    if rectangles:
        # pool_rows_for's re-compaction gather; it keeps the buffer's row count.
        select_rows(hidden_states, torch.zeros(hidden_states.shape[0], dtype=torch.int64))


def _warm_encoder_inline_paths(runner: "SpyreModelRunnerV2") -> None:
    """One block graph per declared rectangle, plus the one that keeps attention opaque."""
    rectangles = encoder_rectangles(runner.vllm_config)
    model_state = runner.model_state
    if not encoder_inline_active() or not rectangles or not hasattr(model_state, "forced_rect"):
        return
    budget = encoder_shape_tables(runner.vllm_config).budget
    groups = encoder_group_shapes(runner.vllm_config)
    for rect in sorted(rectangles, reverse=True):
        model_state.forced_rect = rect
        try:
            _dummy_run(runner, budget, skip_attn=False)
        finally:
            model_state.forced_rect = None
    if groups:
        model_state.force_ragged = True
        try:
            _dummy_run(runner, budget, skip_attn=False)
        finally:
            model_state.force_ragged = False


def _warm_multimodal(runner: "SpyreModelRunnerV2", sizes: list[int]) -> None:
    """The input embedding at every body size, and the vision tower at every image-batch
    width (V1's _warmup_input_embedding / _warmup_multimodal_encoder).

    Dummy runs hand the model the staging buffer instead of producing it through
    embed_input_ids, and nothing else traces the tower before the compile guard arms.
    A generative VLM's tower is enumerated only when its blocks are compiled: an eager
    tower's ops compile per op on first use, which the guard expects.
    """
    if not runner.supports_mm_inputs:
        return
    mm_config = runner.model_config.multimodal_config
    if not runner.model_config.is_encoder_decoder and not (
        mm_config is not None and mm_config.mm_encoder_only
    ):
        for n in sizes:
            # int32 to match upstream's input_ids buffer; 0 is an id every vocab holds.
            runner.model.embed_input_ids(torch.zeros(n, dtype=torch.int32))
    if not runner.is_pooling_model and not runner.compiled_tower_blocks:
        return
    budget = MultiModalBudget(runner.vllm_config, runner.mm_registry, enable_cache=False)
    encoder_runner = runner.model_state.encoder_runner
    for modality, n_max in budget.mm_max_items_per_batch.items():
        logger.info("Warming multimodal encoder batch widths: %s 1..%d.", modality, n_max)
        for n in range(1, int(n_max) + 1):
            encoder_runner.execute_mm_encoder(
                budget.get_dummy_encoder_profile_inputs(modality, n)
            )


def run(runner: "SpyreModelRunnerV2") -> None:
    """Execute plan(runner), then record attention and the sampler dummy."""
    allocate_staging_buffers(runner.compilation_config.static_forward_context, runner.spyre_device)
    calls = plan(runner)
    logger.info("MRV2 warmup: %d dummy runs %s", len(calls), [c.num_tokens for c in calls])
    t0 = time.time()
    bucketer = runner.spyre_shape_bucketer
    compiled = runner.compilation_config.mode is not CompilationMode.NONE
    with _set_spyre_compilation_settings(runner.vllm_config), torch.inference_mode():
        widest = None
        for call in calls:
            hidden_states, sample_hidden_states = _dummy_run(
                runner, call.num_tokens, call.skip_attn
            )
            if runner.pooling_runner is not None:
                _warm_pooling(runner, hidden_states)
            elif widest is None:
                widest = sample_hidden_states
        if runner.pooling_runner is not None and compiled:
            _warm_encoder_inline_paths(runner)
        _warm_multimodal(runner, [c.num_tokens for c in calls])
        if widest is not None:
            rows = (
                logits_row_buckets(bucketer.bucket_sizes, runner.max_num_reqs)
                if bucketer is not None
                else [widest.shape[0]]
            )
            for n in sorted(rows, reverse=True):
                runner._dummy_sampler_run(widest[:n])
    if bucketer is not None:
        bucketer.mark_warmed_up()
    if runner.spyre_kv_caches:
        # Under inference_mode like MRV2's execute_model: grad mode is a guard on every
        # recorded kernel, so recording outside it misses every one at serving time.
        with _set_spyre_compilation_settings(runner.vllm_config), torch.inference_mode():
            _record_attention_graphs(runner)
    logger.info("MRV2 warmup done in %.3fs.", time.time() - t0)
