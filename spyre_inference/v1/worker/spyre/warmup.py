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

from spyre_inference import envs
from spyre_inference.v1.attention import attn_layer
from spyre_inference.v1.attention.backends.spyre_attn import (
    SpyreAttentionImpl,
    SpyreAttentionMetadataBuilder,
    allocate_staging_buffers,
)
from spyre_inference.v1.worker.spyre_model_runner import (  # shared with V1; moves at the flip
    _set_spyre_compilation_settings,
)
from spyre_inference.v1.worker.spyre_shape_bucketer import logits_row_buckets

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
        num_tokens = min(max(16, max_reqs), runner.max_num_tokens)
        if bucketer is not None:
            num_tokens = bucketer.find_bucket(num_tokens) or num_tokens
        sizes = [num_tokens]
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


def run(runner: "SpyreModelRunnerV2") -> None:
    """Execute plan(runner), then record attention and the sampler dummy."""
    allocate_staging_buffers(runner.compilation_config.static_forward_context, runner.spyre_device)
    calls = plan(runner)
    logger.info("MRV2 warmup: %d dummy runs %s", len(calls), [c.num_tokens for c in calls])
    t0 = time.time()
    bucketer = runner.spyre_shape_bucketer
    with _set_spyre_compilation_settings(runner.vllm_config), torch.inference_mode():
        widest = None
        for call in calls:
            if call.skip_attn:
                attn_layer.publish_null_slots(call.num_tokens)
            hidden_states, sample_hidden_states = runner._dummy_run(
                call.num_tokens, skip_attn=call.skip_attn, is_profile=True
            )
            if runner.pooling_runner is not None:
                runner._dummy_pooler_run(hidden_states)
            elif widest is None:
                widest = sample_hidden_states
        if widest is not None:
            rows = (
                logits_row_buckets(bucketer.bucket_sizes, runner.max_num_reqs)
                if bucketer is not None and not runner.is_pooling_model
                else [widest.shape[0]]
            )
            for n in sorted(rows, reverse=True):
                runner._dummy_sampler_run(widest[:n])
    if bucketer is not None:
        bucketer.mark_warmed_up()
    if runner.spyre_kv_caches:
        with _set_spyre_compilation_settings(runner.vllm_config):
            _record_attention_graphs(runner)
    logger.info("MRV2 warmup done in %.3fs.", time.time() - t0)
