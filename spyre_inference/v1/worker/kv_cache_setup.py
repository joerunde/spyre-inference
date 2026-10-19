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

"""Attention-group setup shared by both model runners."""

from dataclasses import replace
from typing import Any

from vllm.logger import init_logger
from vllm.v1.kv_cache_interface import FullAttentionSpec
from vllm.v1.worker.utils import AttentionGroup

logger = init_logger(__name__)


def split_attn_groups_by_window(
    attn_groups: list[list[AttentionGroup]], layers: dict[str, Any]
) -> None:
    """Give each distinct per-layer sliding window its own attention group, in place."""
    # `FullAttentionSpec.merge` collapses a hybrid decoder onto the one window it finds,
    # clamping its full-attention layers too (gemma-3-1b-it: 512 on all 26). Spyre masks
    # per group, not per layer as upstream does, so the group is what has to split.
    for kv_cache_group_id, groups in enumerate(attn_groups):
        split_groups: list[AttentionGroup] = []
        for group in groups:
            spec = group.kv_cache_spec
            windows: dict[int | None, list[str]] = {}
            for layer_name in group.layer_names:
                window = getattr(layers.get(layer_name), "sliding_window", None)
                windows.setdefault(window, []).append(layer_name)
            # SlidingWindowSpec is single-window by construction; UniformTypeKVCacheSpecs
            # is already unwrapped per layer upstream.
            if not isinstance(spec, FullAttentionSpec) or len(windows) <= 1:
                split_groups.append(group)
                continue
            logger.info(
                "Split %d attention layers into %d groups by sliding window %s.",
                len(group.layer_names),
                len(windows),
                {w: len(n) for w, n in windows.items()},
            )
            for window, layer_names in windows.items():
                split_groups.append(
                    AttentionGroup(
                        group.backend,
                        layer_names,
                        replace(spec, sliding_window=window),
                        kv_cache_group_id,
                    )
                )
        attn_groups[kv_cache_group_id] = split_groups
