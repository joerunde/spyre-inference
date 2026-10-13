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

"""resolve_spyre_model_state puts the Spyre mixin over the class upstream picks."""

import pytest
import torch
import torch.nn as nn
from vllm.model_executor.layers.attention import Attention, CrossAttention
from vllm.v1.attention.backend import AttentionType
from vllm.v1.worker.gpu import model_states as upstream
from vllm.v1.worker.gpu.model_states.default import DefaultModelState
from vllm.v1.worker.gpu.model_states.encoder_only import EncoderOnlyModelState
from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState

from spyre_inference.v1.worker.spyre import model_states as spyre
from spyre_inference.v1.worker.spyre.model_states import (
    resolve_spyre_model_state,
    spyre_model_state_cls,
)
from spyre_inference.v1.worker.spyre.model_states.base import SpyreModelStateMixin
from spyre_inference.v1.worker.spyre.model_states.default import SpyreDefaultModelState
from spyre_inference.v1.worker.spyre.model_states.encoder_only import (
    SpyreEncoderOnlyModelState,
)

from .state_fakes import make_config


class _OwnState:
    """Names itself when constructed, like the patched upstream classes below."""

    def __new__(cls, *args):
        return "_OwnState"


class _OwnStateModel(nn.Module):
    @staticmethod
    def get_model_state_cls():
        return _OwnState


def _bare(cls):
    module = cls.__new__(cls)
    nn.Module.__init__(module)
    return module


def _cross_attention_model() -> nn.Module:
    model = nn.Module()
    model.add_module("cross", _bare(CrossAttention))
    return model


def _encoder_only_context() -> dict[str, Attention]:
    layer = _bare(Attention)
    layer.attn_type = AttentionType.ENCODER_ONLY
    return {"enc.0": layer}


_CASES = {
    "decoder": (make_config(), nn.Module()),
    "encoder-only": (make_config(static_forward_context=_encoder_only_context()), nn.Module()),
    "hybrid": (make_config(is_hybrid=True), nn.Module()),
    "cross-attention": (make_config(), _cross_attention_model()),
    "own-state": (make_config(), _OwnStateModel()),
}


@pytest.mark.parametrize("case", sorted(_CASES))
def test_resolution_matches_upstream_init_model_state(case, monkeypatch):
    """Drift check on the 0.28 copy: patch upstream's classes to name themselves,
    then ask upstream which one it would construct."""
    config, model = _CASES[case]
    import vllm.v1.worker.gpu.model_states.default as default_mod
    import vllm.v1.worker.gpu.model_states.encoder_decoder as encoder_decoder_mod
    import vllm.v1.worker.gpu.model_states.encoder_only as encoder_only_mod
    import vllm.v1.worker.gpu.model_states.mamba_hybrid as mamba_mod

    resolved = spyre._resolve_model_state_cls(config, model)  # ty: ignore[invalid-argument-type]
    for module, name in [
        (default_mod, "DefaultModelState"),
        (encoder_decoder_mod, "EncoderDecoderModelState"),
        (encoder_only_mod, "EncoderOnlyModelState"),
        (mamba_mod, "MambaHybridModelState"),
    ]:
        monkeypatch.setattr(module, name, lambda *args, _name=name: _name)

    constructed = upstream.init_model_state(
        config,  # ty: ignore[invalid-argument-type]
        model,
        None,
        torch.device("cpu"),
    )
    assert constructed == resolved.__name__


def test_dedicated_spyre_states():
    assert spyre_model_state_cls(DefaultModelState) is SpyreDefaultModelState
    assert spyre_model_state_cls(EncoderOnlyModelState) is SpyreEncoderOnlyModelState


def test_other_states_get_the_mixin_over_them():
    cls = spyre_model_state_cls(MambaHybridModelState)
    assert cls.__mro__[1:3] == (SpyreModelStateMixin, MambaHybridModelState)
    assert spyre_model_state_cls(MambaHybridModelState) is cls


def test_constructs_with_both_devices():
    spyre_device = torch.device("meta")
    state = resolve_spyre_model_state(
        make_config(),  # ty: ignore[invalid-argument-type]
        nn.Module(),
        None,
        torch.device("cpu"),
        spyre_device,
    )
    assert type(state) is SpyreDefaultModelState
    assert state.device == torch.device("cpu")
    assert state.spyre_device == spyre_device
    # Upstream's hooks still run: the mixin defines none it does not implement.
    assert state.add_request.__func__ is DefaultModelState.add_request
