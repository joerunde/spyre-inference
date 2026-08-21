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

"""DeviceCommunicator override for IBM Spyre devices.

There are two ways to reach a Spyre collective, and which one a call site
gets depends on whether it is being traced by ``torch.compile``:

* ``torch.ops._c10d_functional.*`` + ``wait_tensor``. Inside a compiled
  graph, torch-spyre lowers these to ``spyre::<op>_async`` + ``spyre::
  wait_work``, so the collective becomes part of the compiled program
  instead of a callback into Python. Outside one, they fall through to the
  generic c10d implementation and land on the spyreccl backend.
* Plain ``dist.*``, which always takes the eager spyreccl path.

``all_reduce`` uses the functional form because it works in *both* modes,
which keeps one code path and — more importantly — gets the reduction
compiled into the model graph, where the bulk of TP traffic lives.

``all_gather`` cannot: see the comments on the method for the two separate
blockers. It stays on eager ``dist.all_gather``, which is fine in practice
because the only all_gather on the TP forward path (vocab-parallel logits)
runs outside the compiled region.

`broadcast`, `send`, and `recv` from `DeviceCommunicatorBase` route through
ops libspyre_comms implements, so they are left alone. `reduce_scatter` is
not implemented and raises.

`tests/test_spyre_comms_native_probes.py` exercises each collective in
isolation on a real spyreccl device_group, and marks the blocked ones
xfail-strict. When torch-spyre or a comms RPM lands the missing piece, the
probe flips to passing, the strict-xfail fails CI, and that's the signal to
delete the matching workaround here.
"""

from __future__ import annotations

import torch
import torch.distributed as dist

from vllm.distributed.device_communicators.base_device_communicator import (
    DeviceCommunicatorBase,
)

from spyre_inference.custom_ops.utils import convert


class SpyreCommunicator(DeviceCommunicatorBase):
    """Spyre-specific DeviceCommunicator.

    See the module docstring. `all_reduce` goes through functional
    collectives so it compiles; `all_gather` stays eager.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Resolve the process group to the string name the `_c10d_functional`
        # ops take, once, at construction. Doing it per call would put a
        # ProcessGroup object in dynamo's path and break the trace.
        self._group_name: str | None = None
        if self.device_group is not None:
            from torch.distributed._functional_collectives import _resolve_group_name

            self._group_name = _resolve_group_name(self.device_group)

    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        if self.world_size == 1:
            return input_
        if input_.device.type == "cpu" or self._group_name is None:
            return super().all_reduce(input_)

        # Out-of-place by contract, unlike the base class's in-place
        # `dist.all_reduce(input_); return input_`. That aliasing is not just
        # untidy: vLLM's `torch.ops.vllm.all_reduce` wrapper declares no
        # mutation, so under torch.compile functionalization never learns the
        # input was overwritten and the graph silently computes garbage.
        # `_c10d_functional.all_reduce` carries the right semantics, and
        # inductor's reinplacing pass still recovers the in-place device op.
        out = torch.ops._c10d_functional.all_reduce(
            input_,  # ty: ignore[invalid-argument-type]
            "sum",  # ty: ignore[invalid-argument-type]
            self._group_name,  # ty: ignore[invalid-argument-type]
        )
        return torch.ops._c10d_functional.wait_tensor(out)

    # libspyre_comms allgather transfers each rank's buffer in 64-element
    # chunks along the gathered dim; a shard whose size along `dim` is not a
    # multiple of 64 gets its tail rounded off, so every following rank's data
    # lands shifted. Pad each rank's contribution up to a 64 multiple before
    # the gather and strip the padding afterward. (E.g. TP=2 vocab-parallel
    # logits with per-rank width 24608 = 384*64 + 32 previously shifted rank 1
    # down by 32, corrupting the argmax for its half of the vocab.)
    #
    # This is not merely a correctness issue: as observed on comms build 121,
    # a list-form `dist.all_gather` of a non-64-multiple fp16 shard faults the
    # card outright and needs a device recovery before the next run. Earlier
    # comms builds were never tested for the fault, so read 121 as "where it
    # was observed", not "where it started". Do not drop this padding without
    # re-checking on real hardware.
    _GATHER_ALIGN = 64

    def all_gather(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        # Two independent reasons this can't use the functional form that
        # `all_reduce` does:
        #  1. Eager `_c10d_functional.all_gather_into_tensor` routes to
        #     `allgather_into_tensor_coalesced`, which the spyreccl backend
        #     rejects outright ("Backend SpyreCCL does not support ...").
        #  2. Compiled, it lowers to `spyre::all_gather_async`, whose
        #     reassembly narrows the output along dim 0 — a storage offset of
        #     `rank * per_rank_numel`. When `per_rank_numel` is not a multiple
        #     of 64 that offset lands inside a stick and the `copy_from_d2d`
        #     lowering rejects it. The alignment padding below is exactly what
        #     would fix that, but building it on device hits the unaligned
        #     `F.pad` bug described further down, so there is no all-device
        #     sequence available today.
        # The one all_gather on the TP forward path (vocab-parallel logits in
        # `SpyreLogitsProcessor`) runs outside the compiled region and moves
        # its result to CPU for sampling anyway, so eager costs us nothing.
        # REPLACE-WITH-NATIVE: when torch-spyre wires up `_allgather_base` /
        # the coalesced entry point and lands stick-offset support, this whole
        # override can go and the base class can gather directly.
        if self.world_size == 1:
            return input_
        if input_.device.type == "cpu":
            return super().all_gather(input_, dim)

        dim = dim % input_.dim()
        orig_size = input_.shape[dim]
        pad = (-orig_size) % self._GATHER_ALIGN
        if not pad:
            output_list = [torch.empty_like(input_) for _ in range(self.world_size)]
            dist.all_gather(  # ty: ignore[possibly-missing-attribute]
                output_list, input_, group=self.device_group
            )
            return torch.cat(output_list, dim=dim)

        # Pad this rank's contribution up to a 64 multiple so the transfer does
        # not round off its tail. The pad is built on CPU on purpose: padding on
        # device writes the tail starting `orig_size % 64` elements into a stick,
        # and torch-spyre's layout pass cannot express a mutation whose write
        # stick carries an offset ("no offset-free alternative stick dim for
        # mutation target"). Strip the padding and re-concatenate on CPU too
        # (Spyre slicing/narrow corrupts memory — see spyre_attn.py), restoring
        # the exact per-rank shard layout the caller expects.
        pad_spec = [0, 0] * (input_.dim() - dim - 1) + [0, pad]
        padded = convert(
            torch.nn.functional.pad(convert(input_, device="cpu"), pad_spec).contiguous(),
            device=input_.device,
        )
        output_list = [torch.empty_like(padded) for _ in range(self.world_size)]
        dist.all_gather(  # ty: ignore[possibly-missing-attribute]
            output_list, padded, group=self.device_group
        )
        stripped = [convert(o, device="cpu").narrow(dim, 0, orig_size) for o in output_list]
        return convert(torch.cat(stripped, dim=dim), device=input_.device)

    def reduce_scatter(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        # Not on the standard TP path; raise loudly if anything tries it.
        if self.world_size == 1:
            return input_
        raise NotImplementedError(
            f"SpyreCommunicator: reduce_scatter has no Spyre implementation and no "
            f"fallback for world_size={self.world_size}. Either wait for the upstream "
            f"comms implementation to land + a comms RPM rebuild, or extend "
            f"SpyreCommunicator with a manual fallback."
        )
