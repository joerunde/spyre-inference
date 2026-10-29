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

import importlib.metadata
import logging
import os

# Defer torch_spyre's autoload until we explicitly trigger it inside
# `TorchSpyreWorker.init_device`. Autoload loads `libspyre_comms.so`,
# which captures `RANK`/`WORLD_SIZE`/`LOCAL_RANK`/`LOCAL_WORLD_SIZE`
# at dlopen time — those env vars are only known per-worker, so the
# library can't load before init_device runs.
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

__version__ = importlib.metadata.version("spyre_inference")


def register():
    """Register the Spyre platform."""
    # vLLM resolves the platform during `import vllm`, before vllm.triton_utils is
    # imported, which is the last point the Triton gate can act.
    from spyre_inference import envs
    from spyre_inference.v1.worker.spyre import triton_gate

    # An exception here would be swallowed by vLLM's plugin probe and leave the
    # platform unset, so an invalid SPYRE_MODEL_RUNNER is rejected in
    # check_and_update_config instead.
    if envs.SPYRE_MODEL_RUNNER == "v2" and envs.SPYRE_MRV2_KERNELS == "triton":
        triton_gate.open()
    elif envs.SPYRE_MODEL_RUNNER == "v2":
        triton_gate.open_host()
    else:
        triton_gate.close()
    return "spyre_inference.platform.TorchSpyrePlatform"


def register_ops():
    """Register the Spyre OOT custom ops and model adaptations."""
    from spyre_inference.custom_ops import register_all
    from spyre_inference.models import register_models

    register_all()
    register_models()


def _init_logging():
    """Route our loggers through vLLM's ``vllm`` logger.

    Since 0.31 vLLM configures logging lazily (``vllm.logger.configure_logging``,
    from the engine, workers and CLI) and only for the ``vllm`` logger, whose format
    needs a field its record factory adds. Parenting ours under it inherits whatever
    handlers and level vLLM installs, whenever it does; configuring our own at import
    would format records before that factory exists.
    """
    logging.getLogger("spyre_inference").parent = logging.getLogger("vllm")


_init_logging()
