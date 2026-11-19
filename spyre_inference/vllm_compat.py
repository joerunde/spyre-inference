import importlib.util

# SPIKE: the one switch for "is vLLM <= 0.33". find_spec, not an import, so
# asking costs nothing and pulls in no MRV1 code.
HAS_MRV1 = importlib.util.find_spec("vllm.v1.worker.gpu_model_runner") is not None
