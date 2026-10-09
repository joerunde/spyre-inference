"""Warmup measurement harness for the MRV2 warmup-option analysis (#906).

Runs one in-process vLLM engine on Spyre and records, per warmup phase:
wall time, attention executions, Dynamo graphs, Inductor FX-graph-cache
hits/misses/bypasses, Spyre kernel requests and real backend (dbo-opt) compiles.

MEAS_MODE:
  B  baseline: today's V1 decoder warmup (dummies skip attention)  == option B
  D  dummies run attention, MRV2-style split (force_attention=True) == option D
  A  dummies run attention with a single request                    == option A
"""

import json
import os
import sys
import time

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

import torch  # noqa: E402
from torch._dynamo.utils import counters  # noqa: E402

MODE = os.environ.get("MEAS_MODE", "B")
OUT = os.environ["MEAS_OUT"]
MODEL = os.environ.get("MEAS_MODEL", "ibm-granite/granite-3.3-8b-instruct")
MAX_LEN = int(os.environ.get("MEAS_MAX_LEN", "2048"))
MAX_SEQS = int(os.environ.get("MEAS_MAX_SEQS", "32"))

from spyre_inference.v1.attention.backends import spyre_attn as sa  # noqa: E402
from spyre_inference.v1.worker import spyre_model_runner as smr  # noqa: E402
from torch_spyre.execution import async_compile as ac  # noqa: E402

S = {"backend": 0, "backend_s": 0.0, "sdsc": 0, "attn": 0, "phase": "init"}
EVENTS = []


def sync():
    try:
        torch.spyre.synchronize()
    except Exception:
        pass


def snap():
    return {
        "unique_graphs": counters["stats"]["unique_graphs"],
        "fx_hit": counters["inductor"]["fxgraph_cache_hit"],
        "fx_miss": counters["inductor"]["fxgraph_cache_miss"],
        "fx_bypass": counters["inductor"]["fxgraph_cache_bypass"],
        "sdsc": S["sdsc"],
        "backend": S["backend"],
        "attn": S["attn"],
    }


def delta(a, b):
    return {k: b[k] - a[k] for k in a}


_orig_submit = ac.SpyreAsyncCompile._submit_backend_compile


def _submit(self, *a, **k):
    S["backend"] += 1
    t = time.perf_counter()
    r = _orig_submit(self, *a, **k)
    S["backend_s"] += time.perf_counter() - t
    return r


ac.SpyreAsyncCompile._submit_backend_compile = _submit

_orig_sdsc = ac.SpyreAsyncCompile.sdsc


def _sdsc(self, *a, **k):
    S["sdsc"] += 1
    return _orig_sdsc(self, *a, **k)


ac.SpyreAsyncCompile.sdsc = _sdsc

_orig_attn = sa.SpyreAttentionImpl._online_softmax_attention


def _attn(self, *a, **k):
    S["attn"] += 1
    return _orig_attn(self, *a, **k)


sa.SpyreAttentionImpl._online_softmax_attention = _attn

import collections  # noqa: E402

from torch._inductor import codecache as _cc  # noqa: E402

BYPASS = collections.Counter()
_orig_bypass_init = _cc.BypassFxGraphCache.__init__


def _bypass_init(self, *a, **k):
    BYPASS[str(a[0])[:160] if a else "?"] += 1
    _orig_bypass_init(self, *a, **k)


_cc.BypassFxGraphCache.__init__ = _bypass_init

R = smr.TorchSpyreModelRunner
_orig_dummy = R._dummy_run
_orig_record = R._record_attention_graphs
_orig_warm = R.warming_up_model


def _dummy(self, *args, **kwargs):
    decoder_warmup = S["phase"] == "warmup" and not self.is_pooling_model
    n = args[0] if args else kwargs.get("num_tokens")
    restore = None
    if decoder_warmup and MODE in ("D", "A"):
        kwargs["force_attention"] = True
        if MODE == "A":
            restore = self.scheduler_config.max_num_seqs
            self.scheduler_config.max_num_seqs = 1
    s0, t0 = snap(), time.perf_counter()
    try:
        out = _orig_dummy(self, *args, **kwargs)
        sync()
    finally:
        if restore is not None:
            self.scheduler_config.max_num_seqs = restore
    EVENTS.append({"what": "dummy", "n": n, "s": time.perf_counter() - t0, **delta(s0, snap())})
    return out


def _record(self, *a, **k):
    s0, t0 = snap(), time.perf_counter()
    out = _orig_record(self, *a, **k)
    sync()
    EVENTS.append({"what": "record_attention", "s": time.perf_counter() - t0, **delta(s0, snap())})
    return out


def _warm(self, *a, **k):
    S["phase"] = "warmup"
    s0, t0 = snap(), time.perf_counter()
    out = _orig_warm(self, *a, **k)
    sync()
    EVENTS.append({"what": "warming_up_model", "s": time.perf_counter() - t0, **delta(s0, snap())})
    S["phase"] = "serve"
    return out


R._dummy_run = _dummy
R._record_attention_graphs = _record
R.warming_up_model = _warm

from vllm import LLM, SamplingParams  # noqa: E402

t0 = time.perf_counter()
s0 = snap()
llm = LLM(model=MODEL, max_model_len=MAX_LEN, max_num_seqs=MAX_SEQS, tensor_parallel_size=1)
init_s = time.perf_counter() - t0
init_delta = delta(s0, snap())

s1, t1 = snap(), time.perf_counter()
outs = llm.generate(["The capital of France is"], SamplingParams(max_tokens=16, temperature=0))
gen = {"s": time.perf_counter() - t1, **delta(s1, snap()), "text": outs[0].outputs[0].text}

res = {
    "mode": MODE, "model": MODEL, "max_model_len": MAX_LEN, "max_num_seqs": MAX_SEQS,
    "kernel_cache": os.environ.get("SPYRE_KERNEL_CACHE", "0"),
    "cache_dir": os.environ.get("TORCHINDUCTOR_CACHE_DIR"),
    "init_s": init_s, "init": init_delta, "backend_compile_s_submit": S["backend_s"],
    "events": EVENTS, "generate": gen, "fx_bypass_reasons": dict(BYPASS),
}
json.dump(res, open(OUT, "w"), indent=1)
print("MEAS_DONE", json.dumps({k: res[k] for k in ("mode", "init_s")}), file=sys.stderr)
