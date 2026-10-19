"""usage: kv_probe.py OUT.json [--blocks N] [--kv-bytes B] [--gmu F] [--max-model-len L] [--max-num-seqs S]"""
import argparse, json, time, os
ap = argparse.ArgumentParser()
ap.add_argument("out"); ap.add_argument("--model", default="/models/granite-3.3-8b-instruct")
ap.add_argument("--blocks", type=int); ap.add_argument("--kv-bytes", type=int); ap.add_argument("--gmu", type=float)
ap.add_argument("--max-model-len", type=int, default=2048); ap.add_argument("--max-num-seqs", type=int, default=4)
ap.add_argument("--in-len", type=int, default=128); ap.add_argument("--out-len", type=int, default=128)
a = ap.parse_args()
import numpy as np
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

def mem(_self=None):
    import torch
    s = torch.accelerator.memory_stats()
    return {"allocated_bytes": s["allocated_bytes.all.current"], "peak": s["allocated_bytes.all.peak"], "n_alloc": s["allocation.all.current"]}

def main():
    kw = {}
    if a.blocks is not None: kw["num_gpu_blocks_override"] = a.blocks
    if a.kv_bytes is not None: kw["kv_cache_memory_bytes"] = a.kv_bytes
    if a.gmu is not None: kw["gpu_memory_utilization"] = a.gmu
    res = {"args": vars(a)}
    t0 = time.perf_counter()
    try:
        llm = LLM(model=a.model, tensor_parallel_size=2, dtype="float16", max_model_len=a.max_model_len, max_num_seqs=a.max_num_seqs, **kw)
    except Exception as e:
        res["error"] = f"{type(e).__name__}: {str(e)[:1500]}"; print(json.dumps(res, indent=1)); json.dump(res, open(a.out, "w"), indent=1); raise SystemExit(1)
    res["init_s"] = time.perf_counter() - t0
    cc = llm.llm_engine.vllm_config.cache_config
    res["num_gpu_blocks"] = cc.num_gpu_blocks; res["override"] = cc.num_gpu_blocks_override; res["kv_cache_memory_bytes"] = cc.kv_cache_memory_bytes
    res["rank_mem"] = llm.collective_rpc(mem)
    rng = np.random.default_rng(0); vocab = llm.llm_engine.model_config.get_vocab_size()
    P = lambda n: [TokensPrompt(prompt_token_ids=rng.integers(100, vocab - 100, a.in_len).tolist()) for _ in range(n)]
    llm.generate(P(a.max_num_seqs), SamplingParams(max_tokens=8, ignore_eos=True), use_tqdm=False)
    t = time.perf_counter()
    o = llm.generate(P(a.max_num_seqs), SamplingParams(max_tokens=1, ignore_eos=True, temperature=0.0), use_tqdm=False); ttft = time.perf_counter() - t
    t = time.perf_counter()
    o = llm.generate(P(a.max_num_seqs), SamplingParams(max_tokens=a.out_len, ignore_eos=True, temperature=0.0), use_tqdm=False); wall = time.perf_counter() - t
    res["prefill_batch_s"] = ttft; res["gen_wall_s"] = wall
    res["decode_ms_per_step"] = (wall - ttft) / (a.out_len - 1) * 1000
    res["tokens0"] = list(o[0].outputs[0].token_ids[:16])
    print("RESULT", json.dumps(res)); json.dump(res, open(a.out, "w"), indent=1)
    llm.llm_engine.engine_core.shutdown()

if __name__ == '__main__':
    main()
