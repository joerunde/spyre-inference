"""CLIP text pooling throughput/latency on Spyre: run once per arm (ragged on main, rect on the prototype)."""
import json, statistics, sys, time
import torch, torch.nn.functional as F

MODEL = "openai/clip-vit-base-patch32"
REV = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
arm = sys.argv[1]
MAX_SEQS = int(sys.argv[2]) if len(sys.argv) > 2 else 32

WORDS = "a photo of the big small red blue green cat dog city skyline at night bowl fruit near window couch sitting on".split()
def prompt(i):
    n = 4 + (i * 7) % 40  # 4..43 words, ~6..60 tokens
    return " ".join(WORDS[(i + k) % len(WORDS)] for k in range(n))

def main():
    from vllm import LLM
    t0 = time.perf_counter()
    llm = LLM(model=MODEL, revision=REV, tokenizer_revision=REV, runner="pooling",
              max_model_len=77, max_num_seqs=MAX_SEQS, enforce_eager=False,
              limit_mm_per_prompt={"image": 1})
    startup = time.perf_counter() - t0
    res = {"arm": arm, "max_num_seqs": MAX_SEQS, "startup_s": round(startup, 1)}

    # correctness vs HF CPU
    from transformers import CLIPModel, CLIPTokenizer
    hf = CLIPModel.from_pretrained(MODEL, revision=REV).eval()
    tok = CLIPTokenizer.from_pretrained(MODEL, revision=REV)
    check = [prompt(i) for i in range(MAX_SEQS)]
    outs = llm.embed(check, use_tqdm=False)
    sims = []
    with torch.inference_mode():
        for p, o in zip(check, outs):
            ref = F.normalize(hf.get_text_features(**tok(p, return_tensors="pt")).pooler_output.float(), dim=-1)[0]
            sims.append(F.cosine_similarity(torch.tensor(o.outputs.embedding), ref, dim=0).item())
    res["cosine_min"] = round(min(sims), 4)

    for conc in sorted({1, 8, MAX_SEQS}):
        batch = [prompt(i) for i in range(conc)]
        llm.embed(batch, use_tqdm=False)  # warm
        reps = max(5, 160 // conc)
        times = []
        for r in range(reps):
            t = time.perf_counter(); llm.embed(batch, use_tqdm=False); times.append(time.perf_counter() - t)
        med = statistics.median(times)
        res[f"c{conc}"] = {"reps": reps, "median_ms": round(med * 1e3, 2),
                           "p90_ms": round(sorted(times)[int(0.9 * (reps - 1))] * 1e3, 2),
                           "prompts_per_s": round(conc / med, 1)}
    # a large offline batch
    big = [prompt(i) for i in range(1024)]
    t = time.perf_counter(); llm.embed(big, use_tqdm=False); dt = time.perf_counter() - t
    res["offline_1024"] = {"s": round(dt, 2), "prompts_per_s": round(1024 / dt, 1)}
    print("BENCH_RESULT " + json.dumps(res), flush=True)

if __name__ == "__main__":
    main()
