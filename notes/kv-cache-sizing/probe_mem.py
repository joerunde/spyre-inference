import torch, torch_spyre, sys
chunk = int(sys.argv[1]) if len(sys.argv) > 1 else 1  # GiB
ts = []
try:
    while True:
        ts.append(torch.empty(chunk * (1 << 30) // 2, dtype=torch.float16, device="spyre"))
        print(f"allocated {len(ts)*chunk} GiB", flush=True)
except Exception as e:
    print("STOP at", len(ts)*chunk, "GiB:", type(e).__name__, str(e)[:400])
print(torch.accelerator.memory_stats() if hasattr(torch.accelerator,'memory_stats') else '')
try: print("get_memory_info", torch.accelerator.get_memory_info())
except Exception as e: print("get_memory_info:", str(e)[:200])
