import json
import sys

M = "/home/senuser/.claude/jobs/709f9980/tmp/meas"
names = sys.argv[1:] or ["B-cold", "D-cold", "A-cold", "B-restart-kc0", "B-cold-kc1", "B-warm-kc1"]
rows = []
for n in names:
    r = json.load(open(f"{M}/res-{n}.json"))
    ev = r["events"]
    dummies = [e for e in ev if e["what"] == "dummy"]
    rec = next(e for e in ev if e["what"] == "record_attention")
    warm = next(e for e in ev if e["what"] == "warming_up_model")
    row = {
        "run": n, "init_s": round(r["init_s"]), "warmup_s": round(warm["s"]),
        "dummies_s": round(sum(e["s"] for e in dummies)), "record_s": round(rec["s"]),
        "other_warm_s": round(warm["s"] - sum(e["s"] for e in dummies) - rec["s"]),
        "graphs": warm["unique_graphs"], "fx_hit": warm["fx_hit"], "fx_miss": warm["fx_miss"],
        "fx_bypass": warm["fx_bypass"], "attn_graphs_dummies": sum(e["fx_bypass"] for e in dummies),
        "attn_graphs_record": rec["fx_bypass"], "backend": warm["backend"],
        "backend_record": rec["backend"], "text": r["generate"]["text"][:20].replace("\n", " "),
    }
    rows.append(row)
cols = list(rows[0])
print("| " + " | ".join(cols) + " |")
print("|" + "---|" * len(cols))
for row in rows:
    print("| " + " | ".join(str(row[c]) for c in cols) + " |")
json.dump(rows, open(f"{M}/summary.json", "w"), indent=1)
