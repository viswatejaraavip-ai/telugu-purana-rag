"""Coverage table for the re-sourced verse layer: for every card corpus, how many of
its verse ids exist in corpus_ws/, and how many are simply absent from the source."""
import json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rows = []
tot_old = tot_new = 0
for f in sorted(os.listdir(os.path.join(ROOT, "corpus"))):
    if not f.endswith(".jsonl") or f.endswith("_adhyayas.jsonl") or f == "tatparyam_cache.jsonl":
        continue
    key = f[:-6]
    old = [json.loads(l)["id"] for l in open(os.path.join(ROOT, "corpus", f), encoding="utf-8")]
    p = os.path.join(ROOT, "corpus_ws", f)
    if not os.path.exists(p):
        rows.append((key, len(old), None, None)); tot_old += len(old); continue
    new = {json.loads(l)["id"] for l in open(p, encoding="utf-8")}
    hit = sum(1 for i in old if i in new)
    rows.append((key, len(old), hit, len(new)))
    tot_old += len(old); tot_new += hit
print(f"{'work':22s} {'card verses':>11s} {'resolved':>9s} {'%':>6s} {'ws verses':>9s}")
for key, n, hit, nn in rows:
    print(f"{key:22s} {n:11d} " + (f"{hit:9d} {100*hit/n:6.1f} {nn:9d}" if hit is not None else "        -      -         -  (no open source)"))
print(f"{'TOTAL':22s} {tot_old:11d} {tot_new:9d} {100*tot_new/tot_old:6.1f}")
