"""Validate every card of a work in one process (fast). usage: validate_all.py <work>"""
import glob, json, os, re, sys
sys.path.insert(0, os.path.dirname(__file__))
from common import OUT, CARDS, read_jsonl
from gen_cards import ChapterCard
work = sys.argv[1]
mans = {m["adhyaya_id"]: m for m in read_jsonl(os.path.join(OUT, f"{work}_adhyayas.jsonl"))}
ok, fails, models = 0, [], {}
for p in sorted(glob.glob(os.path.join(CARDS, work, "*.json"))):
    aid = os.path.basename(p)[:-5]; errs = []
    try:
        r = json.load(open(p, encoding="utf-8")); man = mans[aid]
        for k in ("adhyaya_id", "work", "work_te", "book_no", "adhyaya_no", "n_verses", "citation_te"):
            if r.get(k) != man[k]: errs.append(f"{k}={r.get(k)!r}!={man[k]!r}")
        c = ChapterCard(**r["card"]); ids = set(man["verse_ids"])
        bad = [i for a in c.anchors for i in a.verse_ids if i not in ids]
        if bad: errs.append(f"bad anchors {bad}")
        qmin = 4 if man["n_verses"] < 5 else 8   # one- or two-verse chapters cannot support 8 questions
        if not (qmin <= len(c.questions_te) <= 16): errs.append(f"questions {len(c.questions_te)}")
        if re.search(r"[A-Za-z]", c.title_te + c.summary_te + c.moral_te): errs.append("latin in telugu fields")
        m = r.get("gen", {}).get("model", "?"); models[m] = models.get(m, 0) + 1
    except Exception as e:
        errs.append(f"schema/load: {e}")
    if errs: fails.append((aid, errs))
    else: ok += 1
print(f"{work}: {ok} ok, {len(fails)} fail, missing={len(set(mans) - {os.path.basename(p)[:-5] for p in glob.glob(os.path.join(CARDS, work, '*.json'))})}")
print("by model:", models)
for aid, errs in fails: print("FAIL", aid, errs)
