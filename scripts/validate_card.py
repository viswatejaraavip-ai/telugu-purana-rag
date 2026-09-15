"""Validate a chapter card against the schema and the corpus. Exit 1 on any problem.
usage: .venv/bin/python scripts/validate_card.py <work> <adhyaya_id>"""
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from common import OUT, CARDS, read_jsonl
from gen_cards import ChapterCard
work, aid = sys.argv[1], sys.argv[2]
path = os.path.join(CARDS, work, aid + ".json")
errs = []
try:
    r = json.load(open(path, encoding="utf-8"))
except Exception as e:
    print(f"FAIL {aid}: cannot load {path}: {e}"); sys.exit(1)
man = next(m for m in read_jsonl(os.path.join(OUT, f"{work}_adhyayas.jsonl")) if m["adhyaya_id"] == aid)
for k in ("adhyaya_id", "work", "work_te", "book_no", "adhyaya_no", "n_verses", "citation_te"):
    if r.get(k) != man[k]: errs.append(f"top-level {k}={r.get(k)!r} expected {man[k]!r}")
if r.get("review_status") != "unreviewed": errs.append("review_status must be 'unreviewed'")
if not isinstance(r.get("gen"), dict) or "model" not in r["gen"]: errs.append("gen.model missing")
try:
    c = ChapterCard(**r["card"])
    ids = set(man["verse_ids"])
    bad = [i for a in c.anchors for i in a.verse_ids if i not in ids]
    if bad: errs.append(f"anchor ids not in chapter: {bad}")
    qmin = 4 if man["n_verses"] < 5 else 8
    if not (qmin <= len(c.questions_te) <= 16): errs.append(f"questions_te count {len(c.questions_te)} (want {qmin}-16)")
    if not (3 <= len(c.anchors) <= 10): errs.append(f"anchors count {len(c.anchors)} (want 3-10)")
    if not (5 <= len(c.entities) <= 60): errs.append(f"entities count {len(c.entities)} (want 5-60)")
    import re
    latin = [f for f in ("title_te", "summary_te", "moral_te") if re.search(r"[A-Za-z]", getattr(c, f) or "")]
    if latin: errs.append(f"Latin letters inside Telugu fields: {latin}")
except Exception as e:
    errs.append(f"schema: {e}")
if errs:
    print(f"FAIL {aid}:"); [print("  -", e) for e in errs]; sys.exit(1)
print(f"ok {aid} q={len(c.questions_te)} ent={len(c.entities)} anchors={len(c.anchors)}")
