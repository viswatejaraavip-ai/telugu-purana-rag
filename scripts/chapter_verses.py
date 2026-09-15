"""Print one adhyaya's header + verses in IAST for a card-writing agent.
usage: .venv/bin/python scripts/chapter_verses.py <work> <adhyaya_id>   e.g. bhagavata bhp-3-12"""
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from common import OUT, read_jsonl
work, aid = sys.argv[1], sys.argv[2]
man = next(m for m in read_jsonl(os.path.join(OUT, f"{work}_adhyayas.jsonl")) if m["adhyaya_id"] == aid)
want = set(man["verse_ids"])
verses = [r for r in read_jsonl(os.path.join(OUT, f"{work}.jsonl")) if r["id"] in want]
print("HEADER (copy these top-level fields verbatim into the card record):")
print(json.dumps({k: man[k] for k in ("adhyaya_id", "work", "work_te", "book_no", "adhyaya_no", "n_verses", "citation_te")}, ensure_ascii=False))
print(f"speakers: {', '.join(man['speakers']) or 'none marked'}\n")
for v in verses:
    sp = f" ({v['speaker_iast']})" if v.get("speaker_iast") else ""
    print(f"[{v['id']}]{sp}\n{v['sloka_iast']}")
