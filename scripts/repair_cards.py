"""Repair or discard cards that fail validation.
- anchor ids in GRETIL source style (BrP_114.6) are mapped to corpus ids; ids not in the chapter are dropped;
  empty anchors dropped. If < 3 anchors remain, the card is deleted for regeneration.
- 'latin in telugu fields' or 'questions 0' -> card deleted for regeneration.
usage: repair_cards.py <work>..."""
import glob, json, os, re, sys
sys.path.insert(0, os.path.dirname(__file__))
from common import OUT, CARDS, read_jsonl
from gen_cards import ChapterCard
for work in sys.argv[1:]:
    mans = {m["adhyaya_id"]: m for m in read_jsonl(os.path.join(OUT, f"{work}_adhyayas.jsonl"))}
    for p in sorted(glob.glob(os.path.join(CARDS, work, "*.json"))):
        aid = os.path.basename(p)[:-5]; r = json.load(open(p, encoding="utf-8")); c = r["card"]; man = mans[aid]
        ids = set(man["verse_ids"]); prefix = aid  # e.g. brp-1-114
        reasons = []
        if not c.get("questions_te"): reasons.append("questions 0")
        if re.search(r"[A-Za-z]", c.get("title_te", "") + c.get("summary_te", "") + c.get("moral_te", "")): reasons.append("latin")
        changed = False; new_anchors = []
        for a in c.get("anchors", []):
            keep = []
            for i in a["verse_ids"]:
                if i in ids: keep.append(i); continue
                m = re.search(r"(\d+)$", i.replace(".", "-"))          # BrP_114.6 -> verse 6 ; brp-1-114-108 -> 108
                cand = f"{prefix}-{m.group(1)}" if m else None
                if cand in ids: keep.append(cand); changed = True
                else: changed = True
            if keep: new_anchors.append(dict(a, verse_ids=keep))
        if len(new_anchors) < 3: reasons.append(f"anchors {len(new_anchors)}")
        if reasons:
            os.remove(p); print(f"DELETE {aid}: {reasons}"); continue
        if changed:
            c["anchors"] = new_anchors; ChapterCard(**c); r["gen"]["bad_anchor_ids"] = []; r["gen"]["repaired"] = "anchors"
            json.dump(r, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1); print(f"REPAIRED {aid}: anchors -> {len(new_anchors)}")
