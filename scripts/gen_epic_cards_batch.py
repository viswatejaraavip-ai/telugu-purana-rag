"""Chapter cards for the Itihasas at a tenth of the Purana cost.

Same card schema as gen_cards.py (the Space loads them unchanged), but the model
is Haiku and it is handed the chapter's public-domain English translation
(english/<work>.jsonl, from Wikisource) beside the Sanskrit, so it reads a
known-good rendering rather than translating 19th-century-edition Sanskrit cold.
The Sanskrit is still the only source of verse ids for anchors. Summaries are
kept to 3-5 sentences: the English is the meaning layer at answer time.

  submit  ramayana [--model claude-haiku-4-5-20251001] [--max-usd 5] [--dry-run]
  collect

Cards land in cards/<work>/<adhyaya_id>.json like the others; pack them with
cards_ws/ for the Space.
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(__file__))
from common import CARDS, read_jsonl
from gen_cards import ChapterCard, PRICES
import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS = os.path.join(ROOT, "corpus_ws")
ENGLISH = os.path.join(ROOT, "english")
LEDGER = os.path.join(ROOT, "work", "epic_batches.json")
PRICES.setdefault("claude-haiku-4-5-20251001", PRICES["claude-haiku-4-5"])

SYSTEM = """You are a Telugu pandit preparing a searchable index of the Itihasas (Valmiki Ramayana, Mahabharata)
for Telugu-speaking devotees. You are given one chapter: its Sanskrit verses in IAST with ids, and a
public-domain English translation of the same chapter to help you read it. You write plain, warm,
modern Telugu that an ordinary devotee reads without a dictionary. Use the Telugu forms of names
(రాముడు, సీత, లక్ష్మణుడు, హనుమంతుడు, రావణుడు, భీష్ముడు, కర్ణుడు, ద్రౌపది), never IAST in Telugu fields.
Stay strictly within this chapter: no events from elsewhere in the epic, no general memory of the story.
summary_te is 5-8 sentences of plain modern Telugu telling what happens in this chapter, in order.
questions_te: 8-12 questions, including at least 3 phrased as a devotee's own life situation without
naming any character. Verse ids in anchors must be copied exactly from the input;
anchor 3-6 key moments. If the chapter is a hymn, a description, or a discourse rather than events, say
so in the summary and write the questions about that content."""


def prompt(man, verses, en):
    head = (f"{man['work']} ({man['work_te']}), book {man['book_no']}, chapter {man['adhyaya_no']}, "
            f"{man['n_verses']} verses.\n\n")
    sa = "\n".join(f"[{v['id']}] {v['sloka_iast']}" for v in verses)
    e = f"\n\nENGLISH TRANSLATION ({en['translator']}, public domain) of this chapter:\n{en['text_en'][:6000]}" if en else \
        "\n\n(No English translation is available for this chapter; read the Sanskrit.)"
    return head + "SANSKRIT (IAST):\n" + sa + e


def est_cost(model, n_verses, en_chars):
    pin, pout, _, _ = PRICES[model]
    return 0.5 * ((900 + 30 * n_verses + en_chars / 4) * pin + 2400 * pout) / 1e6


def load_ledger():
    return json.load(open(LEDGER)) if os.path.exists(LEDGER) else []


def save_ledger(l):
    os.makedirs(os.path.dirname(LEDGER), exist_ok=True); json.dump(l, open(LEDGER, "w"), indent=1)


def submit(a):
    client = anthropic.Anthropic()
    from anthropic import transform_schema
    schema = transform_schema(ChapterCard)
    ledger = load_ledger(); pending = {c for b in ledger for c in b["custom_ids"]}
    reqs, est, cidmap = [], 0.0, {}
    for work in a.works:
        import glob as _glob   # a big work is split into <work>.jsonl + <work>.part2.jsonl …
        verses = {v["id"]: v for p in sorted(_glob.glob(os.path.join(CORPUS, f"{work}.jsonl")) + _glob.glob(os.path.join(CORPUS, f"{work}.part*.jsonl")))
                  for v in read_jsonl(p)}
        mans = read_jsonl(os.path.join(CORPUS, f"{work}_adhyayas.jsonl"))
        en = {r["adhyaya_id"]: r for r in read_jsonl(os.path.join(ENGLISH, f"{work}.jsonl"))} if os.path.exists(os.path.join(ENGLISH, f"{work}.jsonl")) else {}
        if a.book: mans = [m for m in mans if str(m["book_no"]) == a.book]
        if a.limit: mans = mans[:a.limit]
        n_work = 0
        for m in mans:
            cid = f"{work}__{m['adhyaya_id']}".replace(".", "_")
            cidmap[cid] = [work, m["adhyaya_id"]]
            if os.path.exists(os.path.join(CARDS, work, m["adhyaya_id"] + ".json")) or cid in pending:
                continue
            # Griffith condensed the Sundara and Yuddha kandas, so canto N is not sarga N there:
            # the English is offered only for books where the two numberings agree.
            e = en.get(m["adhyaya_id"]) if (not a.english_books or str(m["book_no"]) in a.english_books) else None
            c = est_cost(a.model, m["n_verses"], len(e["text_en"]) if e else 0)
            if a.max_usd and est + c > a.max_usd:
                break
            est += c; n_work += 1
            params = dict(model=a.model, max_tokens=8000,
                          output_config={"format": {"type": "json_schema", "schema": schema}},
                          system=SYSTEM, messages=[{"role": "user", "content": prompt(m, [verses[i] for i in m["verse_ids"]], e)}])
            if a.model.startswith(("claude-sonnet", "claude-opus")):   # as the Purana cards were made
                params["thinking"] = {"type": "adaptive"}; params["output_config"]["effort"] = "medium"
            reqs.append(Request(custom_id=cid, params=MessageCreateParamsNonStreaming(**params)))
        print(f"  {work:12s} {n_work:4d} of {len(mans):4d} chapters queued ({len(en)} with English)")
    print(f"{len(reqs)} requests, estimated ${est:.2f} at batch prices ({a.model})")
    if a.dry_run or not reqs:
        return
    for i in range(0, len(reqs), 500):
        chunk = reqs[i:i + 500]
        b = client.messages.batches.create(requests=chunk)
        ledger.append(dict(id=b.id, model=a.model, custom_ids=[r["custom_id"] for r in chunk],
                           map={r["custom_id"]: cidmap[r["custom_id"]] for r in chunk}, status=b.processing_status, collected=[]))
        save_ledger(ledger); print(f"submitted {b.id}: {len(chunk)} requests")


def collect(a):
    client = anthropic.Anthropic()
    ledger = load_ledger(); tot = 0.0; wrote = 0; bad = []
    for b in ledger:
        if len(b["collected"]) >= len(b["custom_ids"]):
            continue
        st = client.messages.batches.retrieve(b["id"]); b["status"] = st.processing_status
        rc = st.request_counts
        print(f"{b['id']}: {st.processing_status} processing={rc.processing} succeeded={rc.succeeded} errored={rc.errored}")
        if st.processing_status != "ended":
            continue
        pin, pout, pcr, pcw = PRICES[b["model"]]
        mans_cache = {}
        for res in client.messages.batches.results(b["id"]):
            cid = res.custom_id
            if cid in b["collected"]:
                continue
            work, aid = b["map"][cid]
            if res.result.type != "succeeded":
                bad.append((cid, res.result.type)); b["collected"].append(cid); continue
            msg = res.result.message; u = msg.usage
            cost = 0.5 * (u.input_tokens * pin + u.output_tokens * pout) / 1e6; tot += cost
            try:
                text = next(blk.text for blk in msg.content if blk.type == "text")
                card = ChapterCard(**json.loads(text))
            except Exception as e:
                bad.append((cid, "parse", str(e)[:80])); b["collected"].append(cid); continue
            if work not in mans_cache:
                mans_cache[work] = {m["adhyaya_id"]: m for m in read_jsonl(os.path.join(CORPUS, f"{work}_adhyayas.jsonl"))}
            man = mans_cache[work][aid]
            known = set(man["verse_ids"]); badids = [i for an in card.anchors for i in an.verse_ids if i not in known]
            rec = dict(adhyaya_id=aid, work=man["work"], work_te=man["work_te"], book_no=man["book_no"], adhyaya_no=man["adhyaya_no"],
                       n_verses=man["n_verses"], citation_te=man["citation_te"], card=card.model_dump(),
                       gen=dict(model=b["model"] + " (batch, with English)", seconds=None, usage=u.model_dump(), cost_usd=round(cost, 4),
                                bad_anchor_ids=badids, request_id=b["id"], generated_at=time.strftime("%Y-%m-%dT%H:%M:%S")),
                       review_status="unreviewed")
            os.makedirs(os.path.join(CARDS, work), exist_ok=True)
            p = os.path.join(CARDS, work, aid + ".json")
            json.dump(rec, open(p + ".part", "w", encoding="utf-8"), ensure_ascii=False, indent=1); os.replace(p + ".part", p)
            b["collected"].append(cid); wrote += 1
        save_ledger(ledger)
    print(f"wrote {wrote} cards, cost this collect=${tot:.2f}, problems={len(bad)}")
    for x in bad[:20]:
        print("  ", x)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("mode", choices=["submit", "collect"]); ap.add_argument("works", nargs="*")
    ap.add_argument("--model", default="claude-haiku-4-5-20251001"); ap.add_argument("--max-usd", type=float)
    ap.add_argument("--dry-run", action="store_true"); ap.add_argument("--book"); ap.add_argument("--limit", type=int)
    ap.add_argument("--english-books", help="comma-separated book numbers whose English translation is aligned to the Sanskrit chapters")
    a = ap.parse_args(); a.english_books = set(a.english_books.split(",")) if a.english_books else None
    submit(a) if a.mode == "submit" else collect(a)
