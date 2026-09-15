"""Chapter cards via the Message Batches API (50% price). Same prompt/schema as gen_cards.py.

  submit  : build requests for every adhyaya of the given works that has no card, submit in
            batches of <=500, record batch ids in work/api_batches.json, print cost estimate.
  collect : poll all recorded batches; for ended ones, validate each result with ChapterCard,
            write cards/<work>/<adhyaya_id>.json, log usage + cost.

usage: gen_cards_batch.py submit vishnu markandeya ... [--model claude-sonnet-5] [--max-usd 19]
       gen_cards_batch.py collect
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(__file__))
from common import OUT, CARDS, read_jsonl
from gen_cards import ChapterCard, SYSTEM, chapter_prompt, PRICES
import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

LEDGER = os.path.join("work", "api_batches.json")


def est_cost(model, n_verses):
    """Batch price = half. Measured Sonnet non-batch: ~4.0k output tokens/chapter, ~25 IAST tokens/verse + 2.7k prompt."""
    pin, pout, _, _ = PRICES[model]
    return 0.5 * ((3500 + 70 * n_verses) * pin + 4200 * pout) / 1e6   # no cache in batch; calibrated on vip-6-8 (9.0k in, 4.5k out)


def load_ledger():
    return json.load(open(LEDGER)) if os.path.exists(LEDGER) else []


def save_ledger(l):
    os.makedirs("work", exist_ok=True); json.dump(l, open(LEDGER, "w"), indent=1)


def submit(a):
    client = anthropic.Anthropic()
    from anthropic import transform_schema
    schema = transform_schema(ChapterCard)
    ledger = load_ledger(); pending = {c for b in ledger for c in b["custom_ids"]}
    reqs, est, plan, cidmap = [], 0.0, [], {}
    for spec in a.works:
        work, _, books = spec.partition(":")          # e.g. garuda:2,3 limits to those book_no values
        verses = {v["id"]: v for v in read_jsonl(os.path.join(OUT, f"{work}.jsonl"))}
        mans = read_jsonl(os.path.join(OUT, f"{work}_adhyayas.jsonl"))
        if books: mans = [m for m in mans if str(m["book_no"]) in books.split(",")]
        n_work = 0
        for m in mans:
            cid = f"{work}__{m['adhyaya_id']}".replace(".", "_")     # batch custom_id: [a-zA-Z0-9_-] only
            cidmap[cid] = [work, m["adhyaya_id"]]
            if os.path.exists(os.path.join(CARDS, work, m["adhyaya_id"] + ".json")) or cid in pending: continue
            c = est_cost(a.model, m["n_verses"])
            if a.max_usd and est + c > a.max_usd: break
            est += c; n_work += 1
            reqs.append(Request(custom_id=cid, params=MessageCreateParamsNonStreaming(
                model=a.model, max_tokens=16000, thinking={"type": "adaptive"}, output_config={"effort": "medium",
                    "format": {"type": "json_schema", "schema": schema}},
                system=SYSTEM, messages=[{"role": "user", "content": chapter_prompt(m, [verses[i] for i in m["verse_ids"]])}])))
        plan.append((work, n_work, len(mans)))
        if a.max_usd and est >= a.max_usd: break
    for work, n, tot in plan: print(f"  {work:22s} {n:4d} of {tot:4d} adhyayas queued")
    print(f"{len(reqs)} requests, estimated ${est:.2f} at batch prices ({a.model})")
    if a.dry_run or not reqs: return
    for i in range(0, len(reqs), 500):
        chunk = reqs[i:i+500]
        b = client.messages.batches.create(requests=chunk)
        ledger.append(dict(id=b.id, model=a.model, created=b.created_at.isoformat() if hasattr(b.created_at, "isoformat") else str(b.created_at),
                           custom_ids=[r["custom_id"] for r in chunk], map={r["custom_id"]: cidmap[r["custom_id"]] for r in chunk},
                           status=b.processing_status, collected=[]))
        save_ledger(ledger); print(f"submitted {b.id}: {len(chunk)} requests, status={b.processing_status}")


def collect(a):
    client = anthropic.Anthropic()
    ledger = load_ledger(); tot_cost = 0.0; wrote = 0; bad = []
    for b in ledger:
        if len(b["collected"]) >= len(b["custom_ids"]): continue
        st = client.messages.batches.retrieve(b["id"]); b["status"] = st.processing_status
        rc = st.request_counts
        print(f"{b['id']}: {st.processing_status} processing={rc.processing} succeeded={rc.succeeded} errored={rc.errored} expired={rc.expired}")
        if st.processing_status != "ended": continue
        pin, pout, pcr, pcw = PRICES[b["model"]]
        for res in client.messages.batches.results(b["id"]):
            cid = res.custom_id
            if cid in b["collected"]: continue
            work, aid = b["map"][cid]
            if res.result.type != "succeeded":
                bad.append((cid, res.result.type, getattr(getattr(res.result, "error", None), "type", None))); b["collected"].append(cid); continue
            msg = res.result.message
            u = msg.usage
            cost = 0.5 * (u.input_tokens * pin + u.output_tokens * pout + (u.cache_read_input_tokens or 0) * pcr + (u.cache_creation_input_tokens or 0) * pcw) / 1e6
            tot_cost += cost
            try:
                text = next(blk.text for blk in msg.content if blk.type == "text")
                card = ChapterCard(**json.loads(text))
            except Exception as e:
                bad.append((cid, "parse", str(e)[:80])); b["collected"].append(cid); continue
            man = next(m for m in read_jsonl(os.path.join(OUT, f"{work}_adhyayas.jsonl")) if m["adhyaya_id"] == aid)
            known = set(man["verse_ids"]); badids = [i for an in card.anchors for i in an.verse_ids if i not in known]
            rec = dict(adhyaya_id=aid, work=man["work"], work_te=man["work_te"], book_no=man["book_no"], adhyaya_no=man["adhyaya_no"],
                       n_verses=man["n_verses"], citation_te=man["citation_te"], card=card.model_dump(),
                       gen=dict(model=b["model"] + " (batch)", effort="medium", seconds=None, usage=u.model_dump(), cost_usd=round(cost, 4),
                                bad_anchor_ids=badids, request_id=b["id"], generated_at=time.strftime("%Y-%m-%dT%H:%M:%S")),
                       review_status="unreviewed")
            os.makedirs(os.path.join(CARDS, work), exist_ok=True)
            p = os.path.join(CARDS, work, aid + ".json"); json.dump(rec, open(p + ".part", "w", encoding="utf-8"), ensure_ascii=False, indent=1); os.replace(p + ".part", p)
            b["collected"].append(cid); wrote += 1
        save_ledger(ledger)
    print(f"wrote {wrote} cards, cost this collect=${tot_cost:.2f}, problems={len(bad)}")
    for x in bad[:20]: print("  ", x)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("mode", choices=["submit", "collect"]); ap.add_argument("works", nargs="*")
    ap.add_argument("--model", default="claude-sonnet-5", choices=list(PRICES)); ap.add_argument("--max-usd", type=float); ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(); submit(a) if a.mode == "submit" else collect(a)
