"""Generate one Telugu *chapter card* per adhyaya - the retrieval surface of the Purana agent.

The Sanskrit stays the corpus. The card is the Telugu entry point: summary, the questions a
devotee would ask that this chapter answers, the dharmic lesson, and canonical entity tags.
Idempotent: cards/<work>/<adhyaya_id>.json is skipped if present. Usage is logged per call.

usage: .venv/bin/python scripts/gen_cards.py bhagavata [--book 1] [--adhyaya 7] [--workers 4]
"""
import argparse, json, os, sys, time, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Literal, Optional
sys.path.insert(0, os.path.dirname(__file__))
from common import OUT, CARDS, read_jsonl
from pydantic import BaseModel, Field
import anthropic

PRICES = {  # $/MTok: input, output, cache read, cache write (5m)
    "claude-opus-5": (5.0, 25.0, 0.5, 6.25),
    "claude-sonnet-5": (2.0, 10.0, 0.2, 2.5),
    "claude-haiku-4-5": (1.0, 5.0, 0.1, 1.25),
}
MODEL = "claude-opus-5"
PRICE_IN, PRICE_OUT, PRICE_CACHE_R, PRICE_CACHE_W = PRICES[MODEL]


class Entity(BaseModel):
    name_te: str = Field(description="Telugu name as a devotee would say it, e.g. కర్ణుడు, ద్రౌపది, ద్వారక")
    canonical_iast: str = Field(description="Canonical Sanskrit stem in IAST, lowercase, no case ending: karṇa, draupadī, dvārakā")
    kind: Literal["character", "deity", "sage", "place", "vrata", "tithi", "concept", "object", "group"]


class Anchor(BaseModel):
    topic_te: str = Field(description="One short Telugu phrase naming an event or teaching")
    verse_ids: List[str] = Field(description="Exact ids copied from the input that carry it, 1-6 ids")


class ChapterCard(BaseModel):
    title_te: str = Field(description="Short Telugu title for the chapter, 3-8 words")
    speaker_te: Optional[str] = Field(None, description="Main narrator in Telugu, e.g. శుకుడు")
    listener_te: Optional[str] = Field(None, description="Main listener in Telugu, e.g. పరీక్షిత్తు")
    summary_te: str = Field(description="6-10 sentences of plain modern Telugu prose telling what happens in this chapter, in order")
    key_events_te: List[str] = Field(description="3-8 bullet-style Telugu sentences, one event or teaching each")
    moral_te: str = Field(description="2-4 sentences: the dharmic lessons a devotee should take from this chapter, each stated as general life guidance (about anger, forgiveness, duty, greed, devotion, death...) and not only as a retelling")
    questions_te: List[str] = Field(description="10-14 questions in natural everyday Telugu that a devotee might ask which THIS chapter answers. Required mix: (a) 4-6 story/why/who questions naming the characters (ఏమి జరిగింది, ఎందుకు, ఎవరు); (b) at least 3 LIFE-SITUATION questions phrased the way a devotee describes their own dilemma WITHOUT naming any character, in the vocabulary of ordinary life - e.g. 'తప్పు చేసిన వాడిని క్షమించాలా, శిక్షించాలా?', 'కోపంలో శపిస్తే/మాట్లాడితే ఏమవుతుంది?', 'పాలకుడు దుష్టులను ఎలా శిక్షించాలి?', 'చావు దగ్గర పడితే ఏం చేయాలి?'; (c) one question containing 'ఫలం' and one containing 'ధర్మం' where the chapter supports it. Use the names people use in Telugu, not Sanskrit stems.")
    themes_te: List[str] = Field(description="3-8 short Telugu theme tags, e.g. భక్తి, క్షమ, పితృధర్మం")
    vratas_tithis_te: List[str] = Field(default_factory=list, description="Any vrata, tithi, festival, or tirtha whose rule or phalam is stated here; empty if none")
    entities: List[Entity] = Field(description="Every named character, deity, sage, place, object, and named concept that appears in this chapter, 8-40 entries; be exhaustive, minor names included")
    anchors: List[Anchor] = Field(description="3-8 anchors mapping key events/teachings to the exact verse ids that state them")


SYSTEM = """You are a Telugu pandit preparing a searchable index of the Puranas for Telugu-speaking devotees.
You read Sanskrit in IAST fluently. You write plain, warm, modern Telugu that an ordinary devotee reads
without a dictionary - not pandit-register, not Sanskrit words in Telugu script when a Telugu word exists.
Use the Telugu forms of names (కృష్ణుడు, పరీక్షిత్తు, ద్రౌపది, శుకుడు, భీష్ముడు), never IAST in Telugu fields.
Stay strictly within what the given verses say. Do not add events from other chapters or your general memory
of the story; if the chapter is a hymn or a teaching rather than a story, say so in the summary and write the
questions as questions about that teaching. Verse ids in anchors must be copied exactly from the input."""


def chapter_prompt(man, verses):
    head = (f"{man['work']} ({man['work_te']}), book {man['book_no']}, adhyaya {man['adhyaya_no']}, "
            f"{man['n_verses']} verses. Speakers seen: {', '.join(man['speakers']) or 'none marked'}.\n\n")
    body = "\n".join(f"[{v['id']}]" + (f" ({v['speaker_iast']})" if v.get('speaker_iast') else "") +
                     f"\n{v['sloka_iast']}" for v in verses)
    return head + body


_lock = threading.Lock()
MODEL_USED = [MODEL]
TOTAL = dict(calls=0, inp=0, out=0, cr=0, cw=0)


def cost(u):
    return (u.input_tokens * PRICE_IN + u.output_tokens * PRICE_OUT +
            (u.cache_read_input_tokens or 0) * PRICE_CACHE_R + (u.cache_creation_input_tokens or 0) * PRICE_CACHE_W) / 1e6


def gen_one(client, man, verses, out_path):
    t0 = time.time()
    resp = client.messages.parse(
        model=MODEL_USED[0], max_tokens=16000,
        thinking={"type": "adaptive"}, output_config={"effort": "medium"},
        system=[{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": chapter_prompt(man, verses)}],
        output_format=ChapterCard,
    )
    if resp.stop_reason == "refusal":
        raise RuntimeError(f"refusal: {resp.stop_details}")
    card = resp.parsed_output
    known = {v["id"] for v in verses}
    bad = [i for a in card.anchors for i in a.verse_ids if i not in known]
    rec = dict(adhyaya_id=man["adhyaya_id"], work=man["work"], work_te=man["work_te"],
               book_no=man["book_no"], adhyaya_no=man["adhyaya_no"], n_verses=man["n_verses"],
               citation_te=man["citation_te"], card=card.model_dump(),
               gen=dict(model=MODEL_USED[0], effort="medium", seconds=round(time.time() - t0, 1),
                        usage=resp.usage.model_dump(), cost_usd=round(cost(resp.usage), 4),
                        bad_anchor_ids=bad, request_id=resp._request_id, generated_at=time.strftime("%Y-%m-%dT%H:%M:%S")),
               review_status="unreviewed")
    tmp = out_path + ".part"
    json.dump(rec, open(tmp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    os.replace(tmp, out_path)
    with _lock:
        u = resp.usage
        TOTAL["calls"] += 1; TOTAL["inp"] += u.input_tokens; TOTAL["out"] += u.output_tokens
        TOTAL["cr"] += u.cache_read_input_tokens or 0; TOTAL["cw"] += u.cache_creation_input_tokens or 0
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("work"); ap.add_argument("--book", type=int); ap.add_argument("--adhyaya", type=int)
    ap.add_argument("--workers", type=int, default=4); ap.add_argument("--limit", type=int)
    ap.add_argument("--model", default=MODEL, choices=list(PRICES)); ap.add_argument("--cards-dir", default=CARDS)
    ap.add_argument("--max-usd", type=float, default=None, help="stop submitting new chapters once spend passes this")
    a = ap.parse_args()
    global PRICE_IN, PRICE_OUT, PRICE_CACHE_R, PRICE_CACHE_W
    MODEL_USED[0] = a.model; PRICE_IN, PRICE_OUT, PRICE_CACHE_R, PRICE_CACHE_W = PRICES[a.model]
    verses = {v["id"]: v for v in read_jsonl(os.path.join(OUT, f"{a.work}.jsonl"))}
    mans = read_jsonl(os.path.join(OUT, f"{a.work}_adhyayas.jsonl"))
    if a.book: mans = [m for m in mans if m["book_no"] == a.book]
    if a.adhyaya: mans = [m for m in mans if m["adhyaya_no"] == a.adhyaya]
    outdir = os.path.join(a.cards_dir, a.work); os.makedirs(outdir, exist_ok=True)
    todo = [m for m in mans if not os.path.exists(os.path.join(outdir, m["adhyaya_id"] + ".json"))]
    if a.limit: todo = todo[:a.limit]
    print(f"{len(mans)} adhyayas selected, {len(todo)} to generate", flush=True)
    client = anthropic.Anthropic(max_retries=4)
    fails = []
    def spent():
        return (TOTAL["inp"] * PRICE_IN + TOTAL["out"] * PRICE_OUT + TOTAL["cr"] * PRICE_CACHE_R + TOTAL["cw"] * PRICE_CACHE_W) / 1e6
    stopped = False
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for chunk_start in range(0, len(todo), a.workers * 2):
            if a.max_usd is not None and spent() >= a.max_usd:
                print(f"budget cap ${a.max_usd:.2f} reached at ${spent():.2f}; {len(todo) - chunk_start} chapters left", flush=True); stopped = True; break
            chunk = todo[chunk_start:chunk_start + a.workers * 2]
            futs = {ex.submit(gen_one, client, m, [verses[i] for i in m["verse_ids"]],
                              os.path.join(outdir, m["adhyaya_id"] + ".json")): m for m in chunk}
            for f in as_completed(futs):
                m = futs[f]
                try:
                    r = f.result()
                    print(f"ok  {m['adhyaya_id']:12s} {m['n_verses']:4d}v  {r['gen']['seconds']:6.1f}s  ${r['gen']['cost_usd']:.3f}  "
                          f"q={len(r['card']['questions_te'])} ent={len(r['card']['entities'])} bad_anchors={len(r['gen']['bad_anchor_ids'])}", flush=True)
                except Exception as e:
                    fails.append(m["adhyaya_id"]); print(f"FAIL {m['adhyaya_id']}: {e}", flush=True)
                    if "credit balance" in str(e): stopped = True
            if stopped: break
    tot = (TOTAL["inp"] * PRICE_IN + TOTAL["out"] * PRICE_OUT + TOTAL["cr"] * PRICE_CACHE_R + TOTAL["cw"] * PRICE_CACHE_W) / 1e6
    print(f"done: {TOTAL['calls']} calls, in={TOTAL['inp']} out={TOTAL['out']} cache_read={TOTAL['cr']} cache_write={TOTAL['cw']}  total=${tot:.2f}  fails={fails}")


if __name__ == "__main__":
    main()
