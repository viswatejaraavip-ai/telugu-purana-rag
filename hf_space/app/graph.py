"""LangGraph Purana agent. Same guarantees as the Gita agent (telugu-rag/app/graph.py):
answers come ONLY from retrieved chapters, every claim carries a verse citation whose quoted
Telugu-script sloka is verified verbatim against the corpus, and the graph refuses rather than
answering from parametric memory.

Coarse-to-fine: Telugu question -> chapter cards (retrieval.search) -> the top chapters' Sanskrit
verses (Telugu script + IAST) -> Telugu answer with verse citations -> verify -> lazy tatparyam
write-back (corpus/tatparyam_cache.jsonl, status=unreviewed).
Run with ~/telugu-rag/.venv:  python app/graph.py "ప్రశ్న"
"""
from __future__ import annotations
import json, os, re, sys, time, unicodedata
from pathlib import Path
from typing import Literal, Optional, TypedDict
sys.path.insert(0, str(Path(__file__).resolve().parent))
from retrieval import LATIN, TELUGU, WORK_PROFILES, chapter_verses, norm, search
from pydantic import BaseModel, Field
from langgraph.graph import END, START, StateGraph

MODEL_FAST = os.environ.get("PURANA_MODEL_FAST", "claude-haiku-4-5")   # transliteration, routing, grading
MODEL_GEN = os.environ.get("PURANA_MODEL_GEN", "claude-sonnet-5")      # Telugu answer with citations
TOP_CARDS = 6          # chapters retrieved
CHAPTERS_TO_READ = 3   # chapters whose verses go to the answer model
MAX_VERSES_PER_CHAPTER = 120
MAX_REGEN = 1
ROOT = Path(os.environ.get("PURANA_DATA") or Path(__file__).resolve().parent.parent)
TATPARYAM_CACHE = ROOT / "corpus" / "tatparyam_cache.jsonl"


# ------------------------------------------------------------------- schemas
class Route(BaseModel):
    kind: Literal["story", "moral", "lookup", "out_of_scope"] = Field(description=(
        "story = what happened / who / why in a Purana narrative; moral = a life dilemma or dharma question "
        "seeking what the Puranas teach; lookup = a fact, list, rule, vrata, phalam or tithi; "
        "out_of_scope = not answerable from Puranas (Gita-only questions, panchangam of a specific date, modern topics)"))
    works: list[str] = Field(default_factory=list, description="0-3 work keys from the list that most likely hold the answer; empty = search all")
    search_query_te: str = Field(description="The question rewritten as one clear Telugu search query using the names devotees use")

class Grade(BaseModel):
    sufficient: bool = Field(description="Do the chapter summaries indicate the retrieved chapters actually contain the answer?")
    reason: str

class Citation(BaseModel):
    verse_id: str = Field(description="exact verse id from the passages, e.g. bhp-1-7-42")
    sloka_te: str = Field(description="VERBATIM contiguous substring (one pada to one full verse) copied from that verse's Telugu-script text")
    tatparyam_te: str = Field(description="Plain Telugu meaning of that verse in ONE sentence")
    claim_te: str = Field(description="the statement in your answer this verse supports, a few words")

class Answer(BaseModel):
    answer_te: str = Field(description="The answer in modern natural Telugu, about 120-180 words. Name the Purana and chapter. If the chapters do not contain the answer, write one sentence saying so.")
    answerable: bool = Field(description="true only if the supplied chapters actually contain what was asked; false if you would have to rely on your own memory")
    citations: list[Citation] = Field(description="3-5 citations, never more than 5; empty if answerable is false")
MAX_CITATIONS = 5


# --------------------------------------------------------------------- state
class State(TypedDict, total=False):
    question: str; question_norm: str; script: str
    route: str; works: list[str]; query: str
    cards: list[dict]; sufficient: bool; grade_reason: str
    passages: list[dict]            # verse records handed to the answer model
    answer: str; citations: list[dict]; verification: dict; answerable: bool
    attempts: int; offline: bool; _on_text: object


# -------------------------------------------------------------------- claude
def _client():
    from anthropic import Anthropic
    return Anthropic()

def _has_key() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))

def ask(schema, system: str, user: str, model: str | None = None, effort: str = "medium", think: int = 0, max_tokens: int = 8000):
    """Structured call. Haiku 4.5 takes budget_tokens thinking and rejects `effort`; Sonnet/Opus take adaptive + effort."""
    model = model or MODEL_FAST
    kw = {}
    if model.startswith("claude-haiku"):
        if think: kw["thinking"] = {"type": "enabled", "budget_tokens": think}
    else:
        kw["thinking"] = {"type": "adaptive"}; kw["output_config"] = {"effort": effort}
    r = _client().messages.parse(model=model, max_tokens=max_tokens, system=system,
                                 messages=[{"role": "user", "content": user}], output_format=schema, **kw)
    if r.stop_reason == "refusal":
        raise RuntimeError(f"model refused: {r.stop_details}")
    return r.parsed_output


# ---------------------------------------------------------------------- nodes
def n_normalize(s: State) -> State:
    q = norm(s["question"])
    te, la = len(TELUGU.findall(q)), len(LATIN.findall(q))
    script = "telugu" if te > la else ("latin" if la and not te else "mixed")
    out = {"question_norm": q, "script": script, "attempts": 0, "offline": not _has_key()}
    if script == "latin" and _has_key():
        class T(BaseModel):
            telugu: str = Field(description="the same question in Telugu script")
        out["question_norm"] = ask(T, "Convert romanized Telugu (Telugu typed in Latin letters, informal spelling) into Telugu "
                                      "script, preserving meaning. If the input is English, translate it to Telugu.", q).telugu
    return out

PROFILES_TXT = "\n".join(f"- {k}: {v}" for k, v in WORK_PROFILES.items())

def n_route(s: State) -> State:
    if s.get("offline"):
        return {"route": "story", "works": [], "query": s["question_norm"]}
    r = ask(Route, "You route Telugu questions for a Purana retrieval system. The corpus is the Sanskrit text of these "
                   f"Puranas, indexed by chapter:\n{PROFILES_TXT}\n\nPick works only when confident; otherwise leave empty.",
            s["question_norm"])
    works = [w for w in r.works if w in WORK_PROFILES]
    return {"route": r.kind, "works": works, "query": r.search_query_te or s["question_norm"]}

def _rrf_merge(lists, k=60):
    sc, first = {}, {}
    for lst in lists:
        for rank, c in enumerate(lst):
            sc[c["adhyaya_id"]] = sc.get(c["adhyaya_id"], 0.0) + 1.0 / (k + rank + 1); first.setdefault(c["adhyaya_id"], c)
    out = []
    for aid, v in sorted(sc.items(), key=lambda kv: -kv[1]):
        c = dict(first[aid]); c["_score"] = round(v, 5); out.append(c)
    return out

def retrieve_cards(query: str, works: list | None, top_k: int = TOP_CARDS) -> list[dict]:
    """The router's Purana hint is a SOFT boost, not a filter: fuse the unfiltered ranking with the
    work-filtered ranking by RRF. Measured on 96 gold questions, a hard filter lost the right chapter
    outright whenever Haiku guessed the wrong Purana (Bharata -> Markandeya, Rukmangada -> Garuda)."""
    allw = search(query, top_k=max(20, top_k * 3))
    if not works: return allw[:top_k]
    hinted = search(query, top_k=max(20, top_k * 3), works=works)
    return _rrf_merge([allw, hinted])[:top_k]

def n_retrieve(s: State) -> State:
    return {"cards": retrieve_cards(s["query"], s.get("works") or None)}

def _card_lines(cards: list[dict]) -> str:
    return "\n\n".join(f"[{c['adhyaya_id']}] {c['citation_te']} — {c['card']['title_te']}\n{c['card']['summary_te']}" for c in cards)

def n_grade(s: State) -> State:
    if not s.get("cards"): return {"sufficient": False, "grade_reason": "no chapters retrieved"}
    if s.get("offline"): return {"sufficient": True, "grade_reason": "offline: skipped grading"}
    g = ask(Grade, "Judge whether at least one of these Purana chapter summaries shows the chapter contains what the question "
                   "asks. For a moral/life question, a chapter whose story or teaching clearly bears on the dilemma counts. "
                   "Only the summaries count, not your own knowledge of the Puranas.",
            f"ప్రశ్న: {s['question_norm']}\n\nCHAPTERS:\n{_card_lines(s['cards'])}")
    return {"sufficient": g.sufficient, "grade_reason": g.reason}

SMALL_CHAPTER = 30      # chapters this short are sent whole
ANCHOR_CONTEXT = 1      # neighbouring verses around each anchored verse

def _select_verses(card: dict, vs: list[dict]) -> list[dict]:
    """Anchored verses (the card maps each key event to its verse ids) plus one verse of context each
    side; whole chapter when it is short. Cuts generation input by more than half versus full chapters."""
    if len(vs) <= SMALL_CHAPTER: return vs
    idx = {v["id"]: i for i, v in enumerate(vs)}; keep = set()
    for a in card.get("anchors", []):
        for vid in a.get("verse_ids", []):
            if vid in idx:
                for j in range(idx[vid] - ANCHOR_CONTEXT, idx[vid] + ANCHOR_CONTEXT + 1):
                    if 0 <= j < len(vs): keep.add(j)
    keep.update(range(0, min(2, len(vs))))                 # opening verses (who speaks to whom)
    if len(keep) < 8: return vs[:MAX_VERSES_PER_CHAPTER]   # card without usable anchors: fall back to the chapter
    return [vs[j] for j in sorted(keep)][:MAX_VERSES_PER_CHAPTER]

def n_fetch_verses(s: State) -> State:
    """Selected verses of the top chapters (Telugu script for quoting, IAST for reading)."""
    passages = []
    for c in s["cards"][:CHAPTERS_TO_READ]:
        allv = chapter_verses(c["_work"], c["adhyaya_id"]); vs = _select_verses(c["card"], allv)
        truncated = len(vs) < len(allv)
        for v in vs:
            passages.append(dict(id=v["id"], adhyaya_id=c["adhyaya_id"], work=c["_work"], citation_te=v["citation_te"],
                                 sloka_te=v["sloka_te_script"], sloka_iast=v["sloka_iast"], speaker_te=v.get("speaker_te"),
                                 source_url=v["source_url"], chapter_title_te=c["card"]["title_te"], chapter_truncated=truncated))
    return {"passages": passages}

def _passages(s: State) -> str:
    out, cur = [], None
    for p in s["passages"]:
        if p["adhyaya_id"] != cur:
            cur = p["adhyaya_id"]; card = next(c for c in s["cards"] if c["adhyaya_id"] == cur)["card"]
            out.append(f"\n=== {cur} | {p['citation_te'].rsplit(',', 1)[0]} | {card['title_te']}\nసారాంశం: {card['summary_te']}\n"
                       f"ముఖ్య ఘటనలు: {' | '.join(card.get('key_events_te', []))}\n(శ్లోకాలు: ముఖ్యమైనవి మాత్రమే ఇవ్వబడ్డాయి)\n")
        sp = f" ({p['speaker_te']})" if p.get("speaker_te") else ""
        out.append(f"[{p['id']}]{sp}\n{p['sloka_te']}\n({p['sloka_iast']})")
    return "\n".join(out)

GEN_SYSTEM = """You answer a devotee's question using ONLY the Purana chapters provided (Sanskrit verses in Telugu script, with IAST for reading, plus a chapter summary).

Rules:
- Answer in modern, natural Telugu (వ్యావహారిక), warm and clear, as a pandit explaining to a family.
- Use ONLY the supplied verses. Your own memory of the Puranas must not enter the answer. If the verses do not support a point, leave it out.
- Tell the story or teaching in order, naming the Purana and chapter. For a moral question, say what the story shows and let the reader draw the conclusion; do not issue personal instructions.
- Every substantive claim needs a citation: verse_id exactly as given, and sloka_te copied VERBATIM as a contiguous substring of that verse's Telugu-script text (at least one full pada; never retype, never transliterate from IAST). tatparyam_te is your plain Telugu meaning of that verse.
- Never invent a verse, a number, or a quotation. If two chapters differ, say so.
- Be concise: about 120-180 Telugu words, 3-5 citations, one-sentence tatparyams. Prefer the verses that directly carry the claim.
- If the supplied chapters do not contain the answer, set answerable=false, give no citations, and say so in one sentence."""

def n_generate(s: State) -> State:
    if s.get("offline"):
        return {"answer": "[offline: ANTHROPIC_API_KEY not set — retrieval only]", "citations": [], "attempts": s.get("attempts", 0) + 1}
    fix = ""
    v = s.get("verification", {})
    if v.get("failed"):
        fix = "\n\nYour previous answer failed citation verification for: " + ", ".join(v["failed"]) + \
              ". Copy sloka_te VERBATIM from the Telugu-script verse text this time, and use only the verse ids shown."
    a = generate_streaming(f"ప్రశ్న: {s['question_norm']}\n\nCHAPTERS AND VERSES:\n{_passages(s)}{fix}", s.get("_on_text"))
    if not a.answerable:
        return {"answer": a.answer_te, "citations": [], "answerable": False, "attempts": s.get("attempts", 0) + 1}
    return {"answer": a.answer_te, "citations": [c.model_dump() for c in a.citations[:MAX_CITATIONS]], "answerable": True,
            "attempts": s.get("attempts", 0) + 1}

_ANSWER_RX = re.compile(r'"answer_te"\s*:\s*"((?:[^"\\]|\\.)*)')

def generate_streaming(user: str, on_text=None) -> Answer:
    """Structured output via the streaming API. answer_te is the first field of the JSON, so partial
    text can be shown while the citations are still being written."""
    from anthropic import transform_schema
    client = _client(); buf = ""
    with client.messages.stream(model=MODEL_GEN, max_tokens=8000, thinking={"type": "adaptive"},
                                output_config={"effort": "medium", "format": {"type": "json_schema", "schema": transform_schema(Answer)}},
                                system=GEN_SYSTEM, messages=[{"role": "user", "content": user}]) as stream:
        for text in stream.text_stream:
            buf += text
            if on_text:
                m = _ANSWER_RX.search(buf)
                if m:
                    try: on_text(json.loads('"' + m.group(1) + '"'))
                    except Exception: pass
        msg = stream.get_final_message()
    if msg.stop_reason == "refusal": raise RuntimeError(f"model refused: {msg.stop_details}")
    return Answer(**json.loads(next(b.text for b in msg.content if b.type == "text")))

def _squash(t: str) -> str:
    return re.sub(r"[\s।॥|.,;:!?\-‌‍]+", "", unicodedata.normalize("NFC", t or ""))

def n_verify(s: State) -> State:
    """Every quoted sloka must literally occur in the verse it cites; verified tatparyams are cached."""
    by_id = {p["id"]: p for p in s.get("passages", [])}
    ok, failed = [], []
    for c in s.get("citations", []):
        p = by_id.get(c["verse_id"])
        if not p: failed.append(f"{c['verse_id']} (not in retrieved chapters)"); continue
        q = _squash(c["sloka_te"])
        if len(q) >= 8 and q in _squash(p["sloka_te"]):
            c.update(citation_te=p["citation_te"], source_url=p["source_url"], chapter_title_te=p["chapter_title_te"]); ok.append(c)
        else:
            failed.append(f"{c['verse_id']} (quote not found in verse)")
    if ok and not s.get("offline"):
        with TATPARYAM_CACHE.open("a", encoding="utf-8") as f:
            for c in ok:
                f.write(json.dumps(dict(verse_id=c["verse_id"], tatparyam_te=c["tatparyam_te"], status="unreviewed", model=MODEL_GEN,
                                        question=s["question_norm"], ts=time.strftime("%Y-%m-%dT%H:%M:%S")), ensure_ascii=False) + "\n")
    return {"citations": ok, "verification": {"verified": ok, "failed": failed, "n_ok": len(ok), "n_failed": len(failed)}}

REFUSAL = ("ఈ ప్రశ్నకు సమాధానం నా దగ్గర ఉన్న పురాణ గ్రంథాలలో దొరకలేదు. "
           "ఈ వ్యవస్థ పదిహేడు పురాణ గ్రంథాల సంస్కృత మూలం నుండి మాత్రమే, శ్లోక ఆధారంతో సమాధానం ఇస్తుంది.")

def n_refuse(s: State) -> State:
    return {"answer": REFUSAL, "citations": [], "verification": {"verified": [], "failed": [], "n_ok": 0, "n_failed": 0}}


# ---------------------------------------------------------------------- edges
def e_route(s: State): return "refuse" if s["route"] == "out_of_scope" else "retrieve"
def e_retrieve(s: State): return "fetch" if s.get("cards") else "refuse"
def e_verify(s: State):
    v = s.get("verification", {})
    if s.get("offline"): return END
    if s.get("answerable") is False: return "refuse"  # the generator judged the chapters insufficient (in-band grade)
    if (v.get("failed") or not v.get("verified")) and s.get("attempts", 0) <= MAX_REGEN: return "generate"
    if not v.get("verified"): return "refuse"        # an answer with zero verified citations is not shipped
    return END

def build():
    g = StateGraph(State)
    for name, fn in [("normalize", n_normalize), ("route", n_route), ("retrieve", n_retrieve),
                     ("fetch", n_fetch_verses), ("generate", n_generate), ("verify", n_verify), ("refuse", n_refuse)]:
        g.add_node(name, fn)
    g.add_edge(START, "normalize"); g.add_edge("normalize", "route")
    g.add_conditional_edges("route", e_route, {"retrieve": "retrieve", "refuse": "refuse"})
    g.add_conditional_edges("retrieve", e_retrieve, {"fetch": "fetch", "refuse": "refuse"})
    g.add_edge("fetch", "generate"); g.add_edge("generate", "verify")
    g.add_conditional_edges("verify", e_verify, {"generate": "generate", "refuse": "refuse", END: END})
    g.add_edge("refuse", END)
    return g.compile()


def answer(question: str, on_text=None) -> dict:
    """on_text(partial_answer_te) is called repeatedly while the answer streams."""
    return build().invoke({"question": question, "_on_text": on_text})


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "ధ్రువుడు ఎందుకు తపస్సు చేశాడు?"
    out = answer(q)
    print(f"\nQ: {q}\nroute={out.get('route')} works={out.get('works')} query={out.get('query')} offline={out.get('offline')}")
    print("\n--- chapters ---")
    for c in out.get("cards", []): print(f"  {c['adhyaya_id']:14s} {c['_score']:.4f}  {c['card']['title_te']}")
    print(f"\n--- answer ---\n{out.get('answer','')}")
    v = out.get("verification", {})
    print(f"\ncitations verified={v.get('n_ok')} failed={v.get('n_failed')} {v.get('failed', [])}")
    for c in out.get("citations", []):
        print(f"  • {c['citation_te']}\n    శ్లోకం: {c['sloka_te'][:90]}\n    తాత్పర్యం: {c['tatparyam_te'][:160]}")
