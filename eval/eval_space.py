"""End-to-end evaluation of the deployed Purana Space.

retrieval : /retrieve on every gold question -> recall@1/5/10, NDCG@10, MRR at adhyaya level (multi-gold aware)
generation: /ask_json on a stratified subset -> Sonnet 5 LLM judge for faithfulness (claims vs cited verses)
            and correctness (vs the gold chapters' card summaries), plus citation and refusal stats.
usage: eval_space.py retrieval|generation|both [--n 24] [--out eval/report_space.md]
Run with ~/telugu-rag/.venv; needs ~/.cache/huggingface/token and ANTHROPIC_API_KEY.
"""
import argparse, json, math, os, random, sys, time, urllib.request
from pathlib import Path
from typing import List
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
BASE = "https://amrithatejaswiservices-telugu-purana-rag.hf.space/gradio_api/call"
TOK = open(os.path.expanduser("~/.cache/huggingface/token")).read().strip()
JUDGE = "claude-sonnet-5"


def call(endpoint: str, data: list, timeout: int = 420):
    req = urllib.request.Request(f"{BASE}/{endpoint}", data=json.dumps({"data": data}).encode(), headers={"Authorization": f"Bearer {TOK}", "Content-Type": "application/json"})
    eid = json.load(urllib.request.urlopen(req, timeout=60))["event_id"]
    req = urllib.request.Request(f"{BASE}/{endpoint}/{eid}", headers={"Authorization": f"Bearer {TOK}"})
    last = None
    for line in urllib.request.urlopen(req, timeout=timeout):
        line = line.decode("utf-8")
        if line.startswith("data:"): last = line[5:].strip()
    out = json.loads(last)                       # gradio returns the list of outputs
    if isinstance(out, list): out = out[0]
    return json.loads(out) if isinstance(out, str) else out


def dcg(ranked, gold, k):
    return sum(1.0 / math.log2(i + 2) for i, d in enumerate(ranked[:k]) if d in gold)


def retrieval_eval(gold, out_rows):
    R1, R5, R10, ND, MRR, per_kind = [], [], [], [], [], {}
    for g in gold:
        try:
            r = call("retrieve", [g["q"], 20], timeout=120)
        except Exception as e:
            print("retrieve failed:", g["q"][:40], e); continue
        ranked = [x["adhyaya_id"] for x in r.get("ranked", [])]; gs = set(g["gold"])
        hit = [d in gs for d in ranked]
        r1, r5, r10 = any(hit[:1]), any(hit[:5]), any(hit[:10])
        ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(gs), 10)))
        nd = dcg(ranked, gs, 10) / ideal if ideal else 0
        rank = next((i + 1 for i, h in enumerate(hit) if h), None); mrr = 1 / rank if rank else 0
        R1.append(r1); R5.append(r5); R10.append(r10); ND.append(nd); MRR.append(mrr)
        per_kind.setdefault(g["kind"], []).append((r1, r5, r10, nd, mrr))
        out_rows.append(dict(q=g["q"], kind=g["kind"], gold=g["gold"], rank=rank, top3=ranked[:3], route=r.get("route"), works=r.get("works"), query=r.get("query")))
        print(f"  rank={str(rank):>4} [{g['kind']:8s}] {g['q'][:60]}", flush=True)
    summ = dict(n=len(R1), recall1=np.mean(R1), recall5=np.mean(R5), recall10=np.mean(R10), ndcg10=np.mean(ND), mrr=np.mean(MRR),
                by_kind={k: dict(n=len(v), recall1=np.mean([x[0] for x in v]), recall5=np.mean([x[1] for x in v]), recall10=np.mean([x[2] for x in v]),
                                 ndcg10=np.mean([x[3] for x in v]), mrr=np.mean([x[4] for x in v])) for k, v in per_kind.items()})
    return summ


# ------------------------------------------------------------------ judge
from pydantic import BaseModel, Field
class Faith(BaseModel):
    claims: List[str] = Field(description="Every substantive factual claim in the answer, as short Telugu sentences (5-15 claims)")
    unsupported: List[str] = Field(description="Those claims NOT supported by the cited verses or the chapter summaries provided (empty if all supported)")
    contradicted: List[str] = Field(description="Claims that CONTRADICT the provided material")
    faithfulness: float = Field(description="fraction of claims supported, 0.0-1.0")

class Correct(BaseModel):
    answers_question: bool = Field(description="Does the answer actually address what was asked?")
    correctness: int = Field(description="1-5: 5 = fully correct and complete per the reference; 3 = mostly right with gaps or minor errors; 1 = wrong or missing")
    missing_te: str = Field(description="What the reference has that the answer lacks, in Telugu; empty if nothing important")
    errors_te: str = Field(description="Statements in the answer that conflict with the reference, in Telugu; empty if none")
    telugu_quality: int = Field(description="1-5 naturalness and clarity of the Telugu for a devotee")


def judge(schema, system, user):
    import anthropic
    r = anthropic.Anthropic().messages.parse(model=JUDGE, max_tokens=4000, thinking={"type": "adaptive"}, output_config={"effort": "medium"},
                                             system=system, messages=[{"role": "user", "content": user}], output_format=schema)
    return r.parsed_output


def generation_eval(gold, n, out_rows):
    random.seed(7)
    by_kind = {}
    for g in gold: by_kind.setdefault(g["kind"], []).append(g)
    quota = {"story": 6, "why": 5, "moral": 5, "lookup": 3, "phalam": 2, "teaching": 1, "dharma": 1, "tenglish": 2}
    sample = []
    for k, q in quota.items():
        sample += random.sample(by_kind.get(k, []), min(q, len(by_kind.get(k, []))))
    sample = sample[:n]
    cards = {}
    for p in ROOT.glob("cards/*/*.json"):
        r = json.load(open(p, encoding="utf-8")); cards[r["adhyaya_id"]] = r
    F, C, TQ, refusals, ncit, ndrop, hits = [], [], [], 0, [], [], []
    for g in sample:
        try:
            r = call("ask_json", [g["q"]])
        except Exception as e:
            print("ask failed:", g["q"][:40], e); continue
        if "error" in r: print("error:", r["error"][:100]); continue
        ans, cites, v = r.get("answer", ""), r.get("citations", []), r.get("verification", {})
        refused = not cites
        used = {c["adhyaya_id"] for c in r.get("chapters", [])[:3]}
        hits.append(bool(used & set(g["gold"])))
        ncit.append(v.get("n_ok", 0)); ndrop.append(v.get("n_failed", 0))
        row = dict(q=g["q"], kind=g["kind"], gold=g["gold"], top3=[c["adhyaya_id"] for c in r.get("chapters", [])[:3]], refused=refused,
                   n_cit=v.get("n_ok", 0), n_dropped=v.get("n_failed", 0), answer=ans)
        if refused:
            refusals += 1; F.append(None); C.append(1); TQ.append(None); row.update(faithfulness=None, correctness=1); out_rows.append(row)
            print(f"  REFUSED [{g['kind']}] {g['q'][:60]}", flush=True); continue
        by_id = {p["id"]: p for p in r.get("passages", [])}
        evidence = "\n\n".join(f"[{c['verse_id']}] {c['citation_te']}\n{c['sloka_te']}\n({by_id.get(c['verse_id'], {}).get('sloka_iast', '')})" for c in cites)
        summaries = "\n\n".join(f"[{c['adhyaya_id']}] {c['title_te']}\n{c['summary_te']}" for c in r.get("chapters", [])[:3])
        f = judge(Faith, "You are a strict Sanskrit-literate auditor. Split the Telugu answer into its factual claims and check each against ONLY the "
                         "provided evidence: the cited Sanskrit verses (Telugu script + IAST) and the chapter summaries. A claim is supported if the "
                         "evidence states it or it follows directly. Your own knowledge of the Puranas does not count as evidence.",
                  f"ప్రశ్న: {g['q']}\n\nANSWER:\n{ans}\n\nCITED VERSES:\n{evidence}\n\nCHAPTER SUMMARIES:\n{summaries}")
        ref = "\n\n".join(f"[{a}] {cards[a]['card']['title_te']}\n{cards[a]['card']['summary_te']}\nముఖ్య ఘటనలు: " + " | ".join(cards[a]['card']['key_events_te'])
                          for a in g["gold"] if a in cards)
        c = judge(Correct, "You grade a Telugu answer about the Puranas against a REFERENCE (summaries of the chapters that hold the answer). "
                           "Judge correctness and completeness relative to the reference, not to your own memory. Be fair: a correct answer drawn "
                           "from a different but valid chapter of the same story should still score well if it does not conflict with the reference.",
                  f"ప్రశ్న: {g['q']}\n\nANSWER:\n{ans}\n\nREFERENCE:\n{ref}")
        F.append(f.faithfulness); C.append(c.correctness); TQ.append(c.telugu_quality)
        row.update(faithfulness=f.faithfulness, unsupported=f.unsupported, contradicted=f.contradicted, correctness=c.correctness,
                   answers_question=c.answers_question, missing=c.missing_te, errors=c.errors_te, telugu_quality=c.telugu_quality)
        out_rows.append(row)
        print(f"  faith={f.faithfulness:.2f} correct={c.correctness} te={c.telugu_quality} cit={v.get('n_ok',0)}/{v.get('n_ok',0)+v.get('n_failed',0)} [{g['kind']:8s}] {g['q'][:50]}", flush=True)
    Fv = [x for x in F if x is not None]; TQv = [x for x in TQ if x is not None]
    return dict(n=len(C), refusals=refusals, faithfulness=np.mean(Fv) if Fv else None, faithful_ge_0_9=np.mean([x >= 0.9 for x in Fv]) if Fv else None,
                correctness=np.mean(C), correct_ge_4=np.mean([x >= 4 for x in C]), telugu_quality=np.mean(TQv) if TQv else None,
                avg_citations=np.mean(ncit), avg_dropped=np.mean(ndrop), gold_in_top3_read=np.mean(hits))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("mode", choices=["retrieval", "generation", "both"]); ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--gold", default=str(ROOT / "eval" / "gold_puranas_all.jsonl")); ap.add_argument("--out", default=str(ROOT / "eval" / "report_space"))
    a = ap.parse_args()
    gold = [json.loads(l) for l in open(a.gold, encoding="utf-8") if l.strip()]
    report = {"gold": a.gold, "n_gold": len(gold), "ts": time.strftime("%Y-%m-%d %H:%M")}
    if a.mode in ("retrieval", "both"):
        rows = []; print(f"retrieval over {len(gold)} questions", flush=True); report["retrieval"] = retrieval_eval(gold, rows)
        json.dump(rows, open(a.out + "_retrieval_rows.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    if a.mode in ("generation", "both"):
        rows = []; print(f"generation + judge on {a.n} questions", flush=True); report["generation"] = generation_eval(gold, a.n, rows)
        json.dump(rows, open(a.out + "_generation_rows.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    json.dump(report, open(a.out + ".json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
    print(json.dumps(report, ensure_ascii=False, indent=1, default=lambda x: round(float(x), 3)))


if __name__ == "__main__":
    main()
