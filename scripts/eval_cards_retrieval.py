"""Adhyaya-level retrieval test over chapter cards.

Retrieval units: every generated question and the summary of every card (dense, BGE-M3), plus
char-4gram BM25 over the whole card text. Scores are folded to the adhyaya (max over its units)
and fused with RRF. Reports hit@1/3/5 and MRR, per surface and fused, per question kind.

Run with the telugu-rag venv (has sentence_transformers + rank_bm25):
  ~/telugu-rag/.venv/bin/python scripts/eval_cards_retrieval.py bhagavata eval/gold_bhagavata_skandha1.jsonl --book 1
"""
import argparse, glob, json, os, sys, unicodedata, warnings
warnings.filterwarnings("ignore")
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL = "BAAI/bge-m3"


def norm(s): return unicodedata.normalize("NFC", s or "").strip()
def grams(s):
    s = norm(s); return [s[i:i+4] for i in range(max(0, len(s) - 3))] + s.split()


def card_text(c):
    return " ".join([c["title_te"], c["summary_te"], c["moral_te"], " ".join(c["key_events_te"]),
                     " ".join(c["questions_te"]), " ".join(c["themes_te"]),
                     " ".join(e["name_te"] for e in c["entities"])])


def rrf(lists, k=60):
    s = {}
    for lst in lists:
        for r, d in enumerate(lst): s[d] = s.get(d, 0) + 1 / (k + r + 1)
    return sorted(s, key=s.get, reverse=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("work"); ap.add_argument("gold"); ap.add_argument("--book", type=int); ap.add_argument("--cards-dir", default=os.path.join(ROOT, "cards")); ap.add_argument("--rerank", type=int, default=0, help="rerank top-N fused chapters with bge-reranker-v2-m3")
    a = ap.parse_args()
    cards = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(a.cards_dir, a.work, "*.json")))]
    if a.book: cards = [c for c in cards if c["book_no"] == a.book]
    gold = [json.loads(l) for l in open(a.gold, encoding="utf-8") if l.strip()]
    ids = [c["adhyaya_id"] for c in cards]
    print(f"{len(cards)} cards, {len(gold)} gold questions")
    # dense units
    units, owner = [], []
    for c in cards:
        cc = c["card"]
        for q in cc["questions_te"]: units.append(norm(q)); owner.append(c["adhyaya_id"])
        units.append(norm(cc["summary_te"])); owner.append(c["adhyaya_id"])
        units.append(norm(cc["moral_te"])); owner.append(c["adhyaya_id"])
    import hashlib, pickle
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(MODEL, device="cpu"); m.max_seq_length = 512
    cache_path = os.path.join(ROOT, "work", "embcache.pkl")
    cache = pickle.load(open(cache_path, "rb")) if os.path.exists(cache_path) else {}
    def enc(texts):
        keys = [hashlib.sha1(t.encode()).hexdigest() for t in texts]
        todo = [t for t, k in zip(texts, keys) if k not in cache]
        if todo:
            vecs = m.encode(todo, normalize_embeddings=True, convert_to_numpy=True, batch_size=8)
            for t, v in zip(todo, vecs): cache[hashlib.sha1(t.encode()).hexdigest()] = v
            os.makedirs(os.path.dirname(cache_path), exist_ok=True); pickle.dump(cache, open(cache_path, "wb"))
        return np.stack([cache[k] for k in keys])
    U = enc(units)
    Q = enc([norm(g["q"]) for g in gold])
    from rank_bm25 import BM25Okapi
    bm = BM25Okapi([grams(card_text(c["card"])) for c in cards])

    def dense_rank(qv):
        sims = U @ qv; best = {}
        for s, o in zip(sims, owner): best[o] = max(best.get(o, -1), s)
        return sorted(best, key=best.get, reverse=True)

    def bm_rank(q):
        sc = bm.get_scores(grams(q)); return [ids[i] for i in np.argsort(-sc)]

    res = {"dense": [], "bm25": [], "fused": []}
    ce = None
    if a.rerank:
        from sentence_transformers import CrossEncoder
        ce = CrossEncoder("BAAI/bge-reranker-v2-m3", max_length=512, device="cpu")
        by_id = {c["adhyaya_id"]: c for c in cards}
        def rr_text(cid):
            cc = by_id[cid]["card"]; return norm(cc["title_te"] + " " + cc["summary_te"] + " " + " ".join(cc["questions_te"]))
        res["rerank"] = []
    rows = []
    for g, qv in zip(gold, Q):
        d, b = dense_rank(qv), bm_rank(g["q"]); f = rrf([d, b])
        surfaces = [("dense", d), ("bm25", b), ("fused", f)]
        if ce is not None:
            pool = f[:a.rerank]
            sc = ce.predict([(norm(g["q"]), rr_text(cid)) for cid in pool], batch_size=8)
            rk = [cid for _, cid in sorted(zip(sc, pool), key=lambda x: -x[0])] + f[a.rerank:]
            surfaces.append(("rerank", rk)); f = rk
        for name, lst in surfaces:
            rank = min((lst.index(x) + 1 for x in g["gold"] if x in lst), default=999)
            res[name].append(rank)
        rows.append((g["kind"], res["fused"][-1], g["q"], f[:3]))
    def summ(ranks):
        r = np.array(ranks); return f"hit@1={np.mean(r<=1):.2f} hit@3={np.mean(r<=3):.2f} hit@5={np.mean(r<=5):.2f} MRR={np.mean(1/r):.3f}"
    for k in res: print(f"{k:6s} {summ(res[k])}")
    kinds = sorted({g["kind"] for g in gold})
    for k in kinds:
        rr = [r for (kk, r, _, _) in rows if kk == k]; print(f"  kind={k:9s} n={len(rr):2d} {summ(rr)}")
    print("\nmisses (fused rank > 1):")
    for kind, r, q, top in rows:
        if r > 1: print(f"  rank={r:3d} [{kind}] {q}  top3={top}")


if __name__ == "__main__":
    main()
