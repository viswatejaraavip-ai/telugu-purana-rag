"""Chapter-card retrieval for the Purana agent: BGE-M3 dense over card units (each generated
question, the summary, the moral, the title) + char-4gram BM25 over the whole card, fused by
RRF at the ADHYAYA level. Fully local. Run with ~/telugu-rag/.venv (has the deps).
"""
from __future__ import annotations
import glob, hashlib, json, os, pickle, re, unicodedata
from pathlib import Path
import numpy as np

ROOT = Path(os.environ.get("PURANA_DATA") or Path(__file__).resolve().parent.parent)
CARDS_DIR = ROOT / "cards"
CORPUS_DIR = ROOT / "corpus"
INDEX_DIR = ROOT / "index"
UNITS_NPY = INDEX_DIR / "units.npy"          # float16 [N, 1024], L2-normalised
UNITS_META = INDEX_DIR / "units_meta.json"   # [{adhyaya_id, work, unit}] aligned with rows
BM25_PATH = INDEX_DIR / "bm25.pkl"
CARDS_PATH = ROOT / "index" / "cards.pkl"
EMB_CACHE = ROOT / "work" / "embcache.pkl"
MODEL_NAME = "BAAI/bge-m3"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
DIM = 1024

TELUGU = re.compile(r"[ఀ-౿]")
LATIN = re.compile(r"[A-Za-z]")

# One line per work for the router; Telugu, the way a pandit would describe the text.
WORK_PROFILES = {
    "bhagavata": "శ్రీమద్భాగవతం — కృష్ణ లీలలు, దశావతారాలు, ప్రహ్లాద ధ్రువ గజేంద్ర అజామిళ కథలు, పరీక్షిత్తు, భక్తి",
    "vishnu": "విష్ణు పురాణం — సృష్టి, మన్వంతరాలు, రాజవంశాలు, ప్రహ్లాదుడు, ధ్రువుడు, కృష్ణ చరిత్ర, కలియుగ ధర్మం",
    "markandeya": "మార్కండేయ పురాణం — దేవీ మాహాత్మ్యం (చండీ సప్తశతి), హరిశ్చంద్రుడు, మదాలస, పక్షుల సంవాదం",
    "garuda": "గరుడ పురాణం — మరణానంతర గతి, ప్రేత కల్పం, శ్రాద్ధం, పితృకార్యాలు, విష్ణు పూజ, వ్రతాలు, రత్నాలు",
    "narasimha": "నృసింహ పురాణం — నృసింహావతారం, ప్రహ్లాదుడు, విష్ణు భక్తి, రామ కథ, యమగీత",
    "shiva": "శివ పురాణం — శివ మాహాత్మ్యం, లింగ పూజ, రుద్రాక్ష, భస్మం, పంచాక్షరి, వాయవీయ సంహిత",
    "kurma": "కూర్మ పురాణం — కూర్మావతారం, శివ-విష్ణు ఐక్యం, ఈశ్వర గీత, వర్ణాశ్రమ ధర్మం, తీర్థాలు",
    "vamana": "వామన పురాణం — వామనావతారం, బలి చక్రవర్తి, శివ పార్వతి కథలు, తీర్థాలు",
    "vamana_saromahatmya": "వామన పురాణం సరోమాహాత్మ్యం — కురుక్షేత్ర తీర్థాలు, స్నాన ఫలం",
    "matsya": "మత్స్య పురాణం — మత్స్యావతారం, మనువు, వంశావళులు, వ్రతాలు, దానాలు, దేవాలయ నిర్మాణం",
    "linga": "లింగ పురాణం — లింగార్చన, శివ కథలు, దక్ష యజ్ఞం, తపస్సు, పాశుపత వ్రతం",
    "agni": "అగ్ని పురాణం — వ్రతాలు, పూజా విధి, దానం, ప్రాయశ్చిత్తం, రామాయణ మహాభారత సారం, జ్యోతిషం, వైద్యం",
    "brahma": "బ్రహ్మ పురాణం — సూర్యుడు, పురుషోత్తమ క్షేత్రం (పూరీ), గౌతమీ (గోదావరి) మాహాత్మ్యం, తీర్థాలు, కృష్ణ చరిత్ర",
    "narada": "నారద పురాణం — భక్తి, వ్రతాలు, గంగా మాహాత్మ్యం, తీర్థాలు, పూజా విధానం",
    "brahmanda": "బ్రహ్మాండ పురాణం — సృష్టి, భువన కోశం, వంశావళులు, పరశురాముడు, లలితోపాఖ్యానం",
    "skanda_revakhanda": "స్కాంద పురాణం రేవా ఖండం — నర్మదా నది తీర్థాలు, స్నాన ఫలం",
    "vayu_revakhanda": "వాయు పురాణం రేవా ఖండం — నర్మదా నది తీర్థాలు, స్నాన ఫలం",
}


def norm(s: str) -> str:
    return unicodedata.normalize("NFC", s or "").strip()


def bm25_tokens(s: str) -> list[str]:
    s = norm(s)
    return [s[i:i+4] for i in range(max(0, len(s) - 3))] + [w for w in s.split() if w]


def load_cards() -> list[dict]:
    out = []
    for p in sorted(glob.glob(str(CARDS_DIR / "*" / "*.json"))):
        r = json.load(open(p, encoding="utf-8")); r["_work"] = Path(p).parent.name; out.append(r)
    return out


def card_units(r: dict) -> list[tuple[str, str]]:
    c = r["card"]
    units = [("title", c["title_te"]), ("summary", c["summary_te"]), ("moral", c["moral_te"])]
    units += [("question", q) for q in c["questions_te"]]
    return [(k, norm(t)) for k, t in units if norm(t)]


def card_text(r: dict) -> str:
    c = r["card"]
    return norm(" ".join([c["title_te"], c["summary_te"], c["moral_te"], " ".join(c["key_events_te"]),
                          " ".join(c["questions_te"]), " ".join(c["themes_te"]), " ".join(c["vratas_tithis_te"]),
                          " ".join(e["name_te"] for e in c["entities"])]))


# ------------------------------------------------------------------ embedding
_model = None
def encoder():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(MODEL_NAME, device="cpu"); _model.max_seq_length = 512
    return _model


_cache = None
def embed(texts: list[str], batch_size: int = 8, progress: bool = False) -> np.ndarray:
    """sha1-keyed cache in work/embcache.pkl so rebuilds and evals never re-embed the same text."""
    global _cache
    if _cache is None:
        _cache = pickle.load(open(EMB_CACHE, "rb")) if EMB_CACHE.exists() else {}
    keys = [hashlib.sha1(t.encode()).hexdigest() for t in texts]
    todo = [(k, t) for k, t in zip(keys, texts) if k not in _cache]
    if todo:
        vecs = encoder().encode([t for _, t in todo], batch_size=batch_size, normalize_embeddings=True,
                                convert_to_numpy=True, show_progress_bar=progress)
        for (k, _), v in zip(todo, vecs): _cache[k] = v
        EMB_CACHE.parent.mkdir(exist_ok=True); pickle.dump(_cache, open(EMB_CACHE, "wb"))
    return np.stack([_cache[k] for k in keys])


# --------------------------------------------------------------------- search
_U = None; _META = None
def matrix():
    """All unit vectors in RAM (~93 MB fp16) plus aligned metadata. One matmul per query."""
    global _U, _META
    if _U is None:
        _U = np.load(UNITS_NPY).astype(np.float32)
        _META = json.load(open(UNITS_META, encoding="utf-8"))
    return _U, _META


_bm = None
def _bm25():
    global _bm
    if _bm is None: _bm = pickle.load(open(BM25_PATH, "rb"))
    return _bm


_cards = None
def cards_by_id() -> dict:
    global _cards
    if _cards is None: _cards = pickle.load(open(CARDS_PATH, "rb"))
    return _cards


def _rrf(ranked_lists, k: int = 60):
    s = {}
    for lst in ranked_lists:
        for rank, d in enumerate(lst): s[d] = s.get(d, 0.0) + 1.0 / (k + rank + 1)
    return s


_ce = None
def reranker():
    global _ce
    if _ce is None:
        from sentence_transformers import CrossEncoder
        _ce = CrossEncoder(RERANK_MODEL, max_length=512, device="cpu")
    return _ce


def search(query: str, top_k: int = 6, pool: int = 30, works: list[str] | None = None, rerank: bool = False) -> list[dict]:
    """Return top_k chapter cards for a Telugu query. Dense (max over a card's units) + BM25, RRF-fused."""
    query = norm(query)
    U, meta = matrix()
    sims = U @ embed([query])[0].astype(np.float32)
    if works:
        mask = np.fromiter((m["work"] in works for m in meta), dtype=bool, count=len(meta)); sims = np.where(mask, sims, -1.0)
    dense_ids, dense_hit = [], {}
    for i in np.argsort(-sims)[: pool * 8]:
        aid = meta[i]["adhyaya_id"]
        if aid not in dense_hit:
            dense_hit[aid] = (meta[i]["unit"], float(sims[i])); dense_ids.append(aid)
        if len(dense_ids) >= pool: break
    bm = _bm25()
    scores = bm["model"].get_scores(bm25_tokens(query))
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    if works: order = [i for i in order if bm["works"][i] in works]
    lex_ids = [bm["ids"][i] for i in order[:pool]]
    fused = sorted(_rrf([dense_ids, lex_ids]).items(), key=lambda kv: -kv[1])
    cards = cards_by_id()
    out = []
    for aid, sc in fused[:pool]:
        c = dict(cards[aid]); c["_score"] = round(sc, 5); c["_matched_unit"] = dense_hit.get(aid); out.append(c)
    if rerank and out:
        ce = reranker()
        texts = [norm(c["card"]["title_te"] + " " + c["card"]["summary_te"] + " " + " ".join(c["card"]["questions_te"])) for c in out]
        rs = ce.predict([(query, t) for t in texts], batch_size=8, show_progress_bar=False)
        for c, r in zip(out, rs): c["_score"] = round(float(r), 5); c["_reranked"] = True
        out.sort(key=lambda c: -c["_score"])
    return out[:top_k]


_verses = {}
def chapter_verses(work: str, adhyaya_id: str) -> list[dict]:
    """All verse records of one adhyaya (loaded lazily per work, cached)."""
    if work not in _verses:
        recs = [json.loads(l) for l in open(CORPUS_DIR / f"{work}.jsonl", encoding="utf-8") if l.strip()]
        by = {}
        for r in recs: by.setdefault(r["id"].rsplit("-", 1)[0], []).append(r)
        _verses[work] = by
    return _verses[work].get(adhyaya_id, [])
