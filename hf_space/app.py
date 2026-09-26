"""Telugu Purana RAG — Hugging Face Space.

Startup: snapshot the private dataset repo (cards + slim corpus [+ index]). If the index is missing,
build it here (GPU via ZeroGPU when available, else CPU), then upload it to the dataset repo so the
next start is instant. Serving: app/graph.py (LangGraph) with vectors in RAM.
"""
from __future__ import annotations
import json, os, sys, threading, time, traceback
from pathlib import Path
import gradio as gr
from huggingface_hub import HfApi, snapshot_download

DATA_REPO = os.environ.get("DATA_REPO", "amrithatejaswiservices/telugu-purana-index")
DATA_DIR = Path(snapshot_download(DATA_REPO, repo_type="dataset", token=os.environ.get("HF_TOKEN")))
os.environ["PURANA_DATA"] = str(DATA_DIR)

# The verse layer served to users is corpus_ws/ (Sanskrit Wikisource, CC BY-SA) from the public
# repo, plus cards_ws/ for chapters the dataset has no card for. A sparse, blobless clone keeps
# startup to the files needed. The dataset's corpus/ (GRETIL, non-commercial) is never read.
TEXT_REPO = os.environ.get("TEXT_REPO", "https://github.com/viswatejaraavip-ai/telugu-purana-rag.git")
TEXT_DIR = Path(os.environ.get("TEXT_DIR", "/tmp/purana-text"))
def _checkout_text():
    import subprocess
    if not (TEXT_DIR / ".git").exists():
        subprocess.run(["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse", TEXT_REPO, str(TEXT_DIR)], check=True)
        subprocess.run(["git", "-C", str(TEXT_DIR), "sparse-checkout", "set", "corpus_ws", "cards_ws"], check=True)
    else:
        subprocess.run(["git", "-C", str(TEXT_DIR), "pull", "--ff-only"], check=False)
_checkout_text()
os.environ["PURANA_CORPUS"] = str(TEXT_DIR / "corpus_ws")
os.environ["PURANA_CARDS_WS"] = str(TEXT_DIR / "cards_ws")
sys.path.insert(0, str(Path(__file__).parent / "app"))
import retrieval as R

def _index_current() -> bool:
    """The index on the dataset must cover exactly the cards we now load; otherwise rebuild."""
    if not (R.UNITS_NPY.exists() and R.CARDS_PATH.exists()):
        return False
    import pickle
    try:
        n_index = len(pickle.load(open(R.CARDS_PATH, "rb")))
    except Exception:
        return False
    return n_index == len(R.load_cards())

_READY = _index_current()
STATUS = {"ready": _READY, "msg": "index loaded" if _READY else "building index…", "done": 0, "total": 0}

try:
    import spaces, torch
    ZERO = True
except Exception:
    spaces = None; ZERO = False


def _embed_chunk_cpu(texts):
    return R.embed(texts, batch_size=16)

if ZERO:
    @spaces.GPU(duration=120)
    def _embed_chunk_gpu(texts):
        import torch
        from sentence_transformers import SentenceTransformer
        if R._model is None or R._model.device.type != "cuda":
            R._model = SentenceTransformer(R.MODEL_NAME, device="cuda"); R._model.max_seq_length = 512
        return R.embed(texts, batch_size=64)


def build_index():
    """Same output as app/build_index.py, chunked so ZeroGPU calls stay short; uploads when done."""
    import numpy as np, pickle
    from rank_bm25 import BM25Okapi
    try:
        cards = R.load_cards(); units, meta = [], []
        for r in cards:
            for kind, text in R.card_units(r):
                units.append(text); meta.append(dict(adhyaya_id=r["adhyaya_id"], work=r["_work"], unit=kind))
        STATUS["total"] = len(units)
        vecs = []
        for i in range(0, len(units), 1500):
            chunk = units[i:i+1500]
            try:
                v = _embed_chunk_gpu(chunk) if ZERO else _embed_chunk_cpu(chunk)
            except Exception as e:
                print("GPU chunk failed, CPU fallback:", e, flush=True); v = _embed_chunk_cpu(chunk)
            vecs.append(np.asarray(v)); STATUS["done"] = i + len(chunk)
            STATUS["msg"] = f"building index… {STATUS['done']}/{STATUS['total']} units"
        U = np.concatenate(vecs).astype(np.float16)
        R.INDEX_DIR.mkdir(exist_ok=True)
        np.save(R.UNITS_NPY, U); json.dump(meta, open(R.UNITS_META, "w", encoding="utf-8"), ensure_ascii=False)
        bm = BM25Okapi([R.bm25_tokens(R.card_text(r)) for r in cards])
        pickle.dump({"model": bm, "ids": [r["adhyaya_id"] for r in cards], "works": [r["_work"] for r in cards]}, open(R.BM25_PATH, "wb"))
        pickle.dump({r["adhyaya_id"]: r for r in cards}, open(R.CARDS_PATH, "wb"))
        STATUS["msg"] = "uploading index to dataset repo…"
        HfApi(token=os.environ.get("HF_TOKEN")).upload_folder(repo_id=DATA_REPO, repo_type="dataset", folder_path=str(R.INDEX_DIR),
                                                             path_in_repo="index", commit_message=f"index: {U.shape[0]} units fp16")
        STATUS.update(ready=True, msg=f"index ready: {U.shape[0]} units")
    except Exception:
        STATUS["msg"] = "index build FAILED:\n" + traceback.format_exc(); print(STATUS["msg"], flush=True)

if not STATUS["ready"]:
    threading.Thread(target=build_index, daemon=True).start()
else:
    # warm the encoder and matrix so the first question is fast
    threading.Thread(target=lambda: (R.matrix(), R.embed(["ధ్రువుడు"])), daemon=True).start()


def _render(out):
    cites = "\n\n".join(f"**{c['citation_te']}**  \n{c['sloka_te']}  \n_{c['tatparyam_te']}_" for c in out.get("citations", []))
    chapters = "\n".join(f"- {c['citation_te']} — {c['card']['title_te']}" for c in out.get("cards", []))
    v = out.get("verification", {})
    meta = f"route: {out.get('route')} · works: {out.get('works') or 'all'} · citations verified {v.get('n_ok', 0)}, dropped {v.get('n_failed', 0)}"
    return out.get("answer", ""), cites + f"\n\n---\n{meta}", chapters


def ask(question: str):
    """Generator: streams the answer text as Sonnet writes it, then the citations once verified."""
    if not STATUS["ready"]:
        yield f"⏳ {STATUS['msg']}", "", ""; return
    import queue
    from graph import answer
    q, box = queue.Queue(), {}
    def run():
        try: box["out"] = answer(question, on_text=lambda t: q.put(t))
        except Exception as e: box["err"] = e
        finally: q.put(None)
    threading.Thread(target=run, daemon=True).start()
    yield "🔎 అధ్యాయాలు వెతుకుతున్నాను…", "", ""
    while True:
        t = q.get()
        if t is None: break
        yield t + " ▌", "", ""
    if "err" in box: yield f"లోపం: {box['err']}", "", ""; return
    yield _render(box["out"])


def retrieve_json(question: str, top_k: int = 20) -> str:
    """Ranked chapters after normalize+route (no generation). For retrieval evals and the orchestrator."""
    if not STATUS["ready"]: return json.dumps({"error": STATUS["msg"]})
    from graph import n_normalize, n_route, retrieve_cards
    s = {"question": question}; s.update(n_normalize(s)); s.update(n_route(s))
    cards = retrieve_cards(s["query"], s.get("works") or None, top_k=int(top_k))
    return json.dumps({"question_norm": s["question_norm"], "route": s["route"], "works": s.get("works"), "query": s["query"],
                       "ranked": [{"adhyaya_id": c["adhyaya_id"], "work": c["_work"], "score": c["_score"], "title_te": c["card"]["title_te"]} for c in cards]},
                      ensure_ascii=False)


def search_api(question: str, top_k: float = 4) -> dict:
    """Passages only, no model calls: for callers that write their own answer and verify
    their own citations (the Prashna app). Top chapters by card retrieval, then the verses
    the card anchors (whole chapter when short), Telugu script + IAST, plus each chapter's
    Telugu summary so a chapter without a verse file (Narasimha, Skanda Reva Khanda) can
    still be answered from and cited. The query must already be in Telugu script."""
    t0 = time.time()
    if not STATUS["ready"]:
        return {"error": STATUS["msg"], "passages": [], "chapters": []}
    from graph import retrieve_cards, _select_verses
    q = (question or "").strip()
    if not q:
        return {"passages": [], "chapters": [], "query_te": "", "elapsed_s": 0.0}
    k = max(1, min(int(top_k or 4), 8))
    cards = retrieve_cards(q, None, top_k=k)
    chapters, passages = [], []
    for c in cards:
        allv = R.chapter_verses(c["_work"], c["adhyaya_id"])
        vs = _select_verses(c["card"], allv) if allv else []
        chapters.append(dict(adhyaya_id=c["adhyaya_id"], work=c["_work"], citation_te=c["citation_te"],
                             title_te=c["card"]["title_te"], summary_te=c["card"]["summary_te"],
                             moral_te=c["card"].get("moral_te"), score=c["_score"], n_verses=len(allv),
                             source_url=(allv[0].get("source_url") if allv else None)))
        for v in vs:
            passages.append(dict(id=v["id"], adhyaya_id=c["adhyaya_id"], work=c["_work"], citation_te=v["citation_te"],
                                 citation_en=v.get("citation_en"), sloka_te=v["sloka_te_script"], sloka_iast=v["sloka_iast"],
                                 speaker_te=v.get("speaker_te"), source_url=v.get("source_url"),
                                 chapter_title_te=c["card"]["title_te"]))
    return {"passages": passages, "chapters": chapters, "query_te": q, "elapsed_s": round(time.time() - t0, 2)}


def ask_json(question: str) -> str:
    """Full pipeline as JSON: answer, verified citations, passages the model saw, chapters, verification."""
    if not STATUS["ready"]: return json.dumps({"error": STATUS["msg"]})
    from graph import answer
    try:
        out = answer(question)
    except Exception as e:
        return json.dumps({"error": str(e)})
    return json.dumps({k: out.get(k) for k in ("question_norm", "route", "works", "query", "answer", "citations", "verification", "grade_reason", "sufficient")} |
                      {"chapters": [{"adhyaya_id": c["adhyaya_id"], "citation_te": c["citation_te"], "title_te": c["card"]["title_te"], "summary_te": c["card"]["summary_te"], "score": c["_score"]} for c in out.get("cards", [])],
                       "passages": [{"id": p["id"], "sloka_te": p["sloka_te"], "sloka_iast": p["sloka_iast"]} for p in out.get("passages", [])]}, ensure_ascii=False)


with gr.Blocks(title="పురాణ సందేహాలు") as demo:
    gr.Markdown("# 📜 పురాణ సందేహాలు\n17 పురాణాల సంస్కృత మూలం నుండి, శ్లోక ఆధారంతో తెలుగులో సమాధానం.")
    status = gr.Markdown(value=lambda: f"**స్థితి:** {STATUS['msg']}", every=15)
    q = gr.Textbox(label="మీ ప్రశ్న (తెలుగు లేదా Tenglish)", placeholder="ధ్రువుడు ఎందుకు తపస్సు చేశాడు?")
    btn = gr.Button("అడగండి", variant="primary")
    ans = gr.Markdown(label="సమాధానం")
    with gr.Accordion("ఆధారాలు (శ్లోకాలు)", open=True): cit = gr.Markdown()
    with gr.Accordion("వెతికిన అధ్యాయాలు", open=False): chap = gr.Markdown()
    btn.click(ask, q, [ans, cit, chap], api_name="ask"); q.submit(ask, q, [ans, cit, chap])
    with gr.Row(visible=False):   # programmatic endpoints
        rq = gr.Textbox(); rk = gr.Number(value=20); rout = gr.Textbox(); jq = gr.Textbox(); jout = gr.Textbox()
        gr.Button().click(retrieve_json, [rq, rk], rout, api_name="retrieve")
        gr.Button().click(ask_json, jq, jout, api_name="ask_json")
        # POST /gradio_api/call/search {"data": ["<telugu question>", 4]} -> {"passages": [...], "chapters": [...]}
        s_q = gr.Textbox(); s_k = gr.Number(value=4); s_out = gr.JSON()
        gr.Button().click(search_api, [s_q, s_k], s_out, api_name="search")
demo.queue().launch()
