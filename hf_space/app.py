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
sys.path.insert(0, str(Path(__file__).parent / "app"))
import retrieval as R

STATUS = {"ready": R.UNITS_NPY.exists(), "msg": "index loaded" if R.UNITS_NPY.exists() else "building index…", "done": 0, "total": 0}

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
demo.queue().launch()
