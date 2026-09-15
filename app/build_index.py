"""Build the in-memory index: index/units.npy (fp16), index/units_meta.json, index/bm25.pkl, index/cards.pkl.
Runs anywhere (laptop CPU, HF Space, HF Job). Uses CUDA if available. Embeddings are cached in work/embcache.pkl."""
import json, pickle, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import retrieval as R
from retrieval import BM25_PATH, CARDS_PATH, INDEX_DIR, UNITS_META, UNITS_NPY, bm25_tokens, card_text, card_units, load_cards

def main():
    try:
        import torch
        if torch.cuda.is_available():
            R._model = None
            from sentence_transformers import SentenceTransformer
            R._model = SentenceTransformer(R.MODEL_NAME, device="cuda"); R._model.max_seq_length = 512; print("embedding on CUDA", flush=True)
    except Exception as e:
        print("no CUDA:", e, flush=True)
    cards = load_cards(); print(f"{len(cards)} cards", flush=True)
    units, meta = [], []
    for r in cards:
        for kind, text in card_units(r):
            units.append(text); meta.append(dict(adhyaya_id=r["adhyaya_id"], work=r["_work"], unit=kind))
    print(f"{len(units)} units", flush=True)
    t0 = time.time(); vecs = R.embed(units, batch_size=64 if R._model is not None and R._model.device.type == "cuda" else 16, progress=True)
    print(f"embedded in {time.time() - t0:.0f}s", flush=True)
    INDEX_DIR.mkdir(exist_ok=True)
    np.save(UNITS_NPY, vecs.astype(np.float16)); json.dump(meta, open(UNITS_META, "w", encoding="utf-8"), ensure_ascii=False)
    from rank_bm25 import BM25Okapi
    bm = BM25Okapi([bm25_tokens(card_text(r)) for r in cards])
    pickle.dump({"model": bm, "ids": [r["adhyaya_id"] for r in cards], "works": [r["_work"] for r in cards]}, open(BM25_PATH, "wb"))
    pickle.dump({r["adhyaya_id"]: r for r in cards}, open(CARDS_PATH, "wb"))
    print(f"index written to {INDEX_DIR}: units {vecs.shape} fp16, bm25 {len(cards)} docs. done.", flush=True)

if __name__ == "__main__":
    main()
