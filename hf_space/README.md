---
title: Telugu Purana RAG
emoji: 📜
colorFrom: yellow
colorTo: red
sdk: gradio
sdk_version: 5.9.1
app_file: app.py
pinned: false
---

# పురాణ సందేహాలు — grounded Telugu Q&A over 17 Puranas

Sanskrit text of 17 Puranas (2,809 adhyayas, ~141k verses, GRETIL) indexed by Telugu *chapter cards*.
Every answer cites verse ids and quotes the Telugu-script sloka verbatim; unverified citations are dropped
and an answer with none is refused.

**First start builds the index** (45k BGE-M3 vectors) on this Space's hardware and saves it to the dataset
repo `DATA_REPO`, so later restarts load it in seconds. Vectors live in RAM (~93 MB fp16).

## Secrets
- `ANTHROPIC_API_KEY` — routing (Haiku 4.5) and generation (Sonnet 5)
- `HF_TOKEN` — read the private dataset repo and write the built index back

## Variables (optional)
- `DATA_REPO` (default `amrithatejaswiservices/telugu-purana-index`)
- `PURANA_MODEL_FAST`, `PURANA_MODEL_GEN`
