# Purana RAG evaluation — deployed Space, 2026-09-15

Target: https://huggingface.co/spaces/amrithatejaswiservices/telugu-purana-rag (17 Puranas, 2,809 chapter cards).
Gold: `eval/gold_puranas_all.jsonl` — 96 hand-written Telugu questions (66 Bhagavata, 30 across 12 other Puranas),
gold = chapter id(s); stories told in several Puranas carry every valid chapter as gold.
Script: `eval/eval_space.py` (calls the Space's `/retrieve` and `/ask_json`; judge = Sonnet 5).

## Retrieval (adhyaya level, top-20 from `/retrieve`, includes Haiku routing + query rewrite)

| version | recall@1 | recall@5 | recall@10 | NDCG@10 | MRR |
|---|---|---|---|---|---|
| hard Purana filter (first deploy) | 0.59 | 0.77 | 0.82 | 0.660 | 0.675 |
| **soft RRF boost (current)** | **0.64** | **0.86** | **0.90** | **0.717** | **0.731** |

By question kind (current):

| kind | n | recall@1 | recall@5 | recall@10 | NDCG@10 |
|---|---|---|---|---|---|
| story | 26 | 0.65 | 0.96 | 1.00 | 0.832 |
| why | 20 | 0.75 | 1.00 | 1.00 | 0.824 |
| lookup | 10 | 0.60 | 1.00 | 1.00 | 0.770 |
| teaching | 8 | 0.88 | 0.88 | 1.00 | 0.838 |
| phalam | 7 | 0.71 | 0.86 | 1.00 | 0.680 |
| romanized Telugu | 11 | 0.82 | 0.91 | 0.91 | 0.772 |
| dharma | 3 | 0.67 | 1.00 | 1.00 | 0.807 |
| **moral / life-situation** | 11 | 0.00 | 0.18 | 0.18 | 0.054 |

Findings
- The router's Purana hint as a *hard filter* was the main failure: a wrong guess (Bharata -> Markandeya,
  Rukmangada -> Garuda) removed the right chapter entirely. As a soft RRF boost, recall@10 rose 0.82 -> 0.90.
- Every kind except moral now has the right chapter in the top 10 at least 90% of the time, and in the top 5
  (what the answer model reads: top 3) 86-100%.
- Moral / life-situation questions are the open problem at 2,809-chapter scale (0.18 recall@10). An abstract
  dilemma ("should a guru's adharmic order be obeyed?") does not embed near any one story's summary, and
  several "misses" are arguably other valid chapters. Candidates: a reranker over the top 20, a dedicated
  moral index over `moral_te` + life-situation questions, or routing moral questions to a judge-selected
  story. Not addressed yet.

## Generation (LLM judge, Sonnet 5; `/ask_json` on a stratified sample)

Only **5 of 24** sampled questions completed before the Anthropic credit ran out (all 5 story questions).

| metric | value |
|---|---|
| faithfulness (fraction of claims supported by cited verses + chapter summaries) | 0.945 |
| answers with faithfulness >= 0.9 | 4 / 5 |
| correctness vs gold-chapter reference (1-5) | 4.6 |
| correctness >= 4 | 5 / 5 |
| Telugu naturalness (1-5) | 5.0 |
| citations per answer (verified / total) | 7.8 / 7.8 (0 dropped) |
| refusals | 0 |
| gold chapter among the 3 chapters read | 5 / 5 |

Caveat: the faithfulness judge saw only the *cited* verses, not every verse the model read. The four
"unsupported" claims flagged (Putana's fall flattening trees for twelve krosas, gopis mistaking her for
Lakshmi, the pranava letters in lingodbhava) are in the chapters supplied to the model but were not cited,
so true faithfulness is higher than 0.945. Fix for the next run: pass all supplied passages as evidence.

## To finish
- Top up Anthropic credit (~$3 covers the remaining 19 judged questions and a re-judge after the latency changes).
- Re-run: `~/telugu-rag/.venv/bin/python eval/eval_space.py generation --n 24`
