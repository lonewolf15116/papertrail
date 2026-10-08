# PaperTrail

Question answering over ML-systems research papers (rematerialization, offloading, distributed training),
with citations to the exact passages used, and an evaluation suite that measures how often it is right.

The point of the project is the evaluation, not the chat demo: every retrieval change is scored on a
hand-labelled question set, and CI fails the build if quality regresses.

## Status

| Milestone | Scope | Status |
|---|---|---|
| 1. Corpus + gold set | 24 papers, 60–100 labelled questions incl. unanswerable and cross-paper | 23 papers ingested (1,074 chunks); 74 questions, 12 verified so far |
| 2. Retrieval ablations | BM25 → vectors → hybrid (RRF) → cross-encoder rerank; recall@5, MRR | done; runs in CI with a regression gate |
| 3. Grounded answers | FastAPI `/ask` with paper/page/section citations and a refusal path | stub |
| 4. Ship | Docker Compose, CI quality gate, deployment, p50/p95 latency and cost | CI, Docker and the retrieval gate in place |
| Later | Agent tools: `search_papers`, `fetch_section`, `compare_papers` | — |

## Results: retrieval

Produced end to end on a clean GitHub Actions runner by [`ablation.yml`](.github/workflows/ablation.yml)
(download corpus and models → ingest → pgvector → score). Each run publishes its full per-question output to the
[`ablation-results`](../../tree/ablation-results) branch.

64 answerable questions (49 factual, 15 cross-paper). **Provisional:** most labels are still drafts awaiting human
verification, so treat the numbers as indicative until the verified run catches up.

| Retriever | Recall@5 | MRR@10 | Wrong-paper@5 (cross-paper) | p50 latency |
|---|---|---|---|---|
| Postgres full-text (`ts_rank_cd`) | 0.258 | 0.232 | 0.213 | 12 ms |
| BM25 | 0.648 | 0.479 | 0.147 | 2 ms |
| BM25 + headers | **0.719** | 0.501 | 0.147 | 2 ms |
| Vector (bge-small) | 0.453 | 0.311 | 0.120 | 27 ms |
| Vector + headers | 0.500 | 0.392 | 0.133 | 27 ms |
| Hybrid (RRF of BM25 + vector) | 0.609 | 0.466 | 0.133 | 27 ms |
| **Hybrid + headers** (default) | 0.703 | **0.506** | **0.093** | 28 ms |
| Hybrid + rerank (MiniLM cross-encoder) | 0.625 | 0.510 | 0.133 | 1,667 ms |
| Hybrid + headers + rerank | 0.641 | 0.512 | 0.133 | 1,665 ms |

"Headers" = each chunk indexed as `<paper title>. <section>. <text>`.

Paired bootstrap (5,000 resamples over questions), 95% intervals:

| Comparison | Δ Recall@5 | 95% CI |
|---|---|---|
| BM25 vs vector | +0.195 | [+0.078, +0.312] |
| Headers on BM25 | +0.070 | [+0.016, +0.133] |
| Headers on hybrid | +0.094 | [−0.016, +0.211] |
| Hybrid + headers vs BM25 + headers | −0.016 | [−0.109, +0.078] |
| Reranker on hybrid + headers | −0.062 | [−0.180, +0.055] |

What the numbers say:

- **Postgres full-text search is not BM25.** `ts_rank_cd` has no inverse document frequency, so common words
  ("memory", "tensor") weigh as much as rare ones ("banishing"). It finds the evidence a third as often.
- **Lexical beats dense here, clearly.** These questions turn on exact terms and numbers (`h_DTR`, "5.1×",
  ZeRO-Offload vs ZeRO-Infinity) that a small general-purpose embedding model blurs.
- **Contextual headers are the cheapest win.** A ZeRO-Offload passage rarely names ZeRO-Offload; prefixing the
  title gives every retriever that identity. Significant for BM25; likely but unproven for the others.
- **The reranker is not worth it.** No measurable gain, 60× the latency on CPU. The MS MARCO model was trained on web
  search, not on scientific passages; a domain-tuned reranker is a separate experiment.
- **Default: hybrid + headers.** It ties BM25 + headers on recall, has the best MRR and the lowest wrong-paper rate on
  cross-paper questions, and its dense half should hold up better on real user phrasing.

Caveat: the questions were drafted from the passages they cite, so they share the papers' vocabulary. That favours
lexical retrieval; paraphrased questions from real users would likely narrow the BM25–dense gap.

The gate: every push that touches retrieval code, the corpus or the questions reruns the ablation and fails if the
default retriever's recall@5 or MRR drops more than 0.02 below [`results/baseline.json`](results/baseline.json).

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
make install
make test           # unit tests
python scripts/download_corpus.py   # fetch the arXiv PDFs (Capuchin is manual)
make ingest         # PDFs -> sections -> chunks (data/processed/chunks.jsonl)
make eval           # validate the gold set and check every evidence quote is found
python scripts/download_models.py    # embedding + reranker models (Hugging Face)
make up             # Postgres + pgvector and the API on :8000
papertrail-index                     # chunks + embeddings into pgvector
papertrail-eval-retrieval --include-drafts   # the retrieval ablation table
curl localhost:8000/health
```

## Layout

```
src/papertrail/
  api.py            FastAPI service (/health, /ask)
  schemas.py        Pydantic contracts: chunks, citations, answers, gold questions
  ingest.py         PDF -> sections -> chunks (section-aware, page-accurate, references dropped)
  retrieval.py      Retriever interface + reciprocal rank fusion
  eval/metrics.py   recall@k, MRR, citation precision, refusal accuracy, wrong-paper rate, percentiles
  eval/evidence.py  maps labelled quotes to chunk ids under any chunking
  eval/gate.py      CI regression gate
  eval/run.py       papertrail-eval entry point
data/corpus/papers.yaml   corpus manifest (PDFs are git-ignored)
data/gold/                hand-labelled questions (see its README)
scripts/init.sql          pgvector schema with full-text and HNSW indexes
```

## Evaluation design

- **Gold set** labelled by hand with verbatim evidence quotes (so labels survive re-chunking), unanswerable questions, and adversarial cross-paper
  questions where papers such as DTR, Checkmate and Capuchin use similar terms for different claims.
- **Retrieval**: recall@5 and MRR per retriever, plus the wrong-paper rate on cross-paper questions.
- **Answers**: citation precision and refusal accuracy from hand labels; faithfulness from an LLM judge
  that is spot-checked, so the evaluation does not rest entirely on another model.
- **Operations**: p50/p95 latency, tokens and estimated cost per question.
