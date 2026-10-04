# PaperTrail

Question answering over ML-systems research papers (rematerialization, offloading, distributed training),
with citations to the exact passages used, and an evaluation suite that measures how often it is right.

The point of the project is the evaluation, not the chat demo: every retrieval change is scored on a
hand-labelled question set, and CI fails the build if quality regresses.

## Status

| Milestone | Scope | Status |
|---|---|---|
| 1. Corpus + gold set | 24 papers, 60–100 labelled questions incl. unanswerable and cross-paper | ingestion done; labelling |
| 2. Retrieval ablations | BM25 → vectors → hybrid (RRF) → cross-encoder rerank; recall@5, MRR | not started |
| 3. Grounded answers | FastAPI `/ask` with paper/page/section citations and a refusal path | stub |
| 4. Ship | Docker Compose, CI quality gate, deployment, p50/p95 latency and cost | CI + Docker in place |
| Later | Agent tools: `search_papers`, `fetch_section`, `compare_papers` | — |

## Results

Filled in as milestones land. Numbers come from `papertrail-eval` on the gold set.

| Retriever | Recall@5 | MRR | Wrong-paper rate (cross-paper Qs) |
|---|---|---|---|
| BM25 | – | – | – |
| Vector | – | – | – |
| Hybrid (RRF) | – | – | – |
| Hybrid + rerank | – | – | – |

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
make install
make test           # unit tests
python scripts/download_corpus.py   # fetch the arXiv PDFs (Capuchin is manual)
make ingest         # PDFs -> sections -> chunks (data/processed/chunks.jsonl)
make eval           # validate the gold set and check every evidence quote is found
make up             # Postgres + pgvector and the API on :8000
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
