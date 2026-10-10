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
| 3. Grounded answers | FastAPI `/ask` with paper/page/section citations and a refusal path | built, tested, scored once on all 74 questions (provisional labels); known weaknesses below |
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

## Results: grounded answers

Four full runs of [`answers-eval.yml`](.github/workflows/answers-eval.yml), 74 questions each, **provisional**
(draft labels included), `gpt-4.1-mini` answers, `gpt-4.1` faithfulness judge, `hybrid_header` retrieval, 0
errors. Run 2 adds a prompt rule (refuse when the question names a model, hardware or metric the sources do not
report); run 3 a math-tolerant citation check; run 4 an ellipsis-aware one. Raw output is on the
[`answers-results`](../../tree/answers-results) branch.

| Metric | Run 1 | Run 2 (+ refusal rule) | Run 3 (+ math check) | Run 4 (+ ellipsis check) |
|---|---|---|---|---|
| Refusal accuracy | 0.878 | 0.919 | 0.919 | 0.932 |
| False-answer rate (unanswerable answered) | 0.400 (4 of 10) | 0.200 (2) | 0.200 (2) | 0.200 (2) |
| False-refusal rate (answerable refused) | 0.078 (5) | 0.062 (4) | 0.062 (4) | 0.047 (3) |
| Citation precision (exact chunk) | 0.489 | 0.474 | 0.514 | 0.471 |
| Citation precision (labelled page) | n/a | n/a | 0.519 | 0.486 |
| Citation hit rate | 0.644 | 0.617 | 0.683 | 0.656 |
| Wrong-paper citation rate | 0.051 | 0.067 | 0.100 | 0.082 |
| Faithfulness (LLM judge) | 0.932 | 0.903 | 0.943 | 0.953 |
| Citations dropped by verification | 34 | 34 | 25 | 18 |
| Latency p50 / p95 | 1.5 s / 2.9 s | 1.5 s / 2.8 s | 1.5 s / 2.7 s | 1.6 s / 2.7 s |
| Cost per question | about $0.001 | about $0.001 | about $0.001 | about $0.001 |

What the numbers say, and what they do not:

- **Run-to-run noise is now visible, and it is as large as most of the changes.** The model is not deterministic
  even at temperature 0. Across the four runs, citation precision ranges 0.47 to 0.51, faithfulness 0.90 to
  0.95, wrong-paper rate 0.05 to 0.10. Only two moves are clearly real: the false-answer rate halving from the
  refusal rule (0.40 to 0.20, then stable for three runs), and citations dropped falling 34 to 18.
- **Near-miss questions were the worst failure; the rule halved it.** In run 1 all four unanswerable questions it
  answered swapped one detail for something the corpus does cover: H100 became V100 (ZeRO-Infinity), TPU v4
  became AWS GPUs (PipeDream-2BW), GPT-3 became ResNet-50 (ActNN), transformers became VGG-16 (vDNN). The
  citations were real, so quote checking cannot catch this. PipeDream-2BW and ActNN are now refused;
  ZeRO-Infinity on H100 and vDNN on transformers are still answered. Ten questions is a small sample: one
  question is 10 points.
- **Faithfulness is high but the judge is the same vendor as the answerer.** Treat 0.90 to 0.95 as an upper
  bound until the spot-check sample is read by a person.
- **A verified quote does not mean the answer is supported.** Quote checking proves the quote is in the cited
  chunk, not that the answer's claims came from it. In run 4, pipe-006 and zero-002 were correct answers
  (they match the paper) scored 0.0 and 0.25 for faithfulness, because the one citation that survived points to
  a chunk that lacks most of the claims, while the quotes that did hold the claims were dropped. Re-attributing
  a quote to the retrieved chunk that actually contains it is the next experiment.
- **Citation precision is low (0.47 to 0.51; 0.37 on cross-paper) and the page-level score is no better.** So
  the gap is not mostly "cited a neighbouring chunk on the right page". Answers average 1.8 citations and
  every non-labelled cited chunk counts against precision, while faithfulness is high. Whether the labels are
  too narrow has not been checked.
- **Wrong-paper citations are mostly a sibling paper cited alongside the right one** (PipeDream with PipeDream-2BW,
  Megatron 2019 with 2021, Chen 2016 with Gruslys 2016: 3 of 5). The other 2 are real misses on cross-paper
  questions (cross-008 and cross-014 miss one of the two gold papers).
- **Why quotes were rejected.** Of 25 recorded in run 3, 13 joined two passages with an ellipsis; run 4's
  per-fragment check cut those to 4 (the remaining two quotes splice passages that sit in different chunks,
  which the check rightly refuses). The math fix cleared the two worst cases in run 1 (remat-003, remat-010).

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

export OPENAI_API_KEY=... PAPERTRAIL_LLM_PROVIDER=openai   # or ANTHROPIC_API_KEY (the default provider)
                                    # the answer step and the faithfulness judge call the API
curl -s localhost:8000/ask -H 'content-type: application/json' \
  -d '{"question": "Which tensor does DTR evict?"}'
papertrail-eval-answers --include-drafts --limit 10   # cheap trial run of the answer evaluation
```

## Layout

```
src/papertrail/
  api.py            FastAPI service (/health, /ask)
  pipeline.py       retrieve -> grounded answer; per-request structured log (latency, tokens, cost)
  answer.py         prompt, forced tool-call output, citation verification, refusal rule
  schemas.py        Pydantic contracts: chunks, citations, answers, gold questions
  ingest.py         PDF -> sections -> chunks (section-aware, page-accurate, references dropped)
  retrieval.py      Retriever interface + reciprocal rank fusion
  eval/metrics.py   recall@k, MRR, citation precision, refusal accuracy, wrong-paper rate, percentiles
  eval/evidence.py  maps labelled quotes to chunk ids under any chunking
  eval/gate.py      CI regression gate
  eval/run.py       papertrail-eval entry point
  eval/answer_eval.py  papertrail-eval-answers: refusal, citation and faithfulness scores
data/corpus/papers.yaml   corpus manifest (PDFs are git-ignored)
data/gold/                hand-labelled questions (see its README)
scripts/init.sql          pgvector schema with full-text and HNSW indexes
```

## How answers are grounded

`/ask` retrieves the top 5 chunks (hybrid + headers, the Week 2 default) and gives them to the model as
numbered sources. The model must call one tool: either an answer whose claims carry citations (a source
number and a quote copied from that source), or a refusal with a reason.

The quote is checked, not trusted. A citation survives only if its quote appears in the chunk it names
(ignoring case, spacing and hyphenation); the response carries the chunk's real paper, section and page.
If an answer is left with no verifiable citation it is turned into a refusal, so the service never returns
an uncited answer. The model never sees the gold labels, and the number of citations dropped is reported on
every response (`dropped_citations`) and in the evaluation.

Failure modes by design: no key or empty index returns 503 (and `/health` still works); an LLM or database
error returns 502; a malformed request is a 422 regardless.

## Evaluation design

- **Gold set** labelled by hand with verbatim evidence quotes (so labels survive re-chunking), unanswerable questions, and adversarial cross-paper
  questions where papers such as DTR, Checkmate and Capuchin use similar terms for different claims.
- **Retrieval**: recall@5 and MRR per retriever, plus the wrong-paper rate on cross-paper questions.
- **Answers**: citation precision and refusal accuracy from hand labels; faithfulness from an LLM judge
  that is spot-checked, so the evaluation does not rest entirely on another model.
- **Operations**: p50/p95 latency, tokens and estimated cost per question.
