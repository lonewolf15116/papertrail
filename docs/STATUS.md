# PaperTrail status and handoff

Read this first when starting a new session. Last updated 2026-10-10.

## Roadmap

- **Week 1: corpus and gold set.** Done. 24 papers in `data/corpus/papers.yaml`; 23 PDFs present. Capuchin manual still missing (add by hand to `data/corpus/pdf/capuchin2020.pdf`). Gold set: 74 questions (49 factual, 15 cross-paper, 10 unanswerable).
- **Week 2: retrieval ablation and CI gate.** Done. Nine retrievers scored on evidence-level recall@5, MRR@10, wrong-paper@5, latency. Default retriever `hybrid_header`. Ablation workflow and quality gate passing (run 37861970784).
- **Week 3: grounded answers with citations and refusal.** Code reviewed and merged to main (merge f5a0dc9; `/ask`, `answer.py`, `pipeline.py`, `eval/answer_eval.py`, manual `answers-eval.yml`), plus a provider switch for Anthropic or OpenAI. All unit tests, ruff and mypy pass. Remaining: the first live scored run (item 6), then read citation precision, refusal accuracy and faithfulness; deploy; log latency and cost.
- **Week 4: agent tools.** `search_papers`, `fetch_section`, `compare_papers`. Only after Week 3 holds up.

## Where the numbers stand

- Gold review applied (commit a5c5ad1): 67 verified, 7 still draft (Needs fix): cross-001, cross-002, cross-003, cross-004, cross-005, remat-001, remat-011.
- Baseline in `results/baseline.json` is still provisional (hybrid_header, recall@5 0.7031, MRR 0.5058, from draft labels). Expect it to move once the verified-only run is read.

## Open items

1. Read the new ablation run on Actions and the `ablation-results` branch.
2. Fix or re-verify the 7 needs-fix questions; decide on cross-004 (overlaps remat-012).
3. Add the Capuchin PDF by hand.
4. ~~Review and merge `week3-grounded-answers`~~ Done (see Week 3 above).
5. Decide on an updated baseline once the verified numbers are final.
6. **First live answer evaluation: done (run 4, 10 questions, OpenAI `gpt-4.1-mini` answers, `gpt-4.1` judge, provisional labels).** Works end to end, 0 errors, ~$0.001 per question. Results are on the `answers-results` branch (artifacts and logs are blocked from the Claude sandbox). Numbers: refusal accuracy 0.80 (2 false refusals), citation precision 0.594, citation hit rate 0.75, wrong-paper citations 0.0, faithfulness 0.919 (same-vendor judge: spot-check `answers-spotcheck-provisional.jsonl` before quoting it), p50 2.5 s. Only 10 factual questions, so unanswerable-question behaviour is untested. Findings: (a) remat-006 hit the 600-token cap and was reported as "passages do not support an answer"; fixed by raising `max_answer_tokens` to 1500 and reporting truncation explicitly. (b) 7 citations dropped, concentrated in remat-003 and remat-010 (answers with LaTeX/math): the verbatim-quote check is strict about math text, so look at how `squash` handles PDF-extracted math before loosening anything. (c) Citation precision is also low where the gold chunk differs from the cited chunk (remat-002, remat-005): check whether the gold labels or the metric are too narrow.

**Next:** rerun with no limit (all 74 questions, includes the 10 unanswerable) to get refusal behaviour, then deal with (b) and (c), then deploy and write up the results table in the README.

## Working notes

- Run `.venv/bin/papertrail-eval` (the venv entry point), not `papertrail-eval` on the system path.
- Local Postgres: pgvector 0.6.0, port 5432, role papertrail/papertrail.
- Models: download with `scripts/download_models.py` (models are cached in CI).
- Model choice: Haiku for routine runs (apply reviews, push, check CI), Sonnet for code, Opus only for design or interpreting results.
