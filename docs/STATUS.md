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
6. **First live answer evaluation.** Sunny has an OpenAI key, no Anthropic key. The OpenAI client is built (`OpenAIClient`, `make_client`, `Settings.answer_llm()/judge_llm()`, `PAPERTRAIL_LLM_PROVIDER=openai`) and unit-tested with fakes only; it has never made a real API call. Next: add the repo secret `OPENAI_API_KEY` (GitHub > Settings > Secrets and variables > Actions; never paste keys in chat or commit them), then run the manual "Answer evaluation" workflow with provider `openai`, include_drafts on, limit 10. The default OpenAI models (`gpt-4.1-mini` answers, `gpt-4.1` judge) and their prices in `config.py` are unverified guesses: check they exist on the account and override with `PAPERTRAIL_LLM_MODEL` / `PAPERTRAIL_JUDGE_MODEL` and the price variables if not. The workflow now publishes results to the `answers-results` branch (fetch it with git; run artifacts and logs are blocked from the Claude sandbox). The first 10-question run (run 38070731191) succeeded but its artifact could not be read, so rerun it once. Answerer and judge are both OpenAI, so lean on the human spot-check export (`answers-spotcheck-*.jsonl`) before quoting the faithfulness number.

## Working notes

- Run `.venv/bin/papertrail-eval` (the venv entry point), not `papertrail-eval` on the system path.
- Local Postgres: pgvector 0.6.0, port 5432, role papertrail/papertrail.
- Models: download with `scripts/download_models.py` (models are cached in CI).
- Model choice: Haiku for routine runs (apply reviews, push, check CI), Sonnet for code, Opus only for design or interpreting results.
