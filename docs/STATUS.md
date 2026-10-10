# PaperTrail status and handoff

Read this first when starting a new session. Last updated 2026-10-10.

## Roadmap

- **Week 1: corpus and gold set.** Done. 24 papers in `data/corpus/papers.yaml`; 23 PDFs present. Capuchin manual still missing (add by hand to `data/corpus/pdf/capuchin2020.pdf`). Gold set: 74 questions (49 factual, 15 cross-paper, 10 unanswerable).
- **Week 2: retrieval ablation and CI gate.** Done. Nine retrievers scored on evidence-level recall@5, MRR@10, wrong-paper@5, latency. Default retriever `hybrid_header`. Ablation workflow and quality gate passing (run 37861970784).
- **Week 3: grounded answers with citations and refusal.** Implementation is already written and unit-tested on the remote branch `week3-grounded-answers` (single commit baef69a on top of 4ac9e41; not merged to main). It adds `answer.py`, `pipeline.py`, the `/ask` endpoint, `eval/answer_eval.py`, tests, and a manual `answers-eval.yml` workflow. Do not re-implement it. A trial merge into current main is clean (main is 2 commits ahead: gold review and this file). Remaining: review the branch, merge, run the first scored answer evaluation (citation precision, refusal accuracy, faithfulness; needs `ANTHROPIC_API_KEY`), deploy, log latency and cost.
- **Week 4: agent tools.** `search_papers`, `fetch_section`, `compare_papers`. Only after Week 3 holds up.

## Where the numbers stand

- Gold review applied (commit a5c5ad1): 67 verified, 7 still draft (Needs fix): cross-001, cross-002, cross-003, cross-004, cross-005, remat-001, remat-011.
- Baseline in `results/baseline.json` is still provisional (hybrid_header, recall@5 0.7031, MRR 0.5058, from draft labels). Expect it to move once the verified-only run is read.

## Open items

1. Read the new ablation run on Actions and the `ablation-results` branch.
2. Fix or re-verify the 7 needs-fix questions; decide on cross-004 (overlaps remat-012).
3. Add the Capuchin PDF by hand.
4. Review `week3-grounded-answers` (check it before writing any Week 3 code), merge to main, then do the first live `papertrail-eval-answers` run (start with `--include-drafts --limit 10` to keep cost low).
5. Decide on an updated baseline once the verified numbers are final.
6. **No Anthropic API key available; Sunny has an OpenAI key.** The Week 3 code is Anthropic-only (`AnthropicClient` in `answer.py`, `anthropic` in `pyproject.toml`, judge in `eval/answer_eval.py`). Plan: add an `OpenAIClient` (function calling, same `Generation` interface), a provider switch in `config.py`, the `openai` dependency, `OPENAI_API_KEY` in `.env.example` and the workflow, and tests with a fake client. Use a different model for the faithfulness judge than for the answerer, and lean on the seeded human spot-check export since both are the same vendor. The key goes in a GitHub Actions secret or a local env var, never in chat or the repo. Update the README claims about the provider when done.

## Working notes

- Run `.venv/bin/papertrail-eval` (the venv entry point), not `papertrail-eval` on the system path.
- Local Postgres: pgvector 0.6.0, port 5432, role papertrail/papertrail.
- Models: download with `scripts/download_models.py` (models are cached in CI).
- Model choice: Haiku for routine runs (apply reviews, push, check CI), Sonnet for code, Opus only for design or interpreting results.
