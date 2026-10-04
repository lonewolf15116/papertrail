# Gold question set

`questions.jsonl` holds one hand-labelled question per line, validated by `papertrail.schemas.GoldQuestion`.
`papertrail-eval` refuses to run if any line is invalid.

## Why labels are quotes, not chunk ids

Chunk ids change whenever chunking changes, so labelling by chunk id would break every label during the
chunk-size ablation. Each answerable question instead records **evidence**: the paper, the page, and a
verbatim quote copied from the PDF. At evaluation time the quote is matched (ignoring case, spacing,
hyphenation and punctuation) to whichever chunks contain it under the current chunking.

`papertrail-eval --chunks data/processed/chunks.jsonl` checks that every quote is found, which catches
typos, wrong paper ids and PDF-extraction problems.

## Fields

| field | required | meaning |
|---|---|---|
| `id` | yes | unique, stable id, e.g. `remat-007` |
| `question` | yes | the question as a user would ask it |
| `kind` | yes | `factual`, `unanswerable` or `cross_paper` |
| `status` | no | `verified` (default) or `draft`. Drafts are validated but **not scored**. |
| `evidence` | unless unanswerable | list of `{"paper_id", "page", "quote"}`; quote is ≥ 20 chars, copied verbatim |
| `distractor_paper_ids` | cross_paper only | papers likely to be retrieved by mistake; never also a gold paper |
| `reference_answer` | no | short answer written from the paper |
| `notes` | no | anything that helps re-check the label |

Example:

```json
{"id": "remat-001", "question": "...", "kind": "factual",
 "evidence": [{"paper_id": "dtr2021", "page": 4, "quote": "exact sentence copied from the PDF"}],
 "reference_answer": "..."}
```

## Targets

60–100 questions: roughly 15% unanswerable and 15–20% cross-paper, mostly from the rematerialization
group, where terms like "checkpointing", "eviction" and "rematerialization" mean different things
in different papers.

Citation correctness and refusals are judged against these hand labels. Only faithfulness uses an LLM
judge, and that judge is spot-checked.

The three `tmpl-*` rows are format examples (status `draft`); delete them once real labels exist.
