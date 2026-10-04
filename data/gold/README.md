# Gold question set

`questions.jsonl` holds one hand-labelled question per line, validated by `papertrail.schemas.GoldQuestion`.
`papertrail-eval` refuses to run if any line is invalid.

| field | required | meaning |
|---|---|---|
| `id` | yes | unique, stable id |
| `question` | yes | the question as a user would ask it |
| `kind` | yes | `factual`, `unanswerable` or `cross_paper` |
| `gold_chunk_ids` | unless unanswerable | chunks that support the answer (`<paper_id>:<chunk>`) |
| `gold_paper_ids` | no | papers the answer comes from |
| `distractor_paper_ids` | cross_paper only | papers likely to be retrieved by mistake |
| `reference_answer` | no | short answer written from the paper |
| `notes` | no | anything that helps re-check the label |

Target: 60–100 questions, with roughly 15% unanswerable and 15–20% cross-paper. Label citations and
refusals by hand; only faithfulness uses an LLM judge, and it gets spot-checked.

The three `tmpl-*` rows are format examples. Replace them once the corpus is ingested and chunk ids exist.
