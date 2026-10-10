64 answerable questions, PROVISIONAL (includes draft labels)

| Retriever | Recall@5 | MRR@10 | Wrong-paper@5 (cross-paper) | p50 ms | p95 ms |
|---|---|---|---|---|---|
| fts | 0.258 | 0.232 | 0.213 | 19 | 43 |
| bm25 | 0.648 | 0.479 | 0.147 | 2 | 3 |
| bm25_header | 0.719 | 0.501 | 0.147 | 2 | 3 |
| vector | 0.453 | 0.311 | 0.120 | 29 | 35 |
| vector_header | 0.500 | 0.392 | 0.133 | 30 | 35 |
| hybrid | 0.609 | 0.466 | 0.133 | 34 | 41 |
| hybrid_header | 0.703 | 0.506 | 0.093 | 34 | 39 |
| hybrid_rerank | 0.625 | 0.510 | 0.133 | 1825 | 1857 |
| hybrid_header_rerank | 0.641 | 0.512 | 0.133 | 1818 | 1864 |
