57 answerable questions, verified labels

| Retriever | Recall@5 | MRR@10 | Wrong-paper@5 (cross-paper) | p50 ms | p95 ms |
|---|---|---|---|---|---|
| fts | 0.263 | 0.218 | 0.180 | 19 | 44 |
| bm25 | 0.614 | 0.470 | 0.180 | 2 | 4 |
| bm25_header | 0.693 | 0.491 | 0.160 | 2 | 4 |
| vector | 0.439 | 0.294 | 0.100 | 29 | 35 |
| vector_header | 0.491 | 0.391 | 0.160 | 30 | 33 |
| hybrid | 0.579 | 0.455 | 0.180 | 33 | 40 |
| hybrid_header | 0.702 | 0.498 | 0.140 | 33 | 39 |
| hybrid_rerank | 0.605 | 0.488 | 0.160 | 1824 | 1859 |
| hybrid_header_rerank | 0.623 | 0.490 | 0.180 | 1815 | 1847 |
