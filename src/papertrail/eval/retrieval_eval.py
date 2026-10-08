"""Score retrievers on the gold set: the Week 2 ablation.

For every answerable question (factual and cross-paper) and every retriever:
  recall@5          share of evidence quotes found in the top 5 (evidence-level, see metrics)
  mrr@10            1 / rank of the first chunk holding any evidence, 0 if not in the top 10
  wrong_paper@5     cross-paper questions only: share of the top 5 from a known distractor paper
  latency p50/p95   wall-clock retrieval time per question

Only verified labels are scored unless --include-drafts is passed; results computed on drafts
are marked provisional in the output so they are never mistaken for the real numbers.

Run: papertrail-eval-retrieval --retrievers bm25,fts,vector,hybrid,hybrid_rerank
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any

from papertrail.eval.evidence import EvidenceIndex
from papertrail.eval.gold import load_gold
from papertrail.eval.metrics import evidence_recall_at_k, first_relevant_rank, percentile
from papertrail.retrieval import Retriever
from papertrail.schemas import Chunk, GoldQuestion, LabelStatus, QuestionKind

K = 5
MRR_DEPTH = 10


@dataclass
class QuestionResult:
    id: str
    kind: str
    recall_at_5: float
    rank: int | None
    wrong_paper_at_5: float | None
    latency_ms: float
    top5: list[str]


@dataclass
class RetrieverResult:
    retriever: str
    n_questions: int
    provisional: bool
    recall_at_5: float
    mrr_at_10: float
    wrong_paper_at_5: float | None
    latency_p50_ms: float
    latency_p95_ms: float
    by_kind: dict[str, dict[str, float]] = field(default_factory=dict)
    questions: list[QuestionResult] = field(default_factory=list)

    def summary(self) -> dict[str, object]:
        d = asdict(self)
        d.pop("questions")
        return d


def scored_questions(gold: Sequence[GoldQuestion], include_drafts: bool) -> list[GoldQuestion]:
    return [
        q
        for q in gold
        if q.kind is not QuestionKind.UNANSWERABLE
        and (include_drafts or q.status is LabelStatus.VERIFIED)
        and not q.id.startswith("tmpl-")
    ]


def evaluate(
    retriever: Retriever,
    questions: Sequence[GoldQuestion],
    index: EvidenceIndex,
    paper_of: Callable[[str], str],
    provisional: bool,
) -> RetrieverResult:
    results: list[QuestionResult] = []
    for q in questions:
        evidence_sets = [index.resolve(e) for e in q.evidence]
        gold_ids = set().union(*evidence_sets)
        start = time.perf_counter()
        got: list[Chunk] = retriever.retrieve(q.question, MRR_DEPTH)
        latency = (time.perf_counter() - start) * 1000
        ids = [c.chunk_id for c in got]
        rank = first_relevant_rank(ids, gold_ids)
        wrong = None
        if q.kind is QuestionKind.CROSS_PAPER:
            top_papers = [paper_of(cid) for cid in ids[:K]]
            bad = set(q.distractor_paper_ids)
            wrong = sum(p in bad for p in top_papers) / len(top_papers) if top_papers else 0.0
        results.append(
            QuestionResult(
                id=q.id,
                kind=q.kind.value,
                recall_at_5=evidence_recall_at_k(ids, evidence_sets, K),
                rank=rank,
                wrong_paper_at_5=wrong,
                latency_ms=latency,
                top5=ids[:K],
            )
        )

    def rr(r: QuestionResult) -> float:
        return 1.0 / r.rank if r.rank is not None and r.rank <= MRR_DEPTH else 0.0

    by_kind: dict[str, dict[str, float]] = {}
    for kind in sorted({r.kind for r in results}):
        sub = [r for r in results if r.kind == kind]
        by_kind[kind] = {
            "n": len(sub),
            "recall_at_5": mean(r.recall_at_5 for r in sub),
            "mrr_at_10": mean(rr(r) for r in sub),
        }
    wrongs = [r.wrong_paper_at_5 for r in results if r.wrong_paper_at_5 is not None]
    lat = [r.latency_ms for r in results]
    return RetrieverResult(
        retriever=retriever.name,
        n_questions=len(results),
        provisional=provisional,
        recall_at_5=mean(r.recall_at_5 for r in results) if results else 0.0,
        mrr_at_10=mean(rr(r) for r in results) if results else 0.0,
        wrong_paper_at_5=mean(wrongs) if wrongs else None,
        latency_p50_ms=percentile(lat, 50) if lat else 0.0,
        latency_p95_ms=percentile(lat, 95) if lat else 0.0,
        by_kind=by_kind,
        questions=results,
    )


def markdown_table(rows: Sequence[RetrieverResult]) -> str:
    head = (
        "| Retriever | Recall@5 | MRR@10 | Wrong-paper@5 (cross-paper) | p50 ms | p95 ms |\n"
        "|---|---|---|---|---|---|\n"
    )
    body = "".join(
        f"| {r.retriever} | {r.recall_at_5:.3f} | {r.mrr_at_10:.3f} | "
        f"{'–' if r.wrong_paper_at_5 is None else f'{r.wrong_paper_at_5:.3f}'} | "
        f"{r.latency_p50_ms:.0f} | {r.latency_p95_ms:.0f} |\n"
        for r in rows
    )
    return head + body


def build_retrievers(names: Sequence[str], chunks: Sequence[Chunk]) -> list[Retriever]:
    from papertrail.config import get_settings
    from papertrail.retrieval import (
        BM25Retriever,
        HybridRetriever,
        PgFullTextRetriever,
        PgVectorRetriever,
        RerankRetriever,
    )

    settings = get_settings()

    def paper_titles() -> dict[str, str]:
        from papertrail.index import load_papers

        return {p["id"]: p["title"] for p in load_papers(Path("data/corpus/papers.yaml"))}

    cache: dict[str, Retriever] = {}
    conn: Any = None
    embedder: Any = None

    def db() -> Any:
        nonlocal conn
        if conn is None:
            from papertrail.index import connect

            conn = connect(settings.database_url)
        return conn

    def emb() -> Any:
        nonlocal embedder
        if embedder is None:
            from papertrail.embed import SentenceTransformerEmbedder

            embedder = SentenceTransformerEmbedder(settings.embedding_model)
        return embedder

    def get(name: str) -> Retriever:
        if name in cache:
            return cache[name]
        r: Retriever
        if name == "bm25":
            r = BM25Retriever(chunks)
        elif name == "bm25_header":
            r = BM25Retriever(chunks, titles=paper_titles())
        elif name == "fts":
            r = PgFullTextRetriever(db())
        elif name == "vector":
            r = PgVectorRetriever(db(), emb())
        elif name == "vector_header":
            r = PgVectorRetriever(db(), emb(), column="embedding_header")
        elif name == "hybrid":
            r = HybridRetriever([get("bm25"), get("vector")], candidates=50)
        elif name == "hybrid_header":
            r = HybridRetriever(
                [get("bm25_header"), get("vector_header")], candidates=50, name="hybrid_header"
            )
        elif name in ("hybrid_rerank", "hybrid_header_rerank"):
            from papertrail.embed import CrossEncoderReranker

            r = RerankRetriever(
                get(name.removesuffix("_rerank")),
                CrossEncoderReranker(settings.reranker_model),
                candidates=settings.rerank_candidates,
                name=name,
            )
        else:
            raise ValueError(f"unknown retriever '{name}'")
        cache[name] = r
        return r

    return [get(n) for n in names]


def main(argv: list[str] | None = None) -> int:
    from papertrail.ingest import load_chunks

    parser = argparse.ArgumentParser(prog="papertrail-eval-retrieval")
    parser.add_argument("--gold", type=Path, default=Path("data/gold/questions.jsonl"))
    parser.add_argument("--chunks", type=Path, default=Path("data/processed/chunks.jsonl"))
    parser.add_argument(
        "--retrievers",
        default="fts,bm25,bm25_header,vector,vector_header,hybrid,hybrid_header,"
        "hybrid_rerank,hybrid_header_rerank",
    )
    parser.add_argument("--include-drafts", action="store_true", help="score draft labels too")
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args(argv)

    chunks = load_chunks(args.chunks)
    paper_by_chunk = {c.chunk_id: c.paper_id for c in chunks}
    index = EvidenceIndex(chunks)
    questions = scored_questions(load_gold(args.gold), args.include_drafts)
    if not questions:
        print("no questions to score: verify some labels, or pass --include-drafts")
        return 1
    problems = index.unresolved(questions)
    if problems:
        print("\n".join(f"UNRESOLVED {p}" for p in problems))
        return 1

    retrievers = build_retrievers([n.strip() for n in args.retrievers.split(",")], chunks)
    for r in retrievers:  # warm up model loads and connections outside the timed loop
        r.retrieve("warm up", 1)
    rows = [
        evaluate(r, questions, index, paper_by_chunk.__getitem__, args.include_drafts)
        for r in retrievers
    ]

    args.out.mkdir(parents=True, exist_ok=True)
    tag = "provisional" if args.include_drafts else "verified"
    for row in rows:
        (args.out / f"retrieval-{row.retriever}-{tag}.json").write_text(
            json.dumps(asdict(row), indent=2), encoding="utf-8"
        )
    summary = {row.retriever: row.summary() for row in rows}
    (args.out / f"retrieval-summary-{tag}.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    label = "PROVISIONAL (includes draft labels)" if args.include_drafts else "verified labels"
    header = f"{len(questions)} answerable questions, {label}\n"
    (args.out / f"retrieval-table-{tag}.md").write_text(
        header + "\n" + markdown_table(rows), encoding="utf-8"
    )
    print(header)
    print(markdown_table(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
