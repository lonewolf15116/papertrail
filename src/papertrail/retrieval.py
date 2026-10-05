"""Retrievers for the Week 2 ablation.

Every retriever returns ranked Chunks through one interface, so the evaluation harness scores
them identically and the ablation table is reproducible:

    bm25           Okapi BM25 over the chunk texts (rank-bm25, in memory)
    fts            Postgres full-text search (ts_rank_cd, OR over query lexemes)
    vector         pgvector cosine similarity over dense embeddings
    hybrid         reciprocal rank fusion of bm25 + vector
    hybrid_rerank  hybrid candidates re-scored by a cross-encoder
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any, Protocol

from papertrail.embed import Embedder, Reranker
from papertrail.schemas import Chunk

TOKEN = re.compile(r"[a-z0-9]+")
STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "does",
        "do",
        "for",
        "from",
        "has",
        "have",
        "how",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "was",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "with",
    ]
)


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN.findall(text.lower()) if t not in STOPWORDS]


class Retriever(Protocol):
    name: str

    def retrieve(self, question: str, k: int) -> list[Chunk]: ...


def reciprocal_rank_fusion(rankings: Sequence[Sequence[str]], k: int = 60) -> list[str]:
    """Fuse several ranked id lists (e.g. BM25 and vector) into one ranking.

    Score for an id = sum over lists of 1 / (k + rank), rank starting at 1.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=lambda cid: (-scores[cid], cid))


def with_header(chunk: Chunk, titles: dict[str, str]) -> str:
    """Chunk text prefixed with its paper title and section ("contextual chunk header").

    A passage from ZeRO-Offload rarely names ZeRO-Offload in its own text; the header gives
    keyword and dense retrievers the paper identity that the question usually mentions."""
    title = titles.get(chunk.paper_id, chunk.paper_id)
    return f"{title}. {chunk.section}. {chunk.text}"


class BM25Retriever:
    def __init__(
        self,
        chunks: Sequence[Chunk],
        titles: dict[str, str] | None = None,
        name: str | None = None,
    ) -> None:
        from rank_bm25 import BM25Okapi

        self.name = name or ("bm25_header" if titles else "bm25")
        self._chunks = list(chunks)
        texts = [with_header(c, titles) if titles else c.text for c in self._chunks]
        self._bm25 = BM25Okapi([tokenize(t) for t in texts])

    def retrieve(self, question: str, k: int) -> list[Chunk]:
        scores = self._bm25.get_scores(tokenize(question))
        order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:k]
        return [self._chunks[i] for i in order]


def _rows_to_chunks(rows: Sequence[Sequence[Any]]) -> list[Chunk]:
    return [Chunk(chunk_id=r[0], paper_id=r[1], section=r[2], page=r[3], text=r[4]) for r in rows]


class PgFullTextRetriever:
    """Postgres full-text search. plainto_tsquery ANDs every term, which finds almost nothing
    for a natural-language question, so the lexemes are OR-ed and ranked by ts_rank_cd."""

    name = "fts"
    SQL = """
        WITH q AS (
          SELECT replace(plainto_tsquery('english', %s)::text, '&', '|')::tsquery AS query
        )
        SELECT c.chunk_id, c.paper_id, c.section, c.page, c.text
        FROM chunks c, q
        WHERE q.query::text <> '' AND c.tsv @@ q.query
        ORDER BY ts_rank_cd(c.tsv, q.query) DESC, c.chunk_id
        LIMIT %s
    """

    def __init__(self, conn: Any) -> None:
        self._conn = conn

    def retrieve(self, question: str, k: int) -> list[Chunk]:
        with self._conn.cursor() as cur:
            cur.execute(self.SQL, (question, k))
            return _rows_to_chunks(cur.fetchall())


class PgVectorRetriever:
    """Cosine nearest neighbours in pgvector. `column` picks the plain or header embeddings."""

    COLUMNS = {"embedding": "vector", "embedding_header": "vector_header"}
    SQL = """
        SELECT chunk_id, paper_id, section, page, text
        FROM chunks
        WHERE {col} IS NOT NULL
        ORDER BY {col} <=> %s, chunk_id
        LIMIT %s
    """

    def __init__(self, conn: Any, embedder: Embedder, column: str = "embedding") -> None:
        if column not in self.COLUMNS:
            raise ValueError(f"unknown embedding column '{column}'")
        self.name = self.COLUMNS[column]
        self._sql = self.SQL.format(col=column)  # column is from a fixed allow-list
        self._conn = conn
        self._embedder = embedder

    def retrieve(self, question: str, k: int) -> list[Chunk]:
        vec = self._embedder.embed_queries([question])[0]
        with self._conn.cursor() as cur:
            cur.execute(self._sql, (vec, k))
            return _rows_to_chunks(cur.fetchall())


class HybridRetriever:
    """Reciprocal rank fusion of several retrievers' top `candidates`."""

    def __init__(
        self, retrievers: Sequence[Retriever], candidates: int = 50, name: str = "hybrid"
    ) -> None:
        self.name = name
        self._retrievers = list(retrievers)
        self._candidates = candidates

    def retrieve(self, question: str, k: int) -> list[Chunk]:
        by_id: dict[str, Chunk] = {}
        rankings = []
        for r in self._retrievers:
            got = r.retrieve(question, self._candidates)
            rankings.append([c.chunk_id for c in got])
            by_id.update((c.chunk_id, c) for c in got)
        return [by_id[cid] for cid in reciprocal_rank_fusion(rankings)[:k]]


class RerankRetriever:
    """Re-score a base retriever's top `candidates` with a cross-encoder."""

    def __init__(
        self, base: Retriever, reranker: Reranker, candidates: int = 20, name: str | None = None
    ) -> None:
        self.name = name or f"{base.name}_rerank"
        self._base = base
        self._reranker = reranker
        self._candidates = candidates

    def retrieve(self, question: str, k: int) -> list[Chunk]:
        pool = self._base.retrieve(question, max(k, self._candidates))
        if not pool:
            return []
        scores = self._reranker.score(question, [c.text for c in pool])
        order = sorted(range(len(pool)), key=lambda i: (-scores[i], i))
        return [pool[i] for i in order[:k]]
