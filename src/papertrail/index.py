"""Load processed chunks (and their embeddings) into Postgres/pgvector.

Idempotent: papers and chunks are upserted, and chunks no longer produced by ingestion are
deleted, so re-running after a chunking change leaves exactly the new chunk set.

Run: papertrail-index [--no-embed]
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml

from papertrail.embed import Embedder
from papertrail.retrieval import with_header
from papertrail.schemas import Chunk


def connect(url: str) -> Any:
    import psycopg
    from pgvector.psycopg import register_vector

    conn = psycopg.connect(url, autocommit=True)
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")  # before registering the type
    register_vector(conn)
    return conn


def load_papers(manifest: Path) -> list[dict[str, Any]]:
    papers: list[dict[str, Any]] = yaml.safe_load(manifest.read_text(encoding="utf-8"))["papers"]
    return papers


def embedding_dim(conn: Any) -> int | None:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT atttypmod FROM pg_attribute "
            "WHERE attrelid = 'chunks'::regclass AND attname = 'embedding'"
        )
        row = cur.fetchone()
    return int(row[0]) if row and row[0] > 0 else None


def index_chunks(
    conn: Any,
    chunks: Sequence[Chunk],
    papers: Sequence[dict[str, Any]],
    embedder: Embedder | None = None,
    batch_size: int = 64,
) -> dict[str, int]:
    paper_rows = [
        (p["id"], p["title"], p.get("arxiv"), p.get("venue"), p.get("group")) for p in papers
    ]
    titles = {p["id"]: p["title"] for p in papers}
    dim = embedding_dim(conn)
    if embedder is not None and dim is not None and embedder.dim != dim:
        raise ValueError(
            f"embedder produces {embedder.dim}-dim vectors but chunks.embedding is VECTOR({dim}); "
            "change scripts/init.sql to match the model"
        )
    with conn.cursor() as cur:
        cur.executemany(
            """INSERT INTO papers (paper_id, title, arxiv_id, venue, grp)
               VALUES (%s, %s, %s, %s, %s)
               ON CONFLICT (paper_id) DO UPDATE SET title = EXCLUDED.title,
                 arxiv_id = EXCLUDED.arxiv_id, venue = EXCLUDED.venue, grp = EXCLUDED.grp""",
            paper_rows,
        )
        for start in range(0, len(chunks), batch_size):
            batch = list(chunks[start : start + batch_size])
            vecs = hvecs = None
            if embedder is not None:
                vecs = embedder.embed_passages([c.text for c in batch])
                hvecs = embedder.embed_passages([with_header(c, titles) for c in batch])
            rows = [
                (
                    c.chunk_id,
                    c.paper_id,
                    c.section,
                    c.page,
                    c.text,
                    vecs[i] if vecs is not None else None,
                    hvecs[i] if hvecs is not None else None,
                )
                for i, c in enumerate(batch)
            ]
            cur.executemany(
                """INSERT INTO chunks
                     (chunk_id, paper_id, section, page, text, embedding, embedding_header)
                   VALUES (%s, %s, %s, %s, %s, %s, %s)
                   ON CONFLICT (chunk_id) DO UPDATE SET paper_id = EXCLUDED.paper_id,
                     section = EXCLUDED.section, page = EXCLUDED.page, text = EXCLUDED.text,
                     embedding = COALESCE(EXCLUDED.embedding, chunks.embedding),
                     embedding_header = COALESCE(EXCLUDED.embedding_header,
                                                 chunks.embedding_header)""",
                rows,
            )
        cur.execute(
            "DELETE FROM chunks WHERE NOT (chunk_id = ANY(%s))", ([c.chunk_id for c in chunks],)
        )
        removed = cur.rowcount
        cur.execute("SELECT count(*), count(embedding) FROM chunks")
        total, embedded = cur.fetchone()
    return {"chunks": int(total), "embedded": int(embedded), "removed": int(removed)}


def main(argv: list[str] | None = None) -> int:
    from papertrail.config import get_settings
    from papertrail.ingest import load_chunks

    settings = get_settings()
    parser = argparse.ArgumentParser(prog="papertrail-index")
    parser.add_argument("--chunks", type=Path, default=Path("data/processed/chunks.jsonl"))
    parser.add_argument("--manifest", type=Path, default=Path("data/corpus/papers.yaml"))
    parser.add_argument("--no-embed", action="store_true", help="index text only (no vectors)")
    args = parser.parse_args(argv)

    chunks = load_chunks(args.chunks)
    embedder: Embedder | None = None
    if not args.no_embed:
        from papertrail.embed import SentenceTransformerEmbedder

        embedder = SentenceTransformerEmbedder(settings.embedding_model)
    conn = connect(settings.database_url)
    started = time.perf_counter()
    stats = index_chunks(conn, chunks, load_papers(args.manifest), embedder)
    print(
        f"indexed {stats['chunks']} chunks ({stats['embedded']} with embeddings, "
        f"{stats['removed']} stale removed) in {time.perf_counter() - started:.1f}s"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
