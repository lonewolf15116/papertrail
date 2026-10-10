"""The answer pipeline: retrieve -> grounded answer with verified citations (or a refusal).

`Pipeline` takes its retriever and LLM client as arguments so tests and the evaluation harness
can swap in fakes; `build_default_pipeline` wires the production pieces (Postgres/pgvector,
the default hybrid+headers retriever, the configured LLM client).
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from papertrail.answer import (
    SYSTEM_PROMPT,
    TOOL_SCHEMA,
    AnswerClient,
    build_response,
    render_user_message,
)
from papertrail.config import Settings, get_settings
from papertrail.retrieval import Retriever
from papertrail.schemas import AnswerStatus, AskResponse, Chunk

log = logging.getLogger("papertrail.pipeline")


class Pipeline:
    def __init__(
        self,
        retriever: Retriever,
        client: AnswerClient,
        titles: dict[str, str],
        settings: Settings | None = None,
    ) -> None:
        self.retriever = retriever
        self.client = client
        self.titles = titles
        self.settings = settings or get_settings()

    def ask(self, question: str, top_k: int | None = None) -> AskResponse:
        s = self.settings
        start = time.perf_counter()
        chunks = self.retriever.retrieve(question, top_k or s.top_k)
        retrieval_ms = (time.perf_counter() - start) * 1000

        if not chunks:
            resp = AskResponse(
                status=AnswerStatus.REFUSED,
                refusal_reason="No passages were retrieved for this question.",
            )
        else:
            generation = self.client.generate(
                SYSTEM_PROMPT,
                render_user_message(question, chunks, self.titles),
                TOOL_SCHEMA,
                s.max_answer_tokens,
            )
            spec = s.answer_llm()
            resp = build_response(
                chunks, generation, (spec.input_price_per_mtok, spec.output_price_per_mtok)
            )

        resp.retrieval_ms = retrieval_ms
        resp.latency_ms = (time.perf_counter() - start) * 1000
        log.info(
            json.dumps(
                {
                    "event": "ask",
                    "status": resp.status.value,
                    "retriever": self.retriever.name,
                    "latency_ms": round(resp.latency_ms, 1),
                    "retrieval_ms": round(retrieval_ms, 1),
                    "input_tokens": resp.input_tokens,
                    "output_tokens": resp.output_tokens,
                    "cost_usd": resp.estimated_cost_usd,
                    "retrieved": resp.retrieved_chunk_ids,
                    "cited": [c.chunk_id for c in resp.citations],
                    "dropped_citations": resp.dropped_citations,
                }
            )
        )
        return resp


def load_index(conn: Any) -> tuple[list[Chunk], dict[str, str]]:
    """All chunks and paper titles from Postgres, so the service needs no files on disk."""
    with conn.cursor() as cur:
        cur.execute("SELECT chunk_id, paper_id, section, page, text FROM chunks ORDER BY chunk_id")
        chunks = [
            Chunk(chunk_id=r[0], paper_id=r[1], section=r[2], page=r[3], text=r[4])
            for r in cur.fetchall()
        ]
        cur.execute("SELECT paper_id, title FROM papers")
        titles = {r[0]: r[1] for r in cur.fetchall()}
    return chunks, titles


def build_default_pipeline(settings: Settings | None = None) -> Pipeline:
    """Production wiring. Raises if the database is empty or the LLM provider's key is missing."""
    import os

    from papertrail.answer import make_client
    from papertrail.eval.retrieval_eval import build_retrievers
    from papertrail.index import connect

    s = settings or get_settings()
    spec = s.answer_llm()
    if not os.environ.get(spec.key_env):
        raise RuntimeError(f"{spec.key_env} is not set")
    chunks, titles = load_index(connect(s.database_url))
    if not chunks:
        raise RuntimeError("the chunks table is empty: run papertrail-ingest and papertrail-index")
    (retriever,) = build_retrievers([s.retriever], chunks, titles=titles)
    retriever.retrieve("warm up", 1)  # load models before the first user request
    return Pipeline(retriever, make_client(spec.provider, spec.model), titles, s)
