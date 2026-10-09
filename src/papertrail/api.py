"""FastAPI service: /health and /ask (retrieve, then answer with verified citations or refuse)."""

import logging
import threading

from fastapi import FastAPI, HTTPException

from papertrail import __version__
from papertrail.pipeline import Pipeline, build_default_pipeline
from papertrail.schemas import AskRequest, AskResponse

log = logging.getLogger("papertrail.api")
app = FastAPI(title="PaperTrail", version=__version__)

_pipeline: Pipeline | None = None
_lock = threading.Lock()


def get_pipeline() -> Pipeline:
    """Build the production pipeline on first use, so /health works without a database or key.

    A failed build is retried on the next request (the usual causes, a missing key or an
    unindexed database, are fixed without restarting the service)."""
    global _pipeline
    with _lock:
        if _pipeline is None:
            try:
                _pipeline = build_default_pipeline()
            except Exception as exc:
                log.error("pipeline unavailable: %s", exc)
                raise HTTPException(status_code=503, detail=f"pipeline unavailable: {exc}") from exc
        return _pipeline


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    # Looked up here, not injected with Depends, so a malformed request is a 422 even when
    # the pipeline is unavailable (FastAPI resolves dependencies before reporting body errors).
    pipeline = get_pipeline()
    try:
        return pipeline.ask(req.question, req.top_k)
    except Exception as exc:  # LLM or database failure: a gateway problem, not a bad request
        log.exception("ask failed")
        raise HTTPException(status_code=502, detail="the answer pipeline failed") from exc
