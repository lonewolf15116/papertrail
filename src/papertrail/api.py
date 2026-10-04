"""FastAPI service. /ask returns a refusal until the pipeline is built in milestone 3."""

import time

from fastapi import FastAPI

from papertrail import __version__
from papertrail.schemas import AnswerStatus, AskRequest, AskResponse

app = FastAPI(title="PaperTrail", version=__version__)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    start = time.perf_counter()
    # Placeholder until milestone 3: retrieve -> rerank -> answer with citations.
    return AskResponse(
        status=AnswerStatus.REFUSED,
        refusal_reason="pipeline not implemented yet",
        latency_ms=(time.perf_counter() - start) * 1000,
    )
