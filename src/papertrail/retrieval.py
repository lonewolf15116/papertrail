"""Retriever interface. Concrete BM25 / vector / hybrid / reranked retrievers land in milestone 2.

Every retriever returns ranked chunk ids so the evaluation harness can score any of them the
same way, which is what makes the ablation table reproducible.
"""

from collections.abc import Sequence
from typing import Protocol

from papertrail.schemas import Chunk


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
