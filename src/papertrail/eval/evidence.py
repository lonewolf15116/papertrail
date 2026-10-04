"""Resolve hand-labelled evidence quotes to chunk ids for the current chunking.

Matching compares lowercase alphanumerics only, so hyphenation, line breaks, ligatures and
spacing differences between a PDF viewer copy and our extraction do not break a label.
A quote that spans a chunk boundary counts for every chunk holding a large enough piece of it.
"""

import re
from collections.abc import Iterable

from papertrail.schemas import Chunk, Evidence, GoldQuestion

MIN_PIECE = 40  # alphanumeric chars a chunk must share with a boundary-spanning quote


def squash(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


class EvidenceIndex:
    def __init__(self, chunks: Iterable[Chunk]) -> None:
        self._by_paper: dict[str, list[tuple[str, str]]] = {}
        for c in chunks:
            self._by_paper.setdefault(c.paper_id, []).append((c.chunk_id, squash(c.text)))

    def resolve(self, ev: Evidence) -> set[str]:
        """Chunk ids in ev.paper_id that contain the quote (or a large piece of it)."""
        q = squash(ev.quote)
        chunks = self._by_paper.get(ev.paper_id, [])
        whole = {cid for cid, text in chunks if q in text}
        if whole:
            return whole
        piece = min(len(q), MIN_PIECE)
        head, tail = q[:piece], q[-piece:]
        return {cid for cid, text in chunks if head in text or tail in text}

    def gold_chunk_ids(self, question: GoldQuestion) -> set[str]:
        return (
            set().union(*(self.resolve(e) for e in question.evidence))
            if question.evidence
            else set()
        )

    def unresolved(self, questions: Iterable[GoldQuestion]) -> list[str]:
        """Labels whose quote is not found in the corpus: typos, wrong paper, or bad extraction."""
        problems = []
        for q in questions:
            for e in q.evidence:
                if e.paper_id not in self._by_paper:
                    problems.append(f"{q.id}: paper '{e.paper_id}' is not in the corpus")
                elif not self.resolve(e):
                    problems.append(f"{q.id}: quote not found in {e.paper_id}: {e.quote[:60]!r}")
        return problems
