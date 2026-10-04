"""Load and validate the hand-labelled gold question set (JSONL, one GoldQuestion per line)."""

import json
from collections import Counter
from pathlib import Path

from papertrail.schemas import GoldQuestion


def load_gold(path: Path) -> list[GoldQuestion]:
    items: list[GoldQuestion] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            items.append(GoldQuestion.model_validate(json.loads(line)))
        except Exception as exc:  # surface the line number for whoever is labelling
            raise ValueError(f"{path}:{lineno}: {exc}") from exc
    dupes = [qid for qid, n in Counter(q.id for q in items).items() if n > 1]
    if dupes:
        raise ValueError(f"duplicate question ids: {dupes}")
    return items
