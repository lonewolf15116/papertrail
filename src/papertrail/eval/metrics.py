"""Retrieval and answer metrics. Pure functions, so they are unit-tested and judge-free."""

from collections.abc import Iterable, Sequence
from statistics import mean


def recall_at_k(retrieved: Sequence[str], gold: Iterable[str], k: int) -> float:
    """Fraction of gold chunk ids that appear in the top-k retrieved ids."""
    gold_set = set(gold)
    if not gold_set:
        raise ValueError("recall is undefined without gold ids")
    return len(gold_set.intersection(retrieved[:k])) / len(gold_set)


def hit_at_k(retrieved: Sequence[str], gold: Iterable[str], k: int) -> float:
    """1.0 if any gold id is in the top k, else 0.0."""
    return float(bool(set(gold).intersection(retrieved[:k])))


def reciprocal_rank(retrieved: Sequence[str], gold: Iterable[str]) -> float:
    """1 / rank of the first gold id in the ranking, 0.0 if none is retrieved."""
    gold_set = set(gold)
    for rank, chunk_id in enumerate(retrieved, start=1):
        if chunk_id in gold_set:
            return 1.0 / rank
    return 0.0


def mean_reciprocal_rank(runs: Iterable[tuple[Sequence[str], Iterable[str]]]) -> float:
    return mean(reciprocal_rank(r, g) for r, g in runs)


def wrong_paper_rate(
    retrieved_paper_ids: Sequence[str], distractors: Iterable[str], k: int
) -> float:
    """Share of the top-k results that come from a known distractor paper.

    The cross-paper failure mode: DTR vs Checkmate vs Capuchin use similar words for different
    claims, so a plausible passage from the wrong paper ranks highly.
    """
    top = retrieved_paper_ids[:k]
    if not top:
        return 0.0
    bad = set(distractors)
    return sum(pid in bad for pid in top) / len(top)


def citation_precision(cited_chunk_ids: Iterable[str], gold: Iterable[str]) -> float:
    """Fraction of cited chunks that are gold supporting passages (hand labels, no LLM)."""
    cited = list(cited_chunk_ids)
    if not cited:
        return 0.0
    gold_set = set(gold)
    return sum(c in gold_set for c in cited) / len(cited)


def refusal_accuracy(predicted_refused: Sequence[bool], should_refuse: Sequence[bool]) -> float:
    if len(predicted_refused) != len(should_refuse):
        raise ValueError("prediction and label lists differ in length")
    if not should_refuse:
        raise ValueError("no items")
    return mean(p == s for p, s in zip(predicted_refused, should_refuse, strict=True))


def percentile(values: Sequence[float], q: float) -> float:
    """Nearest-rank percentile (q in [0, 100]); used for p50/p95 latency."""
    if not values:
        raise ValueError("no values")
    if not 0 <= q <= 100:
        raise ValueError("q must be in [0, 100]")
    ordered = sorted(values)
    rank = max(1, -(-q * len(ordered) // 100))  # ceil(q/100 * n), at least 1
    return ordered[int(rank) - 1]
