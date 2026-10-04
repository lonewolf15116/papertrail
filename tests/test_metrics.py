import pytest

from papertrail.eval.metrics import (
    citation_precision,
    hit_at_k,
    mean_reciprocal_rank,
    percentile,
    recall_at_k,
    reciprocal_rank,
    refusal_accuracy,
    wrong_paper_rate,
)


def test_recall_at_k_counts_only_top_k():
    assert recall_at_k(["a", "b", "c", "d"], ["c", "z"], k=3) == 0.5
    assert recall_at_k(["a", "b", "c", "d"], ["d"], k=3) == 0.0


def test_recall_requires_gold():
    with pytest.raises(ValueError):
        recall_at_k(["a"], [], k=1)


def test_hit_at_k():
    assert hit_at_k(["a", "b"], ["b"], k=2) == 1.0
    assert hit_at_k(["a", "b"], ["b"], k=1) == 0.0


def test_reciprocal_rank_and_mrr():
    assert reciprocal_rank(["x", "y", "g"], ["g"]) == pytest.approx(1 / 3)
    assert reciprocal_rank(["x"], ["g"]) == 0.0
    runs = [(["g"], ["g"]), (["x", "g"], ["g"]), (["x"], ["g"])]
    assert mean_reciprocal_rank(runs) == pytest.approx((1 + 0.5 + 0) / 3)


def test_wrong_paper_rate():
    assert (
        wrong_paper_rate(["dtr", "checkmate", "dtr", "capuchin"], ["checkmate", "capuchin"], k=4)
        == 0.5
    )
    assert wrong_paper_rate([], ["x"], k=5) == 0.0


def test_citation_precision():
    assert citation_precision(["a", "b"], ["a"]) == 0.5
    assert citation_precision([], ["a"]) == 0.0


def test_refusal_accuracy():
    assert refusal_accuracy([True, False, True], [True, True, True]) == pytest.approx(2 / 3)
    with pytest.raises(ValueError):
        refusal_accuracy([True], [True, False])


@pytest.mark.parametrize(("q", "expected"), [(50, 50), (95, 95), (100, 100), (0, 1)])
def test_percentile_nearest_rank(q, expected):
    assert percentile(list(range(1, 101)), q) == expected
