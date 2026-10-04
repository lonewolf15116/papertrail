from papertrail.eval.review import apply_reviews
from papertrail.schemas import Evidence, GoldQuestion

EV = [Evidence(paper_id="dtr2021", page=3, quote="a sufficiently long verbatim quote")]


def _q(qid: str) -> GoldQuestion:
    return GoldQuestion(
        id=qid,
        question=f"question {qid}?",
        kind="factual",
        status="draft",
        evidence=EV,
        reference_answer="old answer",
        notes="drafted",
    )


def test_apply_reviews_handles_each_verdict():
    gold = [_q("a"), _q("b"), _q("c"), _q("d")]
    reviews = {
        "a": {"status": "verified", "question": "question a?", "answer": "better answer"},
        "b": {"status": "rejected"},
        "c": {"status": "needs_fix", "comment": "quote is on page 4"},
        "d": {"status": "draft"},
    }
    out, report = apply_reviews(gold, reviews)
    by_id = {q.id: q for q in out}
    assert set(by_id) == {"a", "c", "d"}
    assert by_id["a"].status == "verified" and by_id["a"].reference_answer == "better answer"
    assert (
        by_id["c"].status == "draft"
        and by_id["c"].notes == "drafted | Reviewer: quote is on page 4"
    )
    assert by_id["d"] == gold[3]
    assert report.verified == ["a"] and report.rejected == ["b"]
    assert report.needs_fix == ["c"] and report.edited == ["a"]


def test_unreviewed_questions_pass_through():
    gold = [_q("a")]
    out, report = apply_reviews(gold, {})
    assert out == gold and not report.verified
