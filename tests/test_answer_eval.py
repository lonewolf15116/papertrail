from typing import Any

import pytest

from papertrail.answer import Generation
from papertrail.eval.answer_eval import (
    evaluate_answers,
    judge_faithfulness,
    markdown_table,
    select_questions,
    spotcheck_sample,
    summarize,
)
from papertrail.eval.evidence import EvidenceIndex
from papertrail.schemas import (
    AnswerStatus,
    AskResponse,
    Chunk,
    Citation,
    Evidence,
    GoldQuestion,
    LabelStatus,
    QuestionKind,
)

DTR_TEXT = "DTR evicts the tensor that is stalest, largest, and cheapest to rematerialize."
CHK_TEXT = "Checkmate solves rematerialization as a mixed integer linear program with a solver."
CHUNKS = [
    Chunk(chunk_id="dtr:1", paper_id="dtr", section="3", page=3, text=DTR_TEXT),
    Chunk(chunk_id="chk:1", paper_id="chk", section="4", page=5, text=CHK_TEXT),
]
BY_ID = {c.chunk_id: c for c in CHUNKS}
INDEX = EvidenceIndex(CHUNKS)
TITLES = {"dtr": "DTR", "chk": "Checkmate"}


def q(qid: str, kind: QuestionKind, paper: str | None = None, text: str = "", **kw: Any):
    ev = [Evidence(paper_id=paper, page=1, quote=text)] if paper else []
    return GoldQuestion(id=qid, question=f"question {qid}?", kind=kind, evidence=ev, **kw)


def answered(chunk: Chunk, answer: str = "An answer.") -> AskResponse:
    cite = Citation(
        paper_id=chunk.paper_id,
        section=chunk.section,
        page=chunk.page,
        chunk_id=chunk.chunk_id,
        quote="x" * 20,
    )
    return AskResponse(
        status=AnswerStatus.ANSWERED,
        answer=answer,
        citations=[cite],
        latency_ms=100.0,
        input_tokens=1000,
        output_tokens=100,
        estimated_cost_usd=0.001,
    )


def refused() -> AskResponse:
    return AskResponse(
        status=AnswerStatus.REFUSED, refusal_reason="no", latency_ms=50.0, estimated_cost_usd=0.0005
    )


class FakeAsker:
    def __init__(self, by_question: dict[str, AskResponse | Exception]) -> None:
        self.by_question = by_question

    def ask(self, question: str, top_k: int | None = None) -> AskResponse:
        got = self.by_question[question]
        if isinstance(got, Exception):
            raise got
        return got


class FakeJudge:
    def __init__(self, claims: list[dict[str, Any]], fail: bool = False) -> None:
        self.claims, self.fail = claims, fail
        self.users: list[str] = []

    def generate(self, system: str, user: str, tool: dict[str, Any], max_tokens: int) -> Generation:
        self.users.append(user)
        if self.fail:
            raise RuntimeError("judge down")
        return Generation(data={"claims": self.claims}, input_tokens=2000, output_tokens=100)


GOLD = [
    q("a", QuestionKind.FACTUAL, "dtr", DTR_TEXT),
    q("b", QuestionKind.FACTUAL, "dtr", DTR_TEXT),
    q("c", QuestionKind.CROSS_PAPER, "dtr", DTR_TEXT, distractor_paper_ids=["chk"]),
    q("d", QuestionKind.UNANSWERABLE),
    q("e", QuestionKind.UNANSWERABLE),
    q("f", QuestionKind.FACTUAL, "dtr", DTR_TEXT),
]
RESPONSES: dict[str, AskResponse | Exception] = {
    "question a?": answered(CHUNKS[0]),  # correct, cites the gold chunk
    "question b?": refused(),  # false refusal
    "question c?": answered(CHUNKS[1]),  # cites the distractor paper
    "question d?": refused(),  # correct refusal
    "question e?": answered(CHUNKS[0]),  # false answer
    "question f?": RuntimeError("boom"),  # pipeline failure
}


def run(judge: FakeJudge | None = None):
    return evaluate_answers(
        FakeAsker(RESPONSES), judge, GOLD, INDEX, BY_ID, TITLES, judge_prices=(3.0, 15.0)
    )


def test_per_question_results():
    r = {x.id: x for x in run(FakeJudge([{"claim": "c", "supported": True}]))}
    assert r["a"].citation_precision == 1.0 and r["a"].citation_hit is True
    assert r["a"].wrong_paper_citation is False
    assert r["b"].refused and r["b"].citation_precision is None
    assert r["c"].citation_precision == 0.0 and r["c"].wrong_paper_citation is True
    assert r["d"].refused and r["d"].should_refuse
    assert not r["e"].refused and r["e"].should_refuse
    assert r["e"].citation_precision is None  # no citation scoring for unanswerable questions
    assert r["f"].error and "boom" in r["f"].error


def test_summary_metrics():
    s = summarize(run(FakeJudge([{"claim": "c", "supported": True}])), provisional=False)
    assert s["n_questions"] == 6 and s["n_errors"] == 1
    assert s["n_answerable"] == 3 and s["n_unanswerable"] == 2  # the errored one is not scored
    # scored: a ok, b wrong (false refusal), c ok (answered), d ok, e wrong (false answer)
    assert s["refusal_accuracy"] == pytest.approx(3 / 5)
    assert s["false_refusal_rate"] == pytest.approx(1 / 3)
    assert s["false_answer_rate"] == pytest.approx(1 / 2)
    assert s["citation_precision"] == pytest.approx(0.5)  # a: 1.0, c: 0.0
    assert s["citation_hit_rate"] == pytest.approx(0.5)
    assert s["wrong_paper_citation_rate"] == pytest.approx(0.5)
    assert s["by_kind"]["cross_paper"]["wrong_paper_citation_rate"] == 1.0
    assert s["latency_p50_ms"] is not None


def test_faithfulness_is_share_of_supported_claims():
    judge = FakeJudge(
        [
            {"claim": "one", "supported": True},
            {"claim": "two", "supported": False},
            {"claim": "three", "supported": True},
            {"claim": "four", "supported": True},
        ]
    )
    results = run(judge)
    assert {r.id: r.faithfulness for r in results if r.faithfulness is not None} == {
        "a": 0.75,
        "c": 0.75,
        "e": 0.75,
    }
    assert summarize(results, False)["faithfulness"] == pytest.approx(0.75)
    assert all("Cited passages" in u and "Answer to check" in u for u in judge.users)
    # judge cost: 2000 in @ $3 + 100 out @ $15 per million, for each of the 3 judged answers
    assert summarize(results, False)["total_judge_cost_usd"] == pytest.approx(3 * 0.0075)


def test_judge_with_no_claims_gives_no_score():
    j = judge_faithfulness(FakeJudge([]), "answer", CHUNKS[:1], TITLES)
    assert j.score is None and j.claims == []


def test_judge_failure_is_recorded_but_keeps_the_other_metrics():
    results = run(FakeJudge([], fail=True))
    a = next(r for r in results if r.id == "a")
    assert a.faithfulness is None and a.error and a.error.startswith("judge failed")
    s = summarize(results, False)
    assert s["citation_precision"] == pytest.approx(0.5)  # still scored
    assert s["faithfulness"] is None
    assert s["n_errors"] == 4  # three judge failures + one pipeline failure


def test_no_judge_means_no_faithfulness():
    s = summarize(run(None), False)
    assert s["faithfulness"] is None and s["n_errors"] == 1


def test_select_questions_filters_drafts_templates_and_limit():
    drafts = [
        q("v1", QuestionKind.UNANSWERABLE, status=LabelStatus.VERIFIED),
        q("d1", QuestionKind.UNANSWERABLE, status=LabelStatus.DRAFT),
        q("tmpl-1", QuestionKind.UNANSWERABLE, status=LabelStatus.VERIFIED),
    ]
    assert [x.id for x in select_questions(drafts, False, None)] == ["v1"]
    assert [x.id for x in select_questions(drafts, True, None)] == ["v1", "d1"]
    assert [x.id for x in select_questions(drafts, True, 1)] == ["v1"]


def test_markdown_table_handles_missing_values():
    s = summarize([], True)
    table = markdown_table(s)
    assert "Refusal accuracy | –" in table and "Faithfulness (LLM judge) | –" in table


def test_spotcheck_is_seeded_and_leaves_human_verdict_blank():
    judge = FakeJudge([{"claim": "c", "supported": True}])
    results = run(judge)
    by_id = {g.id: g for g in GOLD}
    one = spotcheck_sample(results, by_id, BY_ID, n=2, seed=7)
    two = spotcheck_sample(results, by_id, BY_ID, n=2, seed=7)
    assert one == two and len(one) == 2
    assert all(row["human_supported"] is None and row["cited_passages"] for row in one)
    assert len(spotcheck_sample(results, by_id, BY_ID, n=50)) == 3  # only judged answers
