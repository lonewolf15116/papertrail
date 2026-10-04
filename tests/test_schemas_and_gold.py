from pathlib import Path

import pytest
from pydantic import ValidationError

from papertrail.eval.gold import load_gold
from papertrail.schemas import AnswerStatus, AskResponse, GoldQuestion

REPO = Path(__file__).resolve().parents[1]


def test_answered_response_needs_citations():
    with pytest.raises(ValidationError):
        AskResponse(status=AnswerStatus.ANSWERED, answer="yes")


def test_refusal_needs_reason():
    with pytest.raises(ValidationError):
        AskResponse(status=AnswerStatus.REFUSED)


EV = {"paper_id": "dtr2021", "page": 3, "quote": "a sufficiently long verbatim quote"}


def test_unanswerable_cannot_have_evidence():
    with pytest.raises(ValidationError):
        GoldQuestion(id="q", question="??", kind="unanswerable", evidence=[EV])


def test_answerable_needs_evidence():
    with pytest.raises(ValidationError):
        GoldQuestion(id="q", question="??", kind="factual")


def test_cross_paper_needs_distractors():
    with pytest.raises(ValidationError):
        GoldQuestion(id="q", question="??", kind="cross_paper", evidence=[EV])


def test_paper_cannot_be_gold_and_distractor():
    with pytest.raises(ValidationError):
        GoldQuestion(
            id="q",
            question="??",
            kind="cross_paper",
            evidence=[EV],
            distractor_paper_ids=["dtr2021"],
        )


def test_status_defaults_to_verified():
    q = GoldQuestion(id="q", question="??", kind="factual", evidence=[EV])
    assert q.status == "verified" and q.gold_paper_ids == ["dtr2021"]


def test_repo_gold_set_is_valid():
    gold = load_gold(REPO / "data/gold/questions.jsonl")
    assert gold, "gold set is empty"


def test_duplicate_ids_rejected(tmp_path):
    line = '{"id": "a", "question": "why?", "kind": "unanswerable"}\n'
    path = tmp_path / "q.jsonl"
    path.write_text(line * 2)
    with pytest.raises(ValueError, match="duplicate"):
        load_gold(path)
