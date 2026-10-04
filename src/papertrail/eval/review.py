"""Apply verdicts from the gold-set review page back to questions.jsonl.

A review record carries {status, question, answer, comment}:
- verified  -> status becomes verified; any edited question/answer is kept
- rejected  -> the question is removed from the gold set
- needs_fix -> stays a draft; the reviewer's comment is added to notes
- draft     -> unchanged
"""

from dataclasses import dataclass, field
from typing import Any

from papertrail.schemas import GoldQuestion, LabelStatus


@dataclass
class ReviewReport:
    verified: list[str] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)
    needs_fix: list[str] = field(default_factory=list)
    edited: list[str] = field(default_factory=list)


def apply_reviews(
    gold: list[GoldQuestion], reviews: dict[str, dict[str, Any]]
) -> tuple[list[GoldQuestion], ReviewReport]:
    report = ReviewReport()
    out: list[GoldQuestion] = []
    for q in gold:
        rv = reviews.get(q.id) or {}
        status = rv.get("status", "draft")
        if status == "rejected":
            report.rejected.append(q.id)
            continue
        updates: dict[str, Any] = {}
        new_q = (rv.get("question") or "").strip()
        new_a = (rv.get("answer") or "").strip()
        if new_q and new_q != q.question:
            updates["question"] = new_q
        if new_a and new_a != (q.reference_answer or ""):
            updates["reference_answer"] = new_a
        if updates:
            report.edited.append(q.id)
        if status == "verified":
            updates["status"] = LabelStatus.VERIFIED
            report.verified.append(q.id)
        elif status == "needs_fix":
            comment = (rv.get("comment") or "").strip()
            note = f"Reviewer: {comment}" if comment else "Reviewer flagged this for a fix."
            updates["notes"] = f"{q.notes} | {note}" if q.notes else note
            report.needs_fix.append(q.id)
        out.append(q.model_copy(update=updates) if updates else q)
    return out, report
