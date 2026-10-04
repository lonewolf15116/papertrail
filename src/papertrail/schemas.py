"""Typed contracts shared by the API, the pipeline and the evaluation harness."""

from enum import StrEnum

from pydantic import BaseModel, Field, model_validator


class Chunk(BaseModel):
    """A retrievable passage with enough metadata to cite it."""

    chunk_id: str
    paper_id: str
    section: str
    page: int = Field(ge=1)
    text: str


class Citation(BaseModel):
    paper_id: str
    section: str
    page: int = Field(ge=1)
    chunk_id: str
    quote: str = Field(description="Short span from the chunk that supports the claim")


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=50)


class AnswerStatus(StrEnum):
    ANSWERED = "answered"
    REFUSED = "refused"  # sources do not support an answer


class AskResponse(BaseModel):
    status: AnswerStatus
    answer: str | None = None
    citations: list[Citation] = Field(default_factory=list)
    refusal_reason: str | None = None
    latency_ms: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None

    @model_validator(mode="after")
    def _consistent(self) -> "AskResponse":
        if self.status is AnswerStatus.ANSWERED and (not self.answer or not self.citations):
            raise ValueError("an answered response needs an answer and at least one citation")
        if self.status is AnswerStatus.REFUSED and not self.refusal_reason:
            raise ValueError("a refused response needs a refusal_reason")
        return self


class QuestionKind(StrEnum):
    FACTUAL = "factual"
    UNANSWERABLE = "unanswerable"
    CROSS_PAPER = "cross_paper"  # similar terminology, different claims across papers


class GoldQuestion(BaseModel):
    """One hand-labelled evaluation item."""

    id: str
    question: str
    kind: QuestionKind
    gold_chunk_ids: list[str] = Field(default_factory=list)
    gold_paper_ids: list[str] = Field(default_factory=list)
    reference_answer: str | None = None
    distractor_paper_ids: list[str] = Field(
        default_factory=list,
        description="Papers likely to be retrieved by mistake (used for cross-paper questions)",
    )
    notes: str | None = None

    @model_validator(mode="after")
    def _labels_match_kind(self) -> "GoldQuestion":
        if self.kind is QuestionKind.UNANSWERABLE:
            if self.gold_chunk_ids:
                raise ValueError(f"{self.id}: unanswerable questions must have no gold chunks")
        elif not self.gold_chunk_ids:
            raise ValueError(f"{self.id}: answerable questions need at least one gold chunk")
        if self.kind is QuestionKind.CROSS_PAPER and not self.distractor_paper_ids:
            raise ValueError(f"{self.id}: cross-paper questions need distractor_paper_ids")
        return self
