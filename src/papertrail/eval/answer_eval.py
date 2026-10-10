"""Score the full answer pipeline on the gold set (Week 3).

Per question the pipeline is run end to end, then:

  refusal accuracy      did it refuse exactly the unanswerable questions? (hand labels, no LLM)
                        also reported as false-refusal and false-answer rates
  citation precision    share of cited chunks that hold a labelled evidence quote (hand labels)
  citation hit rate     answered questions where at least one gold chunk is cited
  wrong-paper citation  answered questions citing a paper that is not a gold paper
  faithfulness          LLM judge: share of the answer's claims supported by the cited passages

Faithfulness is the only judged metric. A seeded sample of judged answers is written out for a
person to check, so the number is never trusted on the judge's say-so alone.

Only verified labels are scored unless --include-drafts is passed; those runs are marked
provisional in every output.

Run: papertrail-eval-answers [--include-drafts] [--limit N]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any, Protocol

from papertrail.answer import AnswerClient, Generation
from papertrail.eval.evidence import EvidenceIndex
from papertrail.eval.gold import load_gold
from papertrail.eval.metrics import citation_precision, percentile, refusal_accuracy
from papertrail.schemas import (
    AnswerStatus,
    AskResponse,
    Chunk,
    GoldQuestion,
    LabelStatus,
    QuestionKind,
)

JUDGE_SYSTEM = """\
You check whether an answer is faithful to the passages it cites.

Split the answer into its separate factual claims. For each claim decide whether the cited \
passages state it or directly entail it. Use only the passages: ignore anything you know about \
the topic, and mark a claim unsupported if it needs outside knowledge, goes beyond what the \
passages say, or attributes to one paper something a passage says about another. Rephrasing is \
fine; added numbers, causes or comparisons that are not in the passages are not.
Respond only by calling the judge_faithfulness tool."""

JUDGE_TOOL: dict[str, Any] = {
    "name": "judge_faithfulness",
    "description": "Report, claim by claim, whether the answer is supported by the passages.",
    "input_schema": {
        "type": "object",
        "properties": {
            "claims": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "claim": {"type": "string"},
                        "supported": {"type": "boolean"},
                        "reason": {"type": "string", "description": "One short sentence."},
                    },
                    "required": ["claim", "supported"],
                },
            }
        },
        "required": ["claims"],
    },
}


class Asker(Protocol):
    def ask(self, question: str, top_k: int | None = None) -> AskResponse: ...


@dataclass
class Judgement:
    score: float | None  # supported / total claims; None when no claims were found
    claims: list[dict[str, Any]]
    input_tokens: int
    output_tokens: int


def judge_faithfulness(
    client: AnswerClient, answer: str, cited_chunks: Sequence[Chunk], titles: dict[str, str]
) -> Judgement:
    passages = "\n\n".join(
        f"[{i}] {titles.get(c.paper_id, c.paper_id)}, {c.section}, page {c.page}\n{c.text.strip()}"
        for i, c in enumerate(cited_chunks, start=1)
    )
    user = f"Cited passages:\n\n{passages}\n\nAnswer to check:\n{answer}"
    gen: Generation = client.generate(JUDGE_SYSTEM, user, JUDGE_TOOL, 1500)
    claims = [c for c in gen.data.get("claims", []) if isinstance(c, dict)]
    flags = [c.get("supported") is True for c in claims]
    return Judgement(
        score=(sum(flags) / len(flags)) if flags else None,
        claims=claims,
        input_tokens=gen.input_tokens,
        output_tokens=gen.output_tokens,
    )


@dataclass
class AnswerResult:
    id: str
    kind: str
    should_refuse: bool
    refused: bool
    refusal_reason: str | None = None
    answer: str | None = None
    cited_chunk_ids: list[str] = field(default_factory=list)
    citation_precision: float | None = None
    citation_hit: bool | None = None
    wrong_paper_citation: bool | None = None
    faithfulness: float | None = None
    judged_claims: list[dict[str, Any]] = field(default_factory=list)
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    judge_cost_usd: float = 0.0
    dropped_citations: int = 0
    error: str | None = None


def select_questions(
    gold: Sequence[GoldQuestion], include_drafts: bool, limit: int | None
) -> list[GoldQuestion]:
    picked = [
        q
        for q in gold
        if (include_drafts or q.status is LabelStatus.VERIFIED) and not q.id.startswith("tmpl-")
    ]
    return picked[:limit] if limit else picked


def _mean_or_none(values: Sequence[float]) -> float | None:
    return mean(values) if values else None


def evaluate_answers(
    asker: Asker,
    judge: AnswerClient | None,
    questions: Sequence[GoldQuestion],
    index: EvidenceIndex,
    chunk_by_id: dict[str, Chunk],
    titles: dict[str, str],
    judge_prices: tuple[float, float] = (0.0, 0.0),
) -> list[AnswerResult]:
    results: list[AnswerResult] = []
    for q in questions:
        should_refuse = q.kind is QuestionKind.UNANSWERABLE
        try:
            resp = asker.ask(q.question)
        except Exception as exc:  # one failed call must not sink the run; it is counted
            results.append(
                AnswerResult(
                    id=q.id,
                    kind=q.kind.value,
                    should_refuse=should_refuse,
                    refused=False,
                    error=f"{type(exc).__name__}: {exc}"[:300],
                )
            )
            continue
        refused = resp.status is AnswerStatus.REFUSED
        r = AnswerResult(
            id=q.id,
            kind=q.kind.value,
            should_refuse=should_refuse,
            refused=refused,
            refusal_reason=resp.refusal_reason,
            answer=resp.answer,
            cited_chunk_ids=[c.chunk_id for c in resp.citations],
            latency_ms=resp.latency_ms or 0.0,
            input_tokens=resp.input_tokens or 0,
            output_tokens=resp.output_tokens or 0,
            cost_usd=resp.estimated_cost_usd or 0.0,
            dropped_citations=resp.dropped_citations,
        )
        if not refused and not should_refuse:
            gold_ids = index.gold_chunk_ids(q)
            r.citation_precision = citation_precision(r.cited_chunk_ids, gold_ids)
            r.citation_hit = bool(gold_ids.intersection(r.cited_chunk_ids))
            gold_papers = set(q.gold_paper_ids)
            r.wrong_paper_citation = any(c.paper_id not in gold_papers for c in resp.citations)
        if not refused and judge is not None and resp.answer:
            cited = [chunk_by_id[cid] for cid in r.cited_chunk_ids if cid in chunk_by_id]
            try:
                j = judge_faithfulness(judge, resp.answer, cited, titles)
            except Exception as exc:
                r.error = f"judge failed: {type(exc).__name__}: {exc}"[:300]
            else:
                r.faithfulness = j.score
                r.judged_claims = j.claims
                r.judge_cost_usd = (
                    j.input_tokens * judge_prices[0] + j.output_tokens * judge_prices[1]
                ) / 1_000_000
        results.append(r)
    return results


def summarize(results: Sequence[AnswerResult], provisional: bool) -> dict[str, Any]:
    # A pipeline failure leaves no response to score; a judge failure still has one (it only
    # lacks a faithfulness value), so it stays in the other metrics.
    scored = [r for r in results if not r.error or r.error.startswith("judge failed")]
    answerable = [r for r in scored if not r.should_refuse]
    unanswerable = [r for r in scored if r.should_refuse]
    answered = [r for r in answerable if not r.refused]
    out: dict[str, Any] = {
        "provisional": provisional,
        "n_questions": len(results),
        "n_errors": sum(1 for r in results if r.error),
        "n_answerable": len(answerable),
        "n_unanswerable": len(unanswerable),
        "refusal_accuracy": (
            refusal_accuracy([r.refused for r in scored], [r.should_refuse for r in scored])
            if scored
            else None
        ),
        "false_refusal_rate": (
            sum(r.refused for r in answerable) / len(answerable) if answerable else None
        ),
        "false_answer_rate": (
            sum(not r.refused for r in unanswerable) / len(unanswerable) if unanswerable else None
        ),
        "citation_precision": _mean_or_none(
            [r.citation_precision for r in answered if r.citation_precision is not None]
        ),
        "citation_hit_rate": _mean_or_none(
            [float(r.citation_hit) for r in answered if r.citation_hit is not None]
        ),
        "wrong_paper_citation_rate": _mean_or_none(
            [float(r.wrong_paper_citation) for r in answered if r.wrong_paper_citation is not None]
        ),
        "faithfulness": _mean_or_none(
            [r.faithfulness for r in scored if r.faithfulness is not None]
        ),
        "citations_dropped_total": sum(r.dropped_citations for r in scored),
    }
    by_kind: dict[str, dict[str, Any]] = {}
    for kind in sorted({r.kind for r in scored}):
        sub = [r for r in scored if r.kind == kind]
        sub_answered = [r for r in sub if not r.refused and not r.should_refuse]
        by_kind[kind] = {
            "n": len(sub),
            "refused": sum(r.refused for r in sub),
            "citation_precision": _mean_or_none(
                [r.citation_precision for r in sub_answered if r.citation_precision is not None]
            ),
            "wrong_paper_citation_rate": _mean_or_none(
                [
                    float(r.wrong_paper_citation)
                    for r in sub_answered
                    if r.wrong_paper_citation is not None
                ]
            ),
            "faithfulness": _mean_or_none(
                [r.faithfulness for r in sub if r.faithfulness is not None]
            ),
        }
    out["by_kind"] = by_kind
    timed = [r for r in results if r.error is None]
    lat = [r.latency_ms for r in timed]
    out["latency_p50_ms"] = percentile(lat, 50) if lat else None
    out["latency_p95_ms"] = percentile(lat, 95) if lat else None
    out["mean_input_tokens"] = _mean_or_none([float(r.input_tokens) for r in timed])
    out["mean_output_tokens"] = _mean_or_none([float(r.output_tokens) for r in timed])
    out["mean_cost_usd"] = _mean_or_none([r.cost_usd for r in timed])
    out["total_judge_cost_usd"] = sum(r.judge_cost_usd for r in results)
    return out


def _fmt(v: float | None, pct: bool = False) -> str:
    if v is None:
        return "–"
    return f"{v:.1%}" if pct else f"{v:.3f}"


def markdown_table(s: dict[str, Any]) -> str:
    rows = [
        ("Refusal accuracy", _fmt(s["refusal_accuracy"])),
        ("False-answer rate (unanswerable questions answered)", _fmt(s["false_answer_rate"])),
        ("False-refusal rate (answerable questions refused)", _fmt(s["false_refusal_rate"])),
        ("Citation precision", _fmt(s["citation_precision"])),
        ("Citation hit rate", _fmt(s["citation_hit_rate"])),
        ("Wrong-paper citation rate", _fmt(s["wrong_paper_citation_rate"])),
        ("Faithfulness (LLM judge)", _fmt(s["faithfulness"])),
        ("Latency p50 / p95 (ms)", f"{_fmt(s['latency_p50_ms'])} / {_fmt(s['latency_p95_ms'])}"),
        (
            "Mean tokens in / out",
            f"{_fmt(s['mean_input_tokens'])} / {_fmt(s['mean_output_tokens'])}",
        ),
        ("Mean cost per question (USD)", _fmt(s["mean_cost_usd"])),
    ]
    return "| Metric | Value |\n|---|---|\n" + "".join(f"| {k} | {v} |\n" for k, v in rows)


def spotcheck_sample(
    results: Sequence[AnswerResult],
    questions: dict[str, GoldQuestion],
    chunk_by_id: dict[str, Chunk],
    n: int,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """A seeded sample of judged answers, with the passages and the judge's verdicts, for a
    person to mark. `human_supported` is left null on purpose."""
    judged = [r for r in results if r.judged_claims and r.answer]
    rng = random.Random(seed)
    picked = rng.sample(judged, min(n, len(judged)))
    return [
        {
            "id": r.id,
            "question": questions[r.id].question,
            "reference_answer": questions[r.id].reference_answer,
            "answer": r.answer,
            "cited_passages": [
                {"chunk_id": cid, "text": chunk_by_id[cid].text}
                for cid in r.cited_chunk_ids
                if cid in chunk_by_id
            ],
            "judge_faithfulness": r.faithfulness,
            "judge_claims": r.judged_claims,
            "human_supported": None,
        }
        for r in picked
    ]


def main(argv: list[str] | None = None) -> int:
    from papertrail.config import get_settings
    from papertrail.ingest import load_chunks
    from papertrail.pipeline import build_default_pipeline

    parser = argparse.ArgumentParser(prog="papertrail-eval-answers")
    parser.add_argument("--gold", type=Path, default=Path("data/gold/questions.jsonl"))
    parser.add_argument("--chunks", type=Path, default=Path("data/processed/chunks.jsonl"))
    parser.add_argument("--include-drafts", action="store_true", help="score draft labels too")
    parser.add_argument("--limit", type=int, default=None, help="only the first N questions")
    parser.add_argument("--no-judge", action="store_true", help="skip the faithfulness judge")
    parser.add_argument("--spotcheck", type=int, default=15, help="judged answers to export")
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args(argv)

    settings = get_settings()
    chunks = load_chunks(args.chunks)
    index = EvidenceIndex(chunks)
    questions = select_questions(load_gold(args.gold), args.include_drafts, args.limit)
    if not questions:
        print("no questions to score: verify some labels, or pass --include-drafts")
        return 1
    problems = index.unresolved(questions)
    if problems:
        print("\n".join(f"UNRESOLVED {p}" for p in problems))
        return 1

    pipeline = build_default_pipeline(settings)
    answer_spec = settings.answer_llm()
    judge_spec = settings.judge_llm()
    judge: AnswerClient | None = None
    if not args.no_judge:
        import os

        from papertrail.answer import make_client

        if not os.environ.get(judge_spec.key_env):
            print(f"{judge_spec.key_env} is not set (needed for the faithfulness judge)")
            return 1
        if (judge_spec.provider, judge_spec.model) == (answer_spec.provider, answer_spec.model):
            print("WARNING: the judge is the same model as the answerer; faithfulness is biased")
        judge = make_client(judge_spec.provider, judge_spec.model, temperature=None)

    results = evaluate_answers(
        pipeline,
        judge,
        questions,
        index,
        {c.chunk_id: c for c in chunks},
        pipeline.titles,
        (judge_spec.input_price_per_mtok, judge_spec.output_price_per_mtok),
    )
    tag = "provisional" if args.include_drafts else "verified"
    summary = summarize(results, args.include_drafts)
    summary["answer_model"] = answer_spec.model
    summary["answer_provider"] = answer_spec.provider
    summary["judge_model"] = None if args.no_judge else judge_spec.model
    summary["judge_provider"] = None if args.no_judge else judge_spec.provider
    summary["retriever"] = pipeline.retriever.name

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"answers-{tag}.json").write_text(
        json.dumps({"summary": summary, "questions": [asdict(r) for r in results]}, indent=2),
        encoding="utf-8",
    )
    (args.out / f"answers-summary-{tag}.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    label = "PROVISIONAL (includes draft labels)" if args.include_drafts else "verified labels"
    header = f"{len(questions)} questions, {label}, {summary['n_errors']} errors\n"
    table = markdown_table(summary)
    (args.out / f"answers-table-{tag}.md").write_text(header + "\n" + table, encoding="utf-8")
    sample = spotcheck_sample(
        results,
        {q.id: q for q in questions},
        {c.chunk_id: c for c in chunks},
        args.spotcheck,
    )
    (args.out / f"answers-spotcheck-{tag}.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in sample), encoding="utf-8"
    )
    (args.out / f"metrics-answers-{tag}.json").write_text(
        json.dumps(
            {
                "citation_precision": summary["citation_precision"],
                "refusal_accuracy": summary["refusal_accuracy"],
                "faithfulness": summary["faithfulness"],
                "_questions": len(questions),
                "_labels": tag,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(header)
    print(table)
    return 0


if __name__ == "__main__":
    sys.exit(main())
