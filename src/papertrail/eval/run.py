"""Evaluation entry point (`papertrail-eval`).

Milestone 1: validates the gold set and prints its composition.
Milestone 2+: runs each retriever over the gold set, writes results JSON, applies the gate.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from papertrail.eval.evidence import EvidenceIndex
from papertrail.eval.gate import check_regression
from papertrail.eval.gold import load_gold
from papertrail.schemas import LabelStatus

DEFAULT_MARGINS = {"recall@5": 0.02, "mrr": 0.02, "faithfulness": 0.03}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="papertrail-eval")
    parser.add_argument("--gold", type=Path, default=Path("data/gold/questions.jsonl"))
    parser.add_argument("--chunks", type=Path, help="chunks.jsonl; checks every quote is found")
    parser.add_argument(
        "--fix-pages",
        type=Path,
        metavar="PDF_DIR",
        help="rewrite each evidence page to where the quote really is in the PDF",
    )
    parser.add_argument(
        "--apply-reviews",
        type=Path,
        metavar="REVIEWS_JSON",
        help="apply verdicts exported from the review page ({id: review}) to the gold file",
    )
    parser.add_argument("--results", type=Path, help="metrics JSON from this run")
    parser.add_argument("--baseline", type=Path, help="metrics JSON to gate against")
    args = parser.parse_args(argv)

    gold = load_gold(args.gold)
    verified = [q for q in gold if q.status is LabelStatus.VERIFIED]
    kinds = Counter(q.kind.value for q in verified)
    print(
        f"gold set OK: {len(gold)} questions, {len(verified)} verified {dict(kinds)}, "
        f"{len(gold) - len(verified)} drafts (drafts are not scored)"
    )

    if args.apply_reviews:
        from papertrail.eval.review import apply_reviews

        gold, report = apply_reviews(gold, json.loads(args.apply_reviews.read_text()))
        args.gold.write_text(
            "".join(q.model_dump_json(exclude_none=True) + "\n" for q in gold),
            encoding="utf-8",
        )
        print(
            f"reviews applied: {len(report.verified)} verified, {len(report.rejected)} rejected, "
            f"{len(report.needs_fix)} need fixes, {len(report.edited)} edited"
        )

    if args.fix_pages:
        from papertrail.eval.evidence import fix_pages

        changes = fix_pages(gold, args.fix_pages)
        for line in changes:
            print(f"PAGE {line}")
        args.gold.write_text(
            "".join(q.model_dump_json(exclude_none=True) + "\n" for q in gold),
            encoding="utf-8",
        )
        print(f"{len(changes)} page changes written to {args.gold}")

    if args.chunks:
        from papertrail.ingest import load_chunks

        index = EvidenceIndex(load_chunks(args.chunks))
        problems = index.unresolved(q for q in gold if not q.id.startswith("tmpl-"))
        for line in problems:
            print(f"UNRESOLVED {line}")
        if problems:
            return 1
        print("every evidence quote was found in the corpus")

    if args.results and args.baseline:
        current = json.loads(args.results.read_text())
        baseline = json.loads(args.baseline.read_text())
        result = check_regression(current, baseline, DEFAULT_MARGINS)
        for line in result.failures:
            print(f"REGRESSION {line}")
        if not result.passed:
            return 1
        print("quality gate passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
