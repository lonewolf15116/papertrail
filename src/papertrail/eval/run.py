"""Evaluation entry point (`papertrail-eval`).

Milestone 1: validates the gold set and prints its composition.
Milestone 2+: runs each retriever over the gold set, writes results JSON, applies the gate.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from papertrail.eval.gate import check_regression
from papertrail.eval.gold import load_gold

DEFAULT_MARGINS = {"recall@5": 0.02, "mrr": 0.02, "faithfulness": 0.03}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="papertrail-eval")
    parser.add_argument("--gold", type=Path, default=Path("data/gold/questions.jsonl"))
    parser.add_argument("--results", type=Path, help="metrics JSON from this run")
    parser.add_argument("--baseline", type=Path, help="metrics JSON to gate against")
    args = parser.parse_args(argv)

    gold = load_gold(args.gold)
    kinds = Counter(q.kind.value for q in gold)
    print(f"gold set OK: {len(gold)} questions {dict(kinds)}")

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
