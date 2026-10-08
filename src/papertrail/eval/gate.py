"""CI regression gate: fail when a metric drops more than its margin below the baseline."""

from dataclasses import dataclass


@dataclass(frozen=True)
class GateResult:
    passed: bool
    failures: list[str]


def check_regression(
    current: dict[str, float],
    baseline: dict[str, float],
    margins: dict[str, float],
) -> GateResult:
    failures: list[str] = []
    for metric, margin in margins.items():
        if metric not in baseline or metric.startswith("_"):
            continue  # new metric, nothing to regress against yet
        if metric not in current:
            failures.append(f"{metric}: missing from current run")
            continue
        floor = baseline[metric] - margin
        if current[metric] < floor:
            failures.append(
                f"{metric}: {current[metric]:.3f} < {floor:.3f} "
                f"(baseline {baseline[metric]:.3f} - margin {margin:.3f})"
            )
    return GateResult(passed=not failures, failures=failures)
