"""Does confidence rank errors better than random? (specs/06 §3, task T-11)

Selective-risk view: sort documents by confidence, accept the most confident first, and track the error rate
among accepted documents. The area under that risk-coverage curve (AURC, lower is better) is compared with a
random ordering, whose expected AURC equals the overall error rate.
"""
from __future__ import annotations

import random


def risk_coverage(confidences: list[float], correct: list[bool]) -> list[tuple[float, float]]:
    """(coverage, risk) after accepting the k most confident documents, k = 1..n. Ties keep input order."""
    if len(confidences) != len(correct) or not confidences:
        raise ValueError("confidences and correct must be non-empty and the same length")
    order = sorted(range(len(confidences)), key=lambda i: -confidences[i])
    points, errors = [], 0
    for k, i in enumerate(order, start=1):
        errors += not correct[i]
        points.append((k / len(order), errors / k))
    return points


def aurc(confidences: list[float], correct: list[bool]) -> float:
    pts = risk_coverage(confidences, correct)
    return sum(risk for _, risk in pts) / len(pts)


def auroc(confidences: list[float], correct: list[bool]) -> float | None:
    """Probability that a random correct document has higher confidence than a random wrong one (ties count half)."""
    pos = [c for c, ok in zip(confidences, correct) if ok]
    neg = [c for c, ok in zip(confidences, correct) if not ok]
    if not pos or not neg:
        return None
    wins = sum((p > q) + 0.5 * (p == q) for p in pos for q in neg)
    return wins / (len(pos) * len(neg))


def confidence_report(confidences: list[float], correct: list[bool], n_permutations: int = 2000,
                      seed: int = 13) -> dict:
    """AURC against random and the oracle, AUROC, and a permutation p-value for 'better than random'.

    p-value: share of random reorderings whose AURC is at least as good (as low) as the observed one.
    Returns `computable: False` when every document is correct or every one is wrong.
    """
    n = len(confidences)
    errors = sum(not c for c in correct)
    report = {"n": n, "error_rate": errors / n if n else None, "computable": 0 < errors < n}
    if not report["computable"]:
        return report
    observed = aurc(confidences, correct)
    oracle = aurc([1.0 if ok else 0.0 for ok in correct], correct)       # all correct first: best possible ordering
    rng = random.Random(seed)
    shuffled = list(confidences)
    at_least_as_good = 0
    for _ in range(n_permutations):
        rng.shuffle(shuffled)
        at_least_as_good += aurc(shuffled, correct) <= observed
    p_value = (at_least_as_good + 1) / (n_permutations + 1)
    report.update({
        "aurc": observed, "aurc_random": errors / n, "aurc_oracle": oracle,
        "auroc": auroc(confidences, correct), "p_value": p_value,
        "ranks_errors_better_than_random": observed < errors / n and p_value < 0.05,
    })
    return report
