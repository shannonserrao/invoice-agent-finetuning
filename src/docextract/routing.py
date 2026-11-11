"""Threshold fitting and the escalation cascade (specs/06-serving-and-routing.md §5, task T-10).

Fitting reads validation files only. `fit_threshold` is a pure function of the rows it is given; the file wrapper
below names the validation paths itself, so no code path here can read the test split.
"""
from __future__ import annotations

import json
import os

from . import metrics as M
from . import schema as S
from .feedback import read_jsonl

NO_THRESHOLD = 1.01        # confidence never exceeds 1, so nothing is routed to auto


def _passes_checks(doc, cfg) -> bool:
    routing = cfg["routing"]
    checks = S.run_checks(doc, cfg["eval"]["amount_tolerance"])
    return S.route(checks, 1.0, 0.0, routing["required_checks"], routing.get("allow_missing_checks")) == "auto"


def _match(gold_rows: list[dict], pred_rows: list[dict]) -> list[tuple[dict, dict]]:
    gold, pred = {r["id"]: r for r in gold_rows}, {r["id"]: r for r in pred_rows}
    if set(gold) != set(pred):
        raise ValueError("prediction ids do not match gold ids")
    return [(gold[i], pred[i]) for i in sorted(gold)]


def fit_threshold(gold_rows: list[dict], pred_rows: list[dict], cfg: dict) -> dict:
    """Lowest confidence threshold at which key-field precision of auto-routed documents meets the target.

    Candidates are the documents whose checks pass; a lower threshold means more straight-through documents.
    """
    target = cfg["routing"]["target_precision"]
    key_fields = cfg["eval"]["key_fields"]
    pairs = _match(gold_rows, pred_rows)
    cands = []
    for gold, pred in pairs:
        doc = pred.get("document")
        if _passes_checks(doc, cfg):
            cands.append((float(pred.get("confidence", 0.0)), M.key_fields_correct(gold["document"], doc, key_fields)))
    chosen, precision, n_auto = None, None, 0
    for t in sorted({c for c, _ in cands}):
        sel = [ok for c, ok in cands if c >= t]
        if sum(sel) / len(sel) >= target:
            chosen, precision, n_auto = t, sum(sel) / len(sel), len(sel)
            break
    return {
        "threshold": chosen if chosen is not None else NO_THRESHOLD, "feasible": chosen is not None,
        "target_precision": target, "split": "validation", "n_documents": len(pairs),
        "n_passing_checks": len(cands), "n_auto_at_threshold": n_auto,
        "straight_through_at_threshold": n_auto / len(pairs) if pairs else 0.0,
        "precision_at_threshold": precision,
    }


def fit_threshold_files(cfg: dict) -> dict:
    """Read validation gold and validation predictions, write artifacts/threshold.json."""
    paths = cfg["paths"]
    gold = read_jsonl(os.path.join(paths["data"], "processed", "validation.jsonl"))
    pred = read_jsonl(os.path.join(paths["artifacts"], "predictions_validation.jsonl"))
    if not gold or not pred:
        raise FileNotFoundError("need data/processed/validation.jsonl and artifacts/predictions_validation.jsonl "
                                "(run predict with --set predict.split=validation)")
    result = fit_threshold(gold, pred, cfg)
    out = os.path.join(paths["artifacts"], "threshold.json")
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))
    return result


def cascade(gold_rows: list[dict], pred_rows: list[dict], baseline_rows: list[dict], cfg: dict,
            threshold: float) -> dict:
    """Review documents go to the cached baseline output; accepted only if it passes the same checks.

    Documents neither the small model nor the baseline can settle go to the human queue.
    """
    base = {r["id"]: r for r in baseline_rows}
    key_fields = cfg["eval"]["key_fields"]
    routing = cfg["routing"]
    counts = {"small": 0, "baseline": 0, "human": 0}
    accepted_correct = 0
    pairs = _match(gold_rows, pred_rows)
    for gold, pred in pairs:
        doc = pred.get("document")
        checks = S.run_checks(doc, cfg["eval"]["amount_tolerance"])
        route = S.route(checks, float(pred.get("confidence", 0.0)), threshold, routing["required_checks"],
                        routing.get("allow_missing_checks"))
        if route == "auto":
            counts["small"] += 1
            accepted_correct += M.key_fields_correct(gold["document"], doc, key_fields)
            continue
        bdoc = base.get(gold["id"], {}).get("document")
        if bdoc is not None and _passes_checks(bdoc, cfg):
            counts["baseline"] += 1
            accepted_correct += M.key_fields_correct(gold["document"], bdoc, key_fields)
        else:
            counts["human"] += 1
    n = len(pairs)
    accepted = counts["small"] + counts["baseline"]
    escalated = n - counts["small"]
    out = {
        "n": n, "auto_small": counts["small"], "accepted_from_baseline": counts["baseline"],
        "to_human": counts["human"], "straight_through_rate": accepted / n,
        "accepted_precision": accepted_correct / accepted if accepted else None, "baseline_calls": escalated,
    }
    per_doc = cfg.get("cost", {}).get("baseline_usd_per_doc")
    if per_doc:
        out["baseline_usd"] = escalated * per_doc
        out["baseline_usd_if_all_docs"] = n * per_doc
    return out
