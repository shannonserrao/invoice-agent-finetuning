"""Extraction, routing and cost metrics (specs/03-evaluation.md)."""
from __future__ import annotations

from collections import Counter
from difflib import SequenceMatcher

from .schema import HEADER_FIELDS, ITEM_FIELDS, normalise


def _match_items(gold_items: list[dict], pred_items: list[dict]) -> list[tuple[int, int]]:
    """Greedy one-to-one matching of line items by description similarity."""
    scored = []
    for gi, g in enumerate(gold_items):
        for pi, p in enumerate(pred_items):
            sim = SequenceMatcher(None, g["description"] or "", p["description"] or "").ratio()
            if g["amount"] is not None and g["amount"] == p["amount"]:
                sim += 0.5
            scored.append((sim, gi, pi))
    pairs, used_g, used_p = [], set(), set()
    for sim, gi, pi in sorted(scored, reverse=True):
        if gi in used_g or pi in used_p or sim < 0.5:
            continue
        pairs.append((gi, pi))
        used_g.add(gi)
        used_p.add(pi)
    return pairs


def field_counts(gold: dict | None, pred: dict | None, fields: list[str] | None = None,
                 unscored_if_null: list[str] | tuple = ()) -> Counter:
    """True positives, false positives and false negatives for one document.

    `fields` limits scoring to the fields the dataset labels. A field in `unscored_if_null` whose gold value is
    null is skipped for this document: the value is unknown (e.g. an ambiguous date), not known to be absent,
    so a prediction there is neither right nor wrong.
    """
    g, p = normalise(gold), normalise(pred)
    use = set(fields) if fields else set(HEADER_FIELDS) | {"line_items"}
    use -= {f for f in unscored_if_null if f in HEADER_FIELDS and g[f] is None}
    c = Counter()

    def compare(gv, pv):
        if gv is None and pv is None:
            return
        if gv is not None and gv == pv:
            c["tp"] += 1
            return
        if pv is not None:
            c["fp"] += 1
        if gv is not None:
            c["fn"] += 1

    for field in HEADER_FIELDS:
        if field in use:
            compare(g[field], p[field])
    if "line_items" in use:
        pairs = _match_items(g["line_items"], p["line_items"])
        for gi, pi in pairs:
            for k in ITEM_FIELDS:
                compare(g["line_items"][gi][k], p["line_items"][pi][k])
        matched_g = {gi for gi, _ in pairs}
        matched_p = {pi for _, pi in pairs}
        for gi, item in enumerate(g["line_items"]):
            if gi not in matched_g:
                c["fn"] += sum(v is not None for v in item.values())
        for pi, item in enumerate(p["line_items"]):
            if pi not in matched_p:
                c["fp"] += sum(v is not None for v in item.values())
    return c


def field_f1(golds: list[dict], preds: list[dict | None], fields: list[str] | None = None,
             scopes: list[tuple[list[str] | None, list[str]]] | None = None) -> dict:
    """Micro precision, recall and F1. `scopes` gives one (fields, unscored_if_null) pair per document."""
    if len(golds) != len(preds) or not golds:
        raise ValueError("golds and preds must be non-empty and the same length")
    if scopes is not None and len(scopes) != len(golds):
        raise ValueError("scopes must have one entry per document")
    total = Counter()
    for i, (g, p) in enumerate(zip(golds, preds)):
        doc_fields, unscored = scopes[i] if scopes is not None else (fields, [])
        total += field_counts(g, p, doc_fields, unscored)
    tp, fp, fn = total["tp"], total["fp"], total["fn"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def per_field_accuracy(golds: list[dict], preds: list[dict | None]) -> dict:
    """Exact-match accuracy per header field, over documents where gold has a value."""
    hits, seen = Counter(), Counter()
    for g, p in zip(golds, preds):
        ng, np_ = normalise(g), normalise(p)
        for field in HEADER_FIELDS:
            if ng[field] is not None:
                seen[field] += 1
                hits[field] += ng[field] == np_[field]
    return {f: {"accuracy": hits[f] / seen[f], "n": seen[f]} for f in HEADER_FIELDS if seen[f]}


def exact_match(gold: dict, pred: dict | None, fields: list[str] | None = None,
                unscored_if_null: list[str] | tuple = ()) -> bool:
    c = field_counts(gold, pred, fields, unscored_if_null)
    return c["fp"] == 0 and c["fn"] == 0


def key_fields_correct(gold: dict, pred: dict | None, key_fields: list[str]) -> bool:
    ng, np_ = normalise(gold), normalise(pred)
    return all(ng[f] == np_[f] for f in key_fields if ng[f] is not None)


def routing_quality(routes: list[str], key_correct: list[bool]) -> dict:
    """Straight-through rate and the precision of documents routed 'auto'."""
    if len(routes) != len(key_correct) or not routes:
        raise ValueError("routes and key_correct must be non-empty and the same length")
    auto = [ok for r, ok in zip(routes, key_correct) if r == "auto"]
    return {
        "straight_through_rate": len(auto) / len(routes),
        "auto_precision": sum(auto) / len(auto) if auto else None,
        "auto_errors": len(auto) - sum(auto),
    }


def cost_per_1k(gpu_usd_per_hour: float, items_per_second: float) -> float:
    if items_per_second <= 0:
        raise ValueError("items_per_second must be positive")
    return gpu_usd_per_hour / 3600.0 / items_per_second * 1000.0


def business_projection(
    docs_per_month: int, manual_minutes_per_doc: float, review_minutes_per_doc: float,
    clerk_usd_per_hour: float, undetected_error_usd: float,
    small_usd_per_doc: float, straight_through_rate: float, auto_precision: float,
    baseline_usd_per_doc: float | None = None,
) -> dict:
    """Projected monthly cost. A projection from test results, not a measurement."""
    minute = clerk_usd_per_hour / 60.0
    manual = docs_per_month * manual_minutes_per_doc * minute
    reviewed = docs_per_month * (1 - straight_through_rate)
    errors = docs_per_month * straight_through_rate * (1 - auto_precision)

    def with_model(usd_per_doc):
        inference = docs_per_month * usd_per_doc
        review = reviewed * review_minutes_per_doc * minute
        error = errors * undetected_error_usd
        return {"inference_usd": inference, "review_usd": review, "error_usd": error,
                "total_usd": inference + review + error}

    out = {"manual_total_usd": manual, "small_model": with_model(small_usd_per_doc)}
    out["net_monthly_saving_vs_manual_usd"] = manual - out["small_model"]["total_usd"]
    if baseline_usd_per_doc is not None:
        out["baseline_model_same_routing"] = with_model(baseline_usd_per_doc)
        out["net_monthly_saving_vs_baseline_usd"] = (
            out["baseline_model_same_routing"]["total_usd"] - out["small_model"]["total_usd"])
    return out
