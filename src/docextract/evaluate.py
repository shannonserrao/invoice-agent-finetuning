"""Evaluation from files on disk (specs/03-evaluation.md). Never fits anything on test data."""
from __future__ import annotations

import json
import os

from . import metrics as M
from . import schema as S
from .baseline import baseline_predictions_path, baseline_summary_path
from .confidence import confidence_report
from .feedback import read_jsonl
from .routing import cascade


def _threshold(cfg: dict) -> float:
    path = os.path.join(cfg["paths"]["artifacts"], "threshold.json")
    if cfg["routing"].get("select") == "validation" and os.path.exists(path):
        with open(path, encoding="utf-8") as handle:
            return float(json.load(handle)["threshold"])
    return float(cfg["routing"]["confidence_threshold"])


def _scopes(rows, cfg):
    """Per-document (labelled fields, unscored_if_null) from data.datasets.<dataset>; all fields when unknown."""
    datasets = cfg.get("data", {}).get("datasets", {})
    out = []
    for row in rows:
        ds = datasets.get(row.get("dataset"), {})
        out.append((ds.get("fields") or None, ds.get("unscored_if_null") or []))
    return out


def _score(gold_rows, pred_rows, cfg, threshold):
    gold_by_id = {r["id"]: r["document"] for r in gold_rows}
    scope_by_id = dict(zip((r["id"] for r in gold_rows), _scopes(gold_rows, cfg)))
    pred_by_id = {r["id"]: r for r in pred_rows}
    if set(gold_by_id) != set(pred_by_id):
        missing = len(set(gold_by_id) - set(pred_by_id))
        extra = len(set(pred_by_id) - set(gold_by_id))
        raise ValueError(f"prediction ids do not match gold ids ({missing} missing, {extra} extra)")
    ids = sorted(gold_by_id)
    golds = [gold_by_id[i] for i in ids]
    scopes = [scope_by_id[i] for i in ids]
    preds = [pred_by_id[i].get("document") for i in ids]
    tol = cfg["eval"]["amount_tolerance"]
    key_fields = cfg["eval"]["key_fields"]
    routing = cfg["routing"]

    checks = [S.run_checks(p, tol) for p in preds]
    routes = [
        S.route(c, float(pred_by_id[i].get("confidence", 0.0)), threshold,
                routing["required_checks"], routing.get("allow_missing_checks"))
        for i, c in zip(ids, checks)
    ]
    key_ok = [M.key_fields_correct(g, p, key_fields) for g, p in zip(golds, preds)]
    f1 = M.field_f1(golds, preds, scopes=scopes)
    n = len(ids)
    metrics = {
        "field_f1": f1["f1"], "field_precision": f1["precision"], "field_recall": f1["recall"],
        "document_exact_match": sum(M.exact_match(g, p, *s) for g, p, s in zip(golds, preds, scopes)) / n,
        "key_fields_accuracy": sum(key_ok) / n,
        "json_validity": sum(bool(pred_by_id[i].get("raw_valid_json")) and c["schema_valid"]
                             for i, c in zip(ids, checks)) / n,
    }
    for name in ("items_sum_ok", "total_ok"):
        known = [c[name] for c in checks if c[name] is not None]
        if known:
            metrics[f"{name}_rate"] = sum(known) / len(known)
    per_field = M.per_field_accuracy(golds, preds)
    for field, row in per_field.items():
        metrics[f"accuracy_{field}"] = row["accuracy"]
    rq = M.routing_quality(routes, key_ok)
    metrics["straight_through_rate"] = rq["straight_through_rate"]
    if rq["auto_precision"] is not None:
        metrics["auto_precision"] = rq["auto_precision"]
    seconds = [r["seconds"] for r in pred_rows if r.get("seconds") is not None]
    if seconds:
        metrics["docs_per_minute"] = 60.0 / (sum(seconds) / len(seconds))
    conf = confidence_report([float(pred_by_id[i].get("confidence", 0.0)) for i in ids], key_ok)
    if conf["computable"]:
        for name in ("aurc", "aurc_random", "auroc"):
            metrics[f"confidence_{name}"] = conf[name]
    return metrics, per_field, rq, n, conf


def evaluate(gold_rows: list[dict], pred_rows: list[dict], cfg: dict,
             baseline_rows: list[dict] | None = None) -> dict:
    threshold = _threshold(cfg)
    metrics, per_field, rq, n, conf = _score(gold_rows, pred_rows, cfg, threshold)
    report = {"metrics": metrics, "per_field": per_field, "routing": rq, "confidence": conf,
              "meta": {"n": n, "threshold": threshold}}
    if baseline_rows:
        base_metrics, *_ = _score(gold_rows, baseline_rows, cfg, threshold)
        for key in ("field_f1", "key_fields_accuracy", "json_validity"):
            metrics[f"baseline_{key}"] = base_metrics[key]
        if cfg["routing"].get("escalate_to_baseline"):
            report["cascade"] = cascade(gold_rows, pred_rows, baseline_rows, cfg, threshold)

    cost, biz = cfg.get("cost", {}), cfg.get("business", {})
    if cost.get("small_docs_per_second") and rq["auto_precision"] is not None:
        small_usd = M.cost_per_1k(cost["gpu_usd_per_hour"], cost["small_docs_per_second"]) / 1000
        metrics["small_usd_per_1k"] = small_usd * 1000
        if cost.get("baseline_usd_per_doc"):
            metrics["baseline_usd_per_1k"] = cost["baseline_usd_per_doc"] * 1000
        report["business"] = M.business_projection(
            biz["docs_per_month"], biz["manual_minutes_per_doc"], biz["review_minutes_per_doc"],
            biz["clerk_usd_per_hour"], biz["undetected_error_usd"], small_usd,
            rq["straight_through_rate"], rq["auto_precision"], cost.get("baseline_usd_per_doc"),
        )
    return report


def score_baseline(gold_rows: list[dict], pred_rows: list[dict], cfg: dict) -> dict:
    """Accuracy metrics of the baseline's predictions on the gold rows. Routing and confidence do not apply."""
    metrics, *_ = _score(gold_rows, pred_rows, cfg, threshold=1.01)
    keep = ("field_f1", "field_precision", "field_recall", "document_exact_match", "key_fields_accuracy",
            "json_validity", "items_sum_ok_rate", "total_ok_rate")
    return {k: v for k, v in metrics.items() if k in keep or k.startswith("accuracy_")}


def evaluate_files(cfg: dict) -> dict:
    paths = cfg["paths"]
    split = cfg.get("predict", {}).get("split", "test")
    gold = read_jsonl(os.path.join(paths["data"], "processed", f"{split}.jsonl"))
    pred = read_jsonl(os.path.join(paths["artifacts"], "predictions.jsonl"))
    if not gold or not pred:
        raise FileNotFoundError("run `make data` and `make predict` before `make eval`")
    baseline = read_jsonl(baseline_predictions_path(cfg, split)) or None
    summary_path = baseline_summary_path(cfg, split)
    if baseline and not cfg.get("cost", {}).get("baseline_usd_per_doc") and os.path.exists(summary_path):
        with open(summary_path, encoding="utf-8") as handle:       # cost per document comes from the baseline run
            per_doc = json.load(handle)["baseline_usd_per_doc"]
        cfg = {**cfg, "cost": {**cfg.get("cost", {}), "baseline_usd_per_doc": per_doc}}
    return evaluate(gold, pred, cfg, baseline)
