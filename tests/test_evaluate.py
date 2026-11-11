import copy

import pytest

from docextract.config import load_config
from docextract.evaluate import evaluate
from test_schema import DOC


def _cfg():
    cfg = load_config("configs/base.yaml", ["routing.select=fixed", "routing.confidence_threshold=0.9"])
    cfg["cost"].update(small_docs_per_second=2.0, baseline_usd_per_doc=0.02)
    return cfg


def test_evaluate_routes_and_scores():
    wrong = copy.deepcopy(DOC)
    wrong["total"] = 99.0            # breaks subtotal + tax = total -> must be routed to review
    gold = [{"id": str(i), "document": DOC} for i in range(4)]
    pred = [
        {"id": "0", "document": copy.deepcopy(DOC), "raw_valid_json": True, "confidence": 0.97, "seconds": 0.5},
        {"id": "1", "document": copy.deepcopy(DOC), "raw_valid_json": True, "confidence": 0.60, "seconds": 0.5},
        {"id": "2", "document": wrong, "raw_valid_json": True, "confidence": 0.99, "seconds": 0.5},
        {"id": "3", "document": None, "raw_valid_json": False, "confidence": 0.0, "seconds": 0.5},
    ]
    report = evaluate(gold, pred, _cfg(), baseline_rows=pred)
    m = report["metrics"]
    assert m["straight_through_rate"] == 0.25      # only doc 0
    assert m["auto_precision"] == 1.0              # the wrong total was caught by the check
    assert m["json_validity"] == 0.75
    assert m["document_exact_match"] == 0.5
    assert m["docs_per_minute"] == pytest.approx(120.0)
    assert m["baseline_field_f1"] == m["field_f1"]
    assert "business" in report


def test_id_mismatch_is_error():
    with pytest.raises(ValueError):
        evaluate([{"id": "1", "document": DOC}],
                 [{"id": "2", "document": DOC, "confidence": 1.0}], _cfg())


def test_evaluate_uses_dataset_scope():
    from docextract import evaluate as E

    gold = {"vendor_name": "Shop", "date": None, "currency": None, "line_items": [],
            "subtotal": None, "tax": None, "total": 9.0}
    pred = {"vendor_name": "Shop", "date": "2018-12-25", "currency": "MYR", "line_items": [],
            "subtotal": 9.0, "tax": 0.0, "total": 9.0}
    gold_rows = [{"id": "a", "dataset": "sroie", "document": gold}]
    pred_rows = [{"id": "a", "document": pred, "raw_valid_json": True, "confidence": 0.9, "seconds": 1.0}]
    cfg = {"paths": {"artifacts": "none"}, "eval": {"amount_tolerance": 0.01,
                                                  "key_fields": ["vendor_name", "date", "total"]},
           "routing": {"select": "fixed", "confidence_threshold": 0.5, "required_checks": ["schema_valid"]},
           "data": {"datasets": {"sroie": {"fields": ["vendor_name", "date", "total"],
                                            "unscored_if_null": ["vendor_name", "date", "total"]}}}}
    assert E.evaluate(gold_rows, pred_rows, cfg)["metrics"]["field_f1"] == 1.0
    del cfg["data"]                                    # no scope configured: strict, extras are false positives
    assert E.evaluate(gold_rows, pred_rows, cfg)["metrics"]["field_f1"] < 1.0
