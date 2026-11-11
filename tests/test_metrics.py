import copy

import pytest

from docextract import metrics as M
from test_schema import DOC


def test_perfect_prediction():
    assert M.field_f1([DOC], [copy.deepcopy(DOC)])["f1"] == 1.0
    assert M.exact_match(DOC, copy.deepcopy(DOC))


def test_formatting_differences_do_not_count():
    pred = copy.deepcopy(DOC)
    pred["vendor_name"] = "CORNER  CAFE"
    pred["total"] = "12.96"
    assert M.exact_match(DOC, pred)


def test_one_wrong_header_field():
    pred = copy.deepcopy(DOC)
    pred["total"] = 13.96
    c = M.field_counts(DOC, pred)
    # 6 header + 8 item values = 14 gold values; one is wrong -> 13 tp, 1 fp, 1 fn
    assert (c["tp"], c["fp"], c["fn"]) == (13, 1, 1)
    assert M.field_f1([DOC], [pred])["f1"] == pytest.approx(13 / 14)
    assert not M.key_fields_correct(DOC, pred, ["total"])
    assert M.key_fields_correct(DOC, pred, ["vendor_name", "date"])


def test_missing_and_extra_line_items():
    pred = copy.deepcopy(DOC)
    pred["line_items"] = pred["line_items"][:1] + [
        {"description": "Muffin", "quantity": 1, "unit_price": 2.0, "amount": 2.0}]
    c = M.field_counts(DOC, pred)
    assert c["fn"] == 4 and c["fp"] == 4       # bagel missed, muffin invented


def test_null_prediction_scores_zero():
    out = M.field_f1([DOC], [None])
    assert out["f1"] == 0.0 and out["recall"] == 0.0


def test_field_subset():
    pred = copy.deepcopy(DOC)
    pred["line_items"] = []
    assert M.exact_match(DOC, pred, fields=["vendor_name", "date", "total"])


def test_per_field_accuracy():
    pred = copy.deepcopy(DOC)
    pred["date"] = "2026-03-05"
    acc = M.per_field_accuracy([DOC, DOC], [pred, DOC])
    assert acc["date"] == {"accuracy": 0.5, "n": 2} and acc["total"]["accuracy"] == 1.0


def test_routing_quality():
    rq = M.routing_quality(["auto", "auto", "review", "auto"], [True, False, False, True])
    assert rq["straight_through_rate"] == 0.75
    assert rq["auto_precision"] == pytest.approx(2 / 3) and rq["auto_errors"] == 1


def test_business_projection():
    out = M.business_projection(
        docs_per_month=1000, manual_minutes_per_doc=3, review_minutes_per_doc=1,
        clerk_usd_per_hour=60, undetected_error_usd=10, small_usd_per_doc=0.001,
        straight_through_rate=0.8, auto_precision=0.99, baseline_usd_per_doc=0.02)
    assert out["manual_total_usd"] == pytest.approx(3000)
    small = out["small_model"]
    assert small["inference_usd"] == pytest.approx(1.0)
    assert small["review_usd"] == pytest.approx(200.0)
    assert small["error_usd"] == pytest.approx(80.0)
    assert out["net_monthly_saving_vs_baseline_usd"] == pytest.approx(19.0)


def test_unscored_if_null_skips_unknown_gold():
    from docextract import metrics as M

    gold = {"vendor_name": "Shop", "date": None, "total": 9.0, "line_items": []}
    pred = {"vendor_name": "Shop", "date": "2018-12-25", "total": 9.0, "line_items": []}
    assert M.field_counts(gold, pred)["fp"] == 1                                   # strict: false positive
    assert M.field_counts(gold, pred, unscored_if_null=["date"])["fp"] == 0       # unknown gold: skipped
    assert M.exact_match(gold, pred, unscored_if_null=["date"])
    wrong = dict(pred, total=8.0)
    assert not M.exact_match(gold, wrong, unscored_if_null=["date"])              # other fields still scored


def test_fields_limit_scope_and_scopes_per_document():
    from docextract import metrics as M

    gold = [{"total": 9.0, "subtotal": None}, {"total": 5.0, "subtotal": 4.0}]
    pred = [{"total": 9.0, "subtotal": 7.0}, {"total": 5.0, "subtotal": 4.0}]
    assert M.field_f1(gold, pred)["precision"] < 1.0
    scopes = [(["total"], []), (["total", "subtotal"], [])]
    assert M.field_f1(gold, pred, scopes=scopes)["f1"] == 1.0
    with pytest.raises(ValueError):
        M.field_f1(gold, pred, scopes=scopes[:1])
