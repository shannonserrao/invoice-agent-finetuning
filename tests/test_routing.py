import copy
import json

import pytest

from docextract import confidence as C
from docextract import routing as R
from docextract.config import load_config
from test_schema import DOC


def cfg(**routing):
    c = load_config("configs/base.yaml", ["routing.select=fixed"])
    c["routing"].update(routing)
    return c


def bad_total():
    d = copy.deepcopy(DOC)
    d["total"] = 99.0            # fails total_ok
    return d


def wrong_vendor():
    d = copy.deepcopy(DOC)
    d["vendor_name"] = "Someone Else"   # passes checks, wrong key field
    return d


def rows(conf_docs):
    gold = [{"id": str(i), "document": DOC} for i in range(len(conf_docs))]
    pred = [{"id": str(i), "document": d, "confidence": c} for i, (c, d) in enumerate(conf_docs)]
    return gold, pred


def test_fit_picks_lowest_threshold_meeting_target():
    # sorted by confidence: .95 ok, .90 ok, .80 WRONG, .70 ok, .60 ok. Precision at >=.70 is 3/4, at >=.90 is 1.0.
    gold, pred = rows([(0.95, DOC), (0.90, DOC), (0.80, wrong_vendor()), (0.70, DOC), (0.60, DOC)])
    r = R.fit_threshold(gold, pred, cfg(target_precision=1.0))
    assert r["feasible"] and r["threshold"] == 0.90 and r["n_auto_at_threshold"] == 2
    assert r["precision_at_threshold"] == 1.0 and r["straight_through_at_threshold"] == 2 / 5
    loose = R.fit_threshold(gold, pred, cfg(target_precision=0.7))
    assert loose["threshold"] == 0.60 and loose["n_auto_at_threshold"] == 5
    assert loose["precision_at_threshold"] == 0.8


def test_documents_failing_checks_never_count():
    gold, pred = rows([(0.99, bad_total()), (0.90, DOC)])
    r = R.fit_threshold(gold, pred, cfg(target_precision=1.0))
    assert r["n_passing_checks"] == 1 and r["threshold"] == 0.90


def test_infeasible_target_routes_nothing_to_auto():
    gold, pred = rows([(0.95, wrong_vendor()), (0.90, wrong_vendor())])
    r = R.fit_threshold(gold, pred, cfg(target_precision=0.98))
    assert not r["feasible"] and r["threshold"] > 1.0 and r["precision_at_threshold"] is None


def test_fit_rejects_id_mismatch():
    with pytest.raises(ValueError):
        R.fit_threshold([{"id": "a", "document": DOC}], [{"id": "b", "document": DOC, "confidence": 1}], cfg())


def test_fit_files_reads_validation_only(tmp_path, monkeypatch):
    c = cfg(target_precision=1.0)
    c["paths"] = {"data": str(tmp_path / "data"), "artifacts": str(tmp_path / "art")}
    (tmp_path / "art").mkdir()
    gold, pred = rows([(0.9, DOC)])
    opened = []

    def fake_read(path):
        opened.append(path.replace("\\", "/"))
        return gold if "processed" in path else pred

    monkeypatch.setattr(R, "read_jsonl", fake_read)
    c["predict"] = {"split": "test"}                      # even if predict targets test, fitting must not
    result = R.fit_threshold_files(c)
    assert result["split"] == "validation" and result["threshold"] == 0.9
    assert [p.rsplit("/", 1)[1] for p in opened] == ["validation.jsonl", "predictions_validation.jsonl"]
    assert json.load(open(tmp_path / "art" / "threshold.json"))["threshold"] == 0.9


def test_cascade_routes_review_documents_to_baseline_then_human():
    # 0: small auto and right. 1: low confidence -> baseline passes checks and is right.
    # 2: low confidence -> baseline fails checks -> human. 3: failing checks -> baseline right.
    gold, pred = rows([(0.95, DOC), (0.40, DOC), (0.40, DOC), (0.99, bad_total())])
    base = [{"id": "1", "document": DOC}, {"id": "2", "document": bad_total()}, {"id": "3", "document": DOC}]
    c = cfg(escalate_to_baseline=True)
    c["cost"]["baseline_usd_per_doc"] = 0.02
    out = R.cascade(gold, pred, base, c, threshold=0.9)
    assert out["auto_small"] == 1 and out["accepted_from_baseline"] == 2 and out["to_human"] == 1
    assert out["straight_through_rate"] == 0.75 and out["accepted_precision"] == 1.0
    assert out["baseline_calls"] == 3 and out["baseline_usd"] == pytest.approx(0.06)
    assert out["baseline_usd_if_all_docs"] == pytest.approx(0.08)


def test_cascade_missing_baseline_row_goes_to_human():
    gold, pred = rows([(0.1, DOC)])
    assert R.cascade(gold, pred, [], cfg(), 0.9)["to_human"] == 1


def test_evaluate_reports_cascade_and_confidence():
    from docextract.evaluate import evaluate

    gold, pred = rows([(0.95, DOC), (0.90, DOC), (0.60, wrong_vendor()), (0.50, wrong_vendor()), (0.40, DOC)])
    for p in pred:
        p.update(raw_valid_json=True, seconds=1.0)
    base = [{"id": r["id"], "document": DOC, "raw_valid_json": True} for r in gold]
    c = cfg(escalate_to_baseline=True, confidence_threshold=0.9)
    report = evaluate(gold, pred, c, baseline_rows=base)
    assert report["cascade"]["accepted_precision"] == 1.0
    assert report["confidence"]["computable"] and report["confidence"]["auroc"] > 0.5
    assert "confidence_aurc" in report["metrics"]
    assert "cascade" not in evaluate(gold, pred, cfg(confidence_threshold=0.9), baseline_rows=base)


# ----------------------------------------------------------------------------- T-11 confidence

def test_risk_coverage_and_aurc():
    pts = C.risk_coverage([0.9, 0.8, 0.7, 0.6], [True, True, False, True])
    assert pts == [(0.25, 0.0), (0.5, 0.0), (0.75, 1 / 3), (1.0, 0.25)]
    assert C.aurc([0.9, 0.8, 0.7, 0.6], [True, True, False, True]) == pytest.approx((0 + 0 + 1 / 3 + 0.25) / 4)
    with pytest.raises(ValueError):
        C.risk_coverage([], [])


def test_auroc_values():
    assert C.auroc([0.9, 0.8, 0.2, 0.1], [True, True, False, False]) == 1.0
    assert C.auroc([0.1, 0.2, 0.8, 0.9], [True, True, False, False]) == 0.0
    assert C.auroc([0.5, 0.5], [True, False]) == 0.5
    assert C.auroc([0.5], [True]) is None


def test_informative_confidence_beats_random():
    correct = [True] * 80 + [False] * 20
    conf = [0.9 - i * 0.001 for i in range(80)] + [0.2 - i * 0.001 for i in range(20)]      # errors rank last
    rep = C.confidence_report(conf, correct, n_permutations=300)
    assert rep["aurc"] == pytest.approx(rep["aurc_oracle"]) and rep["aurc"] < rep["aurc_random"] == 0.2
    assert rep["ranks_errors_better_than_random"] and rep["p_value"] < 0.05 and rep["auroc"] == 1.0


def test_uninformative_confidence_is_not_claimed_better():
    import random

    rng = random.Random(1)
    correct = [rng.random() > 0.2 for _ in range(200)]
    conf = [rng.random() for _ in correct]
    rep = C.confidence_report(conf, correct, n_permutations=300)
    assert not rep["ranks_errors_better_than_random"] and abs(rep["auroc"] - 0.5) < 0.1


def test_inverted_confidence_is_worse_than_random():
    correct = [True] * 50 + [False] * 10
    conf = [0.1] * 50 + [0.9] * 10                         # errors are the most confident
    rep = C.confidence_report(conf, correct, n_permutations=100)
    assert rep["aurc"] > rep["aurc_random"] and not rep["ranks_errors_better_than_random"]


def test_report_not_computable_without_both_classes():
    assert C.confidence_report([0.5, 0.6], [True, True])["computable"] is False
    assert C.confidence_report([0.5, 0.6], [False, False])["computable"] is False
