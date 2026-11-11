import json

import pytest

from docextract import schema as S

DOC = {
    "vendor_name": "Corner Cafe", "date": "2026-03-04", "currency": "USD",
    "line_items": [
        {"description": "Latte", "quantity": 2, "unit_price": 4.5, "amount": 9.0},
        {"description": "Bagel", "quantity": 1, "unit_price": 3.0, "amount": 3.0},
    ],
    "subtotal": 12.0, "tax": 0.96, "total": 12.96,
}


@pytest.mark.parametrize("raw,expected", [
    ("1,234.50", 1234.5), ("1.234,50", 1234.5), ("$12", 12.0), ("12,5", 12.5),
    ("1,234", 1234.0), (7, 7.0), ("n/a", None), (None, None), ("", None),
])
def test_parse_number(raw, expected):
    assert S.parse_number(raw) == expected


def test_normalise_text_and_missing():
    n = S.normalise({"vendor_name": "  CORNER   Cafe. ", "total": "12.960"})
    assert n["vendor_name"] == "corner cafe" and n["total"] == 12.96
    assert n["date"] is None and n["line_items"] == []


def test_schema_matches_json_schema_file():
    with open("schema/document.schema.json", encoding="utf-8") as handle:
        spec = json.load(handle)
    assert set(spec["required"]) == set(S.HEADER_FIELDS) | {"line_items"}
    assert set(spec["properties"]["line_items"]["items"]["required"]) == set(S.ITEM_FIELDS)


def test_schema_errors():
    assert S.schema_errors(DOC) == []
    assert S.schema_errors({**DOC, "total": "12.96"})
    assert S.schema_errors({**DOC, "extra": 1})
    assert S.schema_errors("not a dict")


def test_checks_pass_and_fail():
    ok = S.run_checks(DOC)
    assert ok == {"schema_valid": True, "items_sum_ok": True, "total_ok": True, "date_ok": True}
    bad = S.run_checks({**DOC, "total": 20.0})
    assert bad["total_ok"] is False
    partial = S.run_checks({**DOC, "subtotal": None})
    assert partial["items_sum_ok"] is None and partial["total_ok"] is None


def test_route():
    checks = S.run_checks(DOC)
    req = ["schema_valid", "total_ok"]
    assert S.route(checks, 0.95, 0.9, req) == "auto"
    assert S.route(checks, 0.80, 0.9, req) == "review"
    assert S.route({**checks, "items_sum_ok": False}, 0.99, 0.9, req) == "review"
    missing = {**checks, "total_ok": None}
    assert S.route(missing, 0.99, 0.9, req) == "review"
    assert S.route(missing, 0.99, 0.9, req, allow_missing=["total_ok"]) == "auto"


def test_target_text_is_stable():
    text = S.to_target_text(DOC)
    assert json.loads(text) == DOC and " " not in text.replace("Corner Cafe", "")


def test_line_item_description_must_be_string_or_null():
    bad = {**DOC, "line_items": [{"description": 5, "quantity": 1, "unit_price": 1.0, "amount": 1.0}]}
    assert any("description must be a string" in e for e in S.schema_errors(bad))
    ok = {**DOC, "line_items": [{"description": None, "quantity": 1, "unit_price": 1.0, "amount": 1.0}]}
    assert S.schema_errors(ok) == []
