"""Canonical document handling: normalisation, flattening, validation checks.

The contract is schema/document.schema.json. This module has no third-party dependencies.
"""
from __future__ import annotations

import json
import re
import unicodedata
from datetime import date

HEADER_FIELDS = ["vendor_name", "date", "currency", "subtotal", "tax", "total"]
NUMBER_FIELDS = {"subtotal", "tax", "total"}
ITEM_FIELDS = ["description", "quantity", "unit_price", "amount"]
ITEM_NUMBER_FIELDS = {"quantity", "unit_price", "amount"}


def parse_number(value) -> float | None:
    """Parse '1,234.50', '$12', '12,50' (decimal comma) or a number. None if absent/unparseable."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return round(float(value), 2)
    text = re.sub(r"[^0-9,.\-]", "", str(value))
    if not re.search(r"[0-9]", text):
        return None
    if "," in text and "." in text:
        if text.rfind(",") > text.rfind("."):
            text = text.replace(".", "").replace(",", ".")   # 1.234,50
        else:
            text = text.replace(",", "")                     # 1,234.50
    elif "," in text:
        head, _, tail = text.rpartition(",")
        text = f"{head.replace(',', '')}.{tail}" if len(tail) in (1, 2) else text.replace(",", "")
    try:
        return round(float(text), 2)
    except ValueError:
        return None


def norm_text(value) -> str | None:
    if value is None:
        return None
    text = unicodedata.normalize("NFKC", str(value)).lower()
    text = re.sub(r"\s+", " ", text).strip(" .,:;-_*#")
    return text or None


def norm_date(value) -> str | None:
    text = norm_text(value)
    if not text:
        return None
    try:
        return date.fromisoformat(text[:10]).isoformat()
    except ValueError:
        return text  # not ISO: kept as text so it can only match an identical string


def normalise(doc: dict | None) -> dict:
    """Return a canonical copy with every field present and normalised."""
    doc = doc or {}
    out = {}
    for field in HEADER_FIELDS:
        raw = doc.get(field)
        if field in NUMBER_FIELDS:
            out[field] = parse_number(raw)
        elif field == "date":
            out[field] = norm_date(raw)
        else:
            out[field] = norm_text(raw)
    items = []
    for item in doc.get("line_items") or []:
        if not isinstance(item, dict):
            continue
        clean = {k: (parse_number(item.get(k)) if k in ITEM_NUMBER_FIELDS else norm_text(item.get(k)))
                 for k in ITEM_FIELDS}
        if any(v is not None for v in clean.values()):
            items.append(clean)
    out["line_items"] = items
    return out


def to_target_text(doc: dict) -> str:
    """Training target: canonical key order, compact separators."""
    ordered = {k: doc.get(k) for k in ["vendor_name", "date", "currency"]}
    ordered["line_items"] = [{k: it.get(k) for k in ITEM_FIELDS} for it in doc.get("line_items") or []]
    for k in ["subtotal", "tax", "total"]:
        ordered[k] = doc.get(k)
    return json.dumps(ordered, ensure_ascii=False, separators=(",", ":"))


def schema_errors(doc) -> list[str]:
    """Structural validation equivalent to schema/document.schema.json."""
    if not isinstance(doc, dict):
        return ["document is not an object"]
    errors = []
    allowed = set(HEADER_FIELDS) | {"line_items"}
    errors += [f"unexpected key {k!r}" for k in doc if k not in allowed]
    errors += [f"missing key {k!r}" for k in allowed if k not in doc]
    for k in HEADER_FIELDS:
        v = doc.get(k)
        if v is None:
            continue
        if k in NUMBER_FIELDS and (isinstance(v, bool) or not isinstance(v, (int, float))):
            errors.append(f"{k} must be a number or null")
        if k not in NUMBER_FIELDS and not isinstance(v, str):
            errors.append(f"{k} must be a string or null")
    items = doc.get("line_items")
    if "line_items" in doc and not isinstance(items, list):
        errors.append("line_items must be a list")
    for i, item in enumerate(items if isinstance(items, list) else []):
        if not isinstance(item, dict):
            errors.append(f"line_items[{i}] is not an object")
            continue
        errors += [f"line_items[{i}] unexpected key {k!r}" for k in item if k not in ITEM_FIELDS]
        errors += [f"line_items[{i}] missing key {k!r}" for k in ITEM_FIELDS if k not in item]
        for k in ITEM_NUMBER_FIELDS:
            v = item.get(k)
            if v is not None and (isinstance(v, bool) or not isinstance(v, (int, float))):
                errors.append(f"line_items[{i}].{k} must be a number or null")
        description = item.get("description")
        if description is not None and not isinstance(description, str):
            errors.append(f"line_items[{i}].description must be a string or null")
    return errors


def run_checks(doc, tolerance: float = 0.01) -> dict:
    """Validation checks. Each value is True, False, or None when its inputs are absent."""
    result = {"schema_valid": not schema_errors(doc), "items_sum_ok": None, "total_ok": None,
              "date_ok": None}
    if not isinstance(doc, dict):
        return result
    n = normalise(doc)
    amounts = [it["amount"] for it in n["line_items"]]
    if amounts and all(a is not None for a in amounts) and n["subtotal"] is not None:
        result["items_sum_ok"] = abs(sum(amounts) - n["subtotal"]) <= tolerance
    if n["subtotal"] is not None and n["total"] is not None:
        result["total_ok"] = abs(n["subtotal"] + (n["tax"] or 0.0) - n["total"]) <= tolerance
    if n["date"] is not None:
        result["date_ok"] = bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}", n["date"]))
    return result


def route(checks: dict, confidence: float, threshold: float, required: list[str],
          allow_missing: list[str] | None = None) -> str:
    """'auto' only if every required check passes and confidence meets the threshold."""
    allow_missing = set(allow_missing or [])
    for name in required:
        value = checks.get(name)
        if value is None and name in allow_missing:
            continue
        if value is not True:
            return "review"
    for name, value in checks.items():
        if value is False:
            return "review"
    return "auto" if confidence >= threshold else "review"
