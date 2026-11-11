"""Side-by-side review page (task T-15, specs/05-human-feedback.md).

Turns `feedback/review_queue.csv` into one self-contained local HTML file: the document image beside editable
JSON, a verdict per item, and a button that downloads `reviewed.csv` in the format `make feedback-ingest` reads.
The page is written to disk and opened from disk. Nothing is uploaded, so no document leaves the machine.
"""
from __future__ import annotations

import csv
import json
import os
from importlib import resources
from urllib.parse import quote


def _asset(name: str) -> str:
    return resources.files("docextract").joinpath("assets", name).read_text(encoding="utf-8")


def load_queue(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _resolve_image(ref: str, roots: list[str]) -> str | None:
    if not ref:
        return None
    for candidate in [ref] + [os.path.join(root, ref) for root in roots]:
        if os.path.isfile(candidate):
            return candidate
    return None


def _script_json(value) -> str:
    """JSON safe to place inside a <script> element."""
    return json.dumps(value, ensure_ascii=True).replace("</", "<\\/").replace("<!--", "<\\!--")


def build_review_page(queue_path: str, out_path: str, image_roots: list[str] | None = None) -> dict:
    """Write the page. Images are referenced by relative path, so keep the page where `out_path` puts it."""
    rows = load_queue(queue_path)
    out_dir = os.path.dirname(os.path.abspath(out_path))
    items, missing = [], 0
    for row in rows:
        image = _resolve_image(row.get("input_ref", ""), image_roots or [])
        src = None
        if image:
            src = quote(os.path.relpath(os.path.abspath(image), out_dir).replace(os.sep, "/"))
        else:
            missing += 1
        items.append({
            "id": row["id"], "input_ref": row.get("input_ref", ""), "src": src,
            "prediction": row.get("prediction", ""), "confidence": row.get("confidence", ""),
            "model_version": row.get("model_version", ""), "reason": row.get("reason", ""),
        })
    page = _asset("review.html").replace("/*__LOGIC__*/", _asset("review_logic.js"))
    # Data goes in last so it is never re-scanned for template placeholders.
    page = page.replace("__QUEUE__", _script_json(items))
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(page)
    return {"items": len(items), "images_missing": missing, "path": out_path}
