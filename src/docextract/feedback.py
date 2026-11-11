"""Human feedback loop (specs/05-human-feedback.md).

Flow: predictions log (JSONL) -> review queue (CSV) -> reviewed CSV -> corrections (JSONL).
"""
from __future__ import annotations

import csv
import json
import os
import random
import time

from .schema import schema_errors

QUEUE_FIELDS = [
    "id", "input_ref", "prediction", "confidence", "model_version",
    "reason", "verdict", "corrected", "reviewer", "note",
]
VERDICTS = {"correct", "incorrect", "unsure"}


def read_jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def log_prediction(path: str, record: dict) -> None:
    """Append one prediction. Required keys: id, input_ref, prediction, confidence, model_version."""
    missing = {"id", "input_ref", "prediction", "confidence", "model_version"} - record.keys()
    if missing:
        raise ValueError(f"prediction record missing keys: {sorted(missing)}")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def select_for_review(
    records: list[dict], threshold: float, audit_rate: float, already_reviewed: set[str],
    seed: int = 0,
) -> list[dict]:
    """Pick every record below `threshold`, plus a random `audit_rate` share of the rest."""
    rng = random.Random(seed)
    selected = []
    for rec in records:
        if rec["id"] in already_reviewed:
            continue
        if rec["confidence"] < threshold:
            selected.append({**rec, "reason": "low_confidence"})
        elif rng.random() < audit_rate:
            selected.append({**rec, "reason": "random_audit"})
    return selected


def export_review_queue(
    log_path: str, queue_path: str, corrections_path: str,
    threshold: float, audit_rate: float, seed: int = 0,
) -> int:
    done = {c["id"] for c in read_jsonl(corrections_path)}
    rows = select_for_review(read_jsonl(log_path), threshold, audit_rate, done, seed)
    os.makedirs(os.path.dirname(queue_path) or ".", exist_ok=True)
    with open(queue_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=QUEUE_FIELDS)
        writer.writeheader()
        for row in rows:
            pred = row["prediction"]
            if not isinstance(pred, str):
                pred = json.dumps(pred, ensure_ascii=False)
            writer.writerow({
                "id": row["id"], "input_ref": row["input_ref"], "prediction": pred,
                "confidence": f"{row['confidence']:.4f}", "model_version": row["model_version"],
                "reason": row["reason"], "verdict": "", "corrected": "", "reviewer": "", "note": "",
            })
    return len(rows)


def ingest_reviewed(reviewed_path: str, corrections_path: str) -> dict:
    """Validate a reviewed CSV and append new rows to corrections. Returns counts."""
    existing = {c["id"] for c in read_jsonl(corrections_path)}
    counts = {"added": 0, "skipped_duplicate": 0, "skipped_blank": 0}
    new_rows = []
    with open(reviewed_path, newline="", encoding="utf-8") as handle:
        for line_no, row in enumerate(csv.DictReader(handle), start=2):
            verdict = (row.get("verdict") or "").strip().lower()
            if not verdict:
                counts["skipped_blank"] += 1
                continue
            if verdict not in VERDICTS:
                raise ValueError(f"line {line_no}: verdict must be one of {sorted(VERDICTS)}")
            if verdict == "incorrect" and not (row.get("corrected") or "").strip():
                raise ValueError(f"line {line_no}: 'incorrect' needs a value in 'corrected'")
            if verdict == "incorrect":
                try:
                    corrected = json.loads(row["corrected"])
                except ValueError as err:
                    raise ValueError(f"line {line_no}: 'corrected' is not valid JSON: {err}") from err
                if errors := schema_errors(corrected):
                    raise ValueError(f"line {line_no}: 'corrected' fails the schema: {errors}")
            if not (row.get("reviewer") or "").strip():
                raise ValueError(f"line {line_no}: reviewer is required")
            if row["id"] in existing:
                counts["skipped_duplicate"] += 1
                continue
            existing.add(row["id"])
            new_rows.append({
                "id": row["id"], "input_ref": row["input_ref"], "prediction": row["prediction"],
                "verdict": verdict, "corrected": row.get("corrected", "").strip(),
                "reviewer": row["reviewer"].strip(), "note": row.get("note", "").strip(),
                "model_version": row.get("model_version", ""), "reason": row.get("reason", ""),
                "reviewed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            })
    if new_rows:
        os.makedirs(os.path.dirname(corrections_path) or ".", exist_ok=True)
        with open(corrections_path, "a", encoding="utf-8") as handle:
            for row in new_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    counts["added"] = len(new_rows)
    return counts


def review_summary(corrections_path: str) -> dict:
    """Agreement rate between model and reviewers, overall and for the random audit sample."""
    rows = [r for r in read_jsonl(corrections_path) if r["verdict"] != "unsure"]

    def rate(items):
        return sum(r["verdict"] == "correct" for r in items) / len(items) if items else None

    audit = [r for r in rows if r.get("reason") == "random_audit"]
    return {
        "reviewed": len(rows),
        "agreement_all": rate(rows),
        "audit_reviewed": len(audit),
        "agreement_audit": rate(audit),
    }
