"""Download datasets, map to the canonical schema, split, check, write the manifest.

Implements specs/01-data.md (T-03, T-04). SROIE (T-05) and corrections merge (T-13) plug in through
`MAPPERS`/`LOADERS` and `build_dataset`.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import re
from datetime import date
from collections.abc import Callable, Iterable

from .feedback import read_jsonl
from .schema import parse_number, run_checks, schema_errors

SPLITS = ("train", "validation", "test")
CAP_KEYS = {"train": "max_train_samples", "validation": "max_eval_samples", "test": "max_eval_samples"}


# --------------------------------------------------------------------------- mappers

def _as_list(value) -> list:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _text(value) -> str | None:
    if not isinstance(value, str):
        return None
    value = " ".join(value.split())
    return value or None


def _num(value) -> float | None:
    return parse_number(value) if isinstance(value, (str, int, float)) else None


def map_cord(raw: dict) -> dict:
    """CORD v2 `ground_truth` (full JSON or its `gt_parse` part) -> canonical document.

    CORD labels no vendor, date or currency: those stay null (listed in the data card).
    Line items come from `menu`; nested `sub` modifier rows are not items and are dropped.
    `service_price`, `discount_price` and `etc` have no canonical field and are not folded into tax.
    """
    gt = raw.get("gt_parse", raw)
    items = []
    for entry in _as_list(gt.get("menu")):
        if not isinstance(entry, dict):
            continue
        item = {
            "description": _text(entry.get("nm")),
            "quantity": _num(entry.get("cnt")),
            "unit_price": _num(entry.get("unitprice")),
            "amount": _num(entry.get("price")),
        }
        if any(v is not None for v in item.values()):
            items.append(item)
    sub_total = gt.get("sub_total") if isinstance(gt.get("sub_total"), dict) else {}
    total = gt.get("total") if isinstance(gt.get("total"), dict) else {}
    return {
        "vendor_name": None, "date": None, "currency": None,
        "line_items": items,
        "subtotal": _num(sub_total.get("subtotal_price")),
        "tax": _num(sub_total.get("tax_price")),
        "total": _num(total.get("total_price")),
    }


_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], start=1)}


def _iso(year: int, month: int, day: int) -> str | None:
    try:
        return date(year, month, day).isoformat()
    except ValueError:
        return None


def _year(text: str) -> int:
    return int(text) if len(text) == 4 else 2000 + int(text)


def parse_date(value) -> str | None:
    """Text date -> ISO 8601, or None when unparseable or ambiguous (never guessed).

    All-numeric `a/b/yy` dates are read as day/month or month/day only when one part exceeds 12;
    if both parts are 12 or less and differ, the date is ambiguous and returns None.
    Two-digit years are read as 20yy.
    """
    if not isinstance(value, str):
        return None
    text = value.strip().strip("()").lower()
    if m := re.fullmatch(r"(\d{4})[-/.]?(\d{2})[-/.]?(\d{2})", text):           # 2018-03-23, 20180304
        return _iso(int(m[1]), int(m[2]), int(m[3]))
    if m := re.fullmatch(r"(\d{1,2})[-/. ]([a-z]{3})[a-z]*[-/. ](\d{2}|\d{4})", text):   # 05 MAR 2018
        return _iso(_year(m[3]), _MONTHS[m[2]], int(m[1])) if m[2] in _MONTHS else None
    if m := re.fullmatch(r"([a-z]{3})[a-z]*\.? (\d{1,2}),? (\d{4})", text):        # OCT 3, 2016
        return _iso(int(m[3]), _MONTHS[m[1]], int(m[2])) if m[1] in _MONTHS else None
    if m := re.fullmatch(r"(\d{1,2})[-/.](\d{1,2})[-/.](\d{2}|\d{4})", text):
        a, b, year = int(m[1]), int(m[2]), _year(m[3])
        if a > 12 >= b:
            return _iso(year, b, a)
        if b > 12 >= a:
            return _iso(year, a, b)
        if a == b:
            return _iso(year, a, b)
    return None


def map_sroie(raw: dict) -> dict:
    """SROIE `entities` -> canonical document. Header fields only.

    Labelled: company -> vendor_name, date (ISO or null if ambiguous), total. Address is not in the schema.
    Not labelled, so null or empty and excluded from metrics: currency, line_items, subtotal, tax.
    """
    ent = raw.get("entities", raw)
    return {
        "vendor_name": _text(ent.get("company")),
        "date": parse_date(ent.get("date")),
        "currency": None,
        "line_items": [],
        "subtotal": None, "tax": None,
        "total": _num(ent.get("total")),
    }


MAPPERS: dict[str, Callable[[dict], dict]] = {"cord": map_cord, "sroie": map_sroie}


# --------------------------------------------------------------------------- loading

def load_cord(ds_cfg: dict) -> dict[str, Iterable[dict]]:
    """Yield `{"image_bytes": bytes, "raw": dict}` per published split from the pinned Hub revision."""
    from datasets import Image, load_dataset

    if not ds_cfg.get("revision"):
        raise ValueError("data.datasets.cord.revision must be pinned (see docs/data_card.md)")
    ds = load_dataset(ds_cfg["dataset_id"], revision=ds_cfg["revision"])
    out = {}
    for split in SPLITS:
        part = ds[split].cast_column("image", Image(decode=False))
        out[split] = ({"image_bytes": r["image"]["bytes"], "raw": json.loads(r["ground_truth"])} for r in part)
    return out


def load_sroie(ds_cfg: dict) -> dict[str, Iterable[dict]]:
    """Published SROIE test split only (361 receipts). Used as a second test set, never for training."""
    from datasets import Image, load_dataset

    if not ds_cfg.get("revision"):
        raise ValueError("data.datasets.sroie.revision must be pinned (see docs/data_card.md)")
    part = load_dataset(ds_cfg["dataset_id"], revision=ds_cfg["revision"], split="test")
    part = part.cast_column("image", Image(decode=False))
    return {"test": ({"image_bytes": r["image"]["bytes"], "raw": {"entities": r["entities"]}} for r in part)}


LOADERS: dict[str, Callable[[dict], dict[str, Iterable[dict]]]] = {"cord": load_cord, "sroie": load_sroie}


# --------------------------------------------------------------------------- helpers

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _image_ext(data: bytes) -> str:
    from PIL import Image

    fmt = (Image.open(io.BytesIO(data)).format or "PNG").lower()
    return {"jpeg": "jpg"}.get(fmt, fmt)


def _write_if_changed(path: str, data: bytes) -> None:
    if os.path.exists(path) and sha256_file(path) == sha256_bytes(data):
        return
    with open(path, "wb") as handle:
        handle.write(data)


def _write_jsonl(path: str, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


# --------------------------------------------------------------------------- corrections (T-13)

def _parse_label(item_id: str, raw) -> dict:
    try:
        doc = json.loads(raw) if isinstance(raw, str) else raw
    except ValueError as err:
        raise ValueError(f"correction {item_id}: not valid JSON: {err}") from err
    errs = schema_errors(doc)
    if errs:
        raise ValueError(f"correction {item_id}: fails schema: {errs}")
    return doc


def merge_corrections(corrections: list[dict], rows: dict[str, list[dict]], image_hashes: dict[str, set[str]],
                      processed: str, include_confirmed: bool = False) -> dict:
    """Merge reviewed corrections into the training rows only (spec 05 HF-5). Mutates `rows` and `image_hashes`.

    - `incorrect` with a corrected document is used; `correct` is used only when `include_confirmed`; `unsure` never.
    - An id already in the train split is relabelled in place; an unknown id is a new document and becomes a train
      row (its image is copied to images/feedback/).
    - Ids from validation or test are excluded and reported. A new document whose image equals a validation or
      test image is excluded as a leak; one equal to a train image is skipped as a duplicate.
    """
    by_id = {g: {r["id"]: r for r in rs} for g, rs in rows.items()}
    excluded: dict[str, list[str]] = {k: [] for k in
                                      ("validation", "test", "leak", "duplicate_of_train", "image_missing")}
    applied: list[str] = []
    finger = hashlib.sha256()
    for item in sorted(corrections, key=lambda c: c["id"]):
        verdict = item.get("verdict")
        if verdict == "incorrect":
            raw = item.get("corrected")
        elif verdict == "correct" and include_confirmed:
            raw = item.get("prediction")
        else:
            continue
        item_id = item["id"]
        doc = _parse_label(item_id, raw)
        group = next((g for g in rows if item_id in by_id[g]), None)
        if group == "train":
            by_id["train"][item_id]["document"] = doc
        elif group == "validation":
            excluded["validation"].append(item_id)
            continue
        elif group is not None:
            excluded["test"].append(item_id)
            continue
        else:
            ref = item.get("input_ref", "")
            path = next((p for p in (ref, os.path.join(processed, ref)) if ref and os.path.isfile(p)), None)
            if path is None:
                excluded["image_missing"].append(item_id)
                continue
            with open(path, "rb") as handle:
                data = handle.read()
            digest = sha256_bytes(data)
            if any(digest in hashes for g, hashes in image_hashes.items() if g != "train"):
                excluded["leak"].append(item_id)
                continue
            if digest in image_hashes["train"]:
                excluded["duplicate_of_train"].append(item_id)
                continue
            rel = f"images/feedback/{item_id}.{_image_ext(data)}"
            os.makedirs(os.path.join(processed, "images", "feedback"), exist_ok=True)
            _write_if_changed(os.path.join(processed, rel), data)
            row = {"id": item_id, "dataset": "feedback", "image": rel, "document": doc}
            rows["train"].append(row)
            by_id["train"][item_id] = row
            image_hashes["train"].add(digest)
        applied.append(item_id)
        finger.update(f"{item_id}\n{json.dumps(doc, sort_keys=True, ensure_ascii=False)}\n".encode())
    return {"applied": applied, "excluded": excluded, "fingerprint": finger.hexdigest()[:8]}


# --------------------------------------------------------------------------- build

def build_dataset(cfg: dict, loaders: dict | None = None) -> dict:
    """Build data/processed/*.jsonl, images and data/manifest.json. Returns the manifest.

    `loaders` maps dataset name -> callable(ds_cfg) -> {split: iterable of {"image_bytes", "raw"}}; it exists
    so tests run without network access.
    """
    loaders = loaders or LOADERS
    data_cfg = cfg["data"]
    root = cfg.get("paths", {}).get("data", "data")
    processed = os.path.join(root, "processed")
    names = [n for n, d in data_cfg["datasets"].items() if d.get("dataset_id") and n in MAPPERS]
    # Datasets with `eval_only: true` (SROIE) write test_<name>.jsonl and never touch train/validation/test.
    groups = list(SPLITS) + [f"test_{n}" for n in names if data_cfg["datasets"][n].get("eval_only")]
    rows: dict[str, list[dict]] = {g: [] for g in groups}
    sources = {}
    image_hashes: dict[str, set[str]] = {g: set() for g in groups}

    for name in names:
        ds_cfg = data_cfg["datasets"][name]
        sources[name] = {"dataset_id": ds_cfg["dataset_id"], "revision": ds_cfg.get("revision")}
        img_dir = os.path.join(processed, "images", name)
        os.makedirs(img_dir, exist_ok=True)
        eval_only = bool(ds_cfg.get("eval_only"))
        for split, records in loaders[name](ds_cfg).items():
            group = f"test_{name}" if eval_only else split
            cap = data_cfg.get(CAP_KEYS[split])
            for idx, rec in enumerate(records):
                if cap is not None and idx >= cap:
                    break
                doc = MAPPERS[name](rec["raw"])
                errs = schema_errors(doc)
                if errs:
                    raise ValueError(f"{name}/{split}/{idx}: label fails schema: {errs}")
                data = rec["image_bytes"]
                rel = f"images/{name}/{name}-{split}-{idx:04d}.{_image_ext(data)}"
                _write_if_changed(os.path.join(processed, rel), data)
                digest = sha256_bytes(data)
                image_hashes[group].add(digest)
                rows[group].append({"id": f"{name}-{split}-{idx:04d}", "dataset": name, "image": rel,
                                    "document": doc})

    # Corrections go to the training split only, after the caps so they are never capped away (spec 05 HF-5).
    fb = cfg.get("feedback") or {}
    version = data_cfg.get("version", "data-v1")
    merged = {"applied": [], "excluded": {}, "fingerprint": ""}
    if fb.get("corrections"):
        merged = merge_corrections(read_jsonl(fb["corrections"]), rows, image_hashes, processed,
                                   include_confirmed=bool(fb.get("include_confirmed", False)))
        if merged["applied"]:
            version = f"{version}-fb{merged['fingerprint']}"

    # No image may appear in more than one split (spec 01 §6).
    for a in groups:
        for b in groups:
            if a < b and (shared := image_hashes[a] & image_hashes[b]):
                raise ValueError(f"{len(shared)} image(s) shared between {a} and {b}: {sorted(shared)[:3]}")

    manifest = {
        "data_version": version,
        "seed": data_cfg.get("seed"),
        "sources": sources,
        "splits": {},
        "images_sha256": sha256_bytes("\n".join(sorted(set().union(*image_hashes.values()))).encode()),
        "corrections_applied": sorted(merged["applied"]),
        "corrections_excluded": {k: sorted(v) for k, v in merged["excluded"].items() if v},
        "gold_arithmetic": gold_arithmetic(rows, cfg.get("eval", {}).get("amount_tolerance", 0.01)),
    }
    for group in groups:
        path = os.path.join(processed, f"{group}.jsonl")
        _write_jsonl(path, rows[group])
        manifest["splits"][group] = {"rows": len(rows[group]), "sha256": sha256_file(path)}
    with open(os.path.join(root, "manifest.json"), "w", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"splits": manifest["splits"], "gold_arithmetic": manifest["gold_arithmetic"]}, indent=2))
    return manifest


def gold_arithmetic(rows: dict[str, list[dict]], tolerance: float) -> dict:
    """Share of gold documents passing each arithmetic check, among documents where it applies (reported only)."""
    out = {}
    for split, items in rows.items():
        stats = {}
        for check in ("items_sum_ok", "total_ok"):
            results = [run_checks(r["document"], tolerance)[check] for r in items]
            applicable = [x for x in results if x is not None]
            stats[check] = {"applicable": len(applicable),
                            "pass_rate": round(sum(applicable) / len(applicable), 4) if applicable else None}
        out[split] = stats
    return out
