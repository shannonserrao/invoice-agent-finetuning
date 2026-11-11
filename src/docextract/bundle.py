"""Databricks hand-off for the baseline (task T-06).

`build_bundle` packs exactly what the Databricks notebook needs: the documents of one split (images and ids only,
no gold labels), the shared prompt and schema, a copy of the package code, and the notebook with its instructions.
`import_results` takes the results archive the notebook produces, validates it against the local data and prompt,
installs the files where `make eval` expects them, and logs a tracked run tagged `baseline=true`.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import zipfile
from pathlib import Path

from . import baseline as B
from .evaluate import score_baseline
from .feedback import read_jsonl

FIXED_TIME = (2026, 1, 1, 0, 0, 0)        # fixed timestamps: the same inputs always give the same zip
STORED_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".webp"}


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _add(zf: zipfile.ZipFile, arcname: str, data: bytes | None = None, path: str | None = None) -> None:
    info = zipfile.ZipInfo(arcname, FIXED_TIME)
    info.external_attr = 0o644 << 16
    info.compress_type = zipfile.ZIP_STORED if Path(arcname).suffix.lower() in STORED_SUFFIXES else zipfile.ZIP_DEFLATED
    if data is None:
        with open(path, "rb") as handle:
            data = handle.read()
    zf.writestr(info, data)


def build_bundle(cfg: dict, out_path: str | None = None) -> dict:
    """Write the upload bundle for the configured split. Returns {path, split, n, bytes, sha256, ...}."""
    paths, bl = cfg["paths"], cfg["baseline"]
    split = B.split_name(cfg)
    rows = read_jsonl(os.path.join(paths["data"], "processed", f"{split}.jsonl"))
    if not rows:
        raise FileNotFoundError(f"no rows for split {split!r}: run `make data` first")
    notebook_dir = bl.get("databricks_dir", "platforms/databricks/baseline")
    notebook = os.path.join(notebook_dir, "baseline_notebook.py")
    readme = os.path.join(notebook_dir, "README.md")
    for needed in (notebook, readme, cfg["prompt"]["file"], cfg["prompt"]["schema_file"]):
        if not os.path.isfile(needed):
            raise FileNotFoundError(needed)
    package_dir = Path(__file__).parent
    out_path = out_path or os.path.join(paths["artifacts"], f"baseline_bundle_{split}.zip")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    # No gold labels leave the machine: the notebook only needs ids and image paths.
    inputs = [{"id": r["id"], "dataset": r["dataset"], "image": r["image"]} for r in rows]
    meta = {
        "bundle_format": 1, "split": split, "n": len(rows),
        "ids_sha256": hashlib.sha256("\n".join(sorted(r["id"] for r in rows)).encode()).hexdigest(),
        "prompt_sha256": B.prompt_sha256(cfg), "baseline": {k: v for k, v in bl.items() if k != "databricks_dir"},
    }
    manifest_path = os.path.join(paths["data"], "manifest.json")
    manifest = None
    if os.path.exists(manifest_path):
        with open(manifest_path, encoding="utf-8") as handle:
            manifest = handle.read()
        meta["data_version"] = json.loads(manifest).get("data_version")

    with zipfile.ZipFile(out_path, "w") as zf:
        _add(zf, "README.md", path=readme)
        _add(zf, "notebooks/baseline_notebook.py", path=notebook)
        _add(zf, "bundle.json", data=(json.dumps(meta, indent=2) + "\n").encode())
        _add(zf, "configs/prompt.txt", path=cfg["prompt"]["file"])
        _add(zf, "schema/document.schema.json", path=cfg["prompt"]["schema_file"])
        if manifest is not None:
            _add(zf, "data/manifest.json", data=manifest.encode())
        _add(zf, f"data/processed/{split}.jsonl",
             data=("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in inputs)).encode())
        for source in sorted(package_dir.glob("*.py")):
            _add(zf, f"src/docextract/{source.name}", path=str(source))
        for row in sorted(inputs, key=lambda r: r["image"]):
            _add(zf, f"data/processed/{row['image']}", path=os.path.join(paths["data"], "processed", row["image"]))
    return {"path": out_path, "split": split, "n": len(rows), "bytes": os.path.getsize(out_path),
            "sha256": _sha256_file(out_path), "prompt_sha256": meta["prompt_sha256"],
            "data_version": meta.get("data_version")}


# --------------------------------------------------------------------------- import

def _find(root: str, pattern: str) -> list[Path]:
    return sorted(Path(root).rglob(pattern))


def _install(src: str, dst: str, force: bool) -> None:
    if os.path.exists(dst) and not force and _sha256_file(src) != _sha256_file(dst):
        raise FileExistsError(f"{dst} already exists with different content: pass --force to replace it")
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    shutil.copyfile(src, dst)


def import_results(cfg: dict, source: str, tracker=None, force: bool = False) -> dict:
    """Validate and install a results archive (zip or folder) from the Databricks notebook."""
    tmp = None
    if os.path.isfile(source) and zipfile.is_zipfile(source):
        tmp = tempfile.mkdtemp(prefix="baseline_results_")
        with zipfile.ZipFile(source) as zf:
            zf.extractall(tmp)
        root = tmp
    elif os.path.isdir(source):
        root = source
    else:
        raise FileNotFoundError(f"{source} is neither a results zip nor a folder")
    try:
        summaries = _find(root, "baseline_summary*.json")
        if len(summaries) != 1:
            raise ValueError(f"expected one baseline_summary*.json in the results, found {len(summaries)}")
        with open(summaries[0], encoding="utf-8") as handle:
            summary = json.load(handle)
        split = summary["split"]
        local_cfg = {**cfg, "predict": {**cfg.get("predict", {}), "split": split}}
        pred_files = _find(root, os.path.basename(B.baseline_predictions_path(local_cfg, split)))
        if len(pred_files) != 1:
            raise ValueError("results archive has no (or more than one) baseline predictions file")
        pred_rows = read_jsonl(str(pred_files[0]))

        problems = []
        if not summary.get("complete") or summary.get("limit"):
            problems.append(
                f"this is a dry run or incomplete run (n={summary.get('n')} of {summary.get('expected_n')})")
        gold = read_jsonl(os.path.join(cfg["paths"]["data"], "processed", f"{split}.jsonl"))
        if not gold:
            raise FileNotFoundError(f"no local gold for split {split!r}: run `make data` first")
        gold_ids, pred_ids = {r["id"] for r in gold}, {r["id"] for r in pred_rows}
        if gold_ids != pred_ids:
            problems.append(f"ids differ from the local {split} split ({len(gold_ids - pred_ids)} missing, "
                            f"{len(pred_ids - gold_ids)} extra)")
        if summary.get("prompt_sha256") != B.prompt_sha256(cfg):
            problems.append("the prompt or schema changed since the bundle was built; rebuild and rerun")
        manifest_path = os.path.join(cfg["paths"]["data"], "manifest.json")
        if os.path.exists(manifest_path):
            with open(manifest_path, encoding="utf-8") as handle:
                local_version = json.load(handle).get("data_version")
            if summary.get("data_version") != local_version:
                problems.append(f"data version differs (results {summary.get('data_version')}, local {local_version})")
        if sum(r["input_tokens"] for r in pred_rows) != summary.get("input_tokens") or \
                sum(r["output_tokens"] for r in pred_rows) != summary.get("output_tokens"):
            problems.append("token totals in the summary do not match the predictions file")
        if len(pred_rows) != summary.get("n"):
            problems.append("row count in the summary does not match the predictions file")
        if problems:
            raise ValueError("results rejected:\n  - " + "\n  - ".join(problems))

        artifacts = cfg["paths"]["artifacts"]
        _install(str(pred_files[0]), B.baseline_predictions_path(local_cfg, split), force)
        _install(str(summaries[0]), B.baseline_summary_path(local_cfg, split), force)
        cache_name = cfg["baseline"].get("cache_dir", "baseline_cache")
        cache_src = [p for p in _find(root, "*.json") if p.parent.name == cache_name]
        cache_dst = os.path.join(artifacts, cache_name)
        os.makedirs(cache_dst, exist_ok=True)
        for item in cache_src:
            target = os.path.join(cache_dst, item.name)
            if not os.path.exists(target):
                shutil.copyfile(item, target)

        scores = score_baseline(gold, pred_rows, local_cfg)
        result = {"summary": summary, "scores": scores, "cache_files_added": len(cache_src),
                  "source_sha256": _sha256_file(source) if os.path.isfile(source) else None}
        if tracker is not None:
            tracker.log_params({"baseline.split": split, "baseline.backend": summary.get("backend"),
                                "baseline.model": summary.get("model"),
                                "baseline.prompt_sha256": summary["prompt_sha256"],
                                "baseline.data_version": summary.get("data_version"),
                                "baseline.results_sha256": result["source_sha256"]})
            tracker.log_metrics({k: v for k, v in summary.items() if isinstance(v, (int, float)) and v is not None})
            tracker.log_metrics({f"baseline_{k}": v for k, v in scores.items()})
            tracker.log_artifact(B.baseline_predictions_path(local_cfg, split))
            tracker.log_artifact(B.baseline_summary_path(local_cfg, split))
            result["run_id"] = tracker.run_id
        return result
    finally:
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)
