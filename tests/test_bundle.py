import copy
import io
import json
import os
import sys
import zipfile
from pathlib import Path

import pytest

from docextract import baseline as B
from docextract import bundle as BU
from test_schema import DOC

NOTEBOOK = Path("platforms/databricks/baseline/baseline_notebook.py")
GOLD = dict(DOC, vendor_name="SECRET GOLD VENDOR", total=12.96)


def png(color):
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (8, 8), color).save(buf, "PNG")
    return buf.getvalue()


@pytest.fixture
def cfg(tmp_path):
    root = tmp_path / "repo"
    (root / "data" / "processed" / "images" / "cord").mkdir(parents=True)
    (root / "configs").mkdir()
    (root / "configs" / "prompt.txt").write_text("Extract it.", encoding="utf-8")
    (root / "schema").mkdir()
    (root / "schema" / "document.schema.json").write_text(json.dumps({"title": "T"}), encoding="utf-8")
    rows = []
    for i, color in enumerate([(1, 1, 1), (2, 2, 2), (3, 3, 3)]):
        (root / "data" / "processed" / "images" / "cord" / f"{i}.png").write_bytes(png(color))
        rows.append({"id": f"cord-test-{i}", "dataset": "cord", "image": f"images/cord/{i}.png", "document": GOLD})
    (root / "data" / "processed" / "test.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n",
                                                            encoding="utf-8")
    (root / "data" / "manifest.json").write_text(json.dumps({"data_version": "data-v1"}), encoding="utf-8")
    return {"paths": {"data": str(root / "data"), "artifacts": str(root / "artifacts")},
            "prompt": {"file": str(root / "configs" / "prompt.txt"),
                       "schema_file": str(root / "schema" / "document.schema.json")},
            "predict": {"split": "test"},
            "eval": {"amount_tolerance": 0.01, "key_fields": ["vendor_name", "total"]},
            "routing": {"select": "fixed", "confidence_threshold": 0.5, "required_checks": ["schema_valid"]},
            "baseline": {"backend": "anthropic", "model": "claude-opus-5-5", "effort": "medium", "max_tokens": 100,
                         "refusal_fallback": True, "cache_dir": "baseline_cache", "workers": 2,
                         "usd_per_1m_input_tokens": 4.0, "usd_per_1m_output_tokens": 20.0,
                         "databricks_dir": str(NOTEBOOK.parent)}}


class Dbutils:
    """Minimal stand-in for the Databricks `dbutils` object."""

    def __init__(self, values):
        noop = staticmethod(lambda *a, **k: None)
        self.widgets = type("W", (), {"text": noop, "dropdown": noop, "get": staticmethod(lambda n: values[n])})()
        self.secrets = type("S", (), {"get": staticmethod(lambda scope, key: "fake-key")})()
        self.library = type("L", (), {"restartPython": noop})()


def fake_call(prompt_seen=None):
    def call(client, bl, prompt, image_b64, media_type):
        if prompt_seen is not None:
            prompt_seen.append(prompt)
        return {"text": json.dumps(GOLD), "input_tokens": 900, "output_tokens": 120,
                "model_served": bl["model"], "stop_reason": "end_turn"}
    return call


def run_notebook(bundle_zip, out_dir, work_dir, monkeypatch, limit, **widgets):
    """Execute the notebook's source top to bottom, as Databricks would, with stubbed dbutils/display."""
    values = {"bundle_zip": str(bundle_zip), "output_dir": str(out_dir), "backend": "anthropic",
              "model": "claude-opus-5-5", "secret_scope": "scope", "secret_key": "key", "endpoint": "",
              "usd_per_1m_input": "4.0", "usd_per_1m_output": "20.0", "limit": str(limit), "workers": "2", **widgets}
    monkeypatch.setenv("BASELINE_WORK_DIR", str(work_dir))
    cwd, path = os.getcwd(), list(sys.path)
    shown = []
    try:
        exec(compile(NOTEBOOK.read_text(encoding="utf-8"), "baseline_notebook.py", "exec"),
             {"dbutils": Dbutils(values), "display": shown.append, "__name__": "__main__"})
    finally:
        os.chdir(cwd)
        sys.path[:] = path
    return shown


# ----------------------------------------------------------------------------- bundle

def test_bundle_contents_and_no_gold_labels(cfg, tmp_path):
    info = BU.build_bundle(cfg, str(tmp_path / "b.zip"))
    assert info["n"] == 3 and info["split"] == "test" and info["data_version"] == "data-v1"
    with zipfile.ZipFile(tmp_path / "b.zip") as zf:
        names = set(zf.namelist())
        assert {"README.md", "notebooks/baseline_notebook.py", "bundle.json", "configs/prompt.txt",
                "schema/document.schema.json", "data/manifest.json", "data/processed/test.jsonl",
                "src/docextract/baseline.py", "src/docextract/feedback.py", "src/docextract/schema.py",
                "data/processed/images/cord/0.png"} <= names
        for name in names:                                 # no gold label anywhere in the upload
            if name.endswith((".jsonl", ".json", ".md", ".txt")):
                assert b"SECRET GOLD VENDOR" not in zf.read(name), name
        rows = [json.loads(line) for line in zf.read("data/processed/test.jsonl").decode().splitlines()]
        assert [set(r) for r in rows] == [{"id", "dataset", "image"}] * 3
        meta = json.loads(zf.read("bundle.json"))
        assert meta["prompt_sha256"] == B.prompt_sha256(cfg) and meta["n"] == 3
        assert zf.read("data/processed/images/cord/1.png") == png((2, 2, 2))
        assert not any(n.startswith("data/processed/validation") for n in names)


def test_bundle_is_reproducible(cfg, tmp_path):
    a = BU.build_bundle(cfg, str(tmp_path / "a.zip"))
    b = BU.build_bundle(cfg, str(tmp_path / "b.zip"))
    assert a["sha256"] == b["sha256"]
    Path(cfg["prompt"]["file"]).write_text("Changed.", encoding="utf-8")
    assert BU.build_bundle(cfg, str(tmp_path / "c.zip"))["sha256"] != a["sha256"]


def test_bundle_needs_data(cfg, tmp_path):
    os.remove(Path(cfg["paths"]["data"]) / "processed" / "test.jsonl")
    with pytest.raises(FileNotFoundError, match="make data"):
        BU.build_bundle(cfg, str(tmp_path / "b.zip"))


# ----------------------------------------------------------------------------- notebook + import round trip

@pytest.fixture
def databricks_run(cfg, tmp_path, monkeypatch):
    """Build the bundle, then run the real notebook source on an 'uploaded' copy."""
    pytest.importorskip("pandas")
    pytest.importorskip("anthropic")
    seen = []
    monkeypatch.setattr(B, "_call", fake_call(seen))
    BU.build_bundle(cfg, str(tmp_path / "bundle.zip"))
    out = tmp_path / "volume" / "out"
    out.mkdir(parents=True)

    def run(limit):
        shown = run_notebook(tmp_path / "bundle.zip", out, tmp_path / "driver", monkeypatch, limit)
        suffix = "_dry_run" if limit else ""
        return out / f"baseline_results_test{suffix}.zip", shown

    return {"run": run, "out": out, "prompts": seen}


def test_notebook_dry_run_then_full_run_then_import(cfg, databricks_run, monkeypatch):
    dry_zip, shown = databricks_run["run"](limit=2)
    assert dry_zip.exists() and len(shown) == 1 and len(shown[0]) == 2          # the preview table has 2 rows
    full_zip, _ = databricks_run["run"](limit=0)
    assert full_zip.exists()
    assert len(databricks_run["prompts"]) == 3                 # 2 dry-run calls + 1 new: the first 2 reused from cache
    assert "Extract it." in databricks_run["prompts"][0] and '"title":"T"' in databricks_run["prompts"][0]

    with pytest.raises(ValueError, match="dry run or incomplete"):
        BU.import_results(cfg, str(dry_zip))
    assert not os.path.exists(B.baseline_predictions_path(cfg))                  # nothing installed on rejection

    from docextract.tracking import Tracker

    with Tracker(cfg, run_name="baseline", tags={"baseline": "true"}, use_mlflow=False) as tracker:
        result = BU.import_results(cfg, str(full_zip), tracker)
    assert result["summary"]["complete"] and result["summary"]["n"] == 3 and result["run_id"]
    assert result["scores"]["field_f1"] == 1.0 and result["scores"]["key_fields_accuracy"] == 1.0
    assert result["summary"]["baseline_usd_per_doc"] == pytest.approx((900 * 4 + 120 * 20) / 1e6)
    art = Path(cfg["paths"]["artifacts"])
    preds = [json.loads(line) for line in (art / "baseline_predictions.jsonl").read_text().splitlines()]
    assert [p["id"] for p in preds] == ["cord-test-0", "cord-test-1", "cord-test-2"]
    assert (art / "baseline_summary.json").exists() and len(list((art / "baseline_cache").glob("*.json"))) == 3
    log = [json.loads(line) for line in next((art / "runs").glob("*.jsonl")).read_text().splitlines()]
    assert log[0]["tags"] == {"baseline": "true"}
    assert any(e["event"] == "metrics" and "baseline_field_f1" in e["metrics"] for e in log)

    # The installed cache makes a local rerun free: no client, no calls.
    monkeypatch.setattr(B, "_call", lambda *a, **k: pytest.fail("cache should have been used"))
    again = B.baseline(cfg)
    assert again["api_calls"] == 0 and again["cache_hits"] == 3


def test_import_accepts_a_folder_and_refuses_overwrite_without_force(cfg, databricks_run, tmp_path):
    zip_path, _ = databricks_run["run"](limit=0)
    folder = tmp_path / "unzipped"
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(folder)
    BU.import_results(cfg, str(folder))
    BU.import_results(cfg, str(folder))                                    # identical content: fine
    preds = folder / "baseline_predictions.jsonl"
    lines = preds.read_text().splitlines()
    changed = json.loads(lines[0])
    changed["seconds"] = 999.0
    preds.write_text("\n".join([json.dumps(changed)] + lines[1:]) + "\n")
    with pytest.raises(FileExistsError, match="--force"):
        BU.import_results(cfg, str(folder))
    BU.import_results(cfg, str(folder), force=True)


def test_import_rejects_wrong_prompt_ids_version_and_token_totals(cfg, databricks_run, tmp_path):
    zip_path, _ = databricks_run["run"](limit=0)

    def unzip(name):
        folder = tmp_path / name
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(folder)
        return folder

    cfg_prompt = copy.deepcopy(cfg)
    Path(cfg_prompt["prompt"]["file"]).write_text("A different prompt.", encoding="utf-8")
    with pytest.raises(ValueError, match="prompt or schema changed"):
        BU.import_results(cfg_prompt, str(unzip("p")))
    Path(cfg["prompt"]["file"]).write_text("Extract it.", encoding="utf-8")

    manifest = Path(cfg["paths"]["data"]) / "manifest.json"
    manifest.write_text(json.dumps({"data_version": "data-v2"}), encoding="utf-8")
    with pytest.raises(ValueError, match="data version differs"):
        BU.import_results(cfg, str(unzip("v")))
    manifest.write_text(json.dumps({"data_version": "data-v1"}), encoding="utf-8")

    gold_file = Path(cfg["paths"]["data"]) / "processed" / "test.jsonl"
    original = gold_file.read_text()
    gold_file.write_text("\n".join(original.splitlines()[:2]) + "\n")             # local split now has 2 documents
    with pytest.raises(ValueError, match="ids differ"):
        BU.import_results(cfg, str(unzip("i")))
    gold_file.write_text(original)

    folder = unzip("t")
    summary = json.loads((folder / "baseline_summary.json").read_text())
    summary["input_tokens"] += 1
    (folder / "baseline_summary.json").write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="token totals"):
        BU.import_results(cfg, str(folder))
    assert not os.path.exists(B.baseline_predictions_path(cfg))                  # still nothing installed


def test_notebook_rejects_bad_widgets(cfg, tmp_path, monkeypatch):
    BU.build_bundle(cfg, str(tmp_path / "bundle.zip"))
    zip_path = tmp_path / "bundle.zip"
    with pytest.raises(ValueError, match="widget 2"):
        run_notebook(zip_path, "", tmp_path / "w", monkeypatch, 3)
    with pytest.raises(ValueError, match="widgets 4, 5 and 6"):
        run_notebook(zip_path, tmp_path / "o", tmp_path / "w", monkeypatch, 3, secret_scope="")
    with pytest.raises(ValueError, match="widget 7"):
        run_notebook(zip_path, tmp_path / "o", tmp_path / "w", monkeypatch, 3, backend="databricks", endpoint="")
    with pytest.raises(ValueError, match="widget 1"):
        run_notebook(tmp_path / "missing.zip", tmp_path / "o", tmp_path / "w", monkeypatch, 3)


def test_notebook_source_format_and_no_inline_secrets():
    text = NOTEBOOK.read_text(encoding="utf-8")
    assert text.startswith("# Databricks notebook source\n") and text.count("# COMMAND ----------") >= 8
    assert "dbutils.secrets.get" in text and "sk-ant" not in text
