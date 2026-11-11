# Databricks notebook source
# MAGIC %md
# MAGIC # Baseline: zero-shot extraction with the expensive model (task T-06)
# MAGIC
# MAGIC Runs the same prompt, schema and documents the small model will be judged on, through the expensive model,
# MAGIC and packs the answers into one results zip to bring back to the repo (`make baseline-import`).
# MAGIC
# MAGIC **Before you start** (full steps in `README.md` next to this notebook):
# MAGIC 1. Upload `baseline_bundle_<split>.zip` (made locally by `make baseline-bundle`) to a Unity Catalog Volume.
# MAGIC 2. For the `anthropic` backend, store the organisation API key in a secret scope. Never paste it here.
# MAGIC    For the `databricks` backend, have the name of a Claude serving endpoint ready.
# MAGIC 3. Fill in the widgets at the top. Run once with **limit = 3** (dry run), check the table, then **limit = 0** (all).
# MAGIC
# MAGIC No GPU is needed: the model runs behind the API, this notebook only sends requests.

# COMMAND ----------

# MAGIC %pip install -q anthropic

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.text("bundle_zip", "", "1. Bundle zip path, e.g. /Volumes/main/default/baseline/baseline_bundle_test.zip")
dbutils.widgets.text("output_dir", "", "2. Output folder on a Volume, e.g. /Volumes/main/default/baseline/out")
dbutils.widgets.dropdown("backend", "anthropic", ["anthropic", "databricks"], "3. Backend")
dbutils.widgets.text("model", "claude-opus-5-5", "4. Model (anthropic backend)")
dbutils.widgets.text("secret_scope", "", "5. Secret scope (anthropic backend)")
dbutils.widgets.text("secret_key", "", "6. Secret key name (anthropic backend)")
dbutils.widgets.text("endpoint", "", "7. Serving endpoint name (databricks backend)")
dbutils.widgets.text("usd_per_1m_input", "4.0", "8. USD per 1M input tokens (check your price)")
dbutils.widgets.text("usd_per_1m_output", "20.0", "9. USD per 1M output tokens (check your price)")
dbutils.widgets.text("limit", "3", "10. Documents to run (3 = dry run, 0 = all)")
dbutils.widgets.text("workers", "4", "11. Parallel requests (lower if you see rate-limit errors)")

# COMMAND ----------

import json
import os
import shutil
import sys
import zipfile

backend = dbutils.widgets.get("backend")
settings = {
    "bundle_zip": dbutils.widgets.get("bundle_zip").strip(),
    "output_dir": dbutils.widgets.get("output_dir").strip(),
    "model": dbutils.widgets.get("model").strip(),
    "secret_scope": dbutils.widgets.get("secret_scope").strip(),
    "secret_key": dbutils.widgets.get("secret_key").strip(),
    "endpoint": dbutils.widgets.get("endpoint").strip(),
    "price_in": float(dbutils.widgets.get("usd_per_1m_input")),
    "price_out": float(dbutils.widgets.get("usd_per_1m_output")),
    "limit": int(dbutils.widgets.get("limit") or 0),
    "workers": max(1, int(dbutils.widgets.get("workers") or 1)),
}

problems = []
if not os.path.isfile(settings["bundle_zip"]):
    problems.append("widget 1: bundle zip not found at that path")
if not settings["output_dir"]:
    problems.append("widget 2: output folder is required")
if backend == "anthropic" and not (settings["model"] and settings["secret_scope"] and settings["secret_key"]):
    problems.append("anthropic backend needs widgets 4, 5 and 6 (model, secret scope, secret key)")
if backend == "databricks" and not settings["endpoint"]:
    problems.append("databricks backend needs widget 7 (serving endpoint name)")
if problems:
    raise ValueError("Fix the widgets first:\n - " + "\n - ".join(problems))
print("settings ok:", {k: v for k, v in settings.items() if k not in ("secret_scope", "secret_key")})

# COMMAND ----------

# Unpack the bundle on the driver's local disk (fast), and put the packaged code on the import path.
work = os.environ.get("BASELINE_WORK_DIR") or (
    "/local_disk0/baseline_bundle" if os.path.isdir("/local_disk0") else "/tmp/baseline_bundle")
if not os.path.exists(os.path.join(work, "bundle.json")):
    shutil.rmtree(work, ignore_errors=True)
    with zipfile.ZipFile(settings["bundle_zip"]) as zf:
        zf.extractall(work)
sys.path.insert(0, os.path.join(work, "src"))
os.chdir(work)
with open("bundle.json", encoding="utf-8") as handle:
    meta = json.load(handle)
print(f"bundle: split={meta['split']} documents={meta['n']} data_version={meta.get('data_version')}")

# COMMAND ----------

# Results live on the Volume, not the driver, so a restarted cluster resumes from the cached responses.
# One folder for dry and full runs: the dry run's answers are reused (and not paid for twice) by the full run.
run_dir = os.path.join(settings["output_dir"], f"work_{meta['split']}")
os.makedirs(run_dir, exist_ok=True)

cfg = {
    "paths": {"data": "data", "artifacts": run_dir},
    "prompt": {"file": "configs/prompt.txt", "schema_file": "schema/document.schema.json"},
    "predict": {"split": meta["split"]},
    "baseline": {
        **meta["baseline"],
        "backend": backend,
        "model": settings["model"] or None,
        "endpoint": settings["endpoint"] or None,
        "usd_per_1m_input_tokens": settings["price_in"],
        "usd_per_1m_output_tokens": settings["price_out"],
        "workers": settings["workers"],
    },
}
if backend == "anthropic":
    # The key stays in this process's environment; it is never printed or written to disk.
    os.environ["ANTHROPIC_API_KEY"] = dbutils.secrets.get(settings["secret_scope"], settings["secret_key"])

# COMMAND ----------

from docextract.baseline import baseline  # noqa: E402 - comes from the bundle's src folder

summary = baseline(cfg, limit=settings["limit"] or None)

# COMMAND ----------

# Look at what came back before trusting it.
import pandas as pd  # noqa: E402

from docextract.baseline import baseline_predictions_path  # noqa: E402

rows = [json.loads(line) for line in open(baseline_predictions_path(cfg), encoding="utf-8")]
table = pd.DataFrame([{
    "id": r["id"], "parsed": r["document"] is not None, "raw_valid_json": r["raw_valid_json"],
    "stop_reason": r["stop_reason"], "input_tokens": r["input_tokens"], "output_tokens": r["output_tokens"],
    "seconds": round(r["seconds"] or 0, 1), "vendor": (r["document"] or {}).get("vendor_name"),
    "total": (r["document"] or {}).get("total"),
} for r in rows])
display(table)  # noqa: F821 - display is provided by Databricks
print(json.dumps({k: summary[k] for k in (
    "n", "expected_n", "complete", "api_calls", "cache_hits", "images_resized", "refusals", "raw_json_validity",
    "parsed_rate", "baseline_usd_total", "baseline_usd_per_doc", "models_served")}, indent=2))
if settings["limit"]:
    print(f"\nDRY RUN of {settings['limit']} documents. Projected cost for all {summary['expected_n']}: "
          f"about ${summary['baseline_usd_per_doc'] * summary['expected_n']:.2f}. Set widget 10 to 0 and run again.")

# COMMAND ----------

# Pack the results: predictions, summary and the response cache, in one zip on the Volume.
suffix = "_dry_run" if settings["limit"] else ""
archive = os.path.join(settings["output_dir"], f"baseline_results_{meta['split']}{suffix}.zip")
with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
    for name in sorted(os.listdir(run_dir)):
        full = os.path.join(run_dir, name)
        if os.path.isfile(full) and name.startswith("baseline_") and not name.endswith(".tmp"):
            zf.write(full, name)
    cache = os.path.join(run_dir, cfg["baseline"].get("cache_dir", "baseline_cache"))
    for name in sorted(os.listdir(cache)):
        if name.endswith(".json"):
            zf.write(os.path.join(cache, name), f"{cfg['baseline'].get('cache_dir', 'baseline_cache')}/{name}")
print(f"results written to {archive} ({os.path.getsize(archive) / 1e6:.2f} MB)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next steps
# MAGIC * **Dry run looked right?** Set widget 10 to `0` and run the notebook again from the top (the bundle is cached).
# MAGIC * **Full run finished?** Download `baseline_results_<split>.zip` from the output folder (Catalog Explorer →
# MAGIC   your Volume → the file → Download), put it anywhere in the repo checkout (for example `artifacts/`), and run
# MAGIC   `make baseline-import RESULTS=artifacts/baseline_results_test.zip`.
# MAGIC * The import refuses dry runs, results for different documents, and results made with a different prompt.
# MAGIC * The dry-run zip (`..._dry_run.zip`) is only for checking the notebook; it cannot be imported.
