"""Databricks job entry point. Assumes the repo is checked out as a Databricks Repo."""
import subprocess
import sys

CONFIG = "configs/platform/databricks.yaml"
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", ".[train]"], check=True)
for step in ("data", "train", "predict", "evaluate"):
    subprocess.run([sys.executable, "-m", "docextract.cli", step, "--config", CONFIG], check=True)
