"""Kaggle script kernel entry point."""
import os
import subprocess

REPO = os.environ.get("REPO_URL", "https://github.com/<org>/invoice-receipt-extraction.git")  # set me

subprocess.run(["git", "clone", REPO, "/kaggle/working/repo"], check=True)
os.chdir("/kaggle/working/repo")
subprocess.run(["pip", "install", "-q", "-e", ".[train]"], check=True)
for step in ("data", "train", "predict", "evaluate"):
    subprocess.run(
        ["python", "-m", "docextract.cli", step, "--config", "configs/platform/kaggle.yaml"],
        check=True,
    )
