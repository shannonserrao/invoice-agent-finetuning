"""Paste into a Colab cell (GPU runtime). Clones, installs and runs the pipeline."""
import os
import subprocess

REPO = os.environ.get("REPO_URL", "https://github.com/<org>/invoice-receipt-extraction.git")  # set me

try:
    from google.colab import drive  # type: ignore

    drive.mount("/content/drive")
except ImportError:
    print("Not on Colab; skipping Drive mount.")

if not os.path.isdir("invoice-receipt-extraction"):
    subprocess.run(["git", "clone", REPO], check=True)
os.chdir("invoice-receipt-extraction")
subprocess.run(["pip", "install", "-q", "-e", ".[train]"], check=True)
for step in ("data", "train", "predict", "evaluate"):
    subprocess.run(
        ["python", "-m", "docextract.cli", step, "--config", "configs/platform/colab.yaml"],
        check=True,
    )
