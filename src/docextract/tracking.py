"""Experiment tracking wrapper (specs/04-experiment-tracking.md).

Uses MLflow when installed; otherwise appends to artifacts/runs/<run>.jsonl so that
tests and smoke runs work without it.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
import uuid

from .config import flatten

try:  # pragma: no cover - exercised only when mlflow is installed
    import mlflow
except ImportError:  # pragma: no cover
    mlflow = None


def git_info() -> dict:
    def run(args):
        try:
            return subprocess.check_output(args, stderr=subprocess.DEVNULL).decode().strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return ""

    return {
        "git.sha": run(["git", "rev-parse", "HEAD"]) or "unknown",
        "git.dirty": bool(run(["git", "status", "--porcelain"])),
    }


class Tracker:
    def __init__(self, cfg: dict, run_name: str, stage: str = "dev", use_mlflow: bool = True,
                 tags: dict | None = None):
        self.cfg = cfg
        self.tags = dict(tags or {})
        self.run_name = run_name
        self.stage = stage
        self.use_mlflow = bool(use_mlflow and mlflow is not None)
        self.run_id = uuid.uuid4().hex[:12]
        artifacts = cfg.get("paths", {}).get("artifacts", "artifacts")
        self._fallback_path = os.path.join(artifacts, "runs", f"{self.run_id}.jsonl")

    def __enter__(self) -> Tracker:
        tracking = self.cfg.get("tracking", {})
        if self.use_mlflow:
            uri = os.environ.get("MLFLOW_TRACKING_URI") or tracking.get("uri")
            if uri:
                mlflow.set_tracking_uri(uri)
            mlflow.set_experiment(tracking.get("experiment", "default"))
            run = mlflow.start_run(run_name=self.run_name)
            self.run_id = run.info.run_id
            mlflow.set_tag("stage", self.stage)
            for key, value in self.tags.items():
                mlflow.set_tag(key, value)
        else:
            os.makedirs(os.path.dirname(self._fallback_path), exist_ok=True)
            self._write({"event": "start", "run_name": self.run_name, "stage": self.stage, "tags": self.tags})
        self.log_params({**flatten(self.cfg), **git_info()})
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        status = "FAILED" if exc_type else "FINISHED"
        if self.use_mlflow:
            mlflow.end_run(status=status)
        else:
            self._write({"event": "end", "status": status})

    def _write(self, record: dict) -> None:
        record = {"time": time.time(), "run_id": self.run_id, **record}
        with open(self._fallback_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=str) + "\n")

    def log_params(self, params: dict) -> None:
        if self.use_mlflow:
            mlflow.log_params({k: str(v)[:500] for k, v in params.items()})
        else:
            self._write({"event": "params", "params": params})

    def log_metrics(self, metrics: dict, step: int | None = None) -> None:
        numeric = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
        if self.use_mlflow:
            mlflow.log_metrics(numeric, step=step)
        else:
            self._write({"event": "metrics", "step": step, "metrics": numeric})

    def log_artifact(self, path: str) -> None:
        if self.use_mlflow:
            mlflow.log_artifact(path)
        else:
            self._write({"event": "artifact", "path": path})
