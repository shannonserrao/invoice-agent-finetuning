"""Zero-shot baseline with the expensive model (specs/03-evaluation.md §5, task T-06).

The prompt holds the schema and the same instruction the fine-tuned model gets. Responses are cached on disk by
model, prompt hash, image hash and request settings, so reruns are free and the numbers are reproducible. Token
usage is stored with each cached response, so cost is computed from usage whether or not the call was cached.

Two backends: `anthropic` (an API key, for example an organisation key kept in a Databricks secret) and
`databricks` (a Databricks model-serving endpoint, called through MLflow deployments). The same code runs on a
laptop and inside the Databricks notebook in platforms/databricks/baseline/.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from .feedback import read_jsonl

MEDIA_TYPES = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg", "gif": "image/gif",
               "webp": "image/webp"}


# --------------------------------------------------------------------------- file names

def split_name(cfg: dict, split: str | None = None) -> str:
    return split or cfg.get("predict", {}).get("split", "test")


def baseline_predictions_path(cfg: dict, split: str | None = None) -> str:
    split = split_name(cfg, split)
    name = "baseline_predictions.jsonl" if split == "test" else f"baseline_predictions_{split}.jsonl"
    return os.path.join(cfg["paths"]["artifacts"], name)


def baseline_summary_path(cfg: dict, split: str | None = None) -> str:
    split = split_name(cfg, split)
    name = "baseline_summary.json" if split == "test" else f"baseline_summary_{split}.json"
    return os.path.join(cfg["paths"]["artifacts"], name)


# --------------------------------------------------------------------------- prompt and parsing

def build_prompt(cfg: dict) -> str:
    """Shared instruction text plus the JSON Schema. The fine-tuned model's prompt must start from the same text."""
    with open(cfg["prompt"]["file"], encoding="utf-8") as handle:
        instruction = handle.read().strip()
    with open(cfg["prompt"]["schema_file"], encoding="utf-8") as handle:
        schema = json.dumps(json.load(handle), separators=(",", ":"))
    return f"{instruction}\n\nJSON Schema:\n{schema}"


def prompt_sha256(cfg: dict) -> str:
    return hashlib.sha256(build_prompt(cfg).encode()).hexdigest()


def parse_output(text: str) -> tuple[dict | None, bool]:
    """Return (document or None, raw_valid_json). `raw_valid_json` is True only if the text parses as-is.

    Fenced or chatty output is salvaged into a document so it can be scored, but counts as not raw-valid.
    """
    text = (text or "").strip()
    try:
        value = json.loads(text)
        return (value, True) if isinstance(value, dict) else (None, False)
    except ValueError:
        pass
    fenced = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    for candidate in (fenced, text[text.find("{"):text.rfind("}") + 1] if "{" in text else ""):
        try:
            value = json.loads(candidate)
        except ValueError:
            continue
        if isinstance(value, dict):
            return value, False
    return None, False


# --------------------------------------------------------------------------- images

def prepare_image(data: bytes, ext: str, max_bytes: int, max_edge: int) -> tuple[bytes, str, bool]:
    """Bytes and media type to send. The original is kept unless it is too large for the API.

    Claude rejects images over 5 MB and downscales anything beyond about 1568 px on the long edge itself, so a
    larger image only costs upload time. Oversized images are shrunk deterministically: fit within `max_edge`, keep
    PNG if it fits `max_bytes`, otherwise JPEG at falling quality, and if needed a smaller image, until it fits.
    """
    media_type = MEDIA_TYPES[ext.lower()]
    from PIL import Image

    with Image.open(io.BytesIO(data)) as probe:
        width, height = probe.size
    if len(data) <= max_bytes and max(width, height) <= max_edge:
        return data, media_type, False
    with Image.open(io.BytesIO(data)) as img:
        img = img.convert("RGB")
    scale = min(1.0, max_edge / max(width, height))
    while True:
        size = (max(1, round(width * scale)), max(1, round(height * scale)))
        current = img.resize(size, Image.LANCZOS) if size != (width, height) else img
        buf = io.BytesIO()
        current.save(buf, "PNG", optimize=True)
        if buf.tell() <= max_bytes:
            return buf.getvalue(), "image/png", True
        for quality in (90, 80, 70, 60, 50):
            buf = io.BytesIO()
            current.save(buf, "JPEG", quality=quality)
            if buf.tell() <= max_bytes:
                return buf.getvalue(), "image/jpeg", True
        if max(size) <= 64:                      # cannot get smaller; return the smallest attempt
            return buf.getvalue(), "image/jpeg", True
        scale *= 0.8                             # still too large: shrink the image and try again


# --------------------------------------------------------------------------- calls

def cache_key(model: str, prompt_hash: str, image_hash: str, settings: str) -> str:
    return hashlib.sha256("|".join([model, prompt_hash, image_hash, settings]).encode()).hexdigest()


def _make_client(bl: dict):
    if bl.get("backend", "anthropic") == "databricks":
        from mlflow.deployments import get_deploy_client

        return get_deploy_client("databricks")
    import anthropic

    return anthropic.Anthropic()


def _call_anthropic(client, bl: dict, prompt: str, image_b64: str, media_type: str) -> dict:
    params = {
        "model": bl["model"],
        "max_tokens": bl.get("max_tokens", 16000),
        "messages": [{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": image_b64}},
            {"type": "text", "text": prompt},
        ]}],
    }
    if bl.get("effort"):
        params["output_config"] = {"effort": bl["effort"]}
    if bl.get("refusal_fallback"):
        params["betas"] = ["server-side-fallback-2026-07-01"]
        params["fallbacks"] = "default"
    response = client.beta.messages.create(**params)
    text = "".join(b.text for b in response.content if getattr(b, "type", None) == "text")
    return {"text": text, "input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens,
            "model_served": response.model, "stop_reason": response.stop_reason}


def _call_databricks(client, bl: dict, prompt: str, image_b64: str, media_type: str) -> dict:
    """Chat-completions request to a Databricks serving endpoint (OpenAI-compatible format)."""
    inputs = {
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{image_b64}"}},
            {"type": "text", "text": prompt},
        ]}],
        "max_tokens": bl.get("max_tokens", 16000),
        "temperature": 0,
    }
    last = None
    for attempt in range(int(bl.get("retries", 3))):
        try:
            resp = client.predict(endpoint=bl["endpoint"], inputs=inputs)
            break
        except Exception as err:  # noqa: BLE001 - endpoint errors vary; retry with backoff, then raise
            last = err
            time.sleep(2 ** attempt)
    else:
        raise RuntimeError(f"endpoint {bl['endpoint']!r} failed after retries: {last}") from last
    choice = resp["choices"][0]
    content = choice["message"]["content"]
    if isinstance(content, list):
        content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
    usage = resp.get("usage") or {}
    finish = choice.get("finish_reason")
    return {"text": content or "", "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0), "model_served": resp.get("model", bl["endpoint"]),
            "stop_reason": "refusal" if finish == "content_filter" else finish}


def _call(client, bl: dict, prompt: str, image_b64: str, media_type: str) -> dict:
    """One request. Returns {text, input_tokens, output_tokens, model_served, stop_reason}."""
    call = _call_databricks if bl.get("backend", "anthropic") == "databricks" else _call_anthropic
    return call(client, bl, prompt, image_b64, media_type)


# --------------------------------------------------------------------------- run

def baseline(cfg: dict, tracker=None, client=None, now=time.perf_counter, limit: int | None = None) -> dict:
    """Run the baseline over the configured split. Writes the predictions and summary under paths.artifacts.

    `limit` runs only the first N documents (a dry run); the summary records it so a dry run cannot be mistaken
    for a full run.
    """
    bl = cfg["baseline"]
    backend = bl.get("backend", "anthropic")
    model_label = bl.get("endpoint") if backend == "databricks" else bl.get("model")
    if not model_label:
        raise ValueError("set baseline.endpoint (databricks backend) or baseline.model (anthropic backend)")
    if backend == "anthropic" and not bl.get("model"):
        raise ValueError("baseline.model is not set in the config")
    price_in, price_out = bl.get("usd_per_1m_input_tokens"), bl.get("usd_per_1m_output_tokens")
    if price_in is None or price_out is None:
        raise ValueError("set baseline.usd_per_1m_input_tokens and baseline.usd_per_1m_output_tokens")
    paths = cfg["paths"]
    split = split_name(cfg)
    rows = read_jsonl(os.path.join(paths["data"], "processed", f"{split}.jsonl"))
    if not rows:
        raise FileNotFoundError(f"no rows for split {split!r}: run `make data` first")
    all_ids = sorted(r["id"] for r in rows)
    if limit:
        rows = rows[:limit]

    prompt = build_prompt(cfg)
    prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
    max_bytes, max_edge = int(bl.get("max_image_bytes", 3_500_000)), int(bl.get("max_image_edge", 1568))
    settings = json.dumps({"backend": backend, "max_tokens": bl.get("max_tokens"), "effort": bl.get("effort"),
                           "refusal_fallback": bl.get("refusal_fallback"), "max_image_bytes": max_bytes,
                           "max_image_edge": max_edge}, sort_keys=True)
    cache_dir = os.path.join(paths["artifacts"], bl.get("cache_dir", "baseline_cache"))
    os.makedirs(cache_dir, exist_ok=True)
    holder = {"client": client}
    lock = threading.Lock()

    def get_client():
        with lock:                       # created on first real call, once, so cached reruns need no credentials
            if holder["client"] is None:
                holder["client"] = _make_client(bl)
        return holder["client"]

    def one(row: dict) -> tuple[dict, bool, bool]:
        path = os.path.join(paths["data"], "processed", row["image"])
        with open(path, "rb") as handle:
            data = handle.read()
        key = cache_key(model_label, prompt_hash, hashlib.sha256(data).hexdigest(), settings)
        cache_path = os.path.join(cache_dir, f"{key}.json")
        if os.path.exists(cache_path):
            with open(cache_path, encoding="utf-8") as handle:
                result = json.load(handle)
            return result, True, bool(result.get("image_prepared"))
        sent, media_type, prepared = prepare_image(data, os.path.splitext(path)[1].lstrip("."), max_bytes, max_edge)
        client_ = get_client()
        start = now()
        result = _call(client_, {**bl, "model": model_label}, prompt,
                       base64.standard_b64encode(sent).decode(), media_type)
        result["seconds"] = now() - start
        result["image_prepared"] = prepared
        tmp = cache_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(result, handle)
        os.replace(tmp, cache_path)
        return result, False, prepared

    workers = max(1, int(bl.get("workers", 1)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(one, rows))

    out_rows, totals = [], {"input_tokens": 0, "output_tokens": 0, "cache_hits": 0, "api_calls": 0, "refusals": 0,
                            "images_resized": 0}
    for row, (result, hit, prepared) in zip(rows, results):
        document, raw_valid = parse_output(result["text"])
        totals["cache_hits" if hit else "api_calls"] += 1
        totals["input_tokens"] += result["input_tokens"]
        totals["output_tokens"] += result["output_tokens"]
        totals["refusals"] += result["stop_reason"] == "refusal"
        totals["images_resized"] += prepared
        out_rows.append({"id": row["id"], "document": document, "raw_valid_json": raw_valid,
                         "seconds": result.get("seconds"), "input_tokens": result["input_tokens"],
                         "output_tokens": result["output_tokens"], "model_served": result["model_served"],
                         "stop_reason": result["stop_reason"]})

    os.makedirs(paths["artifacts"], exist_ok=True)
    out = baseline_predictions_path(cfg, split)
    with open(out, "w", encoding="utf-8", newline="\n") as handle:
        for r in out_rows:
            handle.write(json.dumps(r, ensure_ascii=False) + "\n")

    usd = (totals["input_tokens"] * price_in + totals["output_tokens"] * price_out) / 1_000_000
    n = len(out_rows)
    manifest_path = os.path.join(paths["data"], "manifest.json")
    data_version = None
    if os.path.exists(manifest_path):
        with open(manifest_path, encoding="utf-8") as handle:
            data_version = json.load(handle).get("data_version")
    summary = {
        "split": split, "n": n, "expected_n": len(all_ids), "limit": limit, "complete": n == len(all_ids),
        "ids_sha256": hashlib.sha256("\n".join(sorted(r["id"] for r in out_rows)).encode()).hexdigest(),
        "backend": backend, "model": model_label, "prompt_sha256": prompt_hash, "settings": json.loads(settings),
        "data_version": data_version, "usd_per_1m_input_tokens": price_in, "usd_per_1m_output_tokens": price_out,
        "baseline_usd_total": usd, "baseline_usd_per_doc": usd / n if n else None,
        "baseline_usd_per_1k": usd / n * 1000 if n else None,
        "raw_json_validity": sum(r["raw_valid_json"] for r in out_rows) / n if n else None,
        "parsed_rate": sum(r["document"] is not None for r in out_rows) / n if n else None,
        "models_served": sorted({r["model_served"] for r in out_rows}),
        **totals,
    }
    with open(baseline_summary_path(cfg, split), "w", encoding="utf-8", newline="\n") as handle:
        json.dump(summary, handle, indent=2)
    if tracker is not None:
        tracker.log_params({"baseline.prompt_sha256": prompt_hash, "baseline.split": split,
                            "baseline.backend": backend, "baseline.model": model_label})
        tracker.log_metrics({k: v for k, v in summary.items() if isinstance(v, (int, float)) and v is not None})
        tracker.log_artifact(out)
    print(json.dumps(summary, indent=2))
    return summary
