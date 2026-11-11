"""Zero-shot baseline with the expensive model (specs/03-evaluation.md §5, task T-06).

The prompt holds the schema and the same instruction the fine-tuned model gets. Responses are cached on disk by
model, prompt hash and image hash, so reruns are free and the numbers are reproducible. Token usage is stored
with each cached response, so cost is computed from usage whether or not the call was cached.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import time

from .feedback import read_jsonl

MEDIA_TYPES = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg", "gif": "image/gif",
               "webp": "image/webp"}


def build_prompt(cfg: dict) -> str:
    """Shared instruction text plus the JSON Schema. The fine-tuned model's prompt must start from the same text."""
    with open(cfg["prompt"]["file"], encoding="utf-8") as handle:
        instruction = handle.read().strip()
    with open(cfg["prompt"]["schema_file"], encoding="utf-8") as handle:
        schema = json.dumps(json.load(handle), separators=(",", ":"))
    return f"{instruction}\n\nJSON Schema:\n{schema}"


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


def cache_key(model: str, prompt_hash: str, image_hash: str, settings: str) -> str:
    return hashlib.sha256("|".join([model, prompt_hash, image_hash, settings]).encode()).hexdigest()


def _make_client():
    import anthropic

    return anthropic.Anthropic()


def _call(client, bl: dict, prompt: str, image_b64: str, media_type: str) -> dict:
    """One request. Returns {text, input_tokens, output_tokens, model_served, stop_reason}."""
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


def baseline(cfg: dict, tracker=None, client=None, now=time.perf_counter) -> dict:
    """Run the baseline over the configured split. Writes artifacts/baseline_predictions.jsonl."""
    bl = cfg["baseline"]
    if not bl.get("model"):
        raise ValueError("baseline.model is not set in the config")
    price_in, price_out = bl.get("usd_per_1m_input_tokens"), bl.get("usd_per_1m_output_tokens")
    if price_in is None or price_out is None:
        raise ValueError("set baseline.usd_per_1m_input_tokens and baseline.usd_per_1m_output_tokens")
    paths = cfg["paths"]
    split = cfg.get("predict", {}).get("split", "test")
    rows = read_jsonl(os.path.join(paths["data"], "processed", f"{split}.jsonl"))
    if not rows:
        raise FileNotFoundError(f"no rows for split {split!r}: run `make data` first")

    prompt = build_prompt(cfg)
    prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
    settings = json.dumps({k: bl.get(k) for k in ("max_tokens", "effort", "refusal_fallback")}, sort_keys=True)
    cache_dir = os.path.join(paths["artifacts"], bl.get("cache_dir", "baseline_cache"))
    os.makedirs(cache_dir, exist_ok=True)

    out_rows, totals = [], {"input_tokens": 0, "output_tokens": 0, "cache_hits": 0, "api_calls": 0, "refusals": 0}
    for row in rows:
        path = os.path.join(paths["data"], "processed", row["image"])
        with open(path, "rb") as handle:
            data = handle.read()
        key = cache_key(bl["model"], prompt_hash, hashlib.sha256(data).hexdigest(), settings)
        cache_path = os.path.join(cache_dir, f"{key}.json")
        if os.path.exists(cache_path):
            with open(cache_path, encoding="utf-8") as handle:
                result = json.load(handle)
            totals["cache_hits"] += 1
        else:
            client = client or _make_client()
            media_type = MEDIA_TYPES[os.path.splitext(path)[1].lstrip(".").lower()]
            start = now()
            result = _call(client, bl, prompt, base64.standard_b64encode(data).decode(), media_type)
            result["seconds"] = now() - start
            with open(cache_path, "w", encoding="utf-8") as handle:
                json.dump(result, handle)
            totals["api_calls"] += 1
        document, raw_valid = parse_output(result["text"])
        totals["input_tokens"] += result["input_tokens"]
        totals["output_tokens"] += result["output_tokens"]
        totals["refusals"] += result["stop_reason"] == "refusal"
        out_rows.append({"id": row["id"], "document": document, "raw_valid_json": raw_valid,
                         "seconds": result.get("seconds"), "input_tokens": result["input_tokens"],
                         "output_tokens": result["output_tokens"], "model_served": result["model_served"],
                         "stop_reason": result["stop_reason"]})

    os.makedirs(paths["artifacts"], exist_ok=True)
    out = os.path.join(paths["artifacts"], "baseline_predictions.jsonl")
    with open(out, "w", encoding="utf-8", newline="\n") as handle:
        for r in out_rows:
            handle.write(json.dumps(r, ensure_ascii=False) + "\n")

    usd = (totals["input_tokens"] * price_in + totals["output_tokens"] * price_out) / 1_000_000
    n = len(out_rows)
    summary = {
        "split": split, "n": n, "model": bl["model"], "prompt_sha256": prompt_hash,
        "baseline_usd_total": usd, "baseline_usd_per_doc": usd / n,
        "baseline_usd_per_1k": usd / n * 1000,
        "raw_json_validity": sum(r["raw_valid_json"] for r in out_rows) / n,
        "parsed_rate": sum(r["document"] is not None for r in out_rows) / n,
        "models_served": sorted({r["model_served"] for r in out_rows}),
        **totals,
    }
    if tracker is not None:
        tracker.log_params({"baseline.prompt_sha256": prompt_hash, "baseline.split": split})
        tracker.log_metrics({k: v for k, v in summary.items() if isinstance(v, (int, float))})
        tracker.log_artifact(out)
    print(json.dumps(summary, indent=2))
    return summary
