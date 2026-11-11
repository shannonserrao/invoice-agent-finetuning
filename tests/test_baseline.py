import io
import json
from types import SimpleNamespace

import pytest

from docextract import baseline as B

DOC = {"vendor_name": "Shop", "date": None, "currency": None, "line_items": [], "subtotal": None, "tax": None,
       "total": 9.0}


def png(color):
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (8, 8), color).save(buf, "PNG")
    return buf.getvalue()


class FakeClient:
    def __init__(self, texts, stop_reason="end_turn"):
        self.texts, self.calls, self.stop_reason = list(texts), [], stop_reason
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self.create))

    def create(self, **kw):
        self.calls.append(kw)
        text = self.texts[len(self.calls) - 1]
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)], model=kw["model"],
                               stop_reason=self.stop_reason,
                               usage=SimpleNamespace(input_tokens=1000, output_tokens=200))


@pytest.fixture
def cfg(tmp_path):
    (tmp_path / "prompt.txt").write_text("Extract it.", encoding="utf-8")
    (tmp_path / "schema.json").write_text(json.dumps({"title": "T"}), encoding="utf-8")
    proc = tmp_path / "data" / "processed"
    (proc / "images" / "cord").mkdir(parents=True)
    rows = []
    for i, color in enumerate([(1, 1, 1), (2, 2, 2)]):
        (proc / "images" / "cord" / f"{i}.png").write_bytes(png(color))
        rows.append({"id": f"cord-test-{i}", "dataset": "cord", "image": f"images/cord/{i}.png", "document": DOC})
    (proc / "test.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return {"paths": {"data": str(tmp_path / "data"), "artifacts": str(tmp_path / "art")},
            "prompt": {"file": str(tmp_path / "prompt.txt"), "schema_file": str(tmp_path / "schema.json")},
            "predict": {"split": "test"},
            "baseline": {"model": "claude-opus-5-5", "effort": "medium", "max_tokens": 100,
                         "refusal_fallback": True, "usd_per_1m_input_tokens": 4.0,
                         "usd_per_1m_output_tokens": 20.0}}


def test_prompt_has_instruction_and_schema(cfg):
    p = B.build_prompt(cfg)
    assert p.startswith("Extract it.") and '{"title":"T"}' in p


@pytest.mark.parametrize("text,valid,has_doc", [
    (json.dumps(DOC), True, True),
    ("```json\n" + json.dumps(DOC) + "\n```", False, True),
    ("Here you go: " + json.dumps(DOC) + " done", False, True),
    ("not json", False, False), ("[1, 2]", False, False), ("", False, False),
])
def test_parse_output(text, valid, has_doc):
    doc, raw_valid = B.parse_output(text)
    assert raw_valid is valid and (doc is not None) is has_doc


def test_run_costs_and_cache(cfg):
    client = FakeClient([json.dumps(DOC), "```json\n" + json.dumps(DOC) + "```"])
    s = B.baseline(cfg, client=client)
    call = client.calls[0]
    assert call["model"] == "claude-opus-5-5" and call["output_config"] == {"effort": "medium"}
    assert call["fallbacks"] == "default" and call["messages"][0]["content"][0]["source"]["media_type"] == "image/png"
    assert s["api_calls"] == 2 and s["cache_hits"] == 0 and s["n"] == 2
    assert s["input_tokens"] == 2000 and s["output_tokens"] == 400
    assert s["baseline_usd_total"] == pytest.approx((2000 * 4 + 400 * 20) / 1e6)
    assert s["baseline_usd_per_1k"] == pytest.approx(s["baseline_usd_per_doc"] * 1000)
    assert s["raw_json_validity"] == 0.5 and s["parsed_rate"] == 1.0
    rows = [json.loads(line) for line in open(f"{cfg['paths']['artifacts']}/baseline_predictions.jsonl")]
    assert [r["id"] for r in rows] == ["cord-test-0", "cord-test-1"] and rows[0]["document"] == DOC

    again = B.baseline(cfg, client=FakeClient([]))        # all cached: no API call, same cost and outputs
    assert again["api_calls"] == 0 and again["cache_hits"] == 2
    assert again["baseline_usd_total"] == s["baseline_usd_total"]


def test_cache_invalidated_by_prompt_change(cfg, tmp_path):
    B.baseline(cfg, client=FakeClient([json.dumps(DOC)] * 2))
    (tmp_path / "prompt.txt").write_text("Different.", encoding="utf-8")
    client = FakeClient([json.dumps(DOC)] * 2)
    assert B.baseline(cfg, client=client)["api_calls"] == 2


def test_refusal_counts_and_scores_unparsed(cfg):
    s = B.baseline(cfg, client=FakeClient(["", ""], stop_reason="refusal"))
    assert s["refusals"] == 2 and s["parsed_rate"] == 0.0


def test_requires_model_and_prices(cfg):
    cfg["baseline"]["model"] = None
    with pytest.raises(ValueError, match="baseline.model"):
        B.baseline(cfg, client=FakeClient([]))
    cfg["baseline"].update(model="m", usd_per_1m_input_tokens=None)
    with pytest.raises(ValueError, match="usd_per_1m"):
        B.baseline(cfg, client=FakeClient([]))


def test_baseline_output_feeds_evaluate(cfg):
    from docextract import evaluate as E

    B.baseline(cfg, client=FakeClient([json.dumps(DOC)] * 2))
    cfg2 = {**cfg, "eval": {"amount_tolerance": 0.01, "key_fields": ["total"]},
            "routing": {"select": "fixed", "confidence_threshold": 0.5, "required_checks": ["schema_valid"]}}
    gold = [json.loads(line) for line in open(f"{cfg['paths']['data']}/processed/test.jsonl")]
    pred = [json.loads(line) for line in open(f"{cfg['paths']['artifacts']}/baseline_predictions.jsonl")]
    assert E.evaluate(gold, pred, cfg2)["metrics"]["field_f1"] == 1.0


# ----------------------------------------------------------------------------- images, backends, workers

def noisy_png(size, seed=0):
    import random

    from PIL import Image

    rng = random.Random(seed)
    img = Image.new("RGB", size)
    img.putdata([(rng.randrange(256), rng.randrange(256), rng.randrange(256)) for _ in range(size[0] * size[1])])
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def test_prepare_image_keeps_small_images_untouched():
    data = png((5, 5, 5))
    out, media, prepared = B.prepare_image(data, "png", max_bytes=10_000, max_edge=100)
    assert out == data and media == "image/png" and prepared is False


def test_prepare_image_shrinks_by_edge_and_by_bytes():
    from PIL import Image

    big = noisy_png((400, 300))
    out, media, prepared = B.prepare_image(big, "png", max_bytes=10_000_000, max_edge=100)   # too wide only
    assert prepared and media == "image/png" and max(Image.open(io.BytesIO(out)).size) == 100
    out, media, prepared = B.prepare_image(big, "png", max_bytes=20_000, max_edge=10_000)    # too many bytes
    assert prepared and len(out) <= 20_000
    assert media in ("image/png", "image/jpeg")
    assert B.prepare_image(big, "png", 20_000, 10_000) == B.prepare_image(big, "png", 20_000, 10_000)   # deterministic


def test_oversized_image_is_resized_before_sending(cfg):
    from pathlib import Path

    first = sorted(Path(cfg["paths"]["data"], "processed", "images", "cord").glob("*.png"))[0]
    first.write_bytes(noisy_png((300, 200)))
    cfg["baseline"]["max_image_bytes"] = 5_000
    client = FakeClient([json.dumps(DOC)] * 2)
    s = B.baseline(cfg, client=client)
    assert s["images_resized"] == 1
    sent = [c["messages"][0]["content"][0]["source"] for c in client.calls]
    assert sorted(len(base64_decode(x["data"])) <= 5_000 for x in sent) == [True, True]


def base64_decode(text):
    import base64

    return base64.b64decode(text)


class FakeEndpoint:
    """Stands in for the MLflow deployments client of a Databricks serving endpoint."""

    def __init__(self, fail_first=0):
        self.calls, self.fail_first = [], fail_first

    def predict(self, endpoint, inputs):
        self.calls.append({"endpoint": endpoint, "inputs": inputs})
        if len(self.calls) <= self.fail_first:
            raise ConnectionError("temporary")
        return {"choices": [{"message": {"content": json.dumps(DOC)}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 700, "completion_tokens": 90}, "model": "claude-served"}


def test_databricks_backend_request_and_parsing(cfg):
    cfg["baseline"].update(backend="databricks", endpoint="my-claude", model=None)
    client = FakeEndpoint()
    s = B.baseline(cfg, client=client)
    call = client.calls[0]
    assert call["endpoint"] == "my-claude" and call["inputs"]["temperature"] == 0
    parts = call["inputs"]["messages"][0]["content"]
    assert parts[0]["image_url"]["url"].startswith("data:image/png;base64,") and parts[1]["type"] == "text"
    assert s["backend"] == "databricks" and s["model"] == "my-claude" and s["models_served"] == ["claude-served"]
    assert s["input_tokens"] == 1400 and s["output_tokens"] == 180 and s["parsed_rate"] == 1.0


def test_databricks_backend_retries_then_gives_up(cfg, monkeypatch):
    monkeypatch.setattr(B.time, "sleep", lambda _s: None)
    cfg["baseline"].update(backend="databricks", endpoint="e", model=None, retries=3)
    flaky = FakeEndpoint(fail_first=2)
    assert B.baseline(cfg, client=flaky)["api_calls"] == 2 and len(flaky.calls) == 4     # 2 retries on the first doc
    cfg2 = {**cfg, "paths": {**cfg["paths"], "artifacts": cfg["paths"]["artifacts"] + "2"}}
    with pytest.raises(RuntimeError, match="failed after retries"):
        B.baseline(cfg2, client=FakeEndpoint(fail_first=99))


def test_databricks_backend_needs_endpoint(cfg):
    cfg["baseline"].update(backend="databricks", endpoint=None, model=None)
    with pytest.raises(ValueError, match="endpoint"):
        B.baseline(cfg, client=FakeEndpoint())


def test_workers_keep_row_order_and_count_calls(cfg):
    from pathlib import Path

    for i in range(6):                                   # more documents so threads actually interleave
        Path(cfg["paths"]["data"], "processed", "images", "cord", f"x{i}.png").write_bytes(png((10 + i, 0, 0)))
    rows = [json.loads(line) for line in open(Path(cfg["paths"]["data"], "processed", "test.jsonl"))]
    rows += [{"id": f"cord-test-x{i}", "dataset": "cord", "image": f"images/cord/x{i}.png", "document": DOC}
             for i in range(6)]
    Path(cfg["paths"]["data"], "processed", "test.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    class Keyed:
        beta = SimpleNamespace(messages=SimpleNamespace(create=lambda **kw: SimpleNamespace(
            content=[SimpleNamespace(type="text", text=json.dumps(DOC))], model=kw["model"], stop_reason="end_turn",
            usage=SimpleNamespace(input_tokens=10, output_tokens=5))))

    cfg["baseline"]["workers"] = 4
    s = B.baseline(cfg, client=Keyed())
    out = [json.loads(line)["id"] for line in open(B.baseline_predictions_path(cfg))]
    assert out == [r["id"] for r in rows] and s["api_calls"] == 8 and s["input_tokens"] == 80


def test_limit_marks_run_incomplete_and_summary_is_written(cfg):
    s = B.baseline(cfg, client=FakeClient([json.dumps(DOC)]), limit=1)
    assert s["n"] == 1 and s["expected_n"] == 2 and s["limit"] == 1 and s["complete"] is False
    saved = json.load(open(B.baseline_summary_path(cfg)))
    assert saved["complete"] is False and saved["prompt_sha256"] == B.prompt_sha256(cfg)
    full = B.baseline(cfg, client=FakeClient([json.dumps(DOC)]))
    assert full["complete"] is True and full["cache_hits"] == 1 and full["api_calls"] == 1     # reuses the dry run


def test_non_test_split_uses_split_named_files(cfg):
    assert B.baseline_predictions_path(cfg).endswith("baseline_predictions.jsonl")
    assert B.baseline_predictions_path(cfg, "test_sroie").endswith("baseline_predictions_test_sroie.jsonl")
    assert B.baseline_summary_path(cfg, "validation").endswith("baseline_summary_validation.json")


def test_evaluate_files_takes_baseline_cost_from_summary(cfg):
    from docextract import evaluate as E

    B.baseline(cfg, client=FakeClient([json.dumps(DOC)] * 2))
    pred = [{"id": f"cord-test-{i}", "document": DOC, "raw_valid_json": True, "confidence": 0.99, "seconds": 1.0}
            for i in range(2)]
    open(f"{cfg['paths']['artifacts']}/predictions.jsonl", "w").write("\n".join(json.dumps(p) for p in pred) + "\n")
    cfg.update(eval={"amount_tolerance": 0.01, "key_fields": ["total"]},
               routing={"select": "fixed", "confidence_threshold": 0.5, "required_checks": ["schema_valid"]},
               cost={"gpu_usd_per_hour": 1.0, "small_docs_per_second": 2.0},
               business={"docs_per_month": 100, "manual_minutes_per_doc": 3, "review_minutes_per_doc": 1,
                         "clerk_usd_per_hour": 20, "undetected_error_usd": 15})
    report = E.evaluate_files(cfg)
    per_doc = json.load(open(B.baseline_summary_path(cfg)))["baseline_usd_per_doc"]
    assert report["metrics"]["baseline_usd_per_1k"] == pytest.approx(per_doc * 1000)
