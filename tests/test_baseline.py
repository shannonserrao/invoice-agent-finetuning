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
