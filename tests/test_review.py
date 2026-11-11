import copy
import csv
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from docextract import feedback
from docextract import review as R
from docextract.schema import schema_errors
from test_schema import DOC

RUNNER = str(Path(__file__).with_name("review_runner.js"))
needs_node = pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")


def js(fn, *args):
    out = subprocess.run(["node", RUNNER], input=json.dumps({"fn": fn, "args": list(args)}), capture_output=True,
                         text=True, encoding="utf-8", check=True)
    return json.loads(out.stdout)


def write_queue(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=feedback.QUEUE_FIELDS)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in feedback.QUEUE_FIELDS})


def queue_row(i, ref="", prediction=None, reason="low_confidence"):
    return {"id": f"doc-{i}", "input_ref": ref, "prediction": json.dumps(prediction or DOC), "confidence": "0.4321",
            "model_version": "v1", "reason": reason}


# ----------------------------------------------------------------------------- page builder

def test_page_embeds_queue_and_relative_image_paths(tmp_path):
    (tmp_path / "data" / "processed" / "images").mkdir(parents=True)
    (tmp_path / "data" / "processed" / "images" / "a b.png").write_bytes(b"\x89PNG")
    queue = tmp_path / "feedback" / "review_queue.csv"
    queue.parent.mkdir()
    evil = dict(DOC, vendor_name="</script><script>alert(1)</script> /*__LOGIC__*/ __QUEUE__")
    write_queue(queue, [queue_row(1, "images/a b.png"), queue_row(2, "images/missing.png", evil, "random_audit")])
    out = tmp_path / "feedback" / "review.html"
    info = R.build_review_page(str(queue), str(out), [str(tmp_path / "data" / "processed")])
    assert info["items"] == 2 and info["images_missing"] == 1
    page = out.read_text(encoding="utf-8")
    assert "ReviewLogic" in page
    assert page.count("/*__LOGIC__*/") == 1                # only the user-supplied text, not the template
    assert page.count("</script>") == 3                    # queue, logic, app: the payload added no closing tag
    blob = page.split('id="queue">')[1].split("</script>")[0]
    items = json.loads(blob)
    assert items[0]["src"] == "../data/processed/images/a%20b.png" and items[1]["src"] is None
    assert items[1]["reason"] == "random_audit" and json.loads(items[1]["prediction"])["vendor_name"].startswith("</")


def test_empty_queue_builds(tmp_path):
    queue = tmp_path / "q.csv"
    write_queue(queue, [])
    assert R.build_review_page(str(queue), str(tmp_path / "p.html"))["items"] == 0


# ----------------------------------------------------------------------------- page logic under node

def variants():
    d = copy.deepcopy
    yield DOC
    yield dict(DOC, vendor_name=None, tax=None)
    yield dict(DOC, total="12.96")                                     # string where number expected
    yield dict(DOC, total=True)
    yield {k: v for k, v in DOC.items() if k != "tax"}
    yield dict(DOC, extra=1)
    yield dict(DOC, line_items="none")
    yield dict(DOC, line_items=[1])
    bad = d(DOC)
    bad["line_items"][0]["quantity"] = "2"
    yield bad
    bad = d(DOC)
    del bad["line_items"][1]["amount"]
    yield bad
    bad = d(DOC)
    bad["line_items"][0]["surprise"] = 1
    yield bad
    bad = d(DOC)
    bad["line_items"][0]["description"] = 5
    yield bad
    yield dict(DOC, date=20181225)
    yield []
    yield "text"
    yield None
    yield dict(DOC, line_items=[])


@needs_node
def test_js_schema_check_agrees_with_python():
    for doc in variants():
        assert (js("schemaErrors", doc) == []) == (schema_errors(doc) == []), doc


@needs_node
def test_js_parse_and_edit_detection():
    pretty = json.dumps(DOC, indent=2)
    assert js("parseDocument", pretty)["errors"] == []
    assert "not valid JSON" in js("parseDocument", "{oops")["errors"][0]
    original = json.dumps(DOC)
    assert js("isEdited", original, json.dumps(DOC, indent=4)) is False        # whitespace only
    reordered = {k: DOC[k] for k in reversed(list(DOC))}
    assert js("isEdited", original, json.dumps(reordered)) is False            # key order only
    assert js("isEdited", original, json.dumps(dict(DOC, total=1.0))) is True
    assert js("isEdited", "not json", "not json ") is False and js("isEdited", "not json", "other") is True


@needs_node
def test_js_csv_cell_quoting():
    assert js("csvCell", "plain") == "plain"
    assert js("csvCell", 'a,"b"\nc') == '"a,""b""\nc"'
    assert js("csvCell", None) == ""


@needs_node
def test_export_round_trips_through_real_ingest(tmp_path):
    fixed = dict(DOC, vendor_name="Fixed, \"Quoted\" Shop", total=99.0)
    items = [queue_row(i) for i in range(4)]
    states = {
        "doc-0": {"verdict": "correct", "text": "ignored", "note": ""},
        "doc-1": {"verdict": "incorrect", "text": json.dumps(fixed, indent=2), "note": "wrong vendor,\nsee image"},
        "doc-2": {"verdict": "unsure", "text": "", "note": ""},
        # doc-3 left unreviewed
    }
    result = js("buildExport", items, states, "ana")
    assert result["problems"] == []
    csv_text = js("toCsv", result["rows"])
    reviewed = tmp_path / "reviewed.csv"
    reviewed.write_bytes(csv_text.encode("utf-8"))
    corrections = tmp_path / "corrections.jsonl"
    counts = feedback.ingest_reviewed(str(reviewed), str(corrections))
    assert counts == {"added": 3, "skipped_duplicate": 0, "skipped_blank": 1}
    saved = {r["id"]: r for r in feedback.read_jsonl(str(corrections))}
    assert saved["doc-1"]["verdict"] == "incorrect" and saved["doc-1"]["reviewer"] == "ana"
    assert saved["doc-1"]["note"] == "wrong vendor,\nsee image"
    corrected = json.loads(saved["doc-1"]["corrected"])
    assert corrected == fixed and schema_errors(corrected) == []
    assert saved["doc-0"]["corrected"] == "" and saved["doc-2"]["verdict"] == "unsure"
    assert saved["doc-0"]["prediction"] == items[0]["prediction"]


@needs_node
def test_export_blocks_bad_rows():
    items = [queue_row(0), queue_row(1), queue_row(2)]
    states = {"doc-0": {"verdict": "incorrect", "text": "{oops"},
              "doc-1": {"verdict": "incorrect", "text": json.dumps({"total": 1})},
              "doc-2": {"verdict": "correct"}}
    problems = js("buildExport", items, states, "ana")["problems"]
    assert len(problems) == 2 and "not valid JSON" in problems[0] and "doc-1" in problems[1]
    missing_name = js("buildExport", items[:1], {"doc-0": {"verdict": "correct"}}, "  ")["problems"]
    assert missing_name == ["doc-0: reviewer name is required"]


@needs_node
def test_queue_fields_match_python():
    assert js("QUEUE_FIELDS") == feedback.QUEUE_FIELDS
    assert js("VERDICTS") == sorted(feedback.VERDICTS, key=["correct", "incorrect", "unsure"].index)


@needs_node
def test_pretty_prediction_uses_schema_key_order():
    shuffled = {k: DOC[k] for k in sorted(DOC)}                          # alphabetical, as stored in jsonl files
    pretty = js("prettyPrediction", json.dumps(shuffled))
    assert list(json.loads(pretty)) == ["vendor_name", "date", "currency", "line_items", "subtotal", "tax", "total"]
    assert list(json.loads(pretty)["line_items"][0]) == ["description", "quantity", "unit_price", "amount"]
    assert js("prettyPrediction", "not json") == "not json"
