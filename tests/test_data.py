import io
import json

import pytest

from docextract import data as D
from docextract.schema import schema_errors

RAW = {"gt_parse": {
    "menu": [
        {"nm": " Nasi  Campur ", "cnt": "2 x", "price": "75,000", "unitprice": "37,500",
         "sub": {"nm": "extra rice"}},
        {"nm": "Tahu", "cnt": "1", "price": "40,000."},
    ],
    "sub_total": {"subtotal_price": "115,000", "tax_price": "11,500", "service_price": "5,750"},
    "total": {"total_price": "126,500"},
}}


def png(color):
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (8, 8), color).save(buf, "PNG")
    return buf.getvalue()


def cfg(tmp_path, **data):
    return {"paths": {"data": str(tmp_path)},
            "data": {"seed": 1, "version": "data-v1", "max_train_samples": None, "max_eval_samples": None,
                     "datasets": {"cord": {"dataset_id": "x/y", "revision": "abc"}}, **data}}


def loader(tmp_splits):
    return {"cord": lambda _cfg: {s: iter(v) for s, v in tmp_splits.items()}}


def splits(train_colors=((1, 1, 1), (2, 2, 2)), val=((3, 3, 3),), test=((4, 4, 4),)):
    rec = lambda c: {"image_bytes": png(c), "raw": RAW}  # noqa: E731
    return {"train": [rec(c) for c in train_colors], "validation": [rec(c) for c in val],
            "test": [rec(c) for c in test]}


def test_map_cord_fields():
    doc = D.map_cord(RAW)
    assert schema_errors(doc) == []
    assert doc["vendor_name"] is None and doc["date"] is None and doc["currency"] is None
    assert doc["line_items"][0] == {"description": "Nasi Campur", "quantity": 2.0, "unit_price": 37500.0,
                                    "amount": 75000.0}
    assert doc["line_items"][1]["amount"] == 40000.0 and doc["line_items"][1]["unit_price"] is None
    assert (doc["subtotal"], doc["tax"], doc["total"]) == (115000.0, 11500.0, 126500.0)


def test_map_cord_dict_menu_and_missing():
    doc = D.map_cord({"menu": {"nm": "Tea", "price": "3,000"}, "total": {"total_price": "3,000"}})
    assert len(doc["line_items"]) == 1 and doc["subtotal"] is None and doc["tax"] is None
    assert D.map_cord({"menu": [], "total": []})["total"] is None


def test_build_is_deterministic(tmp_path):
    c = cfg(tmp_path)
    m1 = D.build_dataset(c, loaders=loader(splits()))
    m2 = D.build_dataset(c, loaders=loader(splits()))
    assert m1 == m2
    assert m1["splits"]["train"]["rows"] == 2
    row = json.loads((tmp_path / "processed" / "train.jsonl").read_text().splitlines()[0])
    assert set(row) == {"id", "dataset", "image", "document"}
    assert (tmp_path / "processed" / row["image"]).exists()
    assert json.loads((tmp_path / "manifest.json").read_text()) == m1


def test_caps(tmp_path):
    m = D.build_dataset(cfg(tmp_path, max_train_samples=1), loaders=loader(splits()))
    assert m["splits"]["train"]["rows"] == 1


def test_leakage_fails(tmp_path):
    with pytest.raises(ValueError, match="shared between"):
        D.build_dataset(cfg(tmp_path), loaders=loader(splits(val=((1, 1, 1),))))


def test_bad_label_fails(tmp_path, monkeypatch):
    monkeypatch.setitem(D.MAPPERS, "cord", lambda raw: {"total": 1})
    with pytest.raises(ValueError, match="fails schema"):
        D.build_dataset(cfg(tmp_path), loaders=loader(splits()))


def test_gold_arithmetic_reported(tmp_path):
    m = D.build_dataset(cfg(tmp_path), loaders=loader(splits()))
    stats = m["gold_arithmetic"]["train"]
    assert stats["total_ok"] == {"applicable": 2, "pass_rate": 1.0}
    assert stats["items_sum_ok"] == {"applicable": 2, "pass_rate": 1.0}


@pytest.mark.parametrize("raw,expected", [
    ("25/12/2018", "2018-12-25"), ("12/25/2018", "2018-12-25"), ("18/03/18", "2018-03-18"),
    ("12-01-19", None), ("05/06/2018", None), ("05/05/2018", "2018-05-05"),
    ("05 MAR 2018", "2018-03-05"), ("28 MAR 18", "2018-03-28"), ("02/JAN/2017", "2017-01-02"),
    ("24-MAR-2018", "2018-03-24"), ("2018-03-23", "2018-03-23"), ("20180304", "2018-03-04"),
    ("2018/02/22", "2018-02-22"), ("(06/12/2016)", None), ("OCT 3, 2016", "2016-10-03"),
    ("31/02/2018", None), ("", None), (None, None), ("garbage", None),
])
def test_parse_date(raw, expected):
    assert D.parse_date(raw) == expected


def test_map_sroie():
    doc = D.map_sroie({"entities": {"company": " BOOK  TA .K SDN BHD", "date": "25/12/2018",
                                    "address": "x", "total": "RM 9.00"}})
    assert schema_errors(doc) == []
    assert doc == {"vendor_name": "BOOK TA .K SDN BHD", "date": "2018-12-25", "currency": None,
                   "line_items": [], "subtotal": None, "tax": None, "total": 9.0}
    assert D.map_sroie({"entities": {"company": "A", "date": "05/06/2018", "total": ""}})["total"] is None


def test_eval_only_dataset_gets_own_file(tmp_path):
    c = cfg(tmp_path)
    c["data"]["datasets"]["sroie"] = {"dataset_id": "a/b", "revision": "r", "eval_only": True}
    ent = {"entities": {"company": "Shop", "date": "25/12/2018", "total": "9.00"}}
    ld = loader(splits())
    ld["sroie"] = lambda _c: {"test": iter([{"image_bytes": png((9, 9, 9)), "raw": ent}])}
    m = D.build_dataset(c, loaders=ld)
    assert m["splits"]["test_sroie"]["rows"] == 1 and m["splits"]["test"]["rows"] == 1
    row = json.loads((tmp_path / "processed" / "test_sroie.jsonl").read_text())
    assert row["id"] == "sroie-test-0000" and row["document"]["vendor_name"] == "Shop"
    ld["sroie"] = lambda _c: {"test": iter([{"image_bytes": png((1, 1, 1)), "raw": ent}])}  # same image as CORD train
    with pytest.raises(ValueError, match="shared between"):
        D.build_dataset(c, loaders={"cord": loader(splits())["cord"], "sroie": ld["sroie"]})


# ----------------------------------------------------------------------------- T-13 corrections

FIXED = {"vendor_name": "Fixed Shop", "date": None, "currency": None, "line_items": [],
         "subtotal": None, "tax": None, "total": 5.0}


def write_corrections(path, items):
    path.write_text("\n".join(json.dumps(i) for i in items) + "\n", encoding="utf-8")


def live_image(tmp_path, name, color):
    path = tmp_path / name
    path.write_bytes(png(color))
    return str(path)


def fb_cfg(tmp_path, items, **extra):
    tmp_path.mkdir(parents=True, exist_ok=True)
    c = cfg(tmp_path / "data")
    corr = tmp_path / "corrections.jsonl"
    write_corrections(corr, items)
    c["feedback"] = {"corrections": str(corr), **extra}
    return c


def inc(item_id, ref="", doc=FIXED, verdict="incorrect"):
    return {"id": item_id, "input_ref": ref, "verdict": verdict, "corrected": json.dumps(doc),
            "prediction": json.dumps(DOC_PRED)}


DOC_PRED = D.map_cord(RAW)


def test_corrections_merge_into_train_only(tmp_path):
    base = D.build_dataset(cfg(tmp_path / "base"), loaders=loader(splits()))
    items = [
        inc("cord-train-0000"),                                                     # relabel an existing train row
        inc("live-1", live_image(tmp_path, "a.png", (50, 50, 50))),                 # new live document
        inc("cord-validation-0000"),                                                # excluded, validation
        inc("cord-test-0000"),                                                      # excluded, test
        inc("live-leak", live_image(tmp_path, "b.png", (4, 4, 4))),                 # same image as the test split
        inc("live-dup", live_image(tmp_path, "c.png", (2, 2, 2))),                  # same image as a train row
        inc("live-missing", str(tmp_path / "nope.png")),
        {"id": "live-unsure", "verdict": "unsure"},
        {"id": "live-ok", "verdict": "correct", "prediction": json.dumps(DOC_PRED),
         "input_ref": live_image(tmp_path, "d.png", (60, 60, 60))},               # confirmed: ignored by default
    ]
    c = fb_cfg(tmp_path, items)
    m = D.build_dataset(c, loaders=loader(splits()))
    train = [json.loads(line) for line in (tmp_path / "data" / "processed" / "train.jsonl").read_text().splitlines()]
    assert [r["id"] for r in train] == ["cord-train-0000", "cord-train-0001", "live-1"]
    assert train[0]["document"] == FIXED and train[2]["dataset"] == "feedback"
    assert (tmp_path / "data" / "processed" / train[2]["image"]).exists()
    assert m["splits"]["validation"] == base["splits"]["validation"] and m["splits"]["test"] == base["splits"]["test"]
    assert m["corrections_applied"] == ["cord-train-0000", "live-1"]
    assert m["corrections_excluded"] == {"validation": ["cord-validation-0000"], "test": ["cord-test-0000"],
                                         "leak": ["live-leak"], "duplicate_of_train": ["live-dup"],
                                         "image_missing": ["live-missing"]}
    assert m["data_version"].startswith("data-v1-fb") and m["data_version"] != base["data_version"]
    assert D.build_dataset(c, loaders=loader(splits())) == m                        # deterministic


def test_version_changes_with_corrections_and_not_without(tmp_path):
    base = D.build_dataset(cfg(tmp_path / "b"), loaders=loader(splits()))
    none = D.build_dataset(fb_cfg(tmp_path / "n", [{"id": "x", "verdict": "unsure"}]), loaders=loader(splits()))
    assert none["data_version"] == base["data_version"] == "data-v1"
    v1 = D.build_dataset(fb_cfg(tmp_path / "1", [inc("cord-train-0000")]), loaders=loader(splits()))
    other = dict(FIXED, total=6.0)
    v2 = D.build_dataset(fb_cfg(tmp_path / "2", [inc("cord-train-0000", doc=other)]), loaders=loader(splits()))
    assert len({base["data_version"], v1["data_version"], v2["data_version"]}) == 3


def test_confirmed_predictions_only_when_enabled(tmp_path):
    item = {"id": "live-ok", "verdict": "correct", "prediction": json.dumps(DOC_PRED),
            "input_ref": live_image(tmp_path, "d.png", (60, 60, 60))}
    off = D.build_dataset(fb_cfg(tmp_path / "off", [item]), loaders=loader(splits()))
    on = D.build_dataset(fb_cfg(tmp_path / "on", [item], include_confirmed=True), loaders=loader(splits()))
    assert off["corrections_applied"] == [] and on["corrections_applied"] == ["live-ok"]
    assert on["splits"]["train"]["rows"] == 3


def test_invalid_correction_fails_the_build(tmp_path):
    bad = {"id": "cord-train-0000", "verdict": "incorrect", "corrected": json.dumps({"total": 1})}
    with pytest.raises(ValueError, match="fails schema"):
        D.build_dataset(fb_cfg(tmp_path, [bad]), loaders=loader(splits()))


def test_corrections_survive_sample_caps(tmp_path):
    items = [inc("live-1", live_image(tmp_path, "a.png", (50, 50, 50)))]
    m = D.build_dataset(fb_cfg(tmp_path, items), loaders=loader(splits()))
    capped_cfg = fb_cfg(tmp_path / "c", items)
    capped_cfg["data"]["max_train_samples"] = 1
    capped = D.build_dataset(capped_cfg, loaders=loader(splits()))
    assert m["splits"]["train"]["rows"] == 3 and capped["splits"]["train"]["rows"] == 2
