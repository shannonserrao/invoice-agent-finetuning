from docextract.config import flatten, load_config


def test_base_config_loads():
    cfg = load_config("configs/base.yaml")
    assert cfg["tracking"]["experiment"]
    assert 0 < cfg["routing"]["confidence_threshold"] < 1


def test_extends_and_override():
    cfg = load_config("configs/smoke.yaml", ["train.batch_size=2", "train.device=cpu"])
    base = load_config("configs/base.yaml")
    assert cfg["train"]["batch_size"] == 2
    assert cfg["train"]["epochs"] == 1
    assert cfg["lora"] == base["lora"] and cfg["prompt"] == base["prompt"]       # inherited untouched
    assert cfg["model"]["base_id"] != base["model"]["base_id"]                   # smoke swaps in the tiny test model
    assert cfg["model"]["base_id"].startswith("trl-internal-testing/tiny-") and cfg["model"]["revision"]
    assert cfg["train"]["lr"] == base["train"]["lr"]                             # other train keys inherited


def test_platform_configs_resolve():
    for name in ("colab", "kaggle", "databricks"):
        cfg = load_config(f"configs/platform/{name}.yaml")
        assert "model" in cfg and "paths" in cfg


def test_flatten():
    assert flatten({"a": {"b": 1}, "c": 2}) == {"a.b": 1, "c": 2}
