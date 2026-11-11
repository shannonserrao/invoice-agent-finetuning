import json
import os

import pytest

from docextract import train as T


def test_plan_arithmetic():
    p = T.plan(n=800, batch_size=2, effective_batch_size=16, epochs=3)
    assert p == {"accum": 8, "micro_per_epoch": 400, "steps_per_epoch": 50, "total_steps": 150}
    assert T.plan(10, 4, 4, 1)["steps_per_epoch"] == 3          # last micro-batch is partial
    assert T.plan(10, 4, 2, 1)["accum"] == 1                     # effective smaller than batch: no accumulation
    with pytest.raises(ValueError):
        T.plan(0, 2, 16, 1)


def test_epoch_order_is_deterministic_permutation():
    a, b = T.epoch_order(50, 13, 0), T.epoch_order(50, 13, 0)
    assert a == b and sorted(a) == list(range(50))
    assert a != T.epoch_order(50, 13, 1) and a != T.epoch_order(50, 14, 0)


def test_config_sha_ignores_run_length_only():
    base = {"model": {"base_id": "m"}, "lora": {"r": 16}, "prompt": {"file": "p"}, "data": {"version": "v1"},
            "train": {"lr": 1e-4, "max_steps": None, "log_every_steps": 10}}
    sha = T.config_sha(base)
    assert T.config_sha({**base, "train": {"lr": 1e-4, "max_steps": 5, "log_every_steps": 1}}) == sha
    assert T.config_sha({**base, "train": {"lr": 2e-4}}) != sha
    assert T.config_sha({**base, "lora": {"r": 8}}) != sha


def test_checkpoint_discovery_and_pruning(tmp_path):
    for step, complete in [(100, True), (200, True), (300, True), (400, False)]:
        d = tmp_path / f"step-{step}"
        d.mkdir()
        if complete:
            (d / "state.json").write_text(json.dumps({"step": step}))
    assert T.latest_checkpoint(str(tmp_path)).endswith("step-300")     # step-400 never finished writing
    T.prune_checkpoints(str(tmp_path), keep=2)
    assert sorted(os.listdir(tmp_path)) == ["step-300", "step-400"]
    assert T.latest_checkpoint(str(tmp_path / "missing")) is None


class FakeTokenizer:
    eos_token = "<eos>"
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=False, return_tensors=None):
        import torch

        return type("E", (), {"input_ids": torch.tensor([[ord(c) for c in text]])})


class FakeProcessor:
    tokenizer = FakeTokenizer()

    def apply_chat_template(self, messages, add_generation_prompt):
        return "P:" + messages[0]["content"][1]["text"]

    def __call__(self, text, images, return_tensors):
        import torch

        return {"input_ids": torch.tensor([[ord(c) for c in text[0]]]), "pixel_values": torch.ones(4, 3),
                "image_grid_thw": torch.tensor([[1, 2, 2]])}


def test_build_example_masks_prompt_and_keeps_target():
    pytest.importorskip("torch")
    ex = T.build_example(FakeProcessor(), object(), "hi", '{"a":1}')
    ids, labels = ex["input_ids"].tolist(), ex["labels"].tolist()
    n_prompt = len("P:hi")
    assert labels[:n_prompt] == [-100] * n_prompt
    assert labels[n_prompt:] == ids[n_prompt:] and len(ids) == n_prompt + len('{"a":1}<eos>')


def test_collate_pads_and_concatenates():
    torch = pytest.importorskip("torch")
    a = {"input_ids": torch.tensor([1, 2, 3]), "labels": torch.tensor([-100, 2, 3]),
         "pixel_values": torch.ones(4, 3), "image_grid_thw": torch.tensor([[1, 2, 2]])}
    b = {"input_ids": torch.tensor([5]), "labels": torch.tensor([5]),
         "pixel_values": torch.ones(2, 3), "image_grid_thw": torch.tensor([[1, 1, 2]])}
    out = T.collate([a, b], pad_id=9)
    assert out["input_ids"].tolist() == [[1, 2, 3], [5, 9, 9]]
    assert out["labels"].tolist() == [[-100, 2, 3], [5, -100, -100]]
    assert out["attention_mask"].tolist() == [[1, 1, 1], [1, 0, 0]]
    assert out["pixel_values"].shape == (6, 3) and out["image_grid_thw"].shape == (2, 3)
