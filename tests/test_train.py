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


def test_mm_token_type_ids_are_carried_through_example_and_batch():
    torch = pytest.importorskip("torch")

    class WithTypes(FakeProcessor):
        def __call__(self, text, images, return_tensors):
            out = super().__call__(text, images, return_tensors)
            n = out["input_ids"].shape[1]
            out["mm_token_type_ids"] = torch.tensor([[0] * (n - 2) + [1, 1]])       # last two prompt tokens are image
            return out

    ex = T.build_example(WithTypes(), object(), "hi", "ab")
    n_prompt, n_target = len("P:hi"), len("ab<eos>")
    assert ex["mm_token_type_ids"].tolist() == [0] * (n_prompt - 2) + [1, 1] + [0] * n_target
    short = {**ex, "input_ids": ex["input_ids"][:5], "labels": ex["labels"][:5],
             "mm_token_type_ids": ex["mm_token_type_ids"][:5]}
    out = T.collate([ex, short], pad_id=0)
    assert out["mm_token_type_ids"].shape == out["input_ids"].shape
    assert out["mm_token_type_ids"][1, 5:].sum() == 0                              # padding is text type
    assert "mm_token_type_ids" not in T.collate([T.build_example(FakeProcessor(), object(), "hi", "ab")], 0)


def _tiny_cached():
    try:
        from huggingface_hub import try_to_load_from_cache

        path = try_to_load_from_cache("trl-internal-testing/tiny-Qwen2VLForConditionalGeneration", "config.json",
                                      revision="61aad1a820880f84bc4af87262b05945610aa80d")
        return isinstance(path, str)
    except Exception:  # noqa: BLE001
        return False


@pytest.mark.skipif(not _tiny_cached(), reason="tiny test model not in the local Hugging Face cache")
def test_target_only_loss_equals_full_loss_on_tiny_model():
    torch = pytest.importorskip("torch")
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    repo, rev = "trl-internal-testing/tiny-Qwen2VLForConditionalGeneration", "61aad1a820880f84bc4af87262b05945610aa80d"
    model = AutoModelForImageTextToText.from_pretrained(repo, revision=rev).eval()
    proc = AutoProcessor.from_pretrained(repo, revision=rev, max_pixels=50176)
    img = Image.new("RGB", (120, 160), (200, 200, 200))
    ex = T.build_example(proc, img, "Extract it.", '{"total":9.0}')
    batch = T.collate([ex], proc.tokenizer.pad_token_id or 0)
    with torch.no_grad():
        full = model(**batch).loss
        fast = T.target_only_loss(model, batch)
    assert torch.allclose(full, fast, atol=1e-4), (float(full), float(fast))
    assert T.batch_loss(model, batch, target_only=False).item() == pytest.approx(float(full), abs=1e-6)


def test_filter_long_targets_drops_and_counts():
    class Tok(FakeTokenizer):
        def __call__(self, text, add_special_tokens=False, return_tensors=None):
            return type("E", (), {"input_ids": list(range(len(text)))})()

    doc_short = {"vendor_name": None, "date": None, "currency": None, "line_items": [], "subtotal": None,
                 "tax": None, "total": 1.0}
    doc_long = dict(doc_short, line_items=[{"description": "x" * 200, "quantity": 1, "unit_price": 1, "amount": 1}])
    rows = [{"id": "a", "document": doc_short}, {"id": "b", "document": doc_long}]
    assert T.filter_long_targets(rows, Tok(), None) == (rows, 0)
    kept, dropped = T.filter_long_targets(rows, Tok(), 150)
    assert [r["id"] for r in kept] == ["a"] and dropped == 1
    assert T.filter_long_targets(rows, Tok(), 10_000) == (rows, 0)


def test_is_stalled_needs_history_and_a_big_jump():
    assert T.is_stalled([100, 110], 5000, 8) is False                    # not enough history yet
    assert T.is_stalled([100, 110, 105], 400, 8) is False                # slower, but within 8x the median
    assert T.is_stalled([100, 110, 105], 2000, 8) is True
    assert T.is_stalled([100, 110, 105], 99999, None) is False           # guard disabled


@pytest.mark.parametrize("n,chunk", [(5, 64), (64, 64), (130, 64), (1, 64), (37, 8)])
def test_chunked_cross_entropy_equals_full_value_and_gradient(n, chunk):
    torch = pytest.importorskip("torch")
    gen = torch.Generator().manual_seed(0)
    base = torch.randn(n, 50, generator=gen)
    labels = torch.randint(0, 50, (n,), generator=gen)
    a = base.clone().requires_grad_(True)
    b = base.clone().requires_grad_(True)
    full = torch.nn.functional.cross_entropy(a.float(), labels)
    chunked = T.chunked_cross_entropy(b, labels, chunk=chunk)
    assert torch.allclose(full, chunked, atol=1e-6)
    full.backward()
    chunked.backward()
    assert torch.allclose(a.grad, b.grad, atol=1e-6)
    half = base.bfloat16().clone().requires_grad_(True)                    # bf16 logits are upcast per chunk
    assert torch.allclose(T.chunked_cross_entropy(half, labels, chunk=chunk),
                          torch.nn.functional.cross_entropy(half.float(), labels), atol=1e-5)
