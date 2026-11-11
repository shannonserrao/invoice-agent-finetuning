"""LoRA fine-tune the vision-language model; save the adapter to artifacts/model/ (specs/02-training.md, T-07).

Heavy imports (torch, transformers, peft) are done inside functions so that the pure helpers (`plan`,
`epoch_order`, `collate`, checkpoint discovery) can be imported and tested without them.
Validation-time generation and best-adapter selection are T-08; this module saves checkpoints and the final adapter.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import shutil
import time

from .feedback import read_jsonl
from .schema import to_target_text
from .tracking import git_info


# --------------------------------------------------------------------------- pure helpers

def read_instruction(cfg: dict) -> str:
    with open(cfg["prompt"]["file"], encoding="utf-8") as handle:
        return handle.read().strip()


def plan(n: int, batch_size: int, effective_batch_size: int, epochs: int) -> dict:
    """Step arithmetic. One optimizer step = `accum` micro-batches (the last one of an epoch may be smaller)."""
    if n <= 0:
        raise ValueError("no training rows")
    accum = max(1, effective_batch_size // batch_size)
    micro_per_epoch = math.ceil(n / batch_size)
    steps_per_epoch = math.ceil(micro_per_epoch / accum)
    return {"accum": accum, "micro_per_epoch": micro_per_epoch, "steps_per_epoch": steps_per_epoch,
            "total_steps": steps_per_epoch * epochs}


def epoch_order(n: int, seed: int, epoch: int) -> list[int]:
    """Deterministic shuffle per epoch, so a resumed run sees the same batches as an uninterrupted one."""
    order = list(range(n))
    random.Random(seed * 1_000_003 + epoch).shuffle(order)
    return order


def config_sha(cfg: dict) -> str:
    """Hash of the settings that must not change between a run and its resumption."""
    keys = {k: cfg.get(k) for k in ("model", "lora", "prompt")}
    keys["train"] = {k: v for k, v in cfg.get("train", {}).items() if k not in ("max_steps", "log_every_steps")}
    keys["data_version"] = cfg.get("data", {}).get("version")
    return hashlib.sha256(json.dumps(keys, sort_keys=True, default=str).encode()).hexdigest()[:16]


def latest_checkpoint(ckpt_dir: str) -> str | None:
    if not os.path.isdir(ckpt_dir):
        return None
    steps = [(int(m.group(1)), name) for name in os.listdir(ckpt_dir)
             if (m := re.fullmatch(r"step-(\d+)", name)) and os.path.exists(os.path.join(ckpt_dir, name, "state.json"))]
    return os.path.join(ckpt_dir, max(steps)[1]) if steps else None


def prune_checkpoints(ckpt_dir: str, keep: int = 2) -> None:
    steps = sorted(int(m.group(1)) for name in os.listdir(ckpt_dir) if (m := re.fullmatch(r"step-(\d+)", name)))
    for step in steps[:-keep]:
        shutil.rmtree(os.path.join(ckpt_dir, f"step-{step}"), ignore_errors=True)


def collate(examples: list[dict], pad_id: int) -> dict:
    """Right-pad token tensors; concatenate the image tensors Qwen-VL expects."""
    import torch

    longest = max(len(e["input_ids"]) for e in examples)
    ids = torch.full((len(examples), longest), pad_id, dtype=torch.long)
    labels = torch.full((len(examples), longest), -100, dtype=torch.long)
    attn = torch.zeros((len(examples), longest), dtype=torch.long)
    for i, e in enumerate(examples):
        n = len(e["input_ids"])
        ids[i, :n], labels[i, :n], attn[i, :n] = e["input_ids"], e["labels"], 1
    return {"input_ids": ids, "labels": labels, "attention_mask": attn,
            "pixel_values": torch.cat([e["pixel_values"] for e in examples]),
            "image_grid_thw": torch.cat([e["image_grid_thw"] for e in examples])}


def build_example(processor, image, prompt: str, target: str) -> dict:
    """Chat-format prompt with one image; labels are -100 everywhere except the target tokens and end token."""
    import torch

    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    enc = processor(text=[text], images=[image], return_tensors="pt")
    tokenizer = processor.tokenizer
    tgt = tokenizer(target + tokenizer.eos_token, add_special_tokens=False, return_tensors="pt").input_ids[0]
    prompt_ids = enc["input_ids"][0]
    return {
        "input_ids": torch.cat([prompt_ids, tgt]),
        "labels": torch.cat([torch.full_like(prompt_ids, -100), tgt]),
        "pixel_values": enc["pixel_values"], "image_grid_thw": enc["image_grid_thw"],
    }


# --------------------------------------------------------------------------- model

def _device(cfg: dict) -> str:
    import torch

    want = cfg["train"].get("device", "auto")
    return ("cuda" if torch.cuda.is_available() else "cpu") if want == "auto" else want


def load_model_and_processor(cfg: dict, device: str):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

    m = cfg["model"]
    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    compute = torch.bfloat16 if use_bf16 else torch.float16 if device == "cuda" else torch.float32
    kwargs = {"revision": m.get("revision")}
    quantised = False
    if m.get("load_in_4bit") and device == "cuda":
        try:
            import bitsandbytes  # noqa: F401
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=compute)
            kwargs["device_map"] = {"": 0}
            quantised = True
        except ImportError:
            print("WARNING: bitsandbytes not available, loading in 16-bit instead of 4-bit")
    if not quantised:
        kwargs["dtype"] = compute
    model = AutoModelForImageTextToText.from_pretrained(m["base_id"], **kwargs)
    if not quantised:
        model.to(device)
    processor = AutoProcessor.from_pretrained(m["base_id"], revision=m.get("revision"),
                                              max_pixels=m["max_image_pixels"])
    return model, processor, quantised, compute


def attach_lora(model, cfg: dict, quantised: bool):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    t = cfg["train"]
    use_gc = bool(t.get("gradient_checkpointing", False))
    if quantised:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=use_gc)
    elif use_gc:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    model.config.use_cache = False
    lc = cfg["lora"]
    model = get_peft_model(model, LoraConfig(r=lc["r"], lora_alpha=lc["alpha"], lora_dropout=lc["dropout"],
                                             target_modules=lc["target_modules"], task_type="CAUSAL_LM"))
    return model


def trainable_parameters(model) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, sum(p.numel() for p in model.parameters())


# --------------------------------------------------------------------------- checkpoints

def save_checkpoint(ckpt_dir, step, model, optimizer, scheduler, scaler, state: dict) -> str:
    import torch

    path = os.path.join(ckpt_dir, f"step-{step}")
    os.makedirs(path, exist_ok=True)
    model.save_pretrained(path)
    torch.save({"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict() if scaler else None, "rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},
               os.path.join(path, "training_state.pt"))
    with open(os.path.join(path, "state.json"), "w", encoding="utf-8") as handle:  # written last: marks it complete
        json.dump({**state, "step": step}, handle)
    prune_checkpoints(ckpt_dir)
    return path


def load_checkpoint(path, model, optimizer, scheduler, scaler) -> dict:
    import torch
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file

    set_peft_model_state_dict(model, load_file(os.path.join(path, "adapter_model.safetensors")))
    blob = torch.load(os.path.join(path, "training_state.pt"), weights_only=False)
    optimizer.load_state_dict(blob["optimizer"])
    scheduler.load_state_dict(blob["scheduler"])
    if scaler and blob["scaler"]:
        scaler.load_state_dict(blob["scaler"])
    torch.set_rng_state(blob["rng"])
    if blob["cuda_rng"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(blob["cuda_rng"])
    with open(os.path.join(path, "state.json"), encoding="utf-8") as handle:
        return json.load(handle)


# --------------------------------------------------------------------------- training

def train(cfg: dict, tracker) -> str:
    import torch
    from PIL import Image
    from transformers import get_cosine_schedule_with_warmup

    t, paths = cfg["train"], cfg["paths"]
    root = os.path.join(paths["data"], "processed")
    rows = read_jsonl(os.path.join(root, "train.jsonl"))
    if not rows:
        raise FileNotFoundError("no training rows: run `make data` first")
    # The test split is never read here (CLAUDE.md rule).

    seed = t["seed"]
    random.seed(seed)
    torch.manual_seed(seed)
    device = _device(cfg)
    prompt = read_instruction(cfg)
    model, processor, quantised, compute = load_model_and_processor(cfg, device)
    model = attach_lora(model, cfg, quantised)
    n_trainable, n_total = trainable_parameters(model)
    tracker.log_params({"trainable_parameters": n_trainable, "total_parameters": n_total, "device": device,
                        "quantised_4bit": quantised, "compute_dtype": str(compute)})

    p = plan(len(rows), t["batch_size"], t["effective_batch_size"], t["epochs"])
    total_steps = min(p["total_steps"], t["max_steps"]) if t.get("max_steps") else p["total_steps"]
    params = [x for x in model.parameters() if x.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=t["lr"], weight_decay=t.get("weight_decay", 0.0))
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(total_steps * t["warmup_ratio"]), total_steps)
    scaler = torch.amp.GradScaler("cuda") if device == "cuda" and compute == torch.float16 else None

    ckpt_dir = os.path.join(paths["artifacts"], "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    sha = config_sha(cfg)
    step, epoch, micro_done = 0, 0, 0
    resume = latest_checkpoint(ckpt_dir)
    if resume:
        state = load_checkpoint(resume, model, optimizer, scheduler, scaler)
        if state["config_sha"] != sha:
            raise ValueError(f"checkpoint {resume} was made with different settings: delete {ckpt_dir} or revert")
        step, epoch, micro_done = state["step"], state["epoch"], state["micro_done"]
        print(f"resumed from {resume} at step {step}")

    def example(row):
        image = Image.open(os.path.join(root, row["image"])).convert("RGB")
        return build_example(processor, image, prompt, to_target_text(row["document"]))

    pad_id = processor.tokenizer.pad_token_id if processor.tokenizer.pad_token_id is not None else 0
    model.train()
    started, window_loss, window_n = time.time(), 0.0, 0
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    while step < total_steps and epoch < t["epochs"]:
        order = epoch_order(len(rows), seed, epoch)
        batches = [order[i:i + t["batch_size"]] for i in range(0, len(order), t["batch_size"])]
        while micro_done < len(batches) and step < total_steps:
            group = batches[micro_done:micro_done + p["accum"]]
            for idx in group:
                batch = collate([example(rows[i]) for i in idx], pad_id)
                batch = {k: v.to(device) for k, v in batch.items()}
                with torch.autocast(device_type=device, dtype=compute, enabled=device == "cuda"):
                    loss = model(**batch).loss / len(group)
                (scaler.scale(loss) if scaler else loss).backward()
                window_loss += loss.item() * len(group)
                window_n += 1
            micro_done += len(group)
            if scaler:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, t.get("max_grad_norm", 1.0))
            (scaler.step(optimizer), scaler.update()) if scaler else optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            if step % t["log_every_steps"] == 0 or step == total_steps:
                metrics = {"train_loss": window_loss / max(window_n, 1), "lr": scheduler.get_last_lr()[0],
                           "seconds": time.time() - started, "epoch": epoch + micro_done / len(batches)}
                if device == "cuda":
                    metrics["peak_gpu_gb"] = torch.cuda.max_memory_allocated() / 1e9
                tracker.log_metrics(metrics, step=step)
                print(f"step {step}/{total_steps} " + " ".join(f"{k}={v:.4g}" for k, v in metrics.items()))
                window_loss, window_n = 0.0, 0
            if step % t["save_every_steps"] == 0 and step < total_steps:
                save_checkpoint(ckpt_dir, step, model, optimizer, scheduler, scaler,
                                {"epoch": epoch, "micro_done": micro_done, "config_sha": sha})
        if micro_done >= len(batches):
            epoch, micro_done = epoch + 1, 0

    return save_final(cfg, model, processor, prompt, step, total_steps, n_trainable, n_total, time.time() - started,
                      tracker)


def save_final(cfg, model, processor, prompt, step, total_steps, n_trainable, n_total, seconds, tracker) -> str:
    out = os.path.join(cfg["paths"]["artifacts"], "model")
    os.makedirs(out, exist_ok=True)
    model.save_pretrained(out)
    processor.save_pretrained(out)
    with open(os.path.join(out, "prompt.txt"), "w", encoding="utf-8") as handle:
        handle.write(prompt + "\n")
    manifest_path = os.path.join(cfg["paths"]["data"], "manifest.json")
    data_version = None
    if os.path.exists(manifest_path):
        with open(manifest_path, encoding="utf-8") as handle:
            data_version = json.load(handle).get("data_version")
    meta = {**git_info(), "data_version": data_version, "base_model": cfg["model"]["base_id"],
            "base_revision": cfg["model"].get("revision"), "steps": step, "planned_steps": total_steps,
            "trainable_parameters": n_trainable, "total_parameters": n_total, "train_seconds": seconds,
            "config": cfg}
    with open(os.path.join(out, "training_meta.json"), "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2, default=str)
    tracker.log_metrics({"train_seconds": seconds, "trainable_parameters": n_trainable})
    tracker.log_artifact(out)
    print(f"saved adapter to {out}")
    return out
