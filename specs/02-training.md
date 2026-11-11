# Spec 02 — Training

## 1. Goal
Fine-tune a small vision-language model with LoRA to emit the canonical JSON, reproducibly and within the compute budget.

## 2. Inputs and outputs
Inputs: processed splits and images; config sections `model`, `lora`, `train`.
Outputs under `artifacts/model/`: LoRA adapter weights and config, processor files, `prompt.txt`, `training_meta.json` (git SHA, data version, config, base model ID and revision).

## 3. Example format
- Prompt: fixed instruction in `configs/prompt.txt` naming the fields and stating "return JSON only".
- Target: the canonical JSON serialised with fixed key order, no extra whitespace (`schema.to_target_text`).
- Loss is computed on the target tokens only.

## 4. Procedure
1. Seeds from `train.seed`.
2. Load the base model, in 4-bit when `model.load_in_4bit` is true.
3. Attach LoRA to the modules in `lora.target_modules` with rank `lora.r`.
4. Cap images at `model.max_image_pixels`, preserving aspect ratio.
5. AdamW, cosine decay with warm-up, gradient accumulation to `train.effective_batch_size`, mixed precision.
6. Every `train.eval_every_steps`, generate on `train.eval_subset` validation documents and compute field F1; keep the best adapter.
7. Save a resumable checkpoint every `train.save_every_steps`.

## 5. Logging
Per spec 04: loss, learning rate, validation field F1 and JSON validity, seconds, peak GPU memory, trainable parameter count.

## 6. Search
One-at-a-time sweeps on validation only: `lora.r` in {8, 16, 32}; `train.lr` in {5e-5, 1e-4, 2e-4}; image cap at two sizes. Record each conclusion in `CHANGELOG.md`.

## 7. Acceptance
- NFR-2 budget met on the reference GPU.
- Resume reproduces the uninterrupted validation score within noise.
- Loading base plus adapter in a fresh process reproduces validation field F1 within 0.5 points.
