# Configuration Reference Guide

## Overview

This document explains every field in `config.yaml` — the single source of truth for the Falcon-7B fine-tuning pipeline.

---

## `model`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | string | `tiiuae/falcon-7b-instruct` | HuggingFace model hub ID |
| `trust_remote_code` | bool | `true` | Required for Falcon models (custom code in the repo) |
| `use_cache` | bool | `false` | Disable KV-cache during training to save VRAM |

---

## `quantization`

Uses **BitsAndBytes** (NF4 4-bit quantization) to load the 7B model in ~5 GB VRAM.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `load_in_4bit` | bool | `true` | Enable 4-bit weight quantization |
| `bnb_4bit_compute_dtype` | string | `float16` | Compute dtype for matrix multiplications |
| `bnb_4bit_use_double_quant` | bool | `true` | Double-quantize constants for extra memory savings |
| `bnb_4bit_quant_type` | string | `nf4` | NF4 (NormalFloat4) — best accuracy for LLMs |

> **Why NF4?** It's an information-theoretically optimal quantization for normally distributed weights (which LLM weights are), outperforming plain INT4.

---

## `lora`

**LoRA (Low-Rank Adaptation)** injects small trainable rank-decomposition matrices into frozen model layers. Only ~0.1% of parameters are trained.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `r` | int | `16` | LoRA rank — higher = more parameters, more capacity |
| `lora_alpha` | int | `32` | Scaling factor; effective LR ≈ `lora_alpha / r` |
| `target_modules` | list | `[query_key_value]` | Which attention projection layers to inject LoRA into |
| `lora_dropout` | float | `0.05` | Dropout applied to LoRA weights for regularization |
| `bias` | string | `none` | Whether to train bias terms (`none` / `all` / `lora_only`) |
| `task_type` | string | `CAUSAL_LM` | PEFT task type — causal LM for auto-regressive generation |

> **Memory math:** With r=16 and target_modules=['query_key_value'], LoRA adds roughly 4M trainable parameters vs the frozen 7B base.

---

## `dataset`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | string | `GBaker/MedQA-USMLE-4-options` | HuggingFace dataset ID |
| `split` | string | `train` | Which split to load from HF Hub |
| `train_samples` | int | `2000` | Number of training examples (Colab-friendly) |
| `eval_samples` | int | `1000` | Number of validation examples |
| `test_size` | float | `0.2` | Fraction used for eval split |
| `max_length` | int | `512` | Maximum token sequence length |
| `seed` | int | `42` | Reproducibility seed for dataset shuffling |

> **Why 2000/1000?** Colab free tier provides ~12 GB VRAM (T4 GPU) with ~77 GB RAM. 2000 samples gives meaningful fine-tuning signal within the session time limit (~90 min).

---

## `training`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `output_dir` | string | `./outputs/results` | Checkpoint directory |
| `logging_dir` | string | `./outputs/logs` | TensorBoard event files |
| `num_train_epochs` | int | `10` | Maximum training epochs (early stopping may terminate earlier) |
| `per_device_train_batch_size` | int | `2` | Per-GPU batch size |
| `per_device_eval_batch_size` | int | `2` | Per-GPU eval batch size |
| `gradient_accumulation_steps` | int | `4` | Accumulate gradients to simulate effective batch of 8 |
| `learning_rate` | float | `2e-4` | Peak learning rate |
| `lr_scheduler_type` | string | `cosine` | LR warmup then cosine decay |
| `warmup_ratio` | float | `0.05` | 5% of total steps used for LR warmup |
| `weight_decay` | float | `0.01` | L2 regularization on non-bias parameters |
| `fp16` | bool | `true` | Mixed-precision training (requires CUDA GPU) |
| `logging_steps` | int | `10` | Log metrics every N steps |
| `evaluation_strategy` | string | `epoch` | Evaluate at end of each epoch |
| `save_strategy` | string | `epoch` | Save checkpoint at end of each epoch |
| `load_best_model_at_end` | bool | `true` | Restore best checkpoint after training |
| `metric_for_best_model` | string | `eval_loss` | Criterion for "best" model |
| `greater_is_better` | bool | `false` | Lower eval_loss = better |
| `save_total_limit` | int | `2` | Keep only last 2 checkpoints to save disk |
| `report_to` | string | `tensorboard` | Enable TensorBoard logging |
| `dataloader_num_workers` | int | `0` | 0 = safe for Colab (no multi-process issues) |

---

## `early_stopping`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `patience` | int | `3` | Stop if eval_loss doesn't improve for 3 consecutive epochs |
| `threshold` | float | `0.0` | Minimum improvement delta |

---

## `saving`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `final_model_dir` | string | `./outputs/final_model` | Full model + tokenizer (for direct inference) |
| `peft_adapter_dir` | string | `./outputs/peft_adapter` | Lightweight LoRA adapter only (~32 MB) |

> **When to use which?**
> - `final_model` → deploy directly, no base model needed
> - `peft_adapter` → portable, combine with base model at inference time

---

## `evaluation`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `num_samples` | int | `50` | Number of test examples for qualitative eval |
| `max_new_tokens` | int | `200` | Max tokens to generate per answer |
| `temperature` | float | `0.7` | Sampling temperature (0 = greedy) |
| `do_sample` | bool | `true` | Enable sampling (vs greedy decoding) |
| `top_p` | float | `0.9` | Nucleus sampling threshold |
| `repetition_penalty` | float | `1.1` | Penalize repeated tokens in generation |

---

## Colab Free Tier Compatibility

| Constraint | Value |
|------------|-------|
| GPU | NVIDIA T4 (15 GB VRAM) |
| RAM | ~12–13 GB usable |
| Session limit | ~90 min active |
| Model VRAM (4-bit) | ~5–6 GB |
| Training peak VRAM | ~10–12 GB |
| Effective batch size | 2 × 4 = 8 |

> **Tip:** Enable Colab Pro or "High-RAM" runtime if you hit OOM errors. With the default config, the pipeline fits comfortably on a free T4.
