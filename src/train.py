"""
src/train.py — Build Trainer, run training, save model and adapter
"""

import os
import time
import shutil
from pathlib import Path

from transformers import (
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)


def build_trainer(model, cfg: dict, train_dataset, eval_dataset, data_collator) -> Trainer:
    """
    Construct HuggingFace Trainer with all settings from config.
    Includes EarlyStoppingCallback and TensorBoard reporting.
    """
    t   = cfg["training"]
    es  = cfg["early_stopping"]

    # logging_dir is deprecated in transformers v5.2 — use env var instead
    os.environ["TENSORBOARD_LOGGING_DIR"] = "/kaggle/working/logs"

    training_args = TrainingArguments(
        output_dir=t["output_dir"],
        num_train_epochs=t["num_train_epochs"],
        per_device_train_batch_size=t["per_device_train_batch_size"],
        per_device_eval_batch_size=t["per_device_eval_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        learning_rate=t["learning_rate"],
        lr_scheduler_type=t["lr_scheduler_type"],
        warmup_steps=t["warmup_steps"],   # replaces deprecated warmup_ratio
        weight_decay=t["weight_decay"],
        fp16=t["fp16"],
        logging_steps=t["logging_steps"],
        eval_strategy=t["evaluation_strategy"],
        save_strategy=t["save_strategy"],
        load_best_model_at_end=t["load_best_model_at_end"],
        metric_for_best_model=t["metric_for_best_model"],
        greater_is_better=t["greater_is_better"],
        save_total_limit=t["save_total_limit"],
        report_to=t["report_to"],
        label_names=t["label_names"],
        dataloader_num_workers=t["dataloader_num_workers"],
        remove_unused_columns=False,
    )

    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=es["patience"],
        early_stopping_threshold=es["threshold"],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=[early_stopping],
    )

    eff_batch = t["per_device_train_batch_size"] * t["gradient_accumulation_steps"]
    print(f"✅ Trainer configured")
    print(f"   Effective batch size : {eff_batch}")
    print(f"   Early stop patience  : {es['patience']} epochs")
    print(f"   LR scheduler         : {t['lr_scheduler_type']}")
    return trainer


def run_training(trainer: Trainer, cfg: dict) -> dict:
    """
    Run training loop and return metrics dict.
    Prints a formatted summary on completion.
    """
    print("\n" + "═" * 60)
    print("🚀 STARTING FINE-TUNING")
    print("═" * 60)

    t0 = time.time()
    result = trainer.train()
    elapsed = time.time() - t0

    h, rem = divmod(int(elapsed), 3600)
    m, s   = divmod(rem, 60)

    print("\n" + "═" * 60)
    print(f"✅ TRAINING COMPLETE — {h:02d}h {m:02d}m {s:02d}s")
    print("═" * 60)

    metrics = result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    print(f"\n  Train loss       : {metrics.get('train_loss', 'N/A'):.4f}")
    print(f"  Steps per second : {metrics.get('train_steps_per_second', 'N/A'):.2f}")
    return metrics


def save_artifacts(trainer: Trainer, model, tokenizer, cfg: dict):
    """
    Save:
      1. Full fine-tuned model + tokenizer  → final_model_dir
      2. Lightweight LoRA adapter           → peft_adapter_dir
      3. Config used for this run           → final_model_dir/training_config.yaml
      4. Final eval metrics
    """
    final_dir = cfg["saving"]["final_model_dir"]
    peft_dir  = cfg["saving"]["peft_adapter_dir"]

    print("\n📦 Saving artifacts...")

    # 1. Full model
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"  ✅ Full model  → {final_dir}")

    # 2. LoRA adapter
    model.save_pretrained(peft_dir)
    tokenizer.save_pretrained(peft_dir)
    print(f"  ✅ LoRA adapter → {peft_dir}  (~32 MB)")

    # 3. Stamp config used
    shutil.copy("configs/config.yaml", f"{final_dir}/training_config.yaml")

    # 4. Final eval
    eval_metrics = trainer.evaluate()
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)

    import math
    print(f"\n  Final eval loss  : {eval_metrics['eval_loss']:.4f}")
    print(f"  Final perplexity : {math.exp(eval_metrics['eval_loss']):.2f}")
    print("\n✅ All artifacts saved!")
    return eval_metrics
