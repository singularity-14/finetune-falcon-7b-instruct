"""
src/evaluate.py — Perplexity, Multiple-Choice Accuracy, ROUGE, Qualitative examples
"""

import json
import math
import numpy as np
import torch
from pathlib import Path
from tqdm.auto import tqdm


# ── Inference helper ───────────────────────────────────────────────────────────

def generate_answer(model, tokenizer, question: str, cfg: dict, device: str = "cuda") -> str:
    """Generate a free-text answer for a question using the fine-tuned model."""
    e = cfg["evaluation"]
    prompt = (
        f"<|system|>You are a medical expert specializing in USMLE board exam questions.\n"
        f"<|user|>Question: {question}\n"
        f"<|assistant|>Answer:"
    )
    inputs = tokenizer(
        prompt, return_tensors="pt", max_length=400, truncation=True
    ).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=e["max_new_tokens"],
            temperature=e["temperature"],
            do_sample=e["do_sample"],
            top_p=e["top_p"],
            repetition_penalty=e["repetition_penalty"],
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    gen = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen, skip_special_tokens=True).strip()


# ── Metric 1: Perplexity ──────────────────────────────────────────────────────

def compute_perplexity(model, tokenizer, dataset, cfg: dict, device: str = "cuda") -> float:
    """
    Compute token-level perplexity = exp(avg cross-entropy loss).
    Lower is better. Runs over the eval dataset with no generation.
    """
    print("📐 Computing perplexity...")

    from src.data import format_prompt

    model.eval()
    total_loss, total_tokens = 0.0, 0

    with torch.no_grad():
        for ex in tqdm(dataset, desc="Perplexity"):
            text = format_prompt(str(ex["question"]), str(ex["answer"]))
            inputs = tokenizer(
                text, return_tensors="pt",
                max_length=cfg["dataset"]["max_length"],
                truncation=True,
            ).to(device)

            out = model(**inputs, labels=inputs["input_ids"])
            n_tokens = inputs["input_ids"].shape[1]
            total_loss   += out.loss.item() * n_tokens
            total_tokens += n_tokens

    avg_loss   = total_loss / total_tokens
    perplexity = math.exp(avg_loss)

    print(f"  Avg loss   : {avg_loss:.4f}")
    print(f"  Perplexity : {perplexity:.2f}  (lower = better)")
    return perplexity


# ── Metric 2: Multiple-Choice Accuracy ───────────────────────────────────────

def _get_true_key(example: dict) -> str:
    """Extract the correct option key (A/B/C/D) from a raw MedQA example."""
    key = str(example.get("answer_idx", "")).strip().upper()
    if not key:
        ans_text = str(example.get("answer", "")).strip().lower()
        for k, v in example.get("options", {}).items():
            if str(v).strip().lower() == ans_text:
                return k.upper()
    return key


def _score_option(model, tokenizer, question: str, option_text: str,
                  max_len: int, device: str) -> float:
    """Return NLL for model generating `option_text` given `question`."""
    prompt = (
        f"<|system|>You are a medical expert specializing in USMLE board exam questions.\n"
        f"<|user|>Question: {question}\n"
        f"<|assistant|>Answer: {option_text}"
    )
    inputs = tokenizer(
        prompt, return_tensors="pt", max_length=max_len, truncation=True
    ).to(device)
    with torch.no_grad():
        out = model(**inputs, labels=inputs["input_ids"])
    return out.loss.item()  # Lower NLL → model prefers this option


def compute_mc_accuracy(model, tokenizer, dataset, cfg: dict, device: str = "cuda") -> dict:
    """
    Score each MCQ option via NLL; predicted = option with lowest NLL.
    Returns dict with accuracy and per-question results.
    """
    print("\n🎯 Computing multiple-choice accuracy (NLL scoring)...")

    max_len = cfg["dataset"]["max_length"]
    model.eval()
    correct, total, results = 0, 0, []

    for i, ex in enumerate(tqdm(dataset, desc="MC Accuracy")):
        options = ex.get("options", {})
        if not options or len(options) < 2:
            continue

        question = str(ex["question"])
        losses   = {k.upper(): _score_option(model, tokenizer, question, str(v), max_len, device)
                    for k, v in options.items()}

        pred_key = min(losses, key=losses.get)
        true_key = _get_true_key(ex)
        is_ok    = (pred_key == true_key)

        if is_ok:
            correct += 1
        total += 1

        results.append({
            "idx": i, "predicted": pred_key, "true": true_key,
            "correct": is_ok, "losses": losses,
        })

    accuracy = correct / total if total > 0 else 0.0
    print(f"  Correct  : {correct} / {total}")
    print(f"  Accuracy : {accuracy * 100:.1f}%  (random baseline = 25%)")
    return {"accuracy": accuracy, "correct": correct, "total": total, "details": results}


# ── Metric 3: ROUGE ───────────────────────────────────────────────────────────

def compute_rouge(model, tokenizer, dataset, cfg: dict, device: str = "cuda") -> dict:
    """
    Generate free-text answers and compute ROUGE-1/2/L against references.
    Also collects qualitative examples for display.
    """
    try:
        from rouge_score import rouge_scorer as rs_lib
    except ImportError:
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "rouge-score"])
        from rouge_score import rouge_scorer as rs_lib

    print("\n📏 Computing ROUGE scores...")

    scorer = rs_lib.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    r1, r2, rl = [], [], []
    qualitative = []

    model.eval()
    for ex in tqdm(dataset, desc="ROUGE"):
        question    = str(ex["question"])
        true_answer = str(ex["answer"])
        gen_answer  = generate_answer(model, tokenizer, question, cfg, device)

        scores = scorer.score(true_answer, gen_answer)
        r1.append(scores["rouge1"].fmeasure)
        r2.append(scores["rouge2"].fmeasure)
        rl.append(scores["rougeL"].fmeasure)

        qualitative.append({
            "question":   question[:200],
            "true":       true_answer,
            "generated":  gen_answer,
            "rouge1":     round(scores["rouge1"].fmeasure, 4),
            "rougeL":     round(scores["rougeL"].fmeasure, 4),
        })

    result = {
        "rouge1": round(float(np.mean(r1)), 4),
        "rouge2": round(float(np.mean(r2)), 4),
        "rougeL": round(float(np.mean(rl)), 4),
        "examples": qualitative,
    }

    print(f"  ROUGE-1 : {result['rouge1']:.4f}")
    print(f"  ROUGE-2 : {result['rouge2']:.4f}")
    print(f"  ROUGE-L : {result['rougeL']:.4f}")
    return result


# ── Save + Print Summary ──────────────────────────────────────────────────────

def save_and_print_summary(perplexity: float, mc: dict, rouge: dict, cfg: dict):
    """Save all metrics to JSON and print the final evaluation card."""
    summary = {
        "model":         cfg["model"]["name"],
        "dataset":       cfg["dataset"]["name"],
        "train_samples": cfg["dataset"]["train_samples"],
        "lora_rank":     cfg["lora"]["r"],
        "perplexity":    round(perplexity, 4),
        "mc_accuracy":   round(mc["accuracy"] * 100, 2),
        "rouge1":        rouge["rouge1"],
        "rouge2":        rouge["rouge2"],
        "rougeL":        rouge["rougeL"],
    }

    Path("outputs").mkdir(exist_ok=True)
    with open("outputs/evaluation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open("outputs/mc_accuracy_results.json", "w") as f:
        json.dump(mc, f, indent=2)
    with open("outputs/rouge_results.json", "w") as f:
        json.dump(rouge, f, indent=2)

    # Pretty print card
    print("\n" + "╔" + "═" * 56 + "╗")
    print("║" + " EVALUATION CARD ".center(56) + "║")
    print("╠" + "═" * 56 + "╣")
    print(f"║  Perplexity   : {perplexity:<38.2f} ║")
    print(f"║  MC Accuracy  : {mc['accuracy'] * 100:<37.1f}% ║")
    print(f"║  ROUGE-1      : {rouge['rouge1']:<38.4f} ║")
    print(f"║  ROUGE-2      : {rouge['rouge2']:<38.4f} ║")
    print(f"║  ROUGE-L      : {rouge['rougeL']:<38.4f} ║")
    print("╠" + "═" * 56 + "╣")
    print("║  Random MC baseline = 25%                        ║")
    print("║  USMLE passing score ≈ 60%                       ║")
    print("╚" + "═" * 56 + "╝")
    print("\n💾 Results saved → outputs/evaluation_summary.json")
    return summary
