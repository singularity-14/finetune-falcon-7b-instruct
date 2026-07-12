"""
src/data.py — Dataset loading, prompt formatting, tokenization, and splitting
"""

from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, DataCollatorForLanguageModeling


PROMPT_TEMPLATE = (
    "<|system|>You are a medical expert specializing in USMLE board exam questions.\n"
    "<|user|>Question: {question}\n"
    "<|assistant|>Answer: {answer}"
)


def format_prompt(question: str, answer: str) -> str:
    """Format a QA pair into Falcon instruct-style prompt."""
    question = " ".join(question) if isinstance(question, list) else question
    answer   = " ".join(answer)   if isinstance(answer,   list) else answer
    return PROMPT_TEMPLATE.format(question=question, answer=answer)


def build_tokenize_fn(tokenizer: AutoTokenizer, max_length: int):
    """
    Return a map-compatible tokenization function.
    Labels = input_ids (standard causal LM objective).
    """
    def tokenize(example):
        text = format_prompt(example["question"], example["answer"])
        encoded = tokenizer(
            text,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors=None,
        )
        encoded["labels"] = encoded["input_ids"].copy()
        return encoded
    return tokenize


def load_and_split(cfg: dict, tokenizer: AutoTokenizer) -> tuple:
    """
    Load MedQA-USMLE, tokenize, and split into train/eval.
    Returns (train_dataset, eval_dataset, data_collator).
    """
    ds_cfg = cfg["dataset"]

    print(f"📚 Loading dataset: {ds_cfg['name']}")
    raw = load_dataset(ds_cfg["name"], split=ds_cfg["split"])
    print(f"  Raw size : {len(raw):,} examples")
    print(f"  Columns  : {raw.column_names}")

    # Tokenize
    print("\n🔄 Tokenizing...")
    tokenize_fn = build_tokenize_fn(tokenizer, ds_cfg["max_length"])
    tokenized = raw.map(
        tokenize_fn,
        remove_columns=raw.column_names,
        desc="Tokenizing",
    )

    # Split
    split = tokenized.train_test_split(
        test_size=ds_cfg["test_size"],
        seed=ds_cfg["seed"],
    )
    train = split["train"].select(range(ds_cfg["train_samples"]))
    eval_ = split["test"].select(range(ds_cfg["eval_samples"]))

    print(f"\n  Train : {len(train):,} examples")
    print(f"  Eval  : {len(eval_):,} examples")

    # Data collator — handles dynamic padding for CLM
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    print("✅ Dataset ready")
    return train, eval_, collator
