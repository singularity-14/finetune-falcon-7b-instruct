"""
src/model.py — Load Falcon-7B with 4-bit quantization and apply LoRA adapter
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, PeftModel, TaskType


def load_tokenizer(model_name: str, trust_remote_code: bool = True) -> AutoTokenizer:
    """
    Load Falcon tokenizer and add [PAD] token if missing.
    Falcon has no pad token by default — required for batch training.
    """
    print(f"📥 Loading tokenizer: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
    )

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        print("  Added [PAD] token")

    # Right-padding required for causal LM training
    tokenizer.padding_side = "right"

    print(f"  Vocab size : {tokenizer.vocab_size:,}")
    print(f"  Pad token  : '{tokenizer.pad_token}' (id={tokenizer.pad_token_id})")
    print("✅ Tokenizer loaded")
    return tokenizer


_DTYPES = {
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float32": torch.float32,
    "fp32": torch.float32,
}


def resolve_compute_dtype(cfg: dict) -> torch.dtype:
    """
    Resolve `quantization.bnb_4bit_compute_dtype` to a torch dtype.
    The same dtype must be used for the 4-bit matmuls AND for every module
    bitsandbytes leaves unquantized - see load_base_model().
    """
    name = str(cfg["quantization"].get("bnb_4bit_compute_dtype", "float16")).lower()
    if name not in _DTYPES:
        raise ValueError(f"Unknown bnb_4bit_compute_dtype: {name!r} (use {sorted(_DTYPES)})")
    return _DTYPES[name]


def align_dtypes(model, dtype: torch.dtype = torch.float16):
    """
    Cast every float32 parameter/buffer down to `dtype`. Inference only.

    4-bit weights are stored as uint8 and are untouched; this normalizes only the
    modules bitsandbytes leaves unquantized - embeddings, layernorms, rotary
    caches, LoRA adapters. Those stay fp32 when the model is loaded without an
    explicit torch_dtype, which is what makes attention blow up with
    "Expected query, key, and value to have the same dtype".

    Do NOT call this on a model you still intend to train - LoRA weights want fp32
    during optimization.
    """
    n = 0
    for _, param in model.named_parameters():
        if param.dtype == torch.float32:
            param.data = param.data.to(dtype)
            n += 1
    for _, buf in model.named_buffers():
        if buf.dtype == torch.float32:
            buf.data = buf.data.to(dtype)
            n += 1
    if n:
        print(f"  Cast {n} fp32 tensors -> {dtype} for inference")
    return model


def build_quant_config(cfg: dict) -> BitsAndBytesConfig:
    """Build BitsAndBytes 4-bit NF4 quantization config from YAML config."""
    q = cfg["quantization"]
    return BitsAndBytesConfig(
        load_in_4bit=q["load_in_4bit"],
        bnb_4bit_compute_dtype=resolve_compute_dtype(cfg),
        bnb_4bit_use_double_quant=q["bnb_4bit_use_double_quant"],
        bnb_4bit_quant_type=q["bnb_4bit_quant_type"],
    )


def load_base_model(
    model_name: str,
    quant_config: BitsAndBytesConfig,
    tokenizer: AutoTokenizer,
    trust_remote_code: bool = True,
    use_cache: bool = False,
    torch_dtype: torch.dtype = torch.float16,
) -> AutoModelForCausalLM:
    """
    Load Falcon-7B base model with 4-bit quantization.
    Resizes token embeddings to account for added [PAD] token.

    `torch_dtype` must match bnb_4bit_compute_dtype. Without it transformers keeps
    the non-quantized modules (embeddings, layernorms, rotary) in fp32 while the
    4-bit linears compute in fp16, and attention then fails with
    "Expected query, key, and value to have the same dtype".
    """
    print(f"\n📥 Loading base model: {model_name}")
    print("   (First run downloads ~14 GB — subsequent runs use cache)")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config=quant_config,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
        use_cache=use_cache,
    )

    # Resize embeddings for the newly added [PAD] token
    model.resize_token_embeddings(len(tokenizer))

    vram_used = torch.cuda.memory_allocated() / 1024**3
    print(f"  VRAM used after load: {vram_used:.1f} GB")
    print("✅ Base model loaded with 4-bit quantization")
    return model


def apply_lora(model: AutoModelForCausalLM, cfg: dict) -> AutoModelForCausalLM:
    """
    Wrap the base model with LoRA adapter via PEFT.
    Only ~0.1% of parameters become trainable.
    """
    print("\n🔧 Applying LoRA adapter...")

    l = cfg["lora"]
    lora_config = LoraConfig(
        r=l["r"],
        lora_alpha=l["lora_alpha"],
        target_modules=l["target_modules"],
        lora_dropout=l["lora_dropout"],
        bias=l["bias"],
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} params  ({100 * trainable / total:.3f}% of total)")
    print("✅ LoRA applied — model ready for training")
    return model


def load_finetuned_model(
    base_model_name: str,
    adapter_dir: str,
    quant_config: BitsAndBytesConfig,
    trust_remote_code: bool = True,
    torch_dtype: torch.dtype = torch.float16,
) -> tuple:
    """
    Load fine-tuned model (base + LoRA adapter) for inference/evaluation.
    Returns (model, tokenizer).
    """
    print(f"📥 Loading tokenizer from: {adapter_dir}")
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)

    print(f"📥 Loading base model: {base_model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map="auto",
        quantization_config=quant_config,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
        use_cache=True,
    )

    print(f"🔌 Loading LoRA adapter from: {adapter_dir}")
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    align_dtypes(model, torch_dtype)   # adapter weights can arrive in fp32
    model.eval()

    print("✅ Fine-tuned model loaded for inference")
    return model, tokenizer
