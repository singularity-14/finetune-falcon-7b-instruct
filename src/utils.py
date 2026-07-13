"""
src/utils.py — Shared utilities: config loading, logging, GPU reporting, seed setting
"""

import os
import gc
import random
import logging
import yaml
import torch
import numpy as np
from pathlib import Path


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load YAML config. Auto-resolves path if called from notebooks/."""
    # If running from notebooks/ subdirectory, go up one level
    if not os.path.exists(config_path) and os.path.exists(f"../{config_path}"):
        os.chdir("..")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Configure root logger with timestamp format."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger("falcon_ft")


def set_seed(seed: int = 42):
    """Reproducibility: seed Python, NumPy, PyTorch, and CUDA."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def gpu_report() -> dict:
    """Print and return GPU memory stats. Raises if no GPU found."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "❌ No GPU detected!\n"
            "   Go to Runtime → Change runtime type → GPU (T4) and re-run."
        )

    name       = torch.cuda.get_device_name(0)
    total_gb   = torch.cuda.get_device_properties(0).total_memory / 1024**3
    free_gb    = torch.cuda.mem_get_info()[0] / 1024**3
    used_gb    = total_gb - free_gb

    print("═" * 50)
    print("GPU ENVIRONMENT")
    print("═" * 50)
    print(f"  GPU     : {name}")
    print(f"  VRAM    : {total_gb:.1f} GB total  |  {free_gb:.1f} GB free  |  {used_gb:.1f} GB used")
    print(f"  CUDA    : {torch.version.cuda}")
    print(f"  PyTorch : {torch.__version__}")
    print("═" * 50)

    if free_gb < 8:
        print("⚠️  < 8 GB free VRAM — consider restarting the runtime.")
    else:
        print("✅ GPU ready!")

    return {"name": name, "total_gb": total_gb, "free_gb": free_gb, "used_gb": used_gb}


def make_output_dirs(cfg: dict):
    """Create all output directories from config. Handles both absolute and relative paths."""
    dirs = [
        cfg["training"]["output_dir"],
        cfg["training"]["logging_dir"],
        cfg["saving"]["final_model_dir"],
        cfg["saving"]["peft_adapter_dir"],
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
        print(f"  📁 {d}")


def free_memory():
    """Release GPU memory and run GC."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    free_gb = torch.cuda.mem_get_info()[0] / 1024**3 if torch.cuda.is_available() else 0
    print(f"✅ Memory freed | VRAM free: {free_gb:.1f} GB")
