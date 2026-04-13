# 🦅 Fine-Tuning Falcon-7B with LoRA for Medical QA

> Parameter-efficient fine-tuning of Falcon-7B-Instruct on USMLE medical questions using LoRA + 4-bit quantization — achieving domain adaptation on a 7B parameter LLM with consumer-grade GPU resources.

---

## 📌 Project Overview

Large language models like Falcon-7B have broad general knowledge but lack precision in specialized domains like medicine. This project **domain-adapts Falcon-7B-Instruct** for medical question answering using two memory-efficient techniques:

- **LoRA (Low-Rank Adaptation)** — fine-tunes only a small fraction of parameters (~0.1%) instead of the full 7B, drastically cutting compute requirements
- **4-bit Quantization (BitsAndBytes)** — compresses model weights to fit within GPU VRAM without sacrificing performance

Fine-tuned on a subset of the **MedQA-USMLE-4-options** dataset — the same exam used to certify medical doctors in the US.

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🧬 Domain adaptation | Fine-tuned on USMLE-style medical MCQs |
| 🏋️ LoRA fine-tuning | Trains only ~0.1% of model parameters for efficiency |
| 📦 4-bit quantization | Loads 7B model in ~5GB VRAM via BitsAndBytes |
| ⏹️ Early stopping | Prevents overfitting with patience-based callback |
| 📈 TensorBoard logging | Tracks loss curves and training metrics in real time |
| 💾 Dual model saving | Exports both full fine-tuned model and lightweight LoRA adapter |

---

## 🚀 Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.8+ |
| Base Model | Falcon-7B-Instruct (`tiiuae/falcon-7b-instruct`) |
| Fine-Tuning | PEFT / LoRA (`peft` library) |
| Quantization | BitsAndBytes (4-bit, double quant) |
| Training | Hugging Face `Trainer` + `EarlyStoppingCallback` |
| Dataset | MedQA-USMLE-4-options (Hugging Face Datasets) |
| Compute | Kaggle GPU (CUDA) |
| Monitoring | TensorBoard |

---

## 🏗️ Training Pipeline

```
Falcon-7B-Instruct (pretrained)
        ↓
4-bit Quantization (BitsAndBytes)
  └── Loads ~7B params in ~5GB VRAM
        ↓
LoRA Adapter Injection
  └── Targets: query_key_value layers
  └── Trainable params: ~0.1% of total
        ↓
Dataset: MedQA-USMLE-4-options
  └── 2,000 train / 1,000 eval examples
  └── Format: "Question: ... Answer: ..."
        ↓
Hugging Face Trainer
  └── fp16 mixed precision
  └── Gradient accumulation (steps=4)
  └── Early stopping (patience=3)
        ↓
Saved Outputs
  ├── ./final_model      (full fine-tuned model + tokenizer)
  └── ./peft_adapter     (lightweight LoRA adapter only)
```

---

## LoRA Configuration

| Parameter | Value |
|-----------|-------|
| Rank (`r`) | 16 |
| Alpha (`lora_alpha`) | 32 |
| Target modules | `query_key_value` |
| Dropout | 0.05 |
| Task type | Causal LM |

---

## ⚙️ Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/singularity-14/finetune-falcon-7b-instruct.git
cd finetune-falcon-7b-instruct
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```
> Requires Python 3.8+ and a CUDA-compatible GPU (Kaggle free tier works)

### 3. Run the notebook

Open `falcon_7b_instruct_fine_tuning.ipynb` in Kaggle or Jupyter:
- Enable **GPU accelerator** in Kaggle settings
- Run all cells sequentially

The notebook will:
1. Load Falcon-7B with 4-bit quantization
2. Preprocess and tokenize the MedQA dataset
3. Apply LoRA and train the model
4. Save the fine-tuned model and LoRA adapter

---

## 📂 Project Structure

```
finetune-falcon-7b-instruct/
│
├── falcon_7b_instruct_fine_tuning.ipynb   # Full training pipeline
├── requirements.txt                        # Python dependencies
└── README.md                               # Project documentation
```

**After training, the following are generated:**
```
├── final_model/        # Full model weights + tokenizer
└── peft_adapter/       # Lightweight LoRA adapter + tokenizer
```

---

## 💡 Key Learnings & Takeaways

- Implemented **parameter-efficient fine-tuning (PEFT)** with LoRA — adapting a 7B model while training less than 0.1% of its parameters
- Applied **4-bit quantization** to enable large model fine-tuning on a single consumer GPU
- Structured a **causal language modeling pipeline** from raw dataset to trained, saved model
- Used **gradient accumulation** to simulate larger batch sizes within limited VRAM
- Understood tradeoffs between saving the **full model vs. a portable LoRA adapter**

---

## 🙏 Acknowledgements

- [Hugging Face](https://huggingface.co/) — Falcon-7B-Instruct model, Datasets, Trainer API
- [PEFT Library](https://github.com/huggingface/peft) — LoRA implementation
- [BitsAndBytes](https://github.com/TimDettmers/bitsandbytes) — 4-bit quantization
- [MedQA Dataset](https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options) — USMLE medical QA benchmark
- [Kaggle](https://www.kaggle.com/) — Free GPU compute platform

---

## ⚠️ Disclaimer

> This model is intended for **research and educational purposes only**. It is not a substitute for professional medical advice, diagnosis, or treatment.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

*Exploring parameter-efficient fine-tuning techniques for adapting large language models to specialized medical domains.*
