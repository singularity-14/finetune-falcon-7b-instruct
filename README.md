# 🦅 Fine-Tuning Falcon-7B-Instruct for Medical QA

> Parameter-efficient fine-tuning of **Falcon-7B-Instruct** on **MedQA-USMLE** using **LoRA + 4-bit quantization** — domain-adapting a 7B LLM for medical question answering on a single free Colab T4 GPU.

---

## 📌 Project Overview

Large language models like Falcon-7B have broad general knowledge but lack precision in specialized domains like medicine. This project **domain-adapts Falcon-7B-Instruct** for USMLE-style medical question answering using two memory-efficient techniques:

- **LoRA (Low-Rank Adaptation)** — fine-tunes only ~0.1% of parameters instead of the full 7B, cutting compute requirements drastically
- **4-bit Quantization (BitsAndBytes NF4)** — compresses model weights to fit within GPU VRAM without sacrificing accuracy

Fine-tuned on the **MedQA-USMLE-4-options** dataset — the same exam used to certify medical doctors in the US.

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🧬 Domain adaptation | Fine-tuned on USMLE-style medical MCQs |
| 🏋️ LoRA (PEFT) | Trains only ~0.1% of model parameters |
| 📦 4-bit quantization | Loads 7B model in ~5 GB VRAM (NF4 + double-quant) |
| ⏹️ Early stopping | Patience-based, monitors eval loss |
| 📈 TensorBoard | Real-time loss, LR, and gradient norm tracking |
| 📊 Full evaluation | Perplexity · MC Accuracy · ROUGE-1/2/L · Qualitative |
| 🐳 Docker serving | FastAPI server with health checks, batch endpoint |
| 🔧 Config-driven | All hyperparameters in `configs/config.yaml` |

---

## 🚀 Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.10 |
| Base Model | `tiiuae/falcon-7b-instruct` |
| Fine-Tuning | PEFT / LoRA |
| Quantization | BitsAndBytes (NF4 4-bit, double-quant) |
| Training | HuggingFace `Trainer` + `EarlyStoppingCallback` |
| Dataset | `GBaker/MedQA-USMLE-4-options` |
| Monitoring | TensorBoard |
| Evaluation | Perplexity · MC Accuracy · ROUGE · Qualitative |
| Serving | FastAPI + Uvicorn |
| Deployment | Docker + Docker Compose |
| Compute | Google Colab Free Tier (T4 · 15 GB VRAM) |

---

## 📂 Project Structure

```
finetune-falcon-7b-instruct/
│
├── 📓 run_pipeline.ipynb                  # ← THE ONLY NOTEBOOK YOU NEED
│                                          #   10 cells. Run top-to-bottom.
│
├── 🐍 src/                               # All logic lives here (importable, testable)
│   ├── utils.py                           # Config loading, GPU check, seed, logging
│   ├── model.py                           # Tokenizer, 4-bit loading, LoRA injection
│   ├── data.py                            # Dataset loading, prompt format, tokenization
│   ├── train.py                           # Trainer builder, training loop, artifact saving
│   ├── evaluate.py                        # Perplexity, MC accuracy, ROUGE, qualitative
│   └── serve.py                           # FastAPI production inference server
│
├── ⚙️ configs/
│   ├── config.yaml                        # All hyperparameters (single source of truth)
│   └── CONFIG.md                          # Field-by-field reference guide
│
├── 🐳 docker/
│   ├── Dockerfile                         # Production image (CUDA 12.1)
│   └── docker-compose.yml                 # GPU + volume mount setup
│
├── requirements-colab.txt                 # Pinned deps for Colab T4 (use this)
├── requirements-serving.txt               # Minimal deps for Docker API server
├── .gitignore
└── README.md
│
└── outputs/  ← (generated after training, git-ignored)
    ├── results/                           # Trainer checkpoints
    ├── logs/                              # TensorBoard event files
    ├── final_model/                       # Full fine-tuned model + tokenizer
    ├── peft_adapter/                      # Lightweight LoRA adapter (~32 MB)
    ├── evaluation_summary.json            # All eval metrics
    ├── mc_accuracy_results.json           # Per-question MC results
    ├── rouge_results.json                 # ROUGE + qualitative examples
    └── training_curves.png                # Loss/perplexity plots
```

---

## 🏗️ Training Pipeline

```
Falcon-7B-Instruct (pretrained, frozen)
        ↓
BitsAndBytes NF4 4-bit Quantization
  └── ~5 GB VRAM (down from ~14 GB)
        ↓
LoRA Adapter Injection
  └── Target: query_key_value layers
  └── Trainable: ~4M / 7B params (0.058%)
        ↓
MedQA-USMLE-4-options Dataset
  └── 2,000 train / 1,000 eval examples
  └── Prompt: "<|system|>...<|user|>Question...<|assistant|>Answer..."
        ↓
HuggingFace Trainer
  └── fp16 mixed precision
  └── Cosine LR schedule (warmup 5%)
  └── Gradient accumulation (steps=4, eff. batch=8)
  └── EarlyStopping (patience=3 epochs)
  └── TensorBoard logging
        ↓
Evaluation
  ├── Perplexity (exp(eval_loss))
  ├── Multiple-choice accuracy (NLL scoring)
  ├── ROUGE-1 / ROUGE-2 / ROUGE-L
  └── Qualitative side-by-side examples
        ↓
Saved Outputs
  ├── outputs/final_model/       (full model — deploy directly)
  └── outputs/peft_adapter/      (LoRA adapter — ~32 MB, portable)
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
| Effective LR scaling | `alpha / r = 2.0` |

---

## ⚙️ How to Run (Colab Free Tier)

### Step 1 — Enable GPU
- Go to **Runtime → Change runtime type → GPU (T4)**

### Step 2 — Open `run_pipeline.ipynb` and run cells top to bottom

```
Cell 1  → git clone + pip install            (~3 min)
Cell 2  → Load config, verify GPU            (~30 sec)
Cell 3  → Load tokenizer + model (4-bit)     (~5 min, downloads 14 GB first run)
Cell 4  → Load & tokenize dataset            (~2 min)
Cell 5  → Train (LoRA + EarlyStopping)       (~60–90 min)
Cell 6  → TensorBoard (live during training) (background)
Cell 7  → Evaluate: PPL + MC Acc + ROUGE     (~15 min)
Cell 8  → Qualitative examples               (instant)
Cell 9  → Plot training curves               (instant)
Cell 10 → Free memory + download artifacts   (instant)
```

> ⚠️ **Colab sessions expire after ~90 min idle.** Keep the browser tab open during training.

---

## 🐳 Docker Deployment (Local Machine)

### Prerequisites
- Docker + Docker Compose installed
- NVIDIA Docker runtime (`nvidia-container-toolkit`)
- Downloaded `peft_adapter/` from Colab

### Build & Run

```bash
# 1. Place your peft_adapter/ inside docker/model/
mkdir -p docker/model
cp -r outputs/peft_adapter docker/model/

# 2. Build image
cd docker
docker build -t falcon-medical-qa:latest -f Dockerfile ..

# 3. Run with GPU
docker run --gpus all -p 8000:8000 \
  -v $(pwd)/model/peft_adapter:/app/model/peft_adapter \
  falcon-medical-qa:latest

# OR use docker-compose
docker-compose up
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API info |
| `/health` | GET | Health check + VRAM stats |
| `/model-info` | GET | Model configuration |
| `/predict` | POST | Single question inference |
| `/batch` | POST | Batch inference (max 5) |
| `/docs` | GET | Swagger UI |

### Example Request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"question": "A 45-year-old man with crushing chest pain and ST elevation in V1-V4. Diagnosis?"}'
```

---

## 📊 Evaluation Metrics

| Metric | Description | Baseline |
|--------|-------------|---------|
| **Perplexity** | `exp(eval_loss)` — lower = better | N/A (language model) |
| **MC Accuracy** | NLL scoring on 4-option MCQs | 25% (random) |
| **ROUGE-1** | Unigram overlap with reference answer | 0.0 |
| **ROUGE-L** | Longest common subsequence overlap | 0.0 |

> USMLE passing score ≈ **60% accuracy**. Fine-tuned on only 2,000 samples — results will improve with more data.

---

## 🔧 Configuration

All hyperparameters are in [`configs/config.yaml`](configs/config.yaml). See [`configs/CONFIG.md`](configs/CONFIG.md) for a full field-by-field reference.

Key tuning knobs:

```yaml
lora:
  r: 16          # ↑ More capacity, ↑ memory
  lora_alpha: 32 # ↑ Stronger LoRA scaling

training:
  num_train_epochs: 10  # EarlyStopping will exit when eval loss stops improving
  learning_rate: 2.0e-4

dataset:
  train_samples: 2000   # ↑ More data = better results, longer training
```

---

## 💡 Key Learnings

- Implemented **PEFT/LoRA** — adapting a 7B model while training less than 0.1% of its parameters
- Applied **NF4 4-bit quantization** to enable large model fine-tuning on a single consumer GPU
- Built a **causal LM pipeline** from raw dataset to evaluated, deployed model
- Used **gradient accumulation** to simulate larger batch sizes within limited VRAM
- Implemented **three evaluation strategies**: perplexity, NLL-based MC accuracy, and ROUGE
- Built a **production FastAPI server** with health checks, batch inference, and Docker support

---

## ⚠️ Disclaimer

This model is intended for **research and educational purposes only**. It is **not a substitute** for professional medical advice, diagnosis, or treatment.

---

## 📄 License

MIT License

---

## 🙏 Acknowledgements

- [HuggingFace](https://huggingface.co/) — Falcon-7B-Instruct, Datasets, Trainer, PEFT
- [BitsAndBytes](https://github.com/TimDettmers/bitsandbytes) — 4-bit quantization
- [MedQA Dataset](https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options) — USMLE benchmark
- [Google Colab](https://colab.research.google.com/) — Free T4 GPU compute
