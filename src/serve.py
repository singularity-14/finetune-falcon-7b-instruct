"""
Falcon-7B Medical QA — FastAPI Inference Server
================================================
Run locally:  uvicorn src.serve:app --host 0.0.0.0 --port 8000
In Docker:    CMD in Dockerfile calls this automatically

Endpoints:
  GET  /health    → Health check
  GET  /model-info → Model metadata
  POST /predict   → Generate answer for a medical question
  POST /batch     → Batch prediction (multiple questions)
"""

import os
import gc
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

import torch
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Load config ───────────────────────────────────────────────────────────────
CONFIG_PATH = Path(__file__).parent.parent / "configs" / "config.yaml"
with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

# ── Global model state ────────────────────────────────────────────────────────
class ModelState:
    model = None
    tokenizer = None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    is_loaded = False
    load_time_s = 0.0

state = ModelState()


# ── Lifespan: load model at startup, free at shutdown ────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the fine-tuned model on startup."""
    logger.info("=" * 60)
    logger.info("Starting Falcon-7B Medical QA Server")
    logger.info("=" * 60)

    ADAPTER_DIR = os.environ.get("ADAPTER_DIR", "/app/model/peft_adapter")
    MODEL_NAME  = cfg["model"]["name"]

    t0 = time.time()

    logger.info(f"Loading tokenizer from: {ADAPTER_DIR}")
    state.tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)

    logger.info(f"Loading base model: {MODEL_NAME}")
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        quantization_config=quant_config,
        trust_remote_code=True,
        use_cache=True,
    )

    logger.info(f"Loading LoRA adapter from: {ADAPTER_DIR}")
    state.model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    state.model.eval()

    state.is_loaded = True
    state.load_time_s = time.time() - t0

    logger.info(f"✅ Model ready in {state.load_time_s:.1f}s | Device: {state.device}")

    yield  # Server runs here

    # Cleanup
    logger.info("Shutting down — freeing GPU memory...")
    del state.model
    del state.tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Server shut down cleanly.")


# ── FastAPI App ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Falcon-7B Medical QA API",
    description=(
        "Fine-tuned Falcon-7B-Instruct for medical question answering. "
        "Trained on MedQA-USMLE-4-options using LoRA + 4-bit quantization. "
        "**For research and educational purposes only.**"
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS (allow any origin — tighten in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic Schemas ──────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    question: str = Field(
        ...,
        description="Medical question to answer",
        example="A 65-year-old man presents with sudden onset severe headache. CT shows subarachnoid hemorrhage. What is the most likely cause?"
    )
    max_new_tokens: int = Field(
        default=200,
        ge=10,
        le=500,
        description="Maximum tokens to generate"
    )
    temperature: float = Field(
        default=0.7,
        ge=0.01,
        le=2.0,
        description="Sampling temperature"
    )
    do_sample: bool = Field(default=True)
    top_p: float = Field(default=0.9, ge=0.1, le=1.0)
    repetition_penalty: float = Field(default=1.1, ge=1.0, le=2.0)


class PredictResponse(BaseModel):
    question: str
    answer: str
    generation_time_s: float
    model: str
    disclaimer: str = "For research/educational use only. Not a substitute for professional medical advice."


class BatchPredictRequest(BaseModel):
    questions: List[str] = Field(..., max_items=5)
    max_new_tokens: int = Field(default=150, ge=10, le=300)


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    vram_used_gb: Optional[float]
    load_time_s: float


class ModelInfoResponse(BaseModel):
    base_model: str
    adapter_type: str
    lora_rank: int
    lora_alpha: int
    dataset: str
    train_samples: int
    quantization: str


# ── Inference Helper ──────────────────────────────────────────────────────────
def generate_answer(request: PredictRequest) -> str:
    """Run inference and return generated text."""
    prompt = (
        f"<|system|>You are a medical expert specializing in USMLE board exam questions.\n"
        f"<|user|>Question: {request.question}\n"
        f"<|assistant|>Answer:"
    )

    inputs = state.tokenizer(
        prompt,
        return_tensors="pt",
        max_length=400,
        truncation=True,
    ).to(state.device)

    with torch.no_grad():
        output_ids = state.model.generate(
            **inputs,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            do_sample=request.do_sample,
            top_p=request.top_p,
            repetition_penalty=request.repetition_penalty,
            eos_token_id=state.tokenizer.eos_token_id,
            pad_token_id=state.tokenizer.pad_token_id,
        )

    gen_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return state.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()


# ── API Endpoints ─────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    """Health check — returns model load status and GPU memory."""
    vram = None
    if torch.cuda.is_available():
        vram = torch.cuda.memory_allocated() / 1024**3

    return HealthResponse(
        status="healthy" if state.is_loaded else "loading",
        model_loaded=state.is_loaded,
        device=state.device,
        vram_used_gb=round(vram, 2) if vram else None,
        load_time_s=round(state.load_time_s, 2),
    )


@app.get("/model-info", response_model=ModelInfoResponse, tags=["System"])
async def model_info():
    """Returns model configuration metadata."""
    return ModelInfoResponse(
        base_model=cfg["model"]["name"],
        adapter_type="LoRA (PEFT)",
        lora_rank=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["lora_alpha"],
        dataset=cfg["dataset"]["name"],
        train_samples=cfg["dataset"]["train_samples"],
        quantization="BitsAndBytes NF4 4-bit",
    )


@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
async def predict(request: PredictRequest):
    """Generate a medical answer for a single question."""
    if not state.is_loaded:
        raise HTTPException(status_code=503, detail="Model is still loading. Try again in a few seconds.")

    logger.info(f"Predict request: {request.question[:80]}...")
    t0 = time.time()

    try:
        answer = generate_answer(request)
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

    elapsed = time.time() - t0
    logger.info(f"Answer generated in {elapsed:.2f}s")

    return PredictResponse(
        question=request.question,
        answer=answer,
        generation_time_s=round(elapsed, 2),
        model=cfg["model"]["name"],
    )


@app.post("/batch", tags=["Inference"])
async def batch_predict(request: BatchPredictRequest):
    """Generate answers for up to 5 questions in one call."""
    if not state.is_loaded:
        raise HTTPException(status_code=503, detail="Model is still loading.")

    results = []
    for q in request.questions:
        pred_req = PredictRequest(
            question=q,
            max_new_tokens=request.max_new_tokens,
        )
        t0 = time.time()
        answer = generate_answer(pred_req)
        results.append({
            "question": q,
            "answer": answer,
            "generation_time_s": round(time.time() - t0, 2),
        })

    return {"results": results, "count": len(results)}


@app.get("/", tags=["System"])
async def root():
    """API root — redirect to docs."""
    return {
        "message": "Falcon-7B Medical QA API",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict",
    }
