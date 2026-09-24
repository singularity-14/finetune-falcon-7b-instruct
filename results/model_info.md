# Falcon-7B-Instruct Fine-Tuned (Medical QA)

## 📊 Evaluation Details
- **Evaluation Examples Used**: 1,000 (from MedQA-USMLE-4-options)
- **Accuracy**: ~48% (Baseline random is 25%, USMLE Passing is ~60%)
- **Training Examples Used**: 2,000

## 📦 Model Size & Memory
- **Base Model (Falcon-7B-Instruct)**: ~14 GB (FP16)
- **Quantized Model (4-bit)**: ~5 GB VRAM
- **LoRA Adapter**: ~32 MB
- **Total VRAM Required during Inference**: ~5 - 6 GB VRAM

## ⚡ Kaggle Optimization Techniques Used
Since this model is trained/run on environments like Kaggle (T4 GPU / P100) with limited VRAM (15-16 GB), the following techniques were heavily utilized:

1. **Low-Rank Adaptation (LoRA)**
   - **Target Modules**: `query_key_value`
   - **Rank (r)**: 16
   - **Alpha**: 32
   - **Parameters Trained**: ~4M out of 7B (only 0.058%)
   - **Why?**: Instead of full fine-tuning which requires hundreds of GBs of VRAM, LoRA only updates small rank decomposition matrices, drastically reducing GPU memory and allowing us to train on a single Kaggle T4.

2. **4-Bit Quantization (BitsAndBytes NF4)**
   - **Technique**: NF4 (Normalized Float 4-bit) with double quantization.
   - **Why?**: Squeezes the large 7B model from ~14 GB down to ~5 GB VRAM, ensuring we do not get Out Of Memory (OOM) errors during inference or training on Kaggle.

3. **Gradient Accumulation**
   - **Steps**: 4
   - **Why?**: Allows us to simulate a larger batch size without consuming excessive GPU memory.

4. **Early Stopping**
   - **Patience**: 3 epochs
   - **Why?**: Prevents overfitting and saves valuable compute time on Kaggle sessions, which have time limits.
