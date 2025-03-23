# **Fine-Tuning Falcon-7B with LoRA for Medical QA**

## **Overview**
This project fine-tunes the **Falcon-7B-Instruct** model using **Low-Rank Adaptation (LoRA)** for **medical question-answering** tasks. It leverages **Hugging Face Transformers**. The model is finetuned on the subset of **MedQA-USMLE-4-options** dataset.

## **Features**
- **GPU Utilization Check**: Ensures CUDA is available for optimal training.
- **Quantization with BitsAndBytes**: Uses **4-bit quantization** for efficient model loading.
- **Medical QA Dataset**: Uses the subset of **MedQA-USMLE-4-options** dataset for fine-tuning.
- **Tokenization & Data Preprocessing**: Tokenizes and structures data for optimal training.
- **LoRA (Low-Rank Adaptation)**: Implements **parameter-efficient fine-tuning** to reduce computational costs.
- **Trainer with Early Stopping**: Uses **Hugging Face Trainer** with **early stopping** to optimize training performance.
- **Model Saving**: Saves both the full fine-tuned model and the LoRA adapter.

## **Installation**
### **Prerequisites**
- Python 3.8 or higher
- CUDA-compatible GPU (recommended for training) (Kaggle)
- Virtual environment (optional but recommended)

### **Setup**
1. **Clone the repository**  
   ```bash
   git clone https://github.com/singularity-14/Falcon-MedQA.git
   cd Falcon-MedQA
   ```

2. **Create a virtual environment** (optional)  
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

## **Usage**
### **Training the Model**
Run the training script to fine-tune the model on the MedQA dataset:
```bash
python train.py
```
This script will:
- Load the **Falcon-7B-Instruct** model with 4-bit quantization.
- Preprocess the MedQA dataset and tokenize inputs.
- Apply **LoRA fine-tuning** on selected model layers.
- Train using **Hugging Face Trainer** with gradient accumulation.

### **Saving & Exporting the Model**
After training completes, the script will automatically save:
- The **full fine-tuned model** in `./final_model`
- The **LoRA adapter** in `./peft_adapter`
- The **tokenizer** in both directories for reuse.

## **Project Structure**
```
📁 Falcon-MedQA
│-- falcon_7b_instruct_fine_tuning.ipynb             # Training script
│-- requirements.txt     # Dependencies
```

## **Technologies Used**
- **Falcon-7B-Instruct** (Hugging Face Transformers)
- **LoRA (Low-Rank Adaptation)**
- **Hugging Face Trainer**
- **FAISS for embedding storage**
- **BitsAndBytes (4-bit quantization)**
- **Kaggle (GPU acceleration)**

## **Acknowledgments**
- **Hugging Face** for providing **Falcon-7B-Instruct**, **MedQA Dataset** for medical question-answering, and Trainer utilities.
- **BitsAndBytes & LoRA** for efficient model fine-tuning.
- **Kaggle** for providing platfrom to finetune the model using GPUs.
