# Falcon-7B-Instruct Medical QA: End-to-End Project Workflow

This flowchart outlines the complete, end-to-end lifecycle of the project, designed to help you explain the architecture and steps during an interview.

```mermaid
flowchart TD
    %% Define styles
    classDef dataset fill:#f9f,stroke:#333,stroke-width:2px;
    classDef model fill:#bbf,stroke:#333,stroke-width:2px;
    classDef process fill:#dfd,stroke:#333,stroke-width:2px;
    classDef evaluation fill:#ffd,stroke:#333,stroke-width:2px;
    classDef deployment fill:#fdf,stroke:#333,stroke-width:2px;

    %% 1. Data Preparation
    subgraph Data Pipeline
        D1[(MedQA-USMLE Dataset)] --> D2[Tokenization & Prompt Formatting]
        D2 --> D3[Train Split: 2,000 samples]
        D2 --> D4[Eval Split: 1,000 samples]
    end
    D1:::dataset
    D2:::process
    D3:::dataset
    D4:::dataset

    %% 2. Model Architecture & Optimization
    subgraph Model Optimization
        M1(Pretrained Falcon-7B-Instruct\n~14 GB VRAM) --> M2[4-bit Quantization\nBitsAndBytes NF4]
        M2 --> M3[Quantized Base Model\n~5 GB VRAM]
        M3 --> M4[Inject LoRA Adapters\nr=16, alpha=32\ntarget: query_key_value]
        M4 --> M5[Trainable Model\n~4M trainable params / 0.058%]
    end
    M1:::model
    M2:::process
    M3:::model
    M4:::process
    M5:::model

    %% 3. Training Loop
    subgraph Fine-Tuning (Kaggle/Colab T4 GPU)
        M5 --> T1((HuggingFace Trainer))
        D3 --> T1
        T1 --> T2[FP16 Mixed Precision]
        T1 --> T3[Gradient Accumulation steps=4]
        T1 --> T4[Early Stopping patience=3]
    end
    T1:::process
    T2:::process
    T3:::process
    T4:::process

    %% 4. Evaluation
    subgraph Evaluation
        T1 --> E1(Evaluate Checkpoints)
        D4 --> E1
        E1 --> E2>Perplexity metric]
        E1 --> E3>MC Accuracy NLL Scoring]
        E1 --> E4>ROUGE Scores]
    end
    E1:::process
    E2:::evaluation
    E3:::evaluation
    E4:::evaluation

    %% 5. Artifacts and Serving
    subgraph Deployment
        T1 --> A1[Save PEFT Adapter\n~32 MB]
        A1 --> S1(FastAPI Server)
        S1 --> S2[Docker Container]
        S2 --> S3((Local Inference Endpoint))
    end
    A1:::dataset
    S1:::deployment
    S2:::deployment
    S3:::deployment
```

## Key Talking Points for Interviews

### 1. Why Falcon-7B & MedQA?
- **Goal:** Domain-adapt a highly capable general LLM (Falcon-7B) to specialized medical knowledge (USMLE board exams).
- **Dataset:** 4-option multiple-choice questions requiring complex reasoning.

### 2. How did you train a 7B model on a single free GPU (Kaggle/Colab T4 15GB VRAM)?
- **4-bit Quantization (BitsAndBytes NF4):** Compressed the massive 14GB base model down to ~5GB in VRAM without significantly hurting accuracy.
- **LoRA (Low-Rank Adaptation):** Instead of fine-tuning all 7B parameters, we targeted specific linear layers (`query_key_value`), reducing the trainable parameters to ~4 Million (just 0.058%).
- **Gradient Accumulation:** Simulated larger batch sizes within the tight memory constraints.

### 3. How was the model evaluated?
- Evaluated on a 1,000-sample holdout test set using 3 strategies:
  - **Perplexity:** To measure the language modeling loss.
  - **Multiple-Choice Accuracy (NLL):** Scored directly against the 4 choices by calculating the negative log-likelihood of each choice.
  - **ROUGE Metrics & Qualitative Checking:** Measured textual overlap on generated answers.

### 4. How did you deploy it?
- The training process generated a tiny ~32 MB LoRA adapter, making it highly portable.
- Wrapped inference logic in a **FastAPI** server containing `/predict` and `/batch` endpoints.
- Containerized the API using **Docker** and `docker-compose` to allow easy local execution with GPU support.
