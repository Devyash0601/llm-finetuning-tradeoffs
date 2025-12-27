# 🧠 LLM Fine-Tuning Trade-offs: Full Fine-Tuning vs LoRA

This project presents a **hands-on comparison between Full Fine-Tuning and LoRA (Low-Rank Adaptation)** for large language models, focused on **news-style text generation**.

Rather than building a production chatbot, the goal is to **demonstrate the real-world trade-offs between model quality, efficiency, and deployment cost** when fine-tuning LLMs.

---

## 🔍 Motivation

Fine-tuning large language models is expensive and often impractical at scale.  
**Parameter-Efficient Fine-Tuning (PEFT)** techniques like **LoRA** significantly reduce training cost by updating only a small subset of parameters.

This project explores:

- How does output quality compare between Full Fine-Tuning and LoRA?
- What are the differences in inference latency?
- When is LoRA a practical alternative to full fine-tuning?

---

## 📊 Overview

### Models
- **Base Model:** `facebook/opt-125m`
- **Fine-Tuning Strategies:**
  - Full Fine-Tuning (100% parameters updated)
  - LoRA Fine-Tuning (~1% trainable parameters)

### Datasets
- **AG News** – short-form news text
- **CNN/DailyMail** – long-form news articles

All data is converted into a **light instruction format** for generative training.

---

## ⚠️ Scope & Limitations

> This system is trained **exclusively on news-style datasets**.  
> It is designed to demonstrate **fine-tuning trade-offs**, not factual question answering.

- Outputs may be fluent but **not guaranteed to be factually correct**
- The model does **not have access to real-time information**
- Limitations are intentional to keep the comparison controlled and interpretable

---

## 🏗️ Project Structure

llm-folder/
├── src/
│   ├── data/               # Dataset loading & preprocessing
│   ├── models/             # Base model & LoRA adapters
│   ├── training/           # Full FT & LoRA training loops
│   ├── evaluation/         # Loss & perplexity comparison
│   └── inference/          # FastAPI inference service
│
├── static/
│   ├── css/                # UI styling
│   └── js/                 # Frontend logic
│
├── templates/
│   └── index.html          # Web UI layout
│
├── experiments/            # Saved trained models
├── results/                # Evaluation outputs
└── README.md

---

## ⚙️ Training Setup

- **Context length:** 256 tokens  
- **Optimizer:** AdamW  
- **Batch size:** Tuned separately for Full FT and LoRA  
- **Hardware:** Apple Silicon (M-series) CPU/GPU  

Both models are trained on the **same data split** to ensure a fair comparison.

---

## 📈 Evaluation Metrics

- Training loss
- Perplexity
- Inference latency

### Observations
- Full Fine-Tuning achieves **lower loss and higher fluency**
- LoRA trades some output quality for **significant efficiency gains**
- Latency differences are visible during live inference

---

## 🌐 Interactive Web Demo

The project includes a **custom-built web interface** (no Streamlit) that:

- Explains the project before interaction
- Allows side-by-side inference comparison
- Displays latency for each model
- Highlights parameter efficiency trade-offs

The UI is served using **FastAPI + Jinja2 templates**.

---

## 🚀 Running the Demo

From the project root:

```bash
uvicorn src.inference.app:app --reload
Open: http://localhost:8000

## 🐳 Running with Docker

The Docker image contains only application code.
Trained models are mounted at runtime as volumes.

### Run locally

```bash
docker build -t llm-finetuning .
docker run \
  -p 8000:8000 \
  -v $(pwd)/experiments:/app/experiments \
  llm-finetuning