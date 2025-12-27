# LLM Fine-Tuning Trade-offs  
### Full Fine-Tuning vs LoRA on OPT-125M

This project demonstrates the **practical trade-offs** between **Full Fine-Tuning** and **Parameter-Efficient Fine-Tuning (LoRA)** for Large Language Models, using **OPT-125M** trained on **news-style instruction data**.

A production-ready **FastAPI + Docker** application is deployed to **Hugging Face Spaces**, allowing real-time comparison of:
- Output quality
- Latency
- Stability
- Parameter efficiency

---

## 🚀 Live Demo

👉 **Hugging Face Space:**  
https://huggingface.co/spaces/devyash06/llm-finetuning-tradeoffs

---

## 🧠 Motivation

Large Language Models are expensive to train and deploy.

- **Full Fine-Tuning** updates all model parameters → best performance, highest cost  
- **LoRA (Low-Rank Adaptation)** updates ~1% of parameters → massive savings, potential quality loss  

This project makes these **trade-offs visible, measurable, and reproducible**.

---

## 🧪 What This Project Shows

| Aspect | Full Fine-Tuning | LoRA |
|------|-----------------|------|
| Trainable Params | 100% | ~0.9% |
| Training Cost | High | Low |
| Inference Quality | Strong | Weaker / noisier |
| Deployment Size | Large | Small |
| Latency | Lower | Slightly higher |

The **LoRA degradation is intentional and informative**, not a bug.

---

## ⚠️ Important Disclaimer

> **Demo Notice**  
> This system is trained exclusively on **news-style datasets** (AG News, CNN/DailyMail).  
> It is designed to demonstrate **fine-tuning trade-offs**, **not factual question answering**.  
> Outputs may be stylistically fluent but are **not guaranteed to be correct or up-to-date**.

---

## 🧱 Architecture
.
├── src
│   ├── data            # Dataset processing
│   ├── models          # Base model utilities
│   ├── training        # Full FT & LoRA training scripts
│   ├── inference       # FastAPI inference engine
│   └── utils
├── experiments
│   └── news_instruction
│       ├── full        # Full fine-tuned model
│       └── lora        # LoRA adapter
├── static              # Frontend assets
├── templates           # HTML UI
├── Dockerfile
├── requirements.txt
└── README.md

---

## 📚 Datasets Used

- **AG News**
- **CNN / DailyMail**
- Instruction-style prompts generated from news articles

All datasets are used **only for research and demonstration purposes**.

---

## 🏋️ Training

### Full Fine-Tuning
```bash
python -m src.training.train_full
```
### LoRA Fine-Tuning
```bash
python -m src.training.train_lora
```

### Local Inference
python - << 'EOF'
from src.inference.predict import InferenceEngine

engine = InferenceEngine(
    "facebook/opt-125m",
    "experiments/news_instruction/lora",
    is_lora=True
)

print(engine.generate("What is happening in global economic news?"))
EOF

## 🐳 Docker
Build : docker build -t llm-finetuning .
Run : docker run -p 8000:8000 llm-finetuning
Then open:
http://localhost:8000

## ☁️ Deployment

Hugging Face Spaces (Docker SDK)
	•	Fully containerized FastAPI app
	•	Models loaded from Hugging Face Hub
	•	CPU-compatible deployment

This avoids:
	•	Railway build timeouts
	•	Cold-start GPU costs
	•	Cloud credential complexity

🧠 Key Takeaways
	•	LoRA is not a free lunch
	•	Parameter efficiency comes with expressiveness loss
	•	Full fine-tuning still matters for domain alignment
	•	Deployment constraints heavily influence model choice

⸻

## 📌 Future Work
	•	Quantitative evaluation (ROUGE / BLEU)
	•	Memory profiling (VRAM / RAM)
	•	Instruction-tuned base models
	•	Multi-task LoRA adapters
	•	GPU-backed inference comparison

⸻

## 👤 Author

Devashish Komiya
B.Tech AIML @ BIT Mesra
Interested in:
	•	LLM systems
	•	Efficient fine-tuning
	•	Applied ML research

GitHub: https://github.com/Devyash0601
Hugging Face: https://huggingface.co/devyash06

⸻

## ⭐ Acknowledgements
	•	Hugging Face Transformers & PEFT
	•	Meta OPT models
	•	Open research on parameter-efficient fine-tuning

