# src/inference/app.py

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from src.inference.predict import InferenceEngine


# -------------------------------------------------------------------
# Paths (robust, absolute)
# -------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]  # project root

STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"


# -------------------------------------------------------------------
# Model configuration (Hugging Face)
# -------------------------------------------------------------------

BASE_MODEL = "facebook/opt-125m"

# Instruction-tuned models (news domain)
FULL_MODEL_ID = "devyash06/opt125m-news-instruction-full"
LORA_MODEL_ID = "devyash06/opt125m-lora-news"


# -------------------------------------------------------------------
# FastAPI app
# -------------------------------------------------------------------

app = FastAPI(
    title="LLM PEFT Comparison API",
    description="Compare Full Fine-Tuning vs LoRA on an instruction-tuned news task",
)


# -------------------------------------------------------------------
# Static files & templates
# -------------------------------------------------------------------

app.mount(
    "/static",
    StaticFiles(directory=STATIC_DIR),
    name="static",
)

templates = Jinja2Templates(directory=TEMPLATES_DIR)


# -------------------------------------------------------------------
# Load models ONCE at startup
# -------------------------------------------------------------------

full_engine = InferenceEngine(
    base_model_name=BASE_MODEL,
    model_id=FULL_MODEL_ID,
    is_lora=False,
)

lora_engine = InferenceEngine(
    base_model_name=BASE_MODEL,
    model_id=LORA_MODEL_ID,
    is_lora=True,
)


# -------------------------------------------------------------------
# Request schema
# -------------------------------------------------------------------

class InferenceRequest(BaseModel):
    text: str
    max_new_tokens: int = 80


# -------------------------------------------------------------------
# Routes
# -------------------------------------------------------------------

# UI route
@app.get("/", response_class=HTMLResponse)
def serve_ui(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request},
    )


# Full fine-tuned model inference
@app.post("/predict/full")
def predict_full(req: InferenceRequest):
    return full_engine.generate(
        text=req.text,
        max_new_tokens=req.max_new_tokens,
    )


# LoRA model inference
@app.post("/predict/lora")
def predict_lora(req: InferenceRequest):
    return lora_engine.generate(
        text=req.text,
        max_new_tokens=req.max_new_tokens,
    )