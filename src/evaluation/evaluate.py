# src/evaluation/evaluate.py

import math
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from src.models.base_model import get_device


def evaluate_model(
    model_path,
    base_model_name,
    dataset_path,
    is_lora=False,
    batch_size=4,
    max_length=256
):
    device = get_device()

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # -------- Load model correctly --------
    if is_lora:
        base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
        model = PeftModel.from_pretrained(base_model, model_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path)

    model.to(device)
    model.eval()

    # -------- Load evaluation data --------
    dataset = load_dataset(
        "json",
        data_files=dataset_path,
        split="train[:1000]"  # small eval slice
    )

    losses = []

    with torch.no_grad():
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i : i + batch_size]

            inputs = tokenizer(
                batch["text"],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length
            ).to(device)

            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                labels=inputs["input_ids"]
            )

            losses.append(outputs.loss.item())

    avg_loss = sum(losses) / len(losses)
    perplexity = math.exp(avg_loss)

    return {
        "loss": round(avg_loss, 4),
        "perplexity": round(perplexity, 2)
    }