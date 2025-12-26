# src/inference/predict.py

import time
import html
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from src.models.base_model import get_device


def clean_text(text: str) -> str:
    """
    Cleans common artifacts learned from news datasets:
    - HTML entities (&#39;, &quot;)
    - tokenizer artifacts
    - broken unicode characters
    - excessive whitespace
    """
    text = html.unescape(text)
    text = text.replace("<endoftext>", "")
    text = text.replace("�", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


class InferenceEngine:
    def __init__(self, base_model_name: str, model_path: str, is_lora: bool = False):
        self.device = get_device()

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Base model
        base_model = AutoModelForCausalLM.from_pretrained(base_model_name)

        # Load fine-tuned weights
        if is_lora:
            print("Loading LoRA-adapted model...")
            self.model = PeftModel.from_pretrained(base_model, model_path)
        else:
            print("Loading full fine-tuned model...")
            self.model = AutoModelForCausalLM.from_pretrained(model_path)

        self.model.to(self.device)
        self.model.eval()

    def _build_prompt(self, text: str) -> str:
        """
        Minimal instruction-style prompt.
        Same prompt for both models → fair comparison.
        """
        return (
            "Answer the following prompt as clearly and concisely as possible.\n\n"
            f"Prompt: {text}\n\n"
            "Answer:"
        )

    @torch.no_grad()
    def generate(self, text: str, max_new_tokens: int = 80):
        start_time = time.time()

        prompt = self._build_prompt(text)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=256,
            padding=True
        ).to(self.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.15,
            no_repeat_ngram_size=3,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id
        )

        decoded = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )

        # Extract only the answer portion
        if "Answer:" in decoded:
            decoded = decoded.split("Answer:", 1)[-1]

        cleaned_output = clean_text(decoded)

        latency = time.time() - start_time

        return {
            "output": cleaned_output,
            "latency_seconds": round(latency, 3)
        }