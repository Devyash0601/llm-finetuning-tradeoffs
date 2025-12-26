# src/training/train_lora.py

from datasets import load_dataset
from transformers import AutoTokenizer
from src.models.base_model import load_base_model
from src.models.lora_adapter import apply_lora
from src.training.trainer import train

MODEL_NAME = "facebook/opt-125m"
DATA_PATH = "data/processed/news_instruction.jsonl"
OUTPUT_PATH = "experiments/news_instruction/lora"

MAX_LENGTH = 256


def tokenize_function(examples, tokenizer):
    tokens = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens


def main():
    # Load base model
    model, device = load_base_model(MODEL_NAME)

    # Apply LoRA
    model = apply_lora(
        model,
        r=32,
        lora_alpha=64,
        lora_dropout=0.05
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load instruction dataset
    dataset = load_dataset(
        "json",
        data_files=DATA_PATH,
        split="train"
    )

    dataset = dataset.shuffle(seed=42)

    dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=["text"]
    )

    dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"]
    )

    # Train LoRA
    trained_model = train(
        model=model,
        dataset=dataset,
        device=device,
        epochs=3,
        batch_size=4,
        learning_rate=2e-4  # higher LR for LoRA
    )

    trained_model.save_pretrained(OUTPUT_PATH)
    print("LoRA fine-tuning completed and model saved.")


if __name__ == "__main__":
    main()