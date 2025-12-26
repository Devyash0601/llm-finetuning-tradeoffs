# src/training/train_full.py

from datasets import load_dataset
from transformers import AutoTokenizer
from src.models.base_model import load_base_model
from src.training.trainer import train

MODEL_NAME = "facebook/opt-125m"
DATA_PATH = "data/processed/news_instruction.jsonl"
OUTPUT_PATH = "experiments/news_instruction/full"

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

    # Full fine-tuning (more conservative than LoRA)
    trained_model = train(
        model=model,
        dataset=dataset,
        device=device,
        epochs=2,            # fewer epochs for full FT
        batch_size=2,        # smaller batch (memory)
        learning_rate=5e-5   # lower LR for stability
    )

    trained_model.save_pretrained(OUTPUT_PATH)
    print("Full fine-tuning completed and model saved.")


if __name__ == "__main__":
    main()