# src/data/prepare_instruction_data.py

import json
import random
from datasets import load_dataset
from pathlib import Path

OUTPUT_PATH = Path("data/processed/news_instruction.jsonl")

AG_NEWS_SAMPLES = 100_000
CNN_SAMPLES = 30_000
MAX_TEXT_LENGTH = 300  # characters, keep things short


def truncate(text, max_len=MAX_TEXT_LENGTH):
    text = text.replace("\n", " ").strip()
    return text[:max_len]


def prepare_ag_news():
    print("Preparing AG News...")
    dataset = load_dataset("ag_news", split="train")
    dataset = dataset.shuffle(seed=42).select(range(AG_NEWS_SAMPLES))

    records = []
    for row in dataset:
        prompt = f"Explain the following topic in a neutral news-reporting style"
        answer = truncate(row["text"])
        records.append({"prompt": prompt, "answer": answer})

    print(f"AG News prepared: {len(records)} samples")
    return records


def prepare_cnn_dailymail():
    print("Preparing CNN/DailyMail...")
    dataset = load_dataset("cnn_dailymail", "3.0.0", split="train")
    dataset = dataset.shuffle(seed=42).select(range(CNN_SAMPLES))

    records = []
    for row in dataset:
        prompt = "Summarize the key facts from the following news article"
        answer = truncate(row["highlights"])
        records.append({"prompt": prompt, "answer": answer})

    print(f"CNN/DailyMail prepared: {len(records)} samples")
    return records


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    data = []
    data.extend(prepare_ag_news())
    data.extend(prepare_cnn_dailymail())

    random.shuffle(data)

    with open(OUTPUT_PATH, "w") as f:
        for item in data:
            record = {
                "text": f"Prompt: {item['prompt']}\nAnswer: {item['answer']}"
            }
            f.write(json.dumps(record) + "\n")

    print(f"\nSTEP 2 COMPLETE")
    print(f"Total samples written: {len(data)}")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()