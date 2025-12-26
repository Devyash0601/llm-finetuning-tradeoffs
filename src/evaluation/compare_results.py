# src/evaluation/compare_results.py

import json


def load_results(path):
    with open(path, "r") as f:
        return json.load(f)


def compare(full_path, lora_path):
    full = load_results(full_path)
    lora = load_results(lora_path)

    print("\n=== Model Comparison ===\n")
    print(f"{'Metric':<15}{'Full FT':<15}{'LoRA':<15}")
    print("-" * 45)
    print(f"{'Loss':<15}{full['loss']:<15.4f}{lora['loss']:<15.4f}")
    print(f"{'Perplexity':<15}{full['perplexity']:<15.2f}{lora['perplexity']:<15.2f}")


if __name__ == "__main__":
    compare(
        "results/metrics/full_ag_news.json",
        "results/metrics/lora_ag_news.json"
    )