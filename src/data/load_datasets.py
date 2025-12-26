# src/data/load_datasets.py

from datasets import load_dataset


def load_ag_news():
    print("\nLoading AG News...")
    dataset = load_dataset("ag_news", split="train")
    print(f"AG News samples: {len(dataset)}")
    print("Sample:", dataset[0])
    return dataset


def load_cnn_dailymail():
    print("\nLoading CNN/DailyMail (3.0.0)...")
    dataset = load_dataset("cnn_dailymail", "3.0.0", split="train")
    print(f"CNN/DailyMail samples: {len(dataset)}")
    print("Sample keys:", dataset[0].keys())
    return dataset


def main():
    load_ag_news()
    load_cnn_dailymail()
    print("\nSTEP 1 COMPLETE: Script-free datasets loaded successfully.")


if __name__ == "__main__":
    main()