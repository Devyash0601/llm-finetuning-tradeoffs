# scripts/prepare_datasets.py

from src.data.load_datasets import load_ag_news, load_squad
from src.data.preprocess import (
    preprocess_ag_news,
    preprocess_squad,
    save_dataset
)
from src.data.tokenize import (
    get_tokenizer,
    tokenize_classification,
    tokenize_qa
)

MODEL_NAME = "facebook/opt-125m"


def prepare_ag_news():
    print("Preparing AG News...")
    dataset = load_ag_news(sample_size=10000)

    tokenizer = get_tokenizer(MODEL_NAME)

    train = dataset["train"].map(preprocess_ag_news)
    test = dataset["test"].map(preprocess_ag_news)

    train = train.map(lambda x: tokenize_classification(x, tokenizer))
    test = test.map(lambda x: tokenize_classification(x, tokenizer))

    save_dataset(train, "data/processed/ag_news/train")
    save_dataset(test, "data/processed/ag_news/test")

    print("AG News done.")


def prepare_squad():
    print("Preparing SQuAD...")
    dataset = load_squad(sample_size=5000)

    tokenizer = get_tokenizer(MODEL_NAME)

    train = dataset["train"].map(preprocess_squad)
    val = dataset["validation"].map(preprocess_squad)

    train = train.map(lambda x: tokenize_qa(x, tokenizer))
    val = val.map(lambda x: tokenize_qa(x, tokenizer))

    save_dataset(train, "data/processed/squad/train")
    save_dataset(val, "data/processed/squad/validation")

    print("SQuAD done.")


if __name__ == "__main__":
    prepare_ag_news()
    prepare_squad()
    print("All datasets prepared successfully.")