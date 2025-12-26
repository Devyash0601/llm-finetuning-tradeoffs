# src/evaluation/metrics.py

import math
import torch


def compute_perplexity(loss: float) -> float:
    """
    Convert cross-entropy loss to perplexity.
    """
    return math.exp(loss)


@torch.no_grad()
def evaluate_loss(model, dataloader, device):
    """
    Evaluate average loss over dataset.
    """
    model.eval()
    total_loss = 0.0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids
        )

        total_loss += outputs.loss.item()

    avg_loss = total_loss / len(dataloader)
    ppl = compute_perplexity(avg_loss)

    return avg_loss, ppl