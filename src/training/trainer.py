# src/training/trainer.py

import time
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW


def train(
    model,
    dataset,
    device,
    epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
):
    """
    Generic training loop for causal language models.
    Works for both Full FT and LoRA.
    """

    model.train()

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )

    optimizer = AdamW(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        epoch_loss = 0.0
        start_time = time.time()

        for batch in dataloader:
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids
            )

            loss = outputs.loss
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        epoch_time = time.time() - start_time
        avg_loss = epoch_loss / len(dataloader)

        print(
            f"Epoch [{epoch+1}/{epochs}] | "
            f"Loss: {avg_loss:.4f} | "
            f"Time: {epoch_time:.2f}s"
        )

    return model