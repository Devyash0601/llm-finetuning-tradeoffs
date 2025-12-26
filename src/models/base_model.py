# src/models/base_model.py

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_device():
    """
    Select the best available device.
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def load_base_model(model_name: str):
    """
    Load pretrained causal language model and move to device.
    """
    device = get_device()

    model = AutoModelForCausalLM.from_pretrained(
        model_name
    )

    model.to(device)

    return model, device

def load_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # OPT does not have pad token by default
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer