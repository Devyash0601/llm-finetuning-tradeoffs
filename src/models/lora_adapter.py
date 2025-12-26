# src/models/lora_adapter.py

from peft import LoraConfig, get_peft_model


def apply_lora(
    model,
    r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05
):
    """
    Apply LoRA adapters to the model.
    """

    config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"]
    )

    lora_model = get_peft_model(model, config)

    lora_model.print_trainable_parameters()

    return lora_model