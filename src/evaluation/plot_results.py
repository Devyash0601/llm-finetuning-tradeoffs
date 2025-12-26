# src/evaluation/plot_results.py

import matplotlib.pyplot as plt


def plot_performance_vs_params():
    # Trainable parameter percentages
    methods = [
        "Full Fine-Tuning",
        "LoRA (r=8)",
        "LoRA (r=16)"
    ]

    trainable_percent = [
        100.0,     # Full FT
        0.23,      # LoRA r=8
        0.47       # LoRA r=16
    ]

    loss_values = [
        1.27,      # Full FT loss
        3.48,      # LoRA r=8 loss
        3.43       # LoRA r=16 loss
    ]

    plt.figure(figsize=(8, 5))
    plt.scatter(trainable_percent, loss_values)

    for i, method in enumerate(methods):
        plt.annotate(
            method,
            (trainable_percent[i], loss_values[i]),
            textcoords="offset points",
            xytext=(5, 5)
        )

    plt.xlabel("Trainable Parameters (%)")
    plt.ylabel("Loss")
    plt.title("Performance vs Trainable Parameters\n(Full FT vs LoRA)")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("results/plots/performance_vs_params.png")
    plt.show()


if __name__ == "__main__":
    plot_performance_vs_params()