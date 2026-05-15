import csv
from pathlib import Path

import matplotlib.pyplot as plt


class LossPlotter:
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save(self, history, name="loss"):
        """
        history: list of dicts with keys 'epoch', 'train_l1', 'val_l1'
        name:    base filename (without extension)
        """
        epochs = [e["epoch"] for e in history]
        train_l1 = [e["train_l1"] for e in history]
        val_l1 = [e["val_l1"] for e in history]

        csv_path = self.output_dir / f"{name}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_l1", "val_l1"])
            for row in history:
                writer.writerow([row["epoch"], row["train_l1"], row["val_l1"]])

        plt.figure(figsize=(10, 5))
        plt.plot(epochs, train_l1, label="Train L1")
        plt.plot(epochs, val_l1, label="Val L1")
        plt.xlabel("Epoch")
        plt.ylabel("L1 Loss")
        plt.title("Loss Curve")
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.output_dir / f"{name}.png")
        plt.close()
