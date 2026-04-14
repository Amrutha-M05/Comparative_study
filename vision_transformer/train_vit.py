from __future__ import annotations

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import json
import pandas as pd

from torch.optim import AdamW
from models_vit import get_vit
from src.train import fit
from src.metrics import compute_classification_metrics as compute_metrics
from src.data_loader import get_loaders


def train_vit(
    train_loader,
    val_loader,
    device,
    epochs=10,
    lr=1e-4,
    weight_decay=1e-4,
    checkpoint_dir="checkpoints_vit",
):

    print("🚀 Initializing ViT model...")

    model = get_vit().to(device)

    criterion = torch.nn.BCEWithLogitsLoss()

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    print("🚀 Starting training loop...")

    model, history = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=device,
        epochs=epochs,
        model_name="vit",
        metric_fn=compute_metrics,
        checkpoint_dir=checkpoint_dir,
    )

    # =========================
    # SAVE FINAL RESULTS (ViT)
    # =========================
    os.makedirs("outputs/logs", exist_ok=True)

    json_path = "outputs/logs/vit_history.json"
    csv_path = "outputs/logs/vit_history.csv"

    with open(json_path, "w") as f:
        json.dump(history, f, indent=4)

    pd.DataFrame(history).to_csv(csv_path, index=False)

    print("\n📊 ViT logs saved:")
    print(f"JSON -> {json_path}")
    print(f"CSV  -> {csv_path}")

    return model, history


# =========================
# MAIN EXECUTION
# =========================
if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("===================================")
    print("🚀 Vision Transformer Training Start")
    print("Device:", device)
    print("===================================")

    print("📦 Loading dataset...")

    train_loader, val_loader = get_loaders(
        root="LAG",
        batch_size=16
    )

    print("✅ Dataset loaded")
    print("🔥 Starting ViT training...")

    model, history = train_vit(
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=10
    )

    print("🎉 ViT training completed successfully!")