# src/evaluate.py

from __future__ import annotations
import json
import pandas as pd
import torch
from tqdm import tqdm

from src.metrics import compute_classification_metrics


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    records = []

    for batch in tqdm(loader, desc="Testing", leave=False):
        images = batch["image"].to(device)
        image_names = batch["image_name"]
        labels = batch["label"]

        outputs = model(images)
        probs = torch.sigmoid(outputs).squeeze(1).cpu().numpy()

        for name, label, prob in zip(image_names, labels.numpy(), probs):
            records.append({
                "image_name": name,
                "label": int(label),
                "probability": float(prob),
            })

    return pd.DataFrame(records)


def evaluate_predictions(df_pred: pd.DataFrame, threshold: float = 0.5):
    metrics = compute_classification_metrics(
        df_pred["label"].values,
        df_pred["probability"].values,
        threshold=threshold,
    )
    return metrics


def save_metrics(metrics: dict, output_path: str):
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)
