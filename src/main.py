from __future__ import annotations

import json
import os

import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import transforms

from src.dataset import LAGDataset, collect_split_samples
from src.evaluate import predict, evaluate_predictions, save_metrics
from src.metrics import compute_classification_metrics
from src.models import build_model
from src.train import fit
from src.utils import ensure_dir, set_seed


# 🔥 Enable faster GPU ops
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False


# ================= CONFIG ================= #
CONFIG = {
    "seed": 42,
    "data_root": r"c:\Users\anand\OneDrive\Desktop\Research\glaucoma_cnn_lag\LAG",
    "output_dir": "outputs",
    "model_name": "mobilenet_v2",
    "batch_size": 16,
    "epochs": 20,
    "lr": 1e-4,
    "num_workers": 0,
    "input_size": 224,
    "threshold": 0.5,
    "pretrained": True,
    "val_size": 0.15,
}


# ================= TRANSFORMS ================= #
def get_transforms(input_size: int, model_name: str):
    if model_name.lower() == "inception_v3":
        input_size = 299

    train_tfms = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    valid_tfms = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    return train_tfms, valid_tfms


# ================= DATALOADER ================= #
def build_dataloaders(config: dict):

    train_samples = collect_split_samples(config["data_root"], "train")
    test_samples = collect_split_samples(config["data_root"], "test")

    if len(train_samples) == 0:
        raise ValueError("No training images found!")

    if len(test_samples) == 0:
        raise ValueError("No test images found!")

    train_df = pd.DataFrame(train_samples)

    train_df, val_df = train_test_split(
        train_df,
        test_size=config["val_size"],
        stratify=train_df["label"],
        random_state=config["seed"],
    )

    train_samples = train_df.to_dict("records")
    val_samples = val_df.to_dict("records")

    train_tfms, valid_tfms = get_transforms(
        config["input_size"],
        config["model_name"],
    )

    train_ds = LAGDataset(train_samples, transform=train_tfms)
    val_ds = LAGDataset(val_samples, transform=valid_tfms)
    test_ds = LAGDataset(test_samples, transform=valid_tfms)

    persistent = config["num_workers"] > 0

    train_loader = DataLoader(
        train_ds,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=True,
        persistent_workers=False,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=True,
        persistent_workers=persistent,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=True,
        persistent_workers=persistent,
    )

    return train_loader, val_loader, test_loader, train_samples, val_samples, test_samples


# ================= MAIN TRAINING ================= #
def run_experiment(config: dict):
    set_seed(config["seed"])

    output_model_dir = os.path.join(config["output_dir"], config["model_name"])
    checkpoint_dir = os.path.join(output_model_dir, "checkpoints")
    prediction_dir = os.path.join(output_model_dir, "predictions")

    ensure_dir(output_model_dir)
    ensure_dir(checkpoint_dir)
    ensure_dir(prediction_dir)

    train_loader, val_loader, test_loader, train_samples, val_samples, test_samples = build_dataloaders(config)

    print(f"\nModel: {config['model_name']}")
    print(f"Train samples: {len(train_samples)}")
    print(f"Validation samples: {len(val_samples)}")
    print(f"Test samples: {len(test_samples)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    model = build_model(
        config["model_name"],
        num_classes=1,
        pretrained=config["pretrained"],
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3,
    )

    # ================= RESUME ================= #
    checkpoint_path = os.path.join(checkpoint_dir, "last.pth")

    start_epoch = 0
    best_auc = -1.0

    if os.path.exists(checkpoint_path):
        print("\n🔄 Resuming from checkpoint...")

        checkpoint = torch.load(checkpoint_path, map_location=device)

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        start_epoch = checkpoint["epoch"] + 1
        best_auc = checkpoint.get("best_auc", -1.0)

        print(f"Resumed from epoch {start_epoch}, best_auc={best_auc:.4f}")

    # ================= TRAIN ================= #
    model, history = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=device,
        epochs=config["epochs"],
        model_name=config["model_name"],
        metric_fn=compute_classification_metrics,
        patience=5,
        min_delta=1e-4,
        checkpoint_dir=checkpoint_dir,
        start_epoch=start_epoch,
        best_auc=best_auc,
    )

    # ================= SAVE HISTORY ================= #
    hist_path = os.path.join(output_model_dir, "history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)

    # ================= TEST ================= #
    df_pred = predict(model, test_loader, device)

    pred_path = os.path.join(prediction_dir, "test_predictions.csv")
    df_pred.to_csv(pred_path, index=False)

    test_metrics = evaluate_predictions(df_pred, threshold=config["threshold"])

    metrics_path = os.path.join(output_model_dir, "test_metrics.json")
    save_metrics(test_metrics, metrics_path)

    print("\nFinal Test Metrics:")
    for k, v in test_metrics.items():
        print(f"{k}: {v}")

    print("\nSaved outputs:")
    print(f"Checkpoints → {checkpoint_dir}")
    print(f"History → {hist_path}")
    print(f"Predictions → {pred_path}")
    print(f"Metrics → {metrics_path}")

    return {
        "model": model,
        "history": history,
        "test_metrics": test_metrics,
    }


# ================= ENTRY ================= #
if __name__ == "__main__":
    run_experiment(CONFIG)