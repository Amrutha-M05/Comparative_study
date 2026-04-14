from __future__ import annotations
import copy
import os
import torch
import torch.nn as nn
from tqdm import tqdm

from torch.amp import autocast, GradScaler
from src.utils import AverageMeter


def train_one_epoch(model, loader, optimizer, criterion, device, model_name: str, scaler, use_amp: bool):
    model.train()
    losses = AverageMeter()

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].float().unsqueeze(1).to(device, non_blocking=True)

        optimizer.zero_grad()

        if use_amp:
            with autocast(device_type="cuda"):
                if model_name.lower() == "inception_v3":
                    outputs, aux_outputs = model(images)
                    loss_main = criterion(outputs, labels)
                    loss_aux = criterion(aux_outputs, labels)
                    loss = loss_main + 0.4 * loss_aux
                else:
                    outputs = model(images)
                    loss = criterion(outputs, labels)
        else:
            if model_name.lower() == "inception_v3":
                outputs, aux_outputs = model(images)
                loss_main = criterion(outputs, labels)
                loss_aux = criterion(aux_outputs, labels)
                loss = loss_main + 0.4 * loss_aux
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)

        if use_amp:
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        losses.update(loss.item(), images.size(0))

    return losses.avg


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    losses = AverageMeter()
    probs_all, labels_all = [], []

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].float().unsqueeze(1).to(device, non_blocking=True)

        outputs = model(images)

        if isinstance(outputs, tuple):
            outputs = outputs[0]

        loss = criterion(outputs, labels)
        probs = torch.sigmoid(outputs).squeeze(1)

        losses.update(loss.item(), images.size(0))
        probs_all.extend(probs.cpu().numpy().tolist())
        labels_all.extend(labels.squeeze(1).cpu().numpy().tolist())

    return losses.avg, probs_all, labels_all


def fit(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    criterion,
    device,
    epochs,
    model_name: str,
    metric_fn,
    patience: int = 5,
    min_delta: float = 1e-4,
    checkpoint_dir: str = None,
    start_epoch: int = 0,
    best_auc: float = -1.0,
):
    use_amp = torch.cuda.is_available()
    scaler = GradScaler(device="cuda") if use_amp else None

    best_model_wts = copy.deepcopy(model.state_dict())
    best_epoch = 0
    history = []

    epochs_no_improve = 0

    for epoch in range(start_epoch, epochs):

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            model_name=model_name,
            scaler=scaler,
            use_amp=use_amp,
        )

        val_loss, val_probs, val_labels = validate(model, val_loader, criterion, device)
        val_metrics = metric_fn(val_labels, val_probs)

        if scheduler is not None:
            try:
                scheduler.step(val_metrics["roc_auc"])
            except Exception:
                scheduler.step()

        current_auc = val_metrics["roc_auc"]

        # ===== BEST MODEL =====
        is_best = current_auc > best_auc + min_delta
        if is_best:
            best_auc = current_auc
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            best_epoch = epoch + 1
        else:
            epochs_no_improve += 1

        # ===== LOG =====
        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_auc={current_auc:.4f} | "
            f"val_sens={val_metrics['sensitivity']:.4f} | "
            f"val_spec={val_metrics['specificity']:.4f} | "
            f"no_improve={epochs_no_improve}"
        )

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **val_metrics,
        })

        # ===== CHECKPOINT SAVE =====
        if checkpoint_dir is not None:
            os.makedirs(checkpoint_dir, exist_ok=True)

            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_auc": best_auc,
            }, os.path.join(checkpoint_dir, "last.pth"))

            if is_best:
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_auc": best_auc,
                }, os.path.join(checkpoint_dir, "best.pth"))

        # ===== EARLY STOP =====
        if epochs_no_improve >= patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs.")
            break

    print(f"\nBest AUC: {best_auc:.4f} at epoch {best_epoch}")

    model.load_state_dict(best_model_wts)
    return model, history