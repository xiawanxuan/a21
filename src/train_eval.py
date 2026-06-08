import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import sys
import json
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config
from src.models import build_model, count_parameters


class CombinedLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.redshift_loss = nn.MSELoss()
        self.classification_loss = nn.CrossEntropyLoss()
        self.redshift_weight = config.red_shift_loss_weight
        self.classification_weight = config.classification_loss_weight

    def forward(self, redshift_pred, redshift_true, class_pred, class_true):
        r_loss = self.redshift_loss(redshift_pred, redshift_true)
        c_loss = self.classification_loss(class_pred, class_true.squeeze())
        total_loss = self.redshift_weight * r_loss + self.classification_weight * c_loss
        return total_loss, r_loss, c_loss


def compute_metrics(redshift_pred, redshift_true, class_pred, class_true):
    redshift_pred_np = redshift_pred.cpu().numpy().flatten()
    redshift_true_np = redshift_true.cpu().numpy().flatten()

    mae = np.mean(np.abs(redshift_pred_np - redshift_true_np))
    rmse = np.sqrt(np.mean((redshift_pred_np - redshift_true_np) ** 2))

    delta_z = np.abs(redshift_pred_np - redshift_true_np) / (1.0 + redshift_true_np)
    sigma_ni = np.percentile(delta_z, 68.27)
    outlier_rate = np.sum(delta_z > 0.15) / len(delta_z)

    class_pred_labels = torch.argmax(class_pred, dim=1)
    class_true_labels = class_true.squeeze()
    correct = (class_pred_labels == class_true_labels).sum().item()
    accuracy = correct / len(class_true_labels)

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "sigma_ni": float(sigma_ni),
        "outlier_rate": float(outlier_rate),
        "accuracy": float(accuracy),
    }


def train_one_epoch(model, train_loader, criterion, optimizer, device, config, epoch):
    model.train()
    total_loss = 0.0
    total_r_loss = 0.0
    total_c_loss = 0.0

    all_redshift_preds = []
    all_redshift_trues = []
    all_class_preds = []
    all_class_trues = []

    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
    for batch_idx, (spectra, redshifts, labels) in enumerate(pbar):
        spectra = spectra.to(device)
        redshifts = redshifts.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        redshift_pred, class_pred = model(spectra)

        loss, r_loss, c_loss = criterion(redshift_pred, redshifts, class_pred, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_r_loss += r_loss.item()
        total_c_loss += c_loss.item()

        all_redshift_preds.append(redshift_pred.detach())
        all_redshift_trues.append(redshifts)
        all_class_preds.append(class_pred.detach())
        all_class_trues.append(labels)

        if (batch_idx + 1) % config.log_interval == 0:
            avg_loss = total_loss / (batch_idx + 1)
            pbar.set_postfix({"loss": f"{avg_loss:.4f}"})

    all_redshift_preds = torch.cat(all_redshift_preds, dim=0)
    all_redshift_trues = torch.cat(all_redshift_trues, dim=0)
    all_class_preds = torch.cat(all_class_preds, dim=0)
    all_class_trues = torch.cat(all_class_trues, dim=0)

    metrics = compute_metrics(
        all_redshift_preds, all_redshift_trues, all_class_preds, all_class_trues
    )

    avg_loss = total_loss / len(train_loader)
    avg_r_loss = total_r_loss / len(train_loader)
    avg_c_loss = total_c_loss / len(train_loader)

    return {
        "total_loss": avg_loss,
        "redshift_loss": avg_r_loss,
        "classification_loss": avg_c_loss,
        **metrics,
    }


def validate(model, val_loader, criterion, device, epoch, phase="Val"):
    model.eval()
    total_loss = 0.0
    total_r_loss = 0.0
    total_c_loss = 0.0

    all_redshift_preds = []
    all_redshift_trues = []
    all_class_preds = []
    all_class_trues = []

    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f"Epoch {epoch} [{phase}]")
        for spectra, redshifts, labels in pbar:
            spectra = spectra.to(device)
            redshifts = redshifts.to(device)
            labels = labels.to(device)

            redshift_pred, class_pred = model(spectra)

            loss, r_loss, c_loss = criterion(redshift_pred, redshifts, class_pred, labels)

            total_loss += loss.item()
            total_r_loss += r_loss.item()
            total_c_loss += c_loss.item()

            all_redshift_preds.append(redshift_pred)
            all_redshift_trues.append(redshifts)
            all_class_preds.append(class_pred)
            all_class_trues.append(labels)

    all_redshift_preds = torch.cat(all_redshift_preds, dim=0)
    all_redshift_trues = torch.cat(all_redshift_trues, dim=0)
    all_class_preds = torch.cat(all_class_preds, dim=0)
    all_class_trues = torch.cat(all_class_trues, dim=0)

    metrics = compute_metrics(
        all_redshift_preds, all_redshift_trues, all_class_preds, all_class_trues
    )

    avg_loss = total_loss / len(val_loader)
    avg_r_loss = total_r_loss / len(val_loader)
    avg_c_loss = total_c_loss / len(val_loader)

    return {
        "total_loss": avg_loss,
        "redshift_loss": avg_r_loss,
        "classification_loss": avg_c_loss,
        **metrics,
    }


def get_optimizer(model, config):
    optimizer = optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    return optimizer


def get_scheduler(optimizer, config, train_loader):
    total_steps = len(train_loader) * config.num_epochs
    warmup_steps = len(train_loader) * config.warmup_epochs

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return scheduler


def save_checkpoint(model, optimizer, epoch, metrics, config, filename=None):
    if filename is None:
        filename = f"checkpoint_epoch_{epoch}.pth"
    save_path = os.path.join(config.model_dir, filename)
    os.makedirs(config.model_dir, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "config": config.__dict__,
    }
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to {save_path}")


def load_checkpoint(model, optimizer, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    metrics = checkpoint.get("metrics", {})
    print(f"Checkpoint loaded from {checkpoint_path} (epoch {epoch})")
    return epoch, metrics


def save_training_history(history, config, filename="training_history.json"):
    save_path = os.path.join(config.result_dir, filename)
    os.makedirs(config.result_dir, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to {save_path}")


def train(config, train_loader, val_loader, model=None):
    device = config.device
    print(f"Using device: {device}")

    if model is None:
        model = build_model(config)
    model = model.to(device)

    total_params, trainable_params = count_parameters(model)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    criterion = CombinedLoss(config)
    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config, train_loader)

    best_val_loss = float("inf")
    best_epoch = 0
    history = {"train": [], "val": []}

    print("Starting training...")
    for epoch in range(1, config.num_epochs + 1):
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, config, epoch
        )
        val_metrics = validate(model, val_loader, criterion, device, epoch, phase="Val")

        history["train"].append(train_metrics)
        history["val"].append(val_metrics)

        print(f"\nEpoch {epoch}/{config.num_epochs}")
        print(f"  Train - Loss: {train_metrics['total_loss']:.4f} | "
              f"MAE: {train_metrics['mae']:.4f} | "
              f"Acc: {train_metrics['accuracy']:.4f}")
        print(f"  Val   - Loss: {val_metrics['total_loss']:.4f} | "
              f"MAE: {val_metrics['mae']:.4f} | "
              f"Acc: {val_metrics['accuracy']:.4f}")

        if val_metrics["total_loss"] < best_val_loss:
            best_val_loss = val_metrics["total_loss"]
            best_epoch = epoch
            save_checkpoint(model, optimizer, epoch, val_metrics, config, filename="best_model.pth")

        if epoch % config.save_interval == 0:
            save_checkpoint(model, optimizer, epoch, val_metrics, config)

        scheduler.step()

    save_training_history(history, config)

    print(f"\nTraining complete. Best model at epoch {best_epoch} with val loss {best_val_loss:.4f}")
    return model, history


def evaluate(model, test_loader, config):
    device = config.device
    model = model.to(device)
    criterion = CombinedLoss(config)

    test_metrics = validate(model, test_loader, criterion, device, epoch=0, phase="Test")

    print("\nTest Results:")
    print(f"  Total Loss: {test_metrics['total_loss']:.4f}")
    print(f"  Redshift MAE: {test_metrics['mae']:.4f}")
    print(f"  Redshift RMSE: {test_metrics['rmse']:.4f}")
    print(f"  Sigma_NI: {test_metrics['sigma_ni']:.4f}")
    print(f"  Outlier Rate: {test_metrics['outlier_rate']:.4f}")
    print(f"  Classification Accuracy: {test_metrics['accuracy']:.4f}")

    return test_metrics
