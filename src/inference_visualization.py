import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config
from src.models import build_model


def load_model(config, checkpoint_path, device=None):
    if device is None:
        device = config.device

    model = build_model(config)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    print(f"Model loaded from {checkpoint_path}")
    return model


def predict_single(model, spectrum, config, device=None):
    if device is None:
        device = config.device

    from src.data_processing import normalize_spectrum
    spectrum_normalized = normalize_spectrum(spectrum)
    spectrum_tensor = torch.FloatTensor(spectrum_normalized).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        redshift_pred, class_pred = model(spectrum_tensor)
        redshift = redshift_pred.cpu().numpy().flatten()[0]
        class_probs = torch.softmax(class_pred, dim=1).cpu().numpy()[0]
        class_idx = np.argmax(class_probs)

    return {
        "redshift": float(redshift),
        "class_index": int(class_idx),
        "class_name": config.class_names[class_idx],
        "class_probabilities": class_probs.tolist(),
    }


def predict_batch(model, spectra, config, device=None, batch_size=None):
    if device is None:
        device = config.device
    if batch_size is None:
        batch_size = config.batch_size

    from src.data_processing import normalize_spectrum
    normalized_spectra = []
    for spec in spectra:
        normalized = normalize_spectrum(spec)
        normalized_spectra.append(normalized)
    normalized_spectra = np.array(normalized_spectra)

    all_redshifts = []
    all_class_preds = []
    all_class_probs = []

    model.eval()
    with torch.no_grad():
        for i in range(0, len(normalized_spectra), batch_size):
            batch = normalized_spectra[i : i + batch_size]
            batch_tensor = torch.FloatTensor(batch).unsqueeze(1).to(device)
            redshift_pred, class_pred = model(batch_tensor)

            redshifts = redshift_pred.cpu().numpy().flatten()
            class_probs = torch.softmax(class_pred, dim=1).cpu().numpy()
            class_indices = np.argmax(class_probs, axis=1)

            all_redshifts.extend(redshifts)
            all_class_preds.extend(class_indices)
            all_class_probs.extend(class_probs)

    results = []
    for i in range(len(spectra)):
        results.append({
            "redshift": float(all_redshifts[i]),
            "class_index": int(all_class_preds[i]),
            "class_name": config.class_names[all_class_preds[i]],
            "class_probabilities": all_class_probs[i].tolist(),
        })

    return results


def plot_spectrum(spectrum, wavelength=None, title="Spectrum", save_path=None, show=False):
    plt.figure(figsize=(10, 4))
    if wavelength is None:
        wavelength = np.arange(len(spectrum))
    plt.plot(wavelength, spectrum, linewidth=0.8, color="steelblue")
    plt.xlabel("Wavelength / Pixel Index")
    plt.ylabel("Flux (Normalized)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


def plot_redshift_comparison(true_redshifts, pred_redshifts, title="Redshift Prediction", save_path=None, show=False):
    plt.figure(figsize=(8, 8))
    max_val = max(np.max(true_redshifts), np.max(pred_redshifts))
    min_val = min(np.min(true_redshifts), np.min(pred_redshifts))
    plt.plot([min_val, max_val], [min_val, max_val], "r--", label="y=x", alpha=0.7)
    plt.scatter(true_redshifts, pred_redshifts, s=10, alpha=0.6, color="steelblue", edgecolors="none")
    plt.xlabel("True Redshift")
    plt.ylabel("Predicted Redshift")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis("equal")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


def plot_residuals(true_redshifts, pred_redshifts, title="Redshift Residuals", save_path=None, show=False):
    residuals = pred_redshifts - true_redshifts
    delta_z = np.abs(residuals) / (1.0 + true_redshifts)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].scatter(true_redshifts, residuals, s=10, alpha=0.6, color="steelblue", edgecolors="none")
    axes[0].axhline(y=0, color="r", linestyle="--", alpha=0.7)
    axes[0].set_xlabel("True Redshift")
    axes[0].set_ylabel("Residual (pred - true)")
    axes[0].set_title("Redshift Residuals")
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(delta_z, bins=50, edgecolor="black", alpha=0.7, color="steelblue")
    axes[1].set_xlabel("|Δz| / (1 + z)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Δz Distribution")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


def plot_confusion_matrix(true_labels, pred_labels, class_names, title="Confusion Matrix", save_path=None, show=False):
    cm = confusion_matrix(true_labels, pred_labels)
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8, 6))
    im = plt.imshow(cm_normalized, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar(im)
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)

    thresh = cm_normalized.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                f"{cm_normalized[i, j]:.2f}\n({cm[i, j]})",
                ha="center",
                va="center",
                color="white" if cm_normalized[i, j] > thresh else "black",
                fontsize=9,
            )

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


def plot_training_history(history, title="Training History", save_path=None, show=False):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    epochs = range(1, len(history["train"]) + 1)

    axes[0, 0].plot(epochs, [h["total_loss"] for h in history["train"]], "b-", label="Train")
    axes[0, 0].plot(epochs, [h["total_loss"] for h in history["val"]], "r-", label="Validation")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Total Loss")
    axes[0, 0].set_title("Loss Curve")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(epochs, [h["mae"] for h in history["train"]], "b-", label="Train")
    axes[0, 1].plot(epochs, [h["mae"] for h in history["val"]], "r-", label="Validation")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("MAE")
    axes[0, 1].set_title("Redshift MAE")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(epochs, [h["accuracy"] for h in history["train"]], "b-", label="Train")
    axes[1, 0].plot(epochs, [h["accuracy"] for h in history["val"]], "r-", label="Validation")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Accuracy")
    axes[1, 0].set_title("Classification Accuracy")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(epochs, [h["sigma_ni"] for h in history["train"]], "b-", label="Train")
    axes[1, 1].plot(epochs, [h["sigma_ni"] for h in history["val"]], "r-", label="Validation")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("σ_NI")
    axes[1, 1].set_title("Redshift σ_NI")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


def generate_all_plots(model, test_loader, config, output_dir=None):
    if output_dir is None:
        output_dir = config.output_dir
    os.makedirs(output_dir, exist_ok=True)

    device = config.device
    model.eval()

    all_true_redshifts = []
    all_pred_redshifts = []
    all_true_labels = []
    all_pred_labels = []
    all_spectra = []

    with torch.no_grad():
        for spectra, redshifts, labels in test_loader:
            spectra = spectra.to(device)
            redshift_pred, class_pred = model(spectra)

            all_true_redshifts.extend(redshifts.numpy().flatten())
            all_pred_redshifts.extend(redshift_pred.cpu().numpy().flatten())
            all_true_labels.extend(labels.numpy().flatten())
            all_pred_labels.extend(torch.argmax(class_pred, dim=1).cpu().numpy())
            all_spectra.extend(spectra.cpu().numpy())

    all_true_redshifts = np.array(all_true_redshifts)
    all_pred_redshifts = np.array(all_pred_redshifts)
    all_true_labels = np.array(all_true_labels)
    all_pred_labels = np.array(all_pred_labels)

    plot_redshift_comparison(
        all_true_redshifts,
        all_pred_redshifts,
        title="Redshift Prediction Results",
        save_path=os.path.join(output_dir, "redshift_comparison.png"),
    )

    plot_residuals(
        all_true_redshifts,
        all_pred_redshifts,
        title="Redshift Residual Analysis",
        save_path=os.path.join(output_dir, "redshift_residuals.png"),
    )

    plot_confusion_matrix(
        all_true_labels,
        all_pred_labels,
        config.class_names,
        title="Classification Confusion Matrix",
        save_path=os.path.join(output_dir, "confusion_matrix.png"),
    )

    for i in range(min(5, len(all_spectra))):
        plot_spectrum(
            all_spectra[i].flatten(),
            title=f"Spectrum Sample {i} - True: {config.class_names[all_true_labels[i]]}, Pred: {config.class_names[all_pred_labels[i]]}\nz_true={all_true_redshifts[i]:.4f}, z_pred={all_pred_redshifts[i]:.4f}",
            save_path=os.path.join(output_dir, f"spectrum_sample_{i}.png"),
        )

    report = classification_report(
        all_true_labels, all_pred_labels, target_names=config.class_names, output_dict=True
    )
    report_path = os.path.join(output_dir, "classification_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"All plots saved to {output_dir}")
    return {
        "true_redshifts": all_true_redshifts,
        "pred_redshifts": all_pred_redshifts,
        "true_labels": all_true_labels,
        "pred_labels": all_pred_labels,
    }
