"""
src/modeling/evaluate.py
──────────────────────────
Comprehensive model evaluation:
  - Per-class metrics
  - Confusion matrix (saved as PNG)
  - Error analysis (worst predictions)
  - Speed benchmark
  - Grad-CAM visualization (trajectory importance)

Usage:
    python src/modeling/evaluate.py \
        --model-path models/spin_classifier_best.pt \
        --features data/processed/features.parquet \
        --output reports/evaluation/
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import yaml
from loguru import logger
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

from src.modeling.model import SPIN_CLASSES, build_model
from src.modeling.train_classifier import SCALAR_FEATURE_COLS, SpinDataset, get_device
from torch.utils.data import DataLoader


def load_checkpoint(path: Path, device: torch.device):
    ckpt = torch.load(path, map_location=device)
    cfg = ckpt["config"]
    clf_cfg = cfg["spin_classifier"]
    scalar_dim = ckpt["scalar_feature_dim"]

    model = build_model(
        model_type=clf_cfg["model_type"],
        scalar_feature_dim=scalar_dim,
        lstm_hidden=clf_cfg["lstm_hidden"],
        lstm_layers=clf_cfg["lstm_layers"],
        dropout=0.0,  # disable dropout at eval
    ).to(device)

    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, cfg, ckpt


def plot_confusion_matrix(cm: np.ndarray, classes: list, output_path: Path):
    fig, ax = plt.subplots(figsize=(8, 6))
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-6)

    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2%",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        ax=ax,
    )
    # Also overlay raw counts
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j + 0.5, i + 0.75, f"(n={cm[i, j]})", ha="center", va="center",
                    fontsize=8, color="gray")

    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title("Spin Classification — Confusion Matrix", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Confusion matrix saved → {output_path}")


def plot_roc_curves(y_true: list, y_probs: np.ndarray, classes: list, output_path: Path):
    y_bin = label_binarize(y_true, classes=list(range(len(classes))))
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]

    for i, (cls, color) in enumerate(zip(classes, colors)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_probs[:, i])
        auc = roc_auc_score(y_bin[:, i], y_probs[:, i])
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{cls} (AUC={auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — Spin Classification", fontsize=14)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"ROC curves saved → {output_path}")


def plot_speed_distribution(df: pd.DataFrame, output_path: Path):
    """Show speed and trajectory features by spin class."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    features_to_plot = [
        ("speed_mean", "Mean Ball Speed"),
        ("y_quadratic_coeff", "Vertical Arc (Quadratic Coeff)"),
        ("lateral_drift_max", "Max Lateral Drift"),
        ("curvature_mean", "Mean Trajectory Curvature"),
    ]

    colors = {"topspin": "#2196F3", "backspin": "#4CAF50", "sidespin": "#FF9800", "float": "#9C27B0"}

    for ax, (col, title) in zip(axes.flat, features_to_plot):
        if col not in df.columns:
            continue
        for spin in SPIN_CLASSES:
            vals = df[df["spin_label"] == spin][col].dropna()
            if len(vals) > 0:
                ax.hist(vals, bins=20, alpha=0.6, color=colors[spin], label=spin, density=True)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(col)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Feature Distributions by Spin Type", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Feature distributions saved → {output_path}")


def benchmark_inference_speed(model, device, n_runs: int = 200) -> dict:
    """Measure inference latency."""
    model.eval()
    traj = torch.randn(1, 30, 2).to(device)
    feats = torch.randn(1, len(SCALAR_FEATURE_COLS)).to(device)

    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model(traj, feats)

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(traj, feats)
        times.append(time.perf_counter() - t0)

    times_ms = [t * 1000 for t in times]
    return {
        "mean_ms": round(float(np.mean(times_ms)), 3),
        "p50_ms": round(float(np.percentile(times_ms, 50)), 3),
        "p95_ms": round(float(np.percentile(times_ms, 95)), 3),
        "p99_ms": round(float(np.percentile(times_ms, 99)), 3),
        "throughput_fps": round(1000 / float(np.mean(times_ms)), 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate the spin classifier")
    parser.add_argument("--model-path", type=str, default="models/spin_classifier_best.pt")
    parser.add_argument("--features", type=str, default="data/processed/features.parquet")
    parser.add_argument("--output", type=str, default="reports/evaluation")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = Path(args.model_path)
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        return

    with open(args.config) as f:
        base_cfg = yaml.safe_load(f)

    device = get_device(base_cfg)
    model, cfg, ckpt = load_checkpoint(model_path, device)
    logger.info(f"Loaded model from epoch {ckpt.get('epoch', '?')} | val_f1={ckpt.get('val_f1', '?'):.4f}")

    # Load data
    df = pd.read_parquet(args.features)
    labeled = df[df["spin_int"] >= 0].copy()

    if len(labeled) == 0:
        logger.error("No labeled data found in features file.")
        return

    test_ds = SpinDataset(labeled, sequence_len=cfg["spin_classifier"]["sequence_length"])
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)

    # Inference
    all_preds, all_labels, all_probs = [], [], []
    model.eval()
    with torch.no_grad():
        for traj, feats, labels in test_loader:
            traj, feats = traj.to(device), feats.to(device)
            logits = model(traj, feats)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
            all_probs.extend(probs)

    y_probs = np.array(all_probs)

    # ── Metrics ──────────────────────────────────────────────────────
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    report = classification_report(all_labels, all_preds, target_names=SPIN_CLASSES, zero_division=0)

    logger.success(f"\n{'='*50}")
    logger.success(f"EVALUATION RESULTS")
    logger.success(f"{'='*50}")
    logger.info(f"Accuracy: {acc:.4f}")
    logger.info(f"Macro F1: {f1:.4f}")
    logger.info(f"\n{report}")

    # ── Plots ─────────────────────────────────────────────────────────
    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm, SPIN_CLASSES, output_dir / "confusion_matrix.png")
    plot_roc_curves(all_labels, y_probs, SPIN_CLASSES, output_dir / "roc_curves.png")

    if "spin_label" in labeled.columns:
        plot_speed_distribution(labeled, output_dir / "feature_distributions.png")

    # ── Inference speed ───────────────────────────────────────────────
    speed = benchmark_inference_speed(model, device)
    logger.info(f"\nInference speed: {speed}")

    # ── Save full report ──────────────────────────────────────────────
    report_data = {
        "accuracy": acc,
        "macro_f1": f1,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "classes": SPIN_CLASSES,
        "inference_speed": speed,
        "n_samples": len(all_labels),
        "model_path": str(model_path),
    }

    with open(output_dir / "evaluation_report.json", "w") as f:
        json.dump(report_data, f, indent=2)
    logger.success(f"\nFull report saved → {output_dir}/evaluation_report.json")


if __name__ == "__main__":
    main()
