"""
src/modeling/train_classifier.py
──────────────────────────────────
Train the spin classifier with MLflow experiment tracking.

Features:
  - Train / val / test split
  - Class-weighted loss (handles imbalance)
  - Cosine annealing LR schedule
  - Early stopping
  - MLflow logging (params, metrics, artifacts, model)
  - Saves best checkpoint

Usage:
    python src/modeling/train_classifier.py \
        --config configs/config.yaml \
        --features data/processed/features.parquet \
        --experiment my_experiment
"""

import argparse
import json
import random
import time
from pathlib import Path

import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from loguru import logger
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from src.modeling.model import SPIN_CLASSES, build_model


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(cfg: dict) -> torch.device:
    device_str = cfg["detector"].get("device", "auto")
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)


# ─── Dataset ──────────────────────────────────────────────────────────────

SCALAR_FEATURE_COLS = [
    "speed_mean", "speed_std", "speed_max", "speed_skew", "speed_kurt",
    "acc_mean", "acc_std", "acc_max", "acc_skew", "acc_kurt",
    "jerk_mean", "jerk_std", "jerk_max",
    "vy_sign_changes", "hv_ratio",
    "arc_length", "chord_length", "sinuosity",
    "curvature_mean", "curvature_max", "y_quadratic_coeff",
    "lateral_drift_max", "lateral_drift_mean",
    "net_y_disp", "net_x_disp",
    "early_10f_y_coeff",
]


class SpinDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        scaler: StandardScaler = None,
        fit_scaler: bool = False,
        sequence_len: int = 30,
    ):
        # Only keep labeled rows
        df = df[df["spin_int"] >= 0].copy()
        self.labels = torch.tensor(df["spin_int"].values, dtype=torch.long)

        # Scalar features
        feat_cols = [c for c in SCALAR_FEATURE_COLS if c in df.columns]
        feats = df[feat_cols].fillna(0).values.astype(np.float32)

        if fit_scaler:
            self.scaler = StandardScaler()
            feats = self.scaler.fit_transform(feats)
        else:
            self.scaler = scaler
            if scaler is not None:
                feats = scaler.transform(feats)

        self.features = torch.tensor(feats, dtype=torch.float32)
        self.scalar_dim = feats.shape[1]

        # Trajectory sequences
        trajectories = []
        for raw in df["norm_traj"]:
            if isinstance(raw, str):
                traj = np.array(json.loads(raw), dtype=np.float32)
            elif isinstance(raw, list):
                traj = np.array(raw, dtype=np.float32)
            else:
                traj = np.zeros((sequence_len, 2), dtype=np.float32)
            if traj.shape[0] != sequence_len:
                # Resample
                t_orig = np.linspace(0, 1, len(traj))
                t_new = np.linspace(0, 1, sequence_len)
                x = np.interp(t_new, t_orig, traj[:, 0])
                y = np.interp(t_new, t_orig, traj[:, 1])
                traj = np.stack([x, y], axis=1)
            trajectories.append(traj)

        self.trajectories = torch.tensor(np.array(trajectories), dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.trajectories[idx], self.features[idx], self.labels[idx]


# ─── Training ─────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, device) -> dict:
    model.train()
    total_loss, all_preds, all_labels = 0.0, [], []

    for traj, feats, labels in loader:
        traj, feats, labels = traj.to(device), feats.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(traj, feats)
        loss = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * len(labels)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    n = len(all_labels)
    return {
        "loss": total_loss / n,
        "accuracy": accuracy_score(all_labels, all_preds),
        "f1": f1_score(all_labels, all_preds, average="macro", zero_division=0),
    }


@torch.no_grad()
def eval_epoch(model, loader, criterion, device) -> dict:
    model.eval()
    total_loss, all_preds, all_labels, all_probs = 0.0, [], [], []

    for traj, feats, labels in loader:
        traj, feats, labels = traj.to(device), feats.to(device), labels.to(device)
        logits = model(traj, feats)
        loss = criterion(logits, labels)
        total_loss += loss.item() * len(labels)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs)

    n = len(all_labels)
    return {
        "loss": total_loss / n,
        "accuracy": accuracy_score(all_labels, all_preds),
        "f1": f1_score(all_labels, all_preds, average="macro", zero_division=0),
        "preds": all_preds,
        "labels": all_labels,
        "probs": all_probs,
    }


def main():
    parser = argparse.ArgumentParser(description="Train spin classifier")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--features", type=str, default="data/processed/features.parquet")
    parser.add_argument("--experiment", type=str, default="spin_classifier")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--model_type", type=str, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["project"]["seed"])
    device = get_device(cfg)
    logger.info(f"Device: {device}")

    clf_cfg = cfg["spin_classifier"]
    model_type = args.model_type or clf_cfg["model_type"]

    # ── Load features ────────────────────────────────────────────────
    features_path = Path(args.features)
    if not features_path.exists():
        logger.error(f"Features file not found: {features_path}")
        logger.info("Run: python src/processing/feature_engineering.py first")
        return

    df = pd.read_parquet(features_path)
    labeled = df[df["spin_int"] >= 0].copy()
    logger.info(f"Loaded {len(labeled)} labeled samples")

    if len(labeled) < 20:
        logger.error("Not enough labeled samples (need ≥ 20). Generate or annotate more data.")
        return

    # ── Splits ───────────────────────────────────────────────────────
    train_df, test_df = train_test_split(
        labeled, test_size=cfg["data"]["test_split"],
        stratify=labeled["spin_int"], random_state=cfg["project"]["seed"]
    )
    train_df, val_df = train_test_split(
        train_df, test_size=cfg["data"]["val_split"] / (1 - cfg["data"]["test_split"]),
        stratify=train_df["spin_int"], random_state=cfg["project"]["seed"]
    )
    logger.info(f"Split: train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    # ── Datasets ─────────────────────────────────────────────────────
    train_ds = SpinDataset(train_df, fit_scaler=True, sequence_len=clf_cfg["sequence_length"])
    val_ds = SpinDataset(val_df, scaler=train_ds.scaler, sequence_len=clf_cfg["sequence_length"])
    test_ds = SpinDataset(test_df, scaler=train_ds.scaler, sequence_len=clf_cfg["sequence_length"])

    # Weighted sampler for class imbalance
    class_counts = np.bincount(train_ds.labels.numpy(), minlength=clf_cfg["num_classes"])
    class_weights = 1.0 / (class_counts + 1e-6)
    sample_weights = class_weights[train_ds.labels.numpy()]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    train_loader = DataLoader(train_ds, batch_size=clf_cfg["batch_size"], sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=clf_cfg["batch_size"], shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=clf_cfg["batch_size"], shuffle=False, num_workers=0)

    # ── Model ────────────────────────────────────────────────────────
    model = build_model(
        model_type=model_type,
        scalar_feature_dim=train_ds.scalar_dim,
        lstm_hidden=clf_cfg["lstm_hidden"],
        lstm_layers=clf_cfg["lstm_layers"],
        dropout=clf_cfg["dropout"],
    ).to(device)

    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights / class_weights.sum(), dtype=torch.float32).to(device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=clf_cfg["learning_rate"],
        weight_decay=clf_cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=clf_cfg["epochs"], eta_min=1e-6
    )

    # ── MLflow ───────────────────────────────────────────────────────
    mlflow.set_tracking_uri(cfg["paths"]["mlflow_uri"])
    mlflow.set_experiment(args.experiment)

    models_dir = Path(cfg["paths"]["models"])
    models_dir.mkdir(parents=True, exist_ok=True)
    best_path = models_dir / "spin_classifier_best.pt"

    with mlflow.start_run(run_name=args.run_name or f"{model_type}_{int(time.time())}"):
        # Log hyperparams
        mlflow.log_params({
            "model_type": model_type,
            "epochs": clf_cfg["epochs"],
            "batch_size": clf_cfg["batch_size"],
            "lr": clf_cfg["learning_rate"],
            "lstm_hidden": clf_cfg.get("lstm_hidden"),
            "dropout": clf_cfg["dropout"],
            "train_samples": len(train_df),
            "val_samples": len(val_df),
            "sequence_len": clf_cfg["sequence_length"],
        })

        best_val_f1 = 0.0
        patience_counter = 0

        for epoch in range(1, clf_cfg["epochs"] + 1):
            t0 = time.time()
            train_metrics = train_epoch(model, train_loader, optimizer, criterion, device)
            val_metrics = eval_epoch(model, val_loader, criterion, device)
            scheduler.step()

            elapsed = time.time() - t0
            logger.info(
                f"Epoch {epoch:3d}/{clf_cfg['epochs']} | "
                f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['accuracy']:.3f} | "
                f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.3f} "
                f"val_f1={val_metrics['f1']:.3f} | {elapsed:.1f}s"
            )

            mlflow.log_metrics({
                "train_loss": train_metrics["loss"],
                "train_acc": train_metrics["accuracy"],
                "train_f1": train_metrics["f1"],
                "val_loss": val_metrics["loss"],
                "val_acc": val_metrics["accuracy"],
                "val_f1": val_metrics["f1"],
                "lr": scheduler.get_last_lr()[0],
            }, step=epoch)

            if val_metrics["f1"] > best_val_f1:
                best_val_f1 = val_metrics["f1"]
                patience_counter = 0
                torch.save({
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_f1": best_val_f1,
                    "scalar_feature_dim": train_ds.scalar_dim,
                    "scalar_feature_cols": SCALAR_FEATURE_COLS,
                    "config": cfg,
                }, best_path)
                logger.success(f"  ✓ New best model saved (val_f1={best_val_f1:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= clf_cfg["early_stopping_patience"]:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

        # ── Test evaluation ──────────────────────────────────────────
        checkpoint = torch.load(best_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        test_metrics = eval_epoch(model, test_loader, criterion, device)

        logger.info(f"\n{'='*50}")
        logger.success(f"TEST RESULTS:")
        logger.info(f"  Accuracy: {test_metrics['accuracy']:.4f}")
        logger.info(f"  Macro F1: {test_metrics['f1']:.4f}")
        report = classification_report(
            test_metrics["labels"], test_metrics["preds"],
            target_names=SPIN_CLASSES, zero_division=0
        )
        logger.info(f"\nClassification Report:\n{report}")

        mlflow.log_metrics({
            "test_acc": test_metrics["accuracy"],
            "test_f1": test_metrics["f1"],
        })

        # Log confusion matrix
        cm = confusion_matrix(test_metrics["labels"], test_metrics["preds"])
        cm_path = models_dir / "confusion_matrix.json"
        with open(cm_path, "w") as f:
            json.dump({
                "matrix": cm.tolist(),
                "classes": SPIN_CLASSES,
                "test_accuracy": test_metrics["accuracy"],
                "test_f1": test_metrics["f1"],
            }, f, indent=2)
        mlflow.log_artifact(str(cm_path))

        # Log model
        mlflow.pytorch.log_model(model, "spin_classifier")
        logger.success(f"Model logged to MLflow. Best val F1: {best_val_f1:.4f}")


if __name__ == "__main__":
    main()
