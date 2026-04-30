"""
src/processing/feature_engineering.py
───────────────────────────────────────
Convert raw ball trajectories → structured feature vectors
suitable for the spin classifier.

Features engineered:
  1. Kinematic features (velocity, acceleration, jerk)
  2. Geometric features (curvature, arc direction, lateral drift)
  3. Statistical features (mean/std/max of the above)
  4. Optical flow magnitude (proxy for ball rotation)
  5. Trajectory normalization (time + spatial)

Usage:
    python src/processing/feature_engineering.py \
        --trajectories data/processed/trajectories/ \
        --output data/processed/features.parquet
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from scipy.signal import savgol_filter
from scipy.stats import kurtosis, skew
from tqdm import tqdm


SPIN_LABELS = ["topspin", "backspin", "sidespin", "float"]
LABEL_TO_INT = {s: i for i, s in enumerate(SPIN_LABELS)}


def normalize_trajectory(positions: np.ndarray, target_len: int = 30) -> np.ndarray:
    """
    Resample trajectory to a fixed length using linear interpolation.
    Also normalize spatial coordinates to [0, 1] based on bounding box.
    """
    n = len(positions)
    if n < 2:
        return np.zeros((target_len, 2))

    # Resample to target_len
    t_orig = np.linspace(0, 1, n)
    t_new = np.linspace(0, 1, target_len)
    x_interp = np.interp(t_new, t_orig, positions[:, 0])
    y_interp = np.interp(t_new, t_orig, positions[:, 1])

    resampled = np.stack([x_interp, y_interp], axis=1)

    # Normalize to bounding box
    min_xy = resampled.min(axis=0)
    max_xy = resampled.max(axis=0)
    diff = max_xy - min_xy
    diff[diff == 0] = 1  # avoid divide by zero
    normalized = (resampled - min_xy) / diff

    return normalized


def smooth_trajectory(positions: np.ndarray, window: int = 5) -> np.ndarray:
    """Apply Savitzky-Golay smoothing to reduce detection noise."""
    if len(positions) < window:
        return positions
    smoothed = np.copy(positions)
    if len(positions) >= window:
        smoothed[:, 0] = savgol_filter(positions[:, 0], min(window, len(positions) - 1 | 1), 2)
        smoothed[:, 1] = savgol_filter(positions[:, 1], min(window, len(positions) - 1 | 1), 2)
    return smoothed


def compute_kinematic_features(positions: np.ndarray, fps: float = 30.0) -> dict:
    """Compute velocity, acceleration, and jerk from position sequence."""
    dt = 1.0 / fps
    vel = np.diff(positions, axis=0) / dt         # (N-1, 2) px/s
    acc = np.diff(vel, axis=0) / dt               # (N-2, 2) px/s^2
    jerk = np.diff(acc, axis=0) / dt              # (N-3, 2) px/s^3

    speed = np.linalg.norm(vel, axis=1)
    acc_mag = np.linalg.norm(acc, axis=1) if len(acc) > 0 else np.array([0.0])
    jerk_mag = np.linalg.norm(jerk, axis=1) if len(jerk) > 0 else np.array([0.0])

    features = {}
    for name, arr in [("speed", speed), ("acc", acc_mag), ("jerk", jerk_mag)]:
        if len(arr) == 0:
            arr = np.array([0.0])
        features[f"{name}_mean"] = float(np.mean(arr))
        features[f"{name}_std"] = float(np.std(arr))
        features[f"{name}_max"] = float(np.max(arr))
        features[f"{name}_skew"] = float(skew(arr)) if len(arr) > 1 else 0.0
        features[f"{name}_kurt"] = float(kurtosis(arr)) if len(arr) > 1 else 0.0

    # Velocity direction changes (sign flips in y-velocity)
    vy = vel[:, 1]
    sign_changes = np.sum(np.diff(np.sign(vy)) != 0)
    features["vy_sign_changes"] = int(sign_changes)

    # Horizontal vs vertical component ratio
    vx_abs = np.abs(vel[:, 0])
    vy_abs = np.abs(vel[:, 1])
    features["hv_ratio"] = float(np.mean(vx_abs / (vy_abs + 1e-6)))

    return features


def compute_geometric_features(positions: np.ndarray) -> dict:
    """Geometric and shape features of the trajectory."""
    features = {}
    n = len(positions)

    # Total arc length
    arc_len = np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))
    features["arc_length"] = float(arc_len)

    # Chord length (start to end)
    chord = np.linalg.norm(positions[-1] - positions[0])
    features["chord_length"] = float(chord)

    # Sinuosity: arc/chord ratio (1=straight, >1=curved)
    features["sinuosity"] = float(arc_len / (chord + 1e-6))

    # Curvature: mean residual from linear fit
    t = np.arange(n)
    x_fit = np.polyfit(t, positions[:, 0], 1)
    y_fit = np.polyfit(t, positions[:, 1], 1)
    x_res = positions[:, 0] - np.polyval(x_fit, t)
    y_res = positions[:, 1] - np.polyval(y_fit, t)
    features["curvature_mean"] = float(np.mean(np.sqrt(x_res**2 + y_res**2)))
    features["curvature_max"] = float(np.max(np.sqrt(x_res**2 + y_res**2)))

    # Vertical quadratic fit: positive coeff → concave down (topspin)
    if n >= 4:
        y_coeffs = np.polyfit(t, positions[:, 1], 2)
        features["y_quadratic_coeff"] = float(y_coeffs[0])
    else:
        features["y_quadratic_coeff"] = 0.0

    # Lateral drift
    v = positions[-1] - positions[0]
    v_norm = v / (np.linalg.norm(v) + 1e-6)
    drifts = []
    for p in positions:
        diff = p - positions[0]
        proj = np.dot(diff, v_norm) * v_norm
        drift = np.linalg.norm(diff - proj)
        drifts.append(drift)
    features["lateral_drift_max"] = float(np.max(drifts))
    features["lateral_drift_mean"] = float(np.mean(drifts))

    # Net vertical displacement (positive = ball moved downward in image)
    features["net_y_disp"] = float(positions[-1, 1] - positions[0, 1])
    features["net_x_disp"] = float(positions[-1, 0] - positions[0, 0])

    return features


def compute_first_n_features(positions: np.ndarray, n: int = 10) -> dict:
    """
    Features computed from only the first N frames.
    Key finding: backspin/topspin often distinguishable early.
    """
    early = positions[:min(n, len(positions))]
    features = {}
    if len(early) >= 3:
        t = np.arange(len(early))
        y_coeffs = np.polyfit(t, early[:, 1], 2)
        features[f"early_{n}f_y_coeff"] = float(y_coeffs[0])
    else:
        features[f"early_{n}f_y_coeff"] = 0.0
    return features


def build_feature_vector(traj_dict: dict, fps: float = 30.0) -> dict | None:
    """Build a single feature dict from a trajectory JSON dict."""
    positions = np.array(traj_dict.get("positions", []))
    if len(positions) < 5:
        return None

    positions = smooth_trajectory(positions)

    features = {
        "track_id": traj_dict["track_id"],
        "trajectory_length": len(positions),
        "spin_label": traj_dict.get("spin_label"),
        "spin_int": LABEL_TO_INT.get(traj_dict.get("spin_label"), -1),
    }

    features.update(compute_kinematic_features(positions, fps))
    features.update(compute_geometric_features(positions))
    features.update(compute_first_n_features(positions, n=10))

    # Include normalized trajectory as flat array (for sequence models)
    norm_traj = normalize_trajectory(positions, target_len=30)
    features["norm_traj"] = norm_traj.tolist()

    return features


def process_trajectory_files(
    trajectories_dir: Path,
    output_path: Path,
) -> pd.DataFrame:
    """Process all trajectory JSON files into a single feature DataFrame."""

    json_files = list(trajectories_dir.rglob("*_trajectories.json"))
    logger.info(f"Found {len(json_files)} trajectory files")

    all_features = []

    for jf in tqdm(json_files, desc="Processing trajectories"):
        with open(jf) as f:
            data = json.load(f)

        fps = data.get("fps", 30.0)
        video_name = Path(data["video"]).name

        for traj in data.get("trajectories", []):
            feats = build_feature_vector(traj, fps)
            if feats is not None:
                feats["video"] = video_name
                all_features.append(feats)

    if not all_features:
        logger.warning("No features extracted. Check trajectory files.")
        return pd.DataFrame()

    df = pd.DataFrame(all_features)

    # Drop rows where spin label is unknown (for unsupervised use, keep them separately)
    labeled = df[df["spin_int"] >= 0].copy()
    unlabeled = df[df["spin_int"] < 0].copy()

    logger.info(f"Labeled samples: {len(labeled)}")
    logger.info(f"Unlabeled samples: {len(unlabeled)}")
    if len(labeled) > 0:
        logger.info(f"Class distribution:\n{labeled['spin_label'].value_counts().to_string()}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.success(f"Features saved → {output_path} ({len(df)} rows, {len(df.columns)} features)")

    # Also save labeled subset as CSV for easy inspection
    csv_path = output_path.with_suffix(".csv")
    labeled.drop(columns=["norm_traj"], errors="ignore").to_csv(csv_path, index=False)
    logger.info(f"Labeled features CSV → {csv_path}")

    return df


def main():
    parser = argparse.ArgumentParser(description="Extract features from ball trajectories")
    parser.add_argument("--trajectories", type=str, default="data/processed/trajectories")
    parser.add_argument("--output", type=str, default="data/processed/features.parquet")
    args = parser.parse_args()

    df = process_trajectory_files(
        trajectories_dir=Path(args.trajectories),
        output_path=Path(args.output),
    )

    if len(df) > 0:
        logger.info(f"\nFeature summary:\n{df.describe().to_string()}")


if __name__ == "__main__":
    main()
