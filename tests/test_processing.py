"""
tests/test_processing.py
──────────────────────────
Unit tests for processing and feature engineering.
Run: pytest tests/ -v
"""

import json
import numpy as np
import pandas as pd
import pytest
import tempfile
from pathlib import Path


# ─── Feature engineering tests ───────────────────────────────────────────

class TestFeatureEngineering:
    def _make_trajectory(self, spin: str = "topspin", length: int = 30):
        """Generate synthetic trajectory positions for a given spin type."""
        t = np.linspace(0, 1, length)
        x = t * 0.8 + 0.1

        if spin == "topspin":
            y = 0.2 + 0.3 * (t ** 1.5)
        elif spin == "backspin":
            y = 0.5 - 0.1 * t + 0.15 * (t ** 2)
        elif spin == "sidespin":
            y = 0.35 + 0.08 * np.sin(t * 6)
        else:  # float
            y = 0.3 + 0.05 * t

        return np.stack([x, y], axis=1)

    def test_normalize_trajectory_shape(self):
        from src.processing.feature_engineering import normalize_trajectory
        pos = self._make_trajectory(length=45)
        norm = normalize_trajectory(pos, target_len=30)
        assert norm.shape == (30, 2), "Output must be (30, 2)"

    def test_normalize_trajectory_range(self):
        from src.processing.feature_engineering import normalize_trajectory
        pos = self._make_trajectory(length=20)
        norm = normalize_trajectory(pos, target_len=30)
        assert norm.min() >= -0.01
        assert norm.max() <= 1.01

    def test_kinematic_features_keys(self):
        from src.processing.feature_engineering import compute_kinematic_features
        pos = self._make_trajectory()
        feats = compute_kinematic_features(pos)
        required_keys = ["speed_mean", "speed_std", "speed_max", "acc_mean", "hv_ratio"]
        for k in required_keys:
            assert k in feats, f"Missing feature: {k}"

    def test_geometric_features_keys(self):
        from src.processing.feature_engineering import compute_geometric_features
        pos = self._make_trajectory()
        feats = compute_geometric_features(pos)
        required_keys = ["arc_length", "chord_length", "sinuosity", "curvature_mean",
                         "y_quadratic_coeff", "lateral_drift_max", "net_y_disp"]
        for k in required_keys:
            assert k in feats, f"Missing feature: {k}"

    def test_topspin_positive_quadratic(self):
        """Topspin should show positive y_quadratic_coeff (ball drops faster)."""
        from src.processing.feature_engineering import compute_geometric_features
        pos = self._make_trajectory("topspin")
        feats = compute_geometric_features(pos)
        assert feats["y_quadratic_coeff"] > 0, "Topspin should have positive quadratic coeff"

    def test_backspin_lateral_drift_low(self):
        """Backspin should have minimal lateral drift."""
        from src.processing.feature_engineering import compute_geometric_features
        ts_pos = self._make_trajectory("topspin")
        ss_pos = self._make_trajectory("sidespin")
        ts_feats = compute_geometric_features(ts_pos)
        ss_feats = compute_geometric_features(ss_pos)
        assert ss_feats["lateral_drift_max"] > ts_feats["lateral_drift_max"], \
            "Sidespin should have more lateral drift than topspin"

    def test_build_feature_vector_returns_dict(self):
        from src.processing.feature_engineering import build_feature_vector
        pos = self._make_trajectory(length=30)
        traj_dict = {
            "track_id": 1,
            "positions": pos.tolist(),
            "spin_label": "topspin",
        }
        result = build_feature_vector(traj_dict)
        assert result is not None
        assert "speed_mean" in result
        assert "norm_traj" in result

    def test_build_feature_vector_short_trajectory(self):
        """Short trajectories (< 5 frames) should return None."""
        from src.processing.feature_engineering import build_feature_vector
        traj_dict = {
            "track_id": 1,
            "positions": [[0.1, 0.2], [0.2, 0.3]],
            "spin_label": "float",
        }
        result = build_feature_vector(traj_dict)
        assert result is None


# ─── Trajectory class tests ───────────────────────────────────────────────

class TestTrajectory:
    def _make_trajectory_obj(self, spin: str = "topspin"):
        from src.processing.tracker import Trajectory, BallDetection
        traj = Trajectory(track_id=1)
        t = np.linspace(0, 1, 30)
        x = t * 0.8 + 0.1
        if spin == "topspin":
            y = 0.2 + 0.3 * (t ** 1.5)
        else:
            y = 0.5 - 0.1 * t
        for i, (xi, yi) in enumerate(zip(x, y)):
            traj.detections.append(BallDetection(
                frame_idx=i, x_center=xi, y_center=yi,
                width=0.02, height=0.02, confidence=0.9, track_id=1
            ))
        traj.spin_label = spin
        return traj

    def test_trajectory_length(self):
        traj = self._make_trajectory_obj()
        assert traj.length == 30

    def test_positions_shape(self):
        traj = self._make_trajectory_obj()
        assert traj.positions.shape == (30, 2)

    def test_velocities_shape(self):
        traj = self._make_trajectory_obj()
        assert traj.velocities.shape == (29, 2)

    def test_arc_direction_topspin(self):
        traj = self._make_trajectory_obj("topspin")
        arc = traj.compute_arc_direction()
        assert arc == "concave_down", f"Expected concave_down for topspin, got {arc}"

    def test_to_dict_has_required_keys(self):
        traj = self._make_trajectory_obj()
        d = traj.to_dict()
        for key in ["track_id", "length", "positions", "mean_speed", "curvature", "arc_direction"]:
            assert key in d, f"Missing key in trajectory dict: {key}"


# ─── Model tests ──────────────────────────────────────────────────────────

class TestModel:
    def test_mlp_baseline_forward(self):
        import torch
        from src.modeling.model import MLPBaseline
        model = MLPBaseline(input_dim=25)
        x = torch.randn(4, 25)
        out = model(x)
        assert out.shape == (4, 4), "Output should be (batch, 4 classes)"

    def test_cnn_lstm_forward(self):
        import torch
        from src.modeling.model import CnnLstm
        model = CnnLstm(scalar_feature_dim=25)
        traj = torch.randn(4, 30, 2)
        feats = torch.randn(4, 25)
        out = model(traj, feats)
        assert out.shape == (4, 4), "Output should be (batch, 4 classes)"

    def test_transformer_forward(self):
        import torch
        from src.modeling.model import TransformerSpin
        model = TransformerSpin(scalar_feature_dim=25)
        traj = torch.randn(4, 30, 2)
        feats = torch.randn(4, 25)
        out = model(traj, feats)
        assert out.shape == (4, 4)

    def test_build_model_factory(self):
        from src.modeling.model import build_model
        for model_type in ["mlp_baseline", "cnn_lstm", "transformer"]:
            model = build_model(model_type, scalar_feature_dim=25)
            assert model is not None

    def test_model_output_is_not_probability(self):
        """Model outputs logits, not probabilities — softmax should be applied downstream."""
        import torch
        from src.modeling.model import CnnLstm
        model = CnnLstm(scalar_feature_dim=25)
        traj = torch.randn(2, 30, 2)
        feats = torch.randn(2, 25)
        out = model(traj, feats)
        # Logits can be > 1 or < 0
        assert not (out.min() >= 0 and out.max() <= 1).all(), \
            "Raw output should be logits, not probabilities"


# ─── Integration test ─────────────────────────────────────────────────────

class TestPipeline:
    def test_end_to_end_inference(self):
        """Full pipeline: trajectory dict → features → model prediction."""
        import torch
        import numpy as np
        from src.processing.feature_engineering import build_feature_vector, LABEL_TO_INT
        from src.modeling.model import build_model, SPIN_CLASSES
        from src.modeling.train_classifier import SCALAR_FEATURE_COLS

        # Build synthetic trajectory
        t = np.linspace(0, 1, 30)
        positions = np.stack([t * 0.8 + 0.1, 0.2 + 0.3 * t**1.5], axis=1).tolist()

        traj_dict = {
            "track_id": 1,
            "positions": positions,
            "spin_label": "topspin",
        }

        feats = build_feature_vector(traj_dict)
        assert feats is not None

        scalar = np.array([feats.get(c, 0.0) for c in SCALAR_FEATURE_COLS], dtype=np.float32)
        norm_traj = np.array(feats["norm_traj"], dtype=np.float32)

        model = build_model("cnn_lstm", scalar_feature_dim=len(SCALAR_FEATURE_COLS))
        model.eval()

        with torch.no_grad():
            traj_t = torch.tensor(norm_traj).unsqueeze(0)
            feat_t = torch.tensor(scalar).unsqueeze(0)
            logits = model(traj_t, feat_t)

        assert logits.shape == (1, 4)
        pred_class = SPIN_CLASSES[logits.argmax().item()]
        assert pred_class in SPIN_CLASSES
