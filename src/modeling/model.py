"""
src/modeling/model.py
──────────────────────
Model definitions for spin classification.

Three architectures provided:
  1. MLPBaseline  — simple feedforward on engineered features (fast baseline)
  2. CnnLstm      — CNN on ball crop patches + LSTM on trajectory (main model)
  3. TransformerSpin — Transformer encoder on trajectory sequence (experimental)

Input to CnnLstm:
  - trajectory: (B, T, 2) normalized x,y positions
  - features:   (B, F) engineered scalar features

Output: (B, 4) logits for [topspin, backspin, sidespin, float]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger


SPIN_CLASSES = ["topspin", "backspin", "sidespin", "float"]
NUM_CLASSES = len(SPIN_CLASSES)


# ─── 1. MLP Baseline ──────────────────────────────────────────────────────

class MLPBaseline(nn.Module):
    """
    Fast baseline that only uses engineered scalar features.
    No sequence modeling. Useful for benchmarking.
    """

    def __init__(self, input_dim: int, hidden_dims: list[int] = [128, 64], dropout: float = 0.3):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, NUM_CLASSES))
        self.net = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor, trajectory: torch.Tensor = None) -> torch.Tensor:
        return self.net(features)


# ─── 2. CNN-LSTM (Main Model) ─────────────────────────────────────────────

class TrajectoryEncoder(nn.Module):
    """
    1D CNN to extract local motion patterns from trajectory sequence.
    Input: (B, T, 2) → processes as (B, 2, T) channels
    Output: (B, T, hidden)
    """

    def __init__(self, hidden: int = 64):
        super().__init__()
        self.conv1 = nn.Conv1d(2, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(64, hidden, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, 2) → (B, 2, T)
        x = x.transpose(1, 2)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        # (B, hidden, T) → (B, T, hidden)
        return x.transpose(1, 2)


class CnnLstm(nn.Module):
    """
    Main model: CNN trajectory encoder + LSTM + scalar feature fusion.

    Architecture:
      Trajectory (B,T,2) → CNN → (B,T,64) → LSTM → h_n (B,256)
                                                           ↓
      Features (B,F) ──────────────────────────────→ Concat(256+F) → MLP → 4 logits
    """

    def __init__(
        self,
        scalar_feature_dim: int,
        lstm_hidden: int = 256,
        lstm_layers: int = 2,
        dropout: float = 0.3,
        cnn_hidden: int = 64,
    ):
        super().__init__()
        self.trajectory_encoder = TrajectoryEncoder(hidden=cnn_hidden)
        self.lstm = nn.LSTM(
            input_size=cnn_hidden,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
            bidirectional=True,
        )

        lstm_out = lstm_hidden * 2  # bidirectional

        self.feature_proj = nn.Sequential(
            nn.Linear(scalar_feature_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        combined = lstm_out + 64
        self.classifier = nn.Sequential(
            nn.Linear(combined, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, NUM_CLASSES),
        )

    def forward(self, trajectory: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            trajectory: (B, T, 2) normalized positions
            features:   (B, F) engineered features
        Returns:
            logits: (B, 4)
        """
        # Trajectory path
        cnn_out = self.trajectory_encoder(trajectory)         # (B, T, 64)
        lstm_out, (h_n, _) = self.lstm(cnn_out)              # (B, T, 512)
        # Use final hidden state from both directions
        h_forward = h_n[-2]                                   # last layer, forward
        h_backward = h_n[-1]                                  # last layer, backward
        traj_repr = torch.cat([h_forward, h_backward], dim=1) # (B, 512)

        # Feature path
        feat_repr = self.feature_proj(features)               # (B, 64)

        # Combine and classify
        combined = torch.cat([traj_repr, feat_repr], dim=1)   # (B, 576)
        return self.classifier(combined)                       # (B, 4)


# ─── 3. Transformer (Experimental) ───────────────────────────────────────

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 100, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TransformerSpin(nn.Module):
    """
    Transformer encoder over trajectory sequence.
    Best accuracy but slower training. Good for longer sequences.
    """

    def __init__(
        self,
        scalar_feature_dim: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(2, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.feature_proj = nn.Linear(scalar_feature_dim, 64)

        self.classifier = nn.Sequential(
            nn.Linear(d_model + 64, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, NUM_CLASSES),
        )

    def forward(self, trajectory: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(trajectory)                     # (B, T, d_model)
        x = self.pos_enc(x)
        x = self.transformer(x)                             # (B, T, d_model)
        traj_repr = x.mean(dim=1)                           # (B, d_model)
        feat_repr = F.relu(self.feature_proj(features))     # (B, 64)
        combined = torch.cat([traj_repr, feat_repr], dim=1)
        return self.classifier(combined)


# ─── Model factory ────────────────────────────────────────────────────────

def build_model(model_type: str, scalar_feature_dim: int, **kwargs) -> nn.Module:
    """
    Factory function to build a spin classifier model.

    Args:
        model_type: One of 'mlp_baseline', 'cnn_lstm', 'transformer'
        scalar_feature_dim: Number of scalar engineered features
        **kwargs: Additional model hyperparameters from config
    """
    if model_type == "mlp_baseline":
        model = MLPBaseline(input_dim=scalar_feature_dim, **kwargs)
    elif model_type == "cnn_lstm":
        model = CnnLstm(scalar_feature_dim=scalar_feature_dim, **kwargs)
    elif model_type == "transformer":
        model = TransformerSpin(scalar_feature_dim=scalar_feature_dim, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Built {model_type} model | {n_params:,} trainable parameters")
    return model
