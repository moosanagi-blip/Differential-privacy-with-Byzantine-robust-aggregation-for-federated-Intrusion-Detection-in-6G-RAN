"""
Intrusion-detection classifiers for the DP-FL-IDS framework.

Primary model (thesis-specified): a compact **LSTM** that reads the 53-dim
feature vector as a length-53 sequence of scalars and classifies it into one of
the 34 CICIoT2023 labels. We use Opacus' ``DPLSTM`` as the recurrent layer so
the *same* architecture works with and without differential privacy — plain
``nn.LSTM`` is not compatible with Opacus' per-sample gradients, whereas
``DPLSTM`` behaves like ``nn.LSTM`` under a normal optimiser and additionally
supports DP-SGD. A fast ``MLP`` baseline and a small ``Transformer`` variant are
provided for ablations and for CPU-bound sweeps.

All models are Opacus-friendly (no BatchNorm; GroupNorm used where needed).
"""

from __future__ import annotations

import torch
import torch.nn as nn

try:  # DPLSTM is a drop-in for nn.LSTM that also supports DP-SGD.
    from opacus.layers import DPLSTM
    _HAVE_DPLSTM = True
except Exception:  # pragma: no cover - opacus always present in this project
    DPLSTM = nn.LSTM  # type: ignore
    _HAVE_DPLSTM = False


class LSTMIDS(nn.Module):
    """LSTM classifier. Input ``(B, F)`` -> treated as sequence ``(B, F, 1)``."""

    def __init__(self, n_features: int, n_classes: int, hidden: int = 64,
                 num_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.n_features = n_features
        self.lstm = DPLSTM(
            input_size=1, hidden_size=hidden, num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, F) -> (B, F, 1): each feature is one timestep.
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        out, _ = self.lstm(x)          # (B, F, hidden)
        last = out[:, -1, :]           # final timestep
        return self.head(last)         # (B, n_classes) logits


class MLPIDS(nn.Module):
    """Fast fully-connected baseline (Opacus-safe)."""

    def __init__(self, n_features: int, n_classes: int, hidden: int = 128,
                 dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden), nn.GroupNorm(1, hidden), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerIDS(nn.Module):
    """Small Transformer-encoder variant (secondary model)."""

    def __init__(self, n_features: int, n_classes: int, d_model: int = 32,
                 nhead: int = 4, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.embed = nn.Linear(1, d_model)
        self.pos = nn.Parameter(torch.zeros(1, n_features, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=2 * d_model,
            dropout=dropout, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(-1)          # (B, F, 1)
        h = self.embed(x) + self.pos     # (B, F, d_model)
        h = self.encoder(h)
        return self.head(h.mean(dim=1))  # global average pool -> logits


def build_model(name: str, n_features: int, n_classes: int, **kw) -> nn.Module:
    """Factory: ``name`` in {'lstm', 'mlp', 'transformer'}."""
    name = name.lower()
    if name == "lstm":
        return LSTMIDS(n_features, n_classes, **kw)
    if name == "mlp":
        return MLPIDS(n_features, n_classes, **kw)
    if name == "transformer":
        # Both engines expose a shared hidden-width configuration.
        if "hidden" in kw:
            hidden = kw.pop("hidden")
            kw.setdefault("d_model", hidden)
        return TransformerIDS(n_features, n_classes, **kw)
    raise ValueError(f"unknown model '{name}' (expected lstm|mlp|transformer)")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
