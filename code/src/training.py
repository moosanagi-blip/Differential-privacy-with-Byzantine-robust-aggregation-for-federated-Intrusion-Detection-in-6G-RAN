"""
Local training / evaluation helpers shared by the centralized baseline and the
Flower FL clients. Kept deliberately small and framework-agnostic so the exact
same ``train_epochs`` / ``evaluate`` code runs whether or not DP or FL is
active. Full metric reporting (6 metrics, confusion, per-category) lives in
``src/evaluation/metrics.py``; this module reports the quick accuracy/F1/loss
needed for training loops and sanity checks.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .data import loaders


def get_device(prefer_gpu: bool = True) -> torch.device:
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int = 42) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epochs(model: nn.Module, loader: DataLoader, *, epochs: int = 1,
                 lr: float = 1e-3, device: torch.device | None = None,
                 optimizer: torch.optim.Optimizer | None = None) -> float:
    """Train ``model`` for ``epochs`` on ``loader``. Returns mean loss.

    If ``optimizer`` is provided (e.g. an Opacus-wrapped optimizer) it is used
    as-is; otherwise a fresh Adam optimizer is created.
    """
    device = device or get_device()
    model.to(device).train()
    opt = optimizer or torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    total, n = 0.0, 0
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss; stopping invalid run")
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(yb)
            n += len(yb)
    return total / max(n, 1)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader,
             device: torch.device | None = None) -> dict:
    """Quick eval → dict(accuracy, macro_f1, loss). Full metrics elsewhere."""
    from sklearn.metrics import accuracy_score, f1_score

    device = device or get_device()
    model.to(device).eval()
    criterion = nn.CrossEntropyLoss()
    ys, ps, tot, n = [], [], 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Non-finite evaluation logits; stopping invalid run")
        tot += float(criterion(logits, yb)) * len(yb)
        n += len(yb)
        ps.append(logits.argmax(1).cpu().numpy())
        ys.append(yb.cpu().numpy())
    y_true = np.concatenate(ys)
    y_pred = np.concatenate(ps)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "loss": tot / max(n, 1),
    }


def pooled_train_loader(batch_size: int = 128, shuffle: bool = True,
                        seed: int | None = 42, data_root: str | None = None,
                        rows_per_client: int | None = None) -> DataLoader:
    """Concatenate all client shards into one loader (centralized baseline).

    Pooling is the thing the whole framework exists to avoid, so this is only a
    reference baseline. ``rows_per_client`` bounds how much of each shard is
    taken: the complete corpus pooled in memory is ~8 GB, which no laptop in
    this deliverable's target will survive.
    """
    root = loaders.DataRoot(data_root)
    xs, ys = [], []
    for c in range(root.n_clients()):
        if rows_per_client:
            total = root.client_rows(c)
            n_chunks = max(1, -(-total // rows_per_client))
            x, y = root.load_client_chunk(c, 0, n_chunks)
        else:
            x, y = root.load_client(c)
        xs.append(x)
        ys.append(y)
    X = np.concatenate(xs)
    Y = np.concatenate(ys)
    return loaders.make_loader(X, Y, batch_size=batch_size, shuffle=shuffle, seed=seed)
