"""
Server-side aggregation rules (pure NumPy, framework-agnostic).

  * ``fedavg``       — weighted mean (baseline; no robustness).
  * ``krum``         — Blanchard et al. (2017), single Krum.
  * ``multi_krum``   — select the m lowest-score updates, then average them.
  * ``trimmed_mean`` — coordinate-wise trimmed mean (Yin et al., 2018).

All operate on a list of client updates, each an ordered ``list[np.ndarray]``
(see ``params.py``). Multi-Krum also returns per-client scores and the indices
it selected — used by RO4's DP × Multi-Krum interaction analysis.
"""

from __future__ import annotations

import numpy as np

from .params import Params, flatten

Update = Params


def fedavg(updates: list[Update], weights: list[float] | None = None) -> Update:
    """Weighted element-wise mean of client parameter lists."""
    n = len(updates)
    if n == 0:
        raise ValueError("no updates to aggregate")
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=np.float64)
    if w.shape != (n,) or not np.isfinite(w).all() or (w < 0).any() or w.sum() <= 0:
        raise ValueError("Aggregation weights must be finite, nonnegative and have positive total")
    w = w / w.sum()
    out: Update = []
    for layer in range(len(updates[0])):
        stack = np.stack([u[layer] for u in updates], axis=0)
        wshape = (n,) + (1,) * (stack.ndim - 1)
        out.append((stack * w.reshape(wshape)).sum(axis=0).astype(updates[0][layer].dtype))
    return out


def _pairwise_sq_dists(vectors: np.ndarray) -> np.ndarray:
    """n×n matrix of squared Euclidean distances between flattened updates."""
    sq = np.sum(vectors ** 2, axis=1)
    d = sq[:, None] + sq[None, :] - 2.0 * (vectors @ vectors.T)
    np.fill_diagonal(d, 0.0)
    return np.maximum(d, 0.0)


def krum_scores(updates: list[Update], n_byzantine: int) -> np.ndarray:
    """Krum score per client = sum of squared distances to its n-f-2 closest
    neighbours. Lower is better (more central / less likely Byzantine)."""
    n = len(updates)
    if int(n_byzantine) != n_byzantine or not 0 <= n_byzantine <= n - 3:
        raise ValueError("Krum requires an integer f with 0 <= f <= n-3; theoretical guarantees have additional assumptions")
    vecs = np.stack([flatten(u) for u in updates], axis=0)
    if not np.isfinite(vecs).all():
        raise FloatingPointError("Non-finite client parameters; Krum scores are undefined")
    dists = _pairwise_sq_dists(vecs)
    # Validated above; do not silently change the configured experiment.
    k = n - int(n_byzantine) - 2
    scores = np.empty(n)
    for i in range(n):
        row = np.sort(dists[i])          # includes the 0 self-distance first
        scores[i] = row[1:k + 1].sum()   # skip self, take k closest
    if not np.isfinite(scores).all():
        raise FloatingPointError("Non-finite Krum scores; refusing to rank invalid updates")
    return scores


def krum(updates: list[Update], n_byzantine: int) -> tuple[Update, int, np.ndarray]:
    """Single-Krum: pick the one lowest-score update. Returns (update, idx, scores)."""
    scores = krum_scores(updates, n_byzantine)
    idx = int(np.argmin(scores))
    return updates[idx], idx, scores


def multi_krum(updates: list[Update], n_byzantine: int,
               n_selected: int | None = None
               ) -> tuple[Update, list[int], np.ndarray]:
    """Multi-Krum: average the ``m`` lowest-score updates.

    Default ``m = n - f`` (Blanchard et al.). Returns
    ``(aggregated_update, selected_indices, scores)``.
    """
    n = len(updates)
    scores = krum_scores(updates, n_byzantine)
    m = n - int(n_byzantine) if n_selected is None else n_selected
    if int(m) != m or not 1 <= m <= n:
        raise ValueError("Multi-Krum selection count must be an integer in [1, n]")
    m = int(m)
    selected = list(np.argsort(scores)[:m])
    agg = fedavg([updates[i] for i in selected])
    return agg, [int(i) for i in selected], scores


def trimmed_mean(updates: list[Update], trim_ratio: float = 0.1) -> Update:
    """Coordinate-wise trimmed mean; trims ``trim_ratio`` from each tail."""
    n = len(updates)
    k = int(np.floor(trim_ratio * n))
    out: Update = []
    for layer in range(len(updates[0])):
        stack = np.stack([u[layer] for u in updates], axis=0)  # (n, *shape)
        s = np.sort(stack, axis=0)
        if 2 * k < n:
            s = s[k:n - k] if k > 0 else s
        out.append(s.mean(axis=0).astype(updates[0][layer].dtype))
    return out


AGGREGATORS = {"fedavg", "uniformavg", "krum", "multikrum", "trimmedmean"}


def aggregate(name: str, updates: list[Update], *, weights=None,
              n_byzantine: int = 0, trim_ratio: float = 0.1):
    """Dispatch by name → (aggregated_update, info_dict)."""
    name = name.lower().replace("-", "").replace("_", "")
    if name == "fedavg":
        return fedavg(updates, weights), {"selected": list(range(len(updates)))}
    if name == "uniformavg":
        return fedavg(updates), {"selected": list(range(len(updates)))}
    if name == "krum":
        agg, idx, scores = krum(updates, n_byzantine)
        return agg, {"selected": [idx], "scores": scores.tolist()}
    if name == "multikrum":
        agg, sel, scores = multi_krum(updates, n_byzantine)
        return agg, {"selected": sel, "scores": scores.tolist()}
    if name == "trimmedmean":
        return trimmed_mean(updates, trim_ratio), {"selected": list(range(len(updates)))}
    raise ValueError(f"unknown aggregator '{name}' (expected {sorted(AGGREGATORS)})")
