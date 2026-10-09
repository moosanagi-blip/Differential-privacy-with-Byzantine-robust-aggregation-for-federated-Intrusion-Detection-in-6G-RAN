"""
Byzantine attack simulator (RO4).

Two families of poisoning, matching the thesis threat model:

  * **Data poisoning** — ``label_flip``: a malicious gNB trains on corrupted
    labels (applied before local training).
  * **Model/update poisoning** — applied to the honest update *delta*
    (new_params − global_params) after local training:
      - ``sign_flip``  : send −scale·delta (drives the model the wrong way)
      - ``scale``/``boost`` : send scale·delta (model-replacement / boosting)
      - ``gaussian``   : add Gaussian noise to the update
      - ``random``     : ignore training, send random parameters

Which clients are malicious is decided by ``select_byzantine(n, fraction)``.
Honest clients pass through unchanged. Distance-based defences (Krum /
Multi-Krum) are expected to down-rank update-poisoning clients — the central
question of RO4, especially once DP noise is also present.
"""

from __future__ import annotations

import numpy as np

from ..fl.params import Params, flatten
from ..fl.aggregation import krum_scores

DATA_ATTACKS = {"label_flip"}
UPDATE_ATTACKS = {"sign_flip", "scale", "boost", "gaussian", "random", "adaptive_krum", "boundary_krum"}
ALL_ATTACKS = {"none"} | DATA_ATTACKS | UPDATE_ATTACKS


def select_byzantine(n_clients: int, fraction: float, seed: int = 0) -> set[int]:
    """Deterministically choose ``round(fraction·n)`` malicious client ids."""
    k = int(round(fraction * n_clients))
    if k <= 0:
        return set()
    rng = np.random.default_rng(seed)
    return set(int(i) for i in rng.choice(n_clients, size=min(k, n_clients),
                                          replace=False))


def maybe_flip_labels(y: np.ndarray, n_classes: int, *, active: bool,
                      seed: int = 0) -> np.ndarray:
    """Return labels with every entry flipped to a different random class."""
    if not active:
        return y
    rng = np.random.default_rng(seed)
    flipped = (y + rng.integers(1, n_classes, size=y.shape)) % n_classes
    return flipped.astype(y.dtype)


def apply_update_attack(name: str, global_params: Params, honest_params: Params,
                        *, scale: float = 10.0, noise_std: float = 1.0,
                        seed: int = 0) -> Params:
    """Transform an honest update into a malicious one. Returns new params."""
    name = name.lower()
    rng = np.random.default_rng(seed)
    if name in ("none", "label_flip", "adaptive_krum", "boundary_krum"):
        return honest_params  # label_flip acts on data, not the update
    delta = [h - g for h, g in zip(honest_params, global_params)]
    if name == "sign_flip":
        return [g - scale * d for g, d in zip(global_params, delta)]
    if name in ("scale", "boost"):
        return [g + scale * d for g, d in zip(global_params, delta)]
    if name == "gaussian":
        return [h + rng.normal(0.0, noise_std, size=h.shape).astype(h.dtype)
                for h in honest_params]
    if name == "random":
        return [rng.normal(0.0, noise_std, size=g.shape).astype(g.dtype)
                for g in global_params]
    raise ValueError(f"unknown attack '{name}' (expected {sorted(ALL_ATTACKS)})")


def craft_adaptive_krum(updates: list[Params], global_params: Params,
                        malicious_positions: list[int], n_byzantine: int,
                        max_scale: float = 2.0) -> list[Params]:
    """Craft a colluding, defence-aware update near the Multi-Krum boundary.

    The adversary observes the submitted updates, replaces its own values with
    a common update opposite the honest mean direction, and keeps the largest
    displacement whose Krum score is no worse than the 75th percentile honest
    score.  This is deliberately a *white-box* stress test, not a claim that
    an attacker can observe honest updates in every deployment.
    """
    if not malicious_positions:
        return updates
    honest = [u for i, u in enumerate(updates) if i not in malicious_positions]
    if not honest:
        return updates
    g = flatten(global_params)
    mean_delta = np.mean([flatten(u) - g for u in honest], axis=0)
    if not np.any(mean_delta):
        return updates
    threshold = np.percentile(krum_scores(updates, n_byzantine)[
        [i for i in range(len(updates)) if i not in malicious_positions]], 75)
    chosen = mean_delta.copy()
    for scale in np.linspace(0.0, max_scale, 41):
        candidate = g + (1.0 - scale) * mean_delta
        trial = _replace_flat(updates, global_params, malicious_positions, candidate)
        scores = krum_scores(trial, n_byzantine)
        if np.max(scores[malicious_positions]) <= threshold:
            chosen = candidate - g
    return _replace_flat(updates, global_params, malicious_positions, g + chosen)


def _replace_flat(updates: list[Params], template: Params, positions: list[int], vector: np.ndarray) -> list[Params]:
    """Return updates with selected entries reconstructed from ``vector``."""
    rebuilt: Params = []
    offset = 0
    for layer in template:
        size = layer.size
        rebuilt.append(vector[offset:offset + size].reshape(layer.shape).astype(layer.dtype))
        offset += size
    return [rebuilt if i in positions else u for i, u in enumerate(updates)]
