"""
Differential-privacy engine (Opacus) for the DP-FL-IDS framework.

Implements RO1's privacy mechanism exactly as stated in the proposal:
**(ε, δ)-DP via the Gaussian mechanism with per-sample gradient clipping**,
realised with Opacus' ``PrivacyEngine`` (per-sample clipping to ``max_grad_norm``
+ Gaussian noise with a given ``noise_multiplier``) and the Rényi-DP (RDP)
accountant for the (ε, δ) budget.

Two entry points:
  * ``attach_dp(...)`` — wrap a model/optimizer/loader for one round of local
    DP-SGD on a gNB client; the returned ``PrivacyEngine`` exposes the ε spent.
  * ``epsilon_from_schedule(...)`` — analytic cumulative ε over a *sequence* of
    rounds (each with its own noise multiplier), used both to track RO3's
    cumulative-budget metric and to calibrate RO2's adaptive-ε scheduler.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch.nn as nn
from opacus import PrivacyEngine
from opacus.accountants import RDPAccountant
from opacus.accountants.utils import get_noise_multiplier
from opacus.validators import ModuleValidator
from torch.utils.data import DataLoader


@dataclass
class DPConfig:
    """Per-round DP hyper-parameters."""
    noise_multiplier: float = 1.0   # σ: Gaussian noise / clip norm
    max_grad_norm: float = 1.0      # C: per-sample gradient clip
    delta: float = 1e-5             # δ target
    enabled: bool = True


def validate_and_fix(model: nn.Module) -> nn.Module:
    """Ensure the module is Opacus-compatible (no-op for our DPLSTM/MLP)."""
    if not ModuleValidator.is_valid(model):
        model = ModuleValidator.fix(model)
    return model


def attach_dp(model: nn.Module, optimizer, data_loader: DataLoader, *,
              noise_multiplier: float, max_grad_norm: float,
              poisson_sampling: bool = True):
    """Make one training setup private.

    Returns ``(privacy_engine, model, optimizer, data_loader)`` where the
    returned objects are the Opacus-wrapped versions to train with. Read the
    round's spent budget afterwards via ``privacy_engine.get_epsilon(delta)``.
    """
    engine = PrivacyEngine(accountant="rdp")
    model, optimizer, data_loader = engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=data_loader,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
        poisson_sampling=poisson_sampling,
    )
    return engine, model, optimizer, data_loader


# A wide RDP order (α) grid so the accountant's optimal order is interior, not
# at a boundary — this yields a tight (ε, δ) bound across our whole σ range and
# silences Opacus' "optimal order is the smallest/largest alpha" warnings.
# Fine spacing near 1 (needed at large ε) + coarse tail to 512 (needed at small ε).
_RDP_ALPHAS = ([1 + x / 20.0 for x in range(1, 200)]   # 1.05 … 10.95
               + list(range(11, 129))                   # 11 … 128
               + [160, 192, 224, 256, 320, 384, 448, 512])


# ── analytic RDP accounting over a schedule of rounds ──────────────────────
def epsilon_from_schedule(schedule: list[tuple[float, float, int]],
                          delta: float = 1e-5) -> float:
    """Cumulative ε over rounds.

    ``schedule`` is a list of ``(noise_multiplier σ, sample_rate q, n_steps)``
    tuples — one entry per round. Steps compose via the RDP accountant. A round
    with zero noise and positive sampling/steps has infinite privacy cost.
    Empty mechanisms (zero sampling or steps) contribute no cost.
    """
    if not 0 < delta < 1:
        raise ValueError("delta must be in (0, 1)")
    acc = RDPAccountant()
    for sigma, q, steps in schedule:
        if not math.isfinite(q) or not 0 <= q <= 1:
            raise ValueError("sample rate must be finite and in [0, 1]")
        if not math.isfinite(steps) or steps < 0 or int(steps) != steps:
            raise ValueError("steps must be a nonnegative integer")
        if q == 0 or steps == 0:
            continue
        if sigma is None or not math.isfinite(sigma) or sigma < 0:
            raise ValueError("noise multiplier must be finite and nonnegative")
        if sigma == 0:
            return float("inf")
        # Opacus' ``step()`` coalesces consecutive identical (σ, q) entries into
        # a single (σ, q, count) history record, so calling it ``steps`` times
        # and appending one record of ``steps`` are mathematically identical.
        # We append, because at full-corpus scale a round is ~57,500 steps and
        # calibration bisects ~60 times over 30 rounds — the loop form costs
        # ~10^8 Python calls to build the exact same history.
        entry = (float(sigma), float(q), int(steps))
        if acc.history and acc.history[-1][:2] == entry[:2]:
            s, q_, n = acc.history[-1]
            acc.history[-1] = (s, q_, n + entry[2])
        else:
            acc.history.append(entry)
    if not acc.history:
        return 0.0
    return float(acc.get_epsilon(delta=delta, alphas=_RDP_ALPHAS))


def noise_for_target_epsilon(target_epsilon: float, *, sample_rate: float,
                             steps: int, delta: float = 1e-5) -> float:
    """Inverse accounting: σ needed to spend exactly ``target_epsilon`` over
    ``steps`` steps at sampling rate ``q``. Used to calibrate a fixed total
    budget (RO2/RO3) to a constant-noise baseline."""
    return float(get_noise_multiplier(
        target_epsilon=target_epsilon, target_delta=delta,
        sample_rate=sample_rate, steps=int(steps)))


def sample_rate(batch_size: int, dataset_size: int) -> float:
    return min(1.0, batch_size / max(dataset_size, 1))
