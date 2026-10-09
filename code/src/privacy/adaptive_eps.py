"""
Adaptive privacy-budget (ε) scheduling — RO2, the central algorithmic novelty.

Standard DP-FL uses one fixed noise multiplier σ for every round, spending the
privacy budget uniformly. This module instead allocates a *per-round* σ_r under
a **fixed total budget** (ε_total, δ): the round-wise noise follows a chosen
shape, and the whole schedule is calibrated (by RDP composition) so the
cumulative ε stays at or below ε_total within the search tolerance. This lets early rounds — where the global
gradient is large and noise-tolerant — absorb more noise, while late rounds —
where the gradient signal is weak — get less, improving the privacy–utility
trade-off at no extra budget.

Schedules (relative noise *shape* over rounds):
  * ``static``            — constant σ (the baseline).
  * ``decreasing_noise``  — more noise early, less late (proposed).
  * ``increasing_noise``  — less noise early, more late (ablation).

``build(...)`` returns the calibrated list ``[σ_0, …, σ_{R-1}]`` whose RDP
composition over ``steps_per_round`` steps hits ``target_epsilon``.
"""

from __future__ import annotations

import numpy as np

from .dp_engine import epsilon_from_schedule

SCHEDULES = {"static", "decreasing_noise", "increasing_noise",
             "decreasing_steep", "cosine_decreasing", "step_decreasing"}


def _shape(kind: str, rounds: int) -> np.ndarray:
    """Relative per-round noise weights (mean-normalised to 1.0).

    All "decreasing" shapes put *more* noise in early rounds (where the global
    gradient is large and noise-tolerant) and *less* late (where the signal is
    weak) — the RO2 hypothesis. They differ only in how aggressively the noise
    falls off, which is exactly what the tuning sweep probes.
    """
    if rounds <= 1:
        return np.ones(max(rounds, 1))
    r = np.arange(rounds, dtype=np.float64)
    if kind == "static":
        s = np.ones(rounds)
    elif kind == "decreasing_noise":
        # linearly decreasing from 1.5 -> 0.5 (more noise early)
        s = np.linspace(1.5, 0.5, rounds)
    elif kind == "increasing_noise":
        s = np.linspace(0.5, 1.5, rounds)
    elif kind == "decreasing_steep":
        # aggressive linear fall-off: 1.9 -> 0.1
        s = np.linspace(1.9, 0.1, rounds)
    elif kind == "cosine_decreasing":
        # smooth cosine taper from high to low noise
        s = 0.1 + 0.9 * (1 + np.cos(np.pi * r / (rounds - 1)))  # 2.0 -> 0.1 shaped
    elif kind == "step_decreasing":
        # high noise for the first half, low for the second half
        s = np.where(r < rounds / 2.0, 1.6, 0.4)
    else:
        raise ValueError(f"unknown schedule '{kind}' (expected {sorted(SCHEDULES)})")
    return s / s.mean()


def build(kind: str, *, rounds: int, target_epsilon: float, sample_rate: float,
          steps_per_round: int, delta: float = 1e-5,
          tol: float = 1e-2, max_iter: int = 60,
          mechanism_profiles: list[list[tuple[float, int]]] | None = None) -> list[float]:
    """Calibrate a per-round σ schedule without exceeding ``target_epsilon``.

    ``σ_r = k · shape_r``; a scalar ``k`` is found by bisection so that the RDP
    composition over all rounds approaches ``target_epsilon`` from below (ε decreases as k
    grows). Returns ``[σ_0, …, σ_{R-1}]``.
    """
    if (not np.isfinite(target_epsilon) or target_epsilon <= 0 or rounds < 1
            or steps_per_round < 1 or not 0 < sample_rate <= 1
            or not 0 < delta < 1 or not 0 < tol < 1 or max_iter < 1):
        raise ValueError("Invalid privacy calibration parameters")
    shape = _shape(kind, rounds)
    profiles = mechanism_profiles or [[(sample_rate, steps_per_round)] * rounds]
    if any(len(profile) != rounds for profile in profiles):
        raise ValueError("Each client profile must contain one mechanism per round")
    # Identical client profiles need only one accountant evaluation.
    profiles = list(dict.fromkeys(tuple(profile) for profile in profiles))

    def total_eps(k: float) -> float:
        return max(epsilon_from_schedule(
            [(k * shape[r], q, steps) for r, (q, steps) in enumerate(profile)],
            delta=delta) for profile in profiles)

    lo, hi = 1e-3, 1e3
    if total_eps(hi) > target_epsilon:
        raise ValueError("Target cannot be bracketed within the noise search range")
    for _ in range(max_iter):
        mid = np.sqrt(lo * hi)              # geometric bisection (σ spans orders of magnitude)
        eps = total_eps(mid)
        if eps > target_epsilon:            # too much privacy spent -> need more noise
            lo = mid
        else:
            hi = mid
            if target_epsilon - eps <= tol * target_epsilon:
                break
    # Always return the privacy-safe side, including when iteration-limited.
    return [float(k) for k in (hi * shape)]


def constant_sigma(target_epsilon: float, *, rounds: int, sample_rate: float,
                   steps_per_round: int, delta: float = 1e-5) -> float:
    """Single σ (static baseline) calibrated to ``target_epsilon`` over the run."""
    sched = build("static", rounds=rounds, target_epsilon=target_epsilon,
                  sample_rate=sample_rate, steps_per_round=steps_per_round,
                  delta=delta)
    return sched[0]
