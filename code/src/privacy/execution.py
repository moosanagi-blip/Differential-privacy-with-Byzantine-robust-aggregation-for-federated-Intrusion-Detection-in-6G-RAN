"""Account executed DP optimizer steps, conditional on fixed prepared client data.

No amplification is claimed from client selection or aggregator rejection.
Sequential composition across disjoint slices is conservative; no raw-data
preprocessing guarantee or protection for malicious clients is established here.
"""
import math

from . import adaptive_eps
from .dp_engine import epsilon_from_schedule


def loader_mechanism(n: int, batch_size: int, epochs: int) -> tuple[float, int]:
    if n < 1 or batch_size < 1 or epochs < 1:
        raise ValueError("DP requires nonempty data, positive batch size and epochs")
    batches = math.ceil(n / batch_size)
    return 1.0 / batches, epochs * batches


def client_profiles(sizes, *, rounds, batch_size, epochs, n_chunks=None):
    if rounds < 1 or (n_chunks is not None and n_chunks < 1):
        raise ValueError("rounds and chunk count must be positive")
    profiles = []
    for size in sizes:
        profile = []
        for r in range(rounds):
            j = r % n_chunks if n_chunks else 0
            n = ((size * (j + 1)) // n_chunks - (size * j) // n_chunks
                 if n_chunks else size)
            profile.append(loader_mechanism(n, batch_size, epochs))
        profiles.append(profile)
    if not profiles:
        raise ValueError("At least one client is required")
    return profiles


def calibrated_schedule(cfg, profiles):
    if cfg.dp_calibrate != "worst":
        raise ValueError("Corrected DP execution requires dp_calibrate='worst'")
    if cfg.target_epsilon is None:
        if cfg.adaptive_eps:
            raise ValueError("Adaptive scheduling requires a target epsilon")
        if not math.isfinite(cfg.noise_multiplier) or cfg.noise_multiplier <= 0:
            raise ValueError("DP requires positive finite noise")
        return [float(cfg.noise_multiplier)] * cfg.rounds
    q, steps = profiles[0][0]
    return adaptive_eps.build(
        cfg.eps_schedule if cfg.adaptive_eps else "static", rounds=cfg.rounds,
        target_epsilon=cfg.target_epsilon, sample_rate=q, steps_per_round=steps,
        delta=cfg.delta, mechanism_profiles=profiles)


class ExecutionLedger:
    def __init__(self, n_clients, delta):
        self.histories = {c: [] for c in range(n_clients)}
        self.delta = delta

    def record(self, client, info, expected, sigma):
        history = info.get("accountant_history")
        if not history:
            raise ValueError("Missing executed Opacus history; nominal accounting is forbidden")
        q, steps = expected
        if sum(h[2] for h in history) != steps or any(
            not math.isclose(h[0], sigma, rel_tol=1e-12)
            or not math.isclose(h[1], q, rel_tol=1e-12) for h in history
        ):
            raise ValueError("Executed DP mechanism differs from calibrated plan")
        self.histories[client].extend(tuple(h) for h in history)

    def epsilon_by_client(self):
        return {c: epsilon_from_schedule(h, delta=self.delta) for c,h in self.histories.items()}
