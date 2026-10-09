"""
RO4 — DP-noise × Multi-Krum interaction analysis.

Central RO4 question: does the Gaussian DP noise added to client updates blur
Multi-Krum's distance-based outlier scoring, letting Byzantine updates slip
through? Two complementary measurements:

1. **Exclusion rate over a full run** — from a run's timeline, the fraction of
   Byzantine clients Multi-Krum kept *out* of the aggregate each round (higher =
   better defence). Compare a no-DP run against a DP run at matched settings.

2. **Score-separation micro-benchmark** — take a fixed pool of honest + Byzantine
   updates, inject increasing DP-scale Gaussian noise, and track the normalised
   gap between Byzantine and honest Krum scores. A shrinking gap as σ grows is
   direct evidence of interference.
"""

from __future__ import annotations

import numpy as np

from ..attacks import byzantine
from ..data import loaders
from ..experiment_config import RunConfig
from ..fl.aggregation import krum_scores, multi_krum
from ..fl.local import local_train
from ..fl.params import flatten, get_params
from ..models.lstm_ids import build_model
from ..training import set_seed, get_device


def exclusion_rate(result: dict) -> float:
    """Mean fraction of Byzantine clients kept OUT of the aggregate per round."""
    f = len(result.get("byzantine_clients", []))
    if f == 0:
        return float("nan")
    kept = [len(t.get("byz_selected", [])) for t in result["timeline"]]
    return float(np.mean([(f - k) / f for k in kept]))


def score_separation(*, attack: str = "sign_flip", byz_fraction: float = 0.3,
                     sigmas=(0.0, 0.5, 1.0, 2.0, 4.0), model: str = "mlp",
                     seed: int = 42, data_root: str | None = None,
                     rows_per_client: int = 50_000) -> dict:
    """Build one pool of client updates, then measure honest/Byzantine Krum-score
    separation as DP-scale Gaussian noise σ increases.

    What is measured is the *geometry* of the Krum scores, which needs one
    representative update per gNB rather than a converged model, so each client
    contributes a bounded slice of its shard — reading all ten full-corpus
    shards would pull ~8 GB into memory to answer a question about distances.
    """
    set_seed(seed)
    device = get_device()
    root = loaders.DataRoot(data_root)
    n_clients = root.n_clients()
    n_features, n_classes = loaders.N_FEATURES, root.num_classes()
    byz = byzantine.select_byzantine(n_clients, byz_fraction, seed=seed)

    # hidden must match the local_train models below (64), else set_params fails.
    base = build_model(model, n_features, n_classes, hidden=64)
    global_params = get_params(base)

    # One honest/Byzantine update per client (one local round).
    updates = []
    for c in range(n_clients):
        total = root.client_rows(c)
        n_chunks = max(1, -(-total // max(rows_per_client, 1)))
        x, y = root.load_client_chunk(c, 0, n_chunks)
        params, _, _ = local_train(
            global_params=global_params, x=x, y=y, n_features=n_features,
            n_classes=n_classes, model_name=model, hidden=64, epochs=1,
            batch_size=64, lr=1e-3, device=device, is_byzantine=(c in byz),
            attack=attack, dp=False, seed=seed + c)
        updates.append(params)

    honest_idx = [c for c in range(n_clients) if c not in byz]
    byz_idx = sorted(byz)
    rng = np.random.default_rng(seed)
    dim = flatten(updates[0]).size

    out = {"sigmas": list(sigmas), "separation": [], "exclusion": [],
           "attack": attack, "byz_fraction": byz_fraction, "byzantine": byz_idx}
    for s in sigmas:
        noisy = [[p + rng.normal(0, s, size=p.shape).astype(p.dtype) for p in u]
                 for u in updates]
        scores = krum_scores(noisy, n_byzantine=len(byz))
        h = scores[honest_idx].mean()
        b = scores[byz_idx].mean()
        spread = scores.std() + 1e-9
        out["separation"].append(float((b - h) / spread))  # >0 => Byzantine look like outliers
        _, selected, _ = multi_krum(noisy, n_byzantine=len(byz))
        byz_kept = len(set(selected) & byz)
        out["exclusion"].append(float((len(byz) - byz_kept) / max(len(byz), 1)))
    return out


def run_interaction(target_epsilon: float = 5.0, attack: str = "sign_flip",
                    byz_fraction: float = 0.3, rounds: int = 10,
                    model: str = "mlp", optimizer: str = "sgd",
                    lr: float | None = None, device: str = "cpu",
                    seed: int = 42, verbose: bool = False,
                    on_event=None, data_root: str | None = None,
                    round_chunks: bool | None = None) -> dict:
    """Matched no-DP vs DP Multi-Krum runs → exclusion + accuracy comparison.

    ``optimizer`` defaults to SGD (canonical DP-SGD) so this analysis matches the
    rest of the campaign; earlier revisions fell through to the ``RunConfig``
    default of Adam, which measured RO4b under a different optimizer than
    RO1/RO3. Pass ``optimizer="adam"`` to reproduce those older numbers.
    """
    from ..fl.simulation import run_federated
    if lr is None:
        lr = 0.1 if optimizer == "sgd" else 1e-3
    if round_chunks is None:
        round_chunks = bool(data_root)
    common = dict(model=model, rounds=rounds, aggregator="multikrum",
                  attack=attack, byzantine_fraction=byz_fraction, device=device,
                  optimizer=optimizer, lr=lr, seed=seed,
                  data_root=data_root, round_chunks=round_chunks)
    no_dp = run_federated(RunConfig(name="krum_noDP", dp=False, **common),
                          verbose=verbose, on_event=on_event)
    dp = run_federated(RunConfig(name="krum_DP", dp=True, target_epsilon=target_epsilon,
                                 **common), verbose=verbose, on_event=on_event)
    return {
        "no_dp": {"exclusion_rate": exclusion_rate(no_dp),
                  "accuracy": no_dp["final_accuracy"]},
        "dp": {"exclusion_rate": exclusion_rate(dp), "accuracy": dp["final_accuracy"],
               "epsilon": dp["final_epsilon"]},
    }
