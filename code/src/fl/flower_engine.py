"""
Flower engine — proposal-fidelity FL implementation.

The thesis proposal specifies **Flower (flwr)** as the federated-learning
framework. This module implements the DP-FL-IDS round loop with *genuine Flower
abstractions*:

  * ``IDSClient(flwr.client.NumPyClient)`` — a gNB client whose ``fit`` reuses the
    exact same ``local_train`` (optional Opacus DP-SGD + optional Byzantine
    attack) as the sequential engine.
  * ``RobustStrategy(flwr.server.strategy.FedAvg)`` — a server strategy whose
    ``aggregate_fit`` reuses our ``aggregation.aggregate`` (FedAvg / Krum /
    Multi-Krum / Trimmed-Mean) and records which clients were selected (for RO4).

Flower's official simulation transport (``start_simulation`` / ``run_simulation``)
needs Ray, which is **not installed here** (and is fragile on Windows — the client
deliverable target). So we drive the *real* Client and Strategy objects with a
small in-process orchestrator (``run_federated_flower``) that speaks Flower's
message types (``FitIns``/``FitRes``/``Parameters``). This keeps full API fidelity
without the Ray dependency, and returns the **same result dict** as the sequential
engine so the two are interchangeable (``cfg.engine`` selects which).
"""

from __future__ import annotations

import time

import numpy as np
import torch
from flwr.client import NumPyClient
from flwr.common import (Code, FitIns, FitRes, Parameters, Status,
                         ndarrays_to_parameters, parameters_to_ndarrays)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from ..data import loaders
from ..evaluation import metrics as M
from ..experiment_config import RunConfig
from ..models.lstm_ids import build_model
from ..training import evaluate, set_seed
from . import aggregation
from .local import local_train
from .params import Params
from .simulation import _noise_schedule, _resolve_device
from .params import get_params, set_params


# ── client ─────────────────────────────────────────────────────────────────
class IDSClient(NumPyClient):
    """One gNB. ``fit`` runs a local round via the shared ``local_train``."""

    def __init__(self, cid: int, x: np.ndarray, y: np.ndarray, cfg: RunConfig,
                 n_features: int, n_classes: int, device: torch.device,
                 is_byzantine: bool):
        self.cid = int(cid)
        self.x, self.y = x, y
        self.cfg = cfg
        self.n_features, self.n_classes = n_features, n_classes
        self.device = device
        self.is_byzantine = is_byzantine
        self._params: Params = []

    def set_data(self, x: np.ndarray, y: np.ndarray) -> None:
        """Hand this gNB the slice of its shard for the round about to start.

        Chunked runs never hold more than one slice per client, which is what
        keeps a full-corpus run inside a laptop's memory.
        """
        self.x, self.y = x, y

    def get_parameters(self, config):  # noqa: D401 - Flower signature
        return self._params

    def fit(self, parameters, config):
        cfg, r = self.cfg, int(config.get("round", 0))
        sigma = float(config.get("noise_multiplier", 0.0))
        new_params, n_ex, info = local_train(
            global_params=parameters, x=self.x, y=self.y,
            n_features=self.n_features, n_classes=self.n_classes,
            model_name=cfg.model, hidden=cfg.hidden,
            epochs=cfg.local_epochs, batch_size=cfg.batch_size, lr=cfg.lr,
            device=self.device, is_byzantine=self.is_byzantine, attack=cfg.attack,
            attack_scale=cfg.attack_scale, attack_noise_std=cfg.attack_noise_std,
            dp=cfg.dp, noise_multiplier=sigma, max_grad_norm=cfg.max_grad_norm,
            seed=cfg.seed * 1000 + r * 100 + self.cid,
            optimizer_name=cfg.optimizer, momentum=cfg.momentum)
        self._params = new_params
        return new_params, n_ex, {"cid": self.cid}


# ── server strategy ────────────────────────────────────────────────────────
class RobustStrategy(FedAvg):
    """FedAvg replaced with our Byzantine-robust aggregation. Records the set of
    clients Multi-Krum selected each round (``last_selected``) for RO4."""

    def __init__(self, aggregator: str, n_byzantine: int, **kw):
        super().__init__(**kw)
        self.aggregator = aggregator
        self.n_byzantine = int(n_byzantine)
        self.last_selected: list[int] = []

    def aggregate_fit(self, server_round, results, failures):
        # results: list[(ClientProxy, FitRes)] -> ordered updates + weights
        order = sorted(range(len(results)),
                       key=lambda i: int(results[i][1].metrics.get("cid", i)))
        updates = [parameters_to_ndarrays(results[i][1].parameters) for i in order]
        weights = [results[i][1].num_examples for i in order]
        cids = [int(results[i][1].metrics.get("cid", i)) for i in order]

        agg, info = aggregation.aggregate(
            self.aggregator, updates, weights=weights, n_byzantine=self.n_byzantine)
        self.last_selected = [cids[j] for j in info.get("selected", range(len(cids)))]
        self.last_scores = info.get("scores")
        return ndarrays_to_parameters(agg), {"selected": str(self.last_selected)}


# ── minimal in-process client proxy (no gRPC/Ray) ──────────────────────────
class _LocalProxy(ClientProxy):
    """Carrier object so ``aggregate_fit`` receives ``(proxy, FitRes)`` pairs."""

    def __init__(self, cid: str):
        super().__init__(cid)

    def get_parameters(self, ins, timeout=None, group_id=None): ...      # unused
    def fit(self, ins, timeout=None, group_id=None): ...                  # unused
    def evaluate(self, ins, timeout=None, group_id=None): ...             # unused
    def get_properties(self, ins, timeout=None, group_id=None): ...       # unused
    def reconnect(self, ins, timeout=None, group_id=None): ...            # unused


# ── driver ─────────────────────────────────────────────────────────────────
def run_federated_flower(cfg: RunConfig, verbose: bool = True,
                         on_event=None) -> dict:
    """Run the DP-FL-IDS loop through genuine Flower Client + Strategy objects.

    Mirrors ``simulation.run_federated`` (same seeds, same metrics, same result
    dict) but routes local updates through ``IDSClient.fit`` and aggregation
    through ``RobustStrategy.aggregate_fit``.

    ``on_event`` is the same read-only progress observer accepted by
    ``simulation.run_federated``; see ``simulation.EventFn``.
    """
    if cfg.dp:
        raise NotImplementedError("Corrected executed DP accounting currently requires the sequential engine")
    if (cfg.rotate_byzantine or cfg.mask_ran_features
            or cfg.n_byzantine_estimate is not None or cfg.attack in ("adaptive_krum", "boundary_krum")
            or cfg.clients_per_round is not None):
        raise NotImplementedError("Revision options require the sequential engine; Flower parity is not validated")
    from .simulation import _emit
    from ..attacks import byzantine
    from ..privacy.dp_engine import epsilon_from_schedule, sample_rate  # noqa: F401
    from ..utils import console

    set_seed(cfg.seed)
    device = _resolve_device(cfg.device)

    # Same dataset handling as the sequential engine, so the two stay comparable
    # on the corpus the run was actually pointed at.
    root = loaders.DataRoot(getattr(cfg, "data_root", None))
    if getattr(root, 'feature_dtype', 'float32') != 'float32':
        raise ValueError('Float64 evidence runs require the validated sequential engine')
    n_clients = min(cfg.num_clients, root.n_clients())
    # Shard sizes come from parquet metadata: no row reads even at 3.7M/shard.
    sizes = [root.client_rows(c) for c in range(n_clients)]
    n_chunks = int(getattr(cfg, "n_chunks", None) or cfg.rounds)
    chunked = bool(getattr(cfg, "round_chunks", False)) and n_chunks > 1
    # Eagerly holding every shard is ~8 GB at full-corpus size, so chunked runs
    # load only the round's slice, per client, per round.
    client_data = None if chunked else [root.load_client(c) for c in range(n_clients)]

    test_loader = root.test_loader(
        batch_size=256, subset=bool(getattr(cfg, "eval_on_subset", True)))
    n_features, n_classes = root.num_features(), root.num_classes()

    # DP accounting uses the records a client actually samples from in one round
    # — the whole shard normally, one chunk when chunked — because the sampling
    # rate q = batch/n is what drives epsilon.
    per_round_n = [max(1, s // n_chunks) for s in sizes] if chunked else sizes
    avg_n = int(min(per_round_n) if getattr(cfg, "dp_calibrate", "mean") == "worst"
                else np.mean(per_round_n))

    byz = byzantine.select_byzantine(n_clients, cfg.byzantine_fraction, seed=cfg.seed)
    f_est = len(byz)
    sched, q, steps = _noise_schedule(cfg, avg_n)

    set_seed(cfg.seed)
    global_model = build_model(cfg.model, n_features, n_classes, hidden=cfg.hidden)
    global_params = get_params(global_model)

    # Chunked clients start empty and are handed their slice at the top of each
    # round; unchunked ones keep their whole shard for the run, as before.
    clients = [IDSClient(c, *(client_data[c] if client_data else (None, None)),
                         cfg, n_features, n_classes, device,
                         is_byzantine=(c in byz)) for c in range(n_clients)]
    proxies = [_LocalProxy(str(c)) for c in range(n_clients)]
    strategy = RobustStrategy(aggregator=cfg.aggregator, n_byzantine=f_est)

    if verbose:
        console.info(f"[FLOWER] clients={n_clients}  byzantine={sorted(byz)} "
                     f"(f={f_est})  aggregator={cfg.aggregator}  dp={cfg.dp}")

    _emit(on_event, "run_start", name=cfg.name, n_clients=n_clients,
          byzantine=sorted(int(c) for c in byz), f_est=f_est,
          aggregator=cfg.aggregator, dp=cfg.dp, rounds=cfg.rounds,
          sigma_schedule=[float(s) for s in sched], engine="flower",
          model=cfg.model, target_epsilon=cfg.target_epsilon)

    timeline: list[dict] = []
    for r in range(cfg.rounds):
        t0 = time.time()
        sigma_r = sched[r] if cfg.dp else 0.0
        fit_config = {"round": r, "noise_multiplier": float(sigma_r)}
        _emit(on_event, "round_start", round=r + 1, rounds=cfg.rounds,
              sigma=float(sigma_r))

        results = []
        for c in range(n_clients):
            # Chunk r of the shard: across cfg.rounds rounds this consumes every
            # row of every client exactly once, and only one chunk is resident.
            if chunked:
                clients[c].set_data(*root.load_client_chunk(c, r % n_chunks, n_chunks))
            t_client = time.time()
            _emit(on_event, "client_start", client=int(c),
                  n_samples=len(clients[c].y), is_byzantine=bool(c in byz),
                  sigma=float(sigma_r), index=c + 1, total=n_clients)
            new_params, n_ex, mets = clients[c].fit(global_params, fit_config)
            fr = FitRes(status=Status(Code.OK, "ok"),
                        parameters=ndarrays_to_parameters(new_params),
                        num_examples=n_ex, metrics=mets)
            results.append((proxies[c], fr))
            _emit(on_event, "client_end", client=int(c), n_examples=int(n_ex),
                  is_byzantine=bool(c in byz), seconds=time.time() - t_client)

        _emit(on_event, "aggregate_start", aggregator=cfg.aggregator,
              n_updates=len(results), f_est=f_est)
        agg_params, _ = strategy.aggregate_fit(r + 1, results, [])
        global_params = parameters_to_ndarrays(agg_params)
        set_params(global_model, global_params)

        ev = evaluate(global_model, test_loader, device=device)
        cum_eps = epsilon_from_schedule(
            [(sched[i], q, steps) for i in range(r + 1)], delta=cfg.delta) if cfg.dp else 0.0
        byz_selected = [c for c in strategy.last_selected if c in byz]
        rec = {
            "round": r + 1, "accuracy": ev["accuracy"], "macro_f1": ev["macro_f1"],
            "loss": ev["loss"], "sigma": float(sigma_r), "cum_epsilon": float(cum_eps),
            "time_s": time.time() - t0,
            "selected": sorted(int(c) for c in strategy.last_selected),
            "byz_selected": [int(c) for c in byz_selected],
        }
        timeline.append(rec)
        _emit(on_event, "aggregate_end", aggregator=cfg.aggregator,
              selected=rec["selected"], byz_selected=rec["byz_selected"],
              excluded=[c for c in range(n_clients) if c not in rec["selected"]],
              f_est=f_est, scores=None)
        _emit(on_event, "round_end", **rec, rounds=cfg.rounds)
        if verbose:
            print(f"    r{r + 1:02d}/{cfg.rounds}  acc={ev['accuracy']:.3f} "
                  f"f1={ev['macro_f1']:.3f}  eps={cum_eps:.2f}  "
                  f"byz_kept={len(byz_selected)}/{f_est}  (flower)")

    # Final metrics always on the COMPLETE held-out set, streamed from parquet,
    # exactly as the sequential engine does — otherwise a parity check would be
    # comparing two different measurements.
    final_loader = (root.stream_test(batch_size=4096, subset=False)
                    if getattr(cfg, "eval_on_subset", True) else test_loader)
    final = M.evaluate_full(global_model, final_loader, device, label_map=root.label_map())
    accs = [t["accuracy"] for t in timeline]
    conv = next((t["round"] for t in timeline if t["accuracy"] >= cfg.conv_threshold), None)
    result = {
        "config": cfg.to_dict(), "engine": "flower", "timeline": timeline,
        "final": final, "convergence_round": conv,
        "final_accuracy": final["accuracy"],
        "final_macro_f1": final["macro_f1"],
        "final_fpr": final["fpr"],
        "final_epsilon": timeline[-1]["cum_epsilon"] if timeline else 0.0,
        "avg_round_time_s": float(np.mean([t["time_s"] for t in timeline])) if timeline else 0.0,
        "byzantine_clients": sorted(int(c) for c in byz),
        "noise_schedule": [float(s) for s in sched],
        "dataset": {
            "root": root.name,
            "n_train": int(sum(sizes)),
            "client_sizes": [int(s) for s in sizes],
            "round_chunks": chunked,
            "rows_per_client_per_round": int(avg_n),
        },
    }
    _emit(on_event, "run_end", name=cfg.name, engine="flower",
          accuracy=result["final_accuracy"], macro_f1=result["final_macro_f1"],
          fpr=result["final_fpr"], epsilon=result["final_epsilon"],
          convergence_round=conv)
    return result
