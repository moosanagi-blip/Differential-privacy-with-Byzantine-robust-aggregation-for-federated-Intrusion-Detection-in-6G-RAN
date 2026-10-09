"""
Federated training engine.

Runs the full DP-FL-IDS round loop for a ``RunConfig`` and returns a rich result
(metric timeline + final metrics + RO4 interaction data). This *sequential*
engine is the reference implementation — deterministic, dependency-light, and
identical in semantics to the Flower engine (``flower_engine.py``), which reuses
the same ``local_train`` and ``aggregation`` code. Choose via ``cfg.engine``.

Per round it: selects clients → each runs ``local_train`` (optionally DP-SGD at
that round's σ, optionally Byzantine) → the server aggregates (FedAvg / Krum /
Multi-Krum / Trimmed-Mean) → the global model is evaluated on the held-out test
set. It also tracks the six RO3 metrics (accuracy, macro-F1, FPR, convergence
speed, cumulative ε, per-round overhead) and, for RO4, which clients Multi-Krum
selected vs which were actually Byzantine.
"""

from __future__ import annotations

import time
from typing import Callable, Optional

import numpy as np
import torch

from ..attacks import byzantine
from ..data import loaders
from ..evaluation import metrics as M
from ..experiment_config import RunConfig
from ..models.lstm_ids import build_model
from ..privacy import adaptive_eps
from ..privacy.dp_engine import epsilon_from_schedule, sample_rate
from ..privacy.execution import client_profiles, calibrated_schedule, ExecutionLedger
from ..training import evaluate, set_seed
from ..utils import console
from ..utils.provenance import capture as capture_provenance
from . import aggregation
from .diagnostics import round_diagnostics
from .local import local_train
from .params import get_params, set_params


# Observer callback: ``fn(event_name, payload)``. Purely for progress reporting
# (the narrated demo uses it to show what happens *inside* a round, which would
# otherwise be up to 50 s of silence). It never influences training.
EventFn = Callable[[str, dict], None]


def _emit(on_event: Optional[EventFn], event: str, /, **payload) -> None:
    """Fire an observer event, never letting a reporting bug kill a run.

    Both leading arguments are positional-only so that payload keys may freely
    use names like ``name`` or ``event``.
    """
    if on_event is None:
        return
    try:
        on_event(event, payload)
    except Exception as exc:  # pragma: no cover - defensive
        if not getattr(_emit, "_warned", False):
            console.warn(f"progress callback failed ({exc!r}); continuing silently")
            _emit._warned = True  # type: ignore[attr-defined]


def _resolve_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _noise_schedule(cfg: RunConfig, avg_n: int) -> tuple[list[float], float, int]:
    """Return (per-round σ list, sample_rate q, steps_per_round)."""
    q = sample_rate(cfg.batch_size, avg_n)
    steps = cfg.local_epochs * int(np.ceil(avg_n / cfg.batch_size))
    if not cfg.dp:
        return [0.0] * cfg.rounds, q, steps
    if cfg.adaptive_eps:
        assert cfg.target_epsilon, "adaptive_eps requires target_epsilon"
        sched = adaptive_eps.build(cfg.eps_schedule, rounds=cfg.rounds,
                                   target_epsilon=cfg.target_epsilon,
                                   sample_rate=q, steps_per_round=steps,
                                   delta=cfg.delta)
    elif cfg.target_epsilon:
        sigma = adaptive_eps.constant_sigma(cfg.target_epsilon, rounds=cfg.rounds,
                                            sample_rate=q, steps_per_round=steps,
                                            delta=cfg.delta)
        sched = [sigma] * cfg.rounds
    else:
        sched = [cfg.noise_multiplier] * cfg.rounds
    return sched, q, steps


def run_federated(cfg: RunConfig, verbose: bool = True, return_model: bool = False,
                  on_event: Optional[EventFn] = None):
    """Run one federated experiment.

    Returns the result dict. If ``return_model=True`` returns
    ``(result_dict, trained_global_model)`` so callers (e.g. the confusion-matrix
    figure) can run predictions with the final model — the dict itself stays
    JSON-serialisable.

    ``cfg.engine == "flower"`` routes to the genuine-Flower engine
    (``flower_engine.run_federated_flower``); the default sequential engine below
    is the reference implementation. Both reuse ``local_train`` + ``aggregation``.

    ``on_event`` is an optional observer (see ``EventFn``) called at run/round/
    client boundaries so a caller can narrate progress live. It is read-only —
    passing one does not change a single number.
    """
    if getattr(cfg, "engine", "sequential") == "flower" and not return_model:
        from .flower_engine import run_federated_flower  # lazy: avoid import cycle
        return run_federated_flower(cfg, verbose=verbose, on_event=on_event)

    set_seed(cfg.seed)
    device = _resolve_device(cfg.device)

    # ── dataset ──
    root = loaders.DataRoot(getattr(cfg, "data_root", None))
    provenance = capture_provenance(cfg, root)
    _emit(on_event, "provenance", **provenance)
    n_clients = min(cfg.num_clients, root.n_clients())
    # Shard sizes come from parquet metadata, so this costs no row reads even
    # when a shard holds millions of rows.
    sizes = [root.client_rows(c) for c in range(n_clients)]
    # Chunks default to one per round, which is what consumes the corpus exactly
    # once; cfg.n_chunks overrides that for short runs that should still train
    # on a normal round's worth of data rather than a whole shard.
    n_chunks = int(getattr(cfg, "n_chunks", None) or cfg.rounds)
    chunked = bool(getattr(cfg, "round_chunks", False)) and n_chunks > 1

    # Eagerly holding every shard is ~8 GB once shards reach full-corpus size,
    # so chunked runs load only the round's slice, per client, per round.
    client_data = None if chunked else [root.load_client(c, mask_ran_features=cfg.mask_ran_features)
                                        for c in range(n_clients)]

    test_loader = root.test_loader(
        batch_size=256, subset=bool(getattr(cfg, "eval_on_subset", True)),
        mask_ran_features=cfg.mask_ran_features)
    n_features, n_classes = root.num_features(), root.num_classes()

    byz = byzantine.select_byzantine(n_clients, cfg.byzantine_fraction, seed=cfg.seed)
    f_est = len(byz) if cfg.n_byzantine_estimate is None else cfg.n_byzantine_estimate
    if int(f_est) != f_est or f_est < 0:
        raise ValueError("Configured attacker estimate must be a nonnegative integer")

    # Legacy descriptive size fields are retained for compatibility only.
    # Corrected DP calibration below uses exact per-client/per-round profiles.
    per_round_n = [max(1, s // n_chunks) for s in sizes] if chunked else sizes
    # Calibrate against the smallest shard when asked, so the stated epsilon is
    # one that holds for every gNB rather than only for an average one.
    calib_n = (min(per_round_n) if getattr(cfg, "dp_calibrate", "mean") == "worst"
               else int(np.mean(per_round_n)))
    avg_n = int(calib_n)
    profiles = client_profiles(sizes, rounds=cfg.rounds, batch_size=cfg.batch_size,
                               epochs=cfg.local_epochs, n_chunks=n_chunks if chunked else None)
    if cfg.dp:
        sched = calibrated_schedule(cfg, profiles)
        q, steps = profiles[0][0]  # descriptive only; actual accounting is per client
    else:
        sched, q, steps = _noise_schedule(cfg, avg_n)
    ledger = ExecutionLedger(n_clients, cfg.delta)

    # ── init global model ──
    set_seed(cfg.seed)
    global_model = build_model(cfg.model, n_features, n_classes, hidden=cfg.hidden)
    if getattr(root, 'feature_dtype', 'float32') == 'float64':
        global_model = global_model.double()
    global_params = get_params(global_model)

    if verbose:
        console.info(f"clients={n_clients}  byzantine={sorted(byz)} "
                     f"(f={f_est})  aggregator={cfg.aggregator}  dp={cfg.dp}"
                     + (f"  adaptive_eps={cfg.eps_schedule}" if cfg.adaptive_eps else ""))

    timeline: list[dict] = []
    m_per_round = cfg.clients_per_round or n_clients

    _emit(on_event, "run_start", name=cfg.name, n_clients=n_clients,
          byzantine=sorted(int(c) for c in byz), f_est=f_est,
          aggregator=cfg.aggregator, dp=cfg.dp, rounds=cfg.rounds,
          sigma_schedule=[float(s) for s in sched], client_sizes=sizes,
          model=cfg.model, target_epsilon=cfg.target_epsilon)

    for r in range(cfg.rounds):
        t0 = time.time()
        sigma_r = sched[r] if cfg.dp else 0.0
        _emit(on_event, "round_start", round=r + 1, rounds=cfg.rounds,
              sigma=float(sigma_r))
        # (all clients participate by default; deterministic subset otherwise)
        rng = np.random.default_rng(cfg.seed + r)
        selected = sorted(rng.choice(n_clients, size=m_per_round, replace=False)) \
            if m_per_round < n_clients else list(range(n_clients))
        byz_round = (byzantine.select_byzantine(n_clients, cfg.byzantine_fraction,
                                                 seed=cfg.seed + r + 1)
                     if cfg.rotate_byzantine else byz)

        updates, weights = [], []
        for idx, c in enumerate(selected, start=1):
            # Chunk r of the shard: across cfg.rounds rounds this consumes every
            # row of every client exactly once, and only one chunk is resident.
            x, y = (root.load_client_chunk(c, r % n_chunks, n_chunks,
                                           mask_ran_features=cfg.mask_ran_features) if chunked
                    else client_data[c])
            t_client = time.time()
            _emit(on_event, "client_start", client=int(c), n_samples=len(y),
                  is_byzantine=bool(c in byz_round), sigma=float(sigma_r),
                  index=idx, total=len(selected))
            params, n_ex, _info = local_train(
                global_params=global_params, x=x, y=y,
                n_features=n_features, n_classes=n_classes,
                model_name=cfg.model, hidden=cfg.hidden,
                epochs=cfg.local_epochs, batch_size=cfg.batch_size, lr=cfg.lr,
                device=device, is_byzantine=(c in byz_round), attack=cfg.attack,
                attack_scale=cfg.attack_scale, attack_noise_std=cfg.attack_noise_std,
                dp=cfg.dp, noise_multiplier=sigma_r, max_grad_norm=cfg.max_grad_norm,
                seed=cfg.seed * 1000 + r * 100 + c,
                optimizer_name=cfg.optimizer, momentum=cfg.momentum)
            updates.append(params)
            weights.append(n_ex)
            if cfg.dp:
                ledger.record(c, _info, profiles[c][r], sigma_r)
            _emit(on_event, "client_end", client=int(c), n_examples=int(n_ex),
                  is_byzantine=bool(c in byz_round), poisoned=bool(_info.get("poisoned")),
                  noise_multiplier=float(_info.get("noise_multiplier", 0.0)),
                  seconds=time.time() - t_client)

        attack_measurement = None
        if cfg.attack in ("adaptive_krum", "boundary_krum"):
            from ..attacks.boundary import craft_boundary, perturbation_metrics
            malicious_positions = [i for i, c in enumerate(selected) if c in byz_round]
            pre_attack = updates
            if cfg.attack == 'boundary_krum':
                if cfg.aggregator != 'multikrum':
                    raise ValueError('Boundary attack currently requires Multi-Krum')
                updates, search = craft_boundary(updates, global_params, malicious_positions,
                                                 f_est, max_scale=cfg.attack_scale)
            else:
                updates = byzantine.craft_adaptive_krum(updates, global_params, malicious_positions,
                                                         f_est, max_scale=cfg.attack_scale)
                search = None
            attack_measurement = perturbation_metrics(pre_attack, updates, global_params, malicious_positions)
            attack_measurement['search'] = search
            for client in attack_measurement['clients']:
                client['client_id'] = int(selected[client['position']])
        _emit(on_event, "aggregate_start", aggregator=cfg.aggregator,
              n_updates=len(updates), f_est=f_est)
        agg, agg_info = aggregation.aggregate(
            cfg.aggregator, updates, weights=weights, n_byzantine=f_est)
        diagnostics = round_diagnostics(updates, selected, byz_round,
            agg_info.get('selected', range(len(selected))), agg_info.get('scores'))
        global_params = agg
        set_params(global_model, global_params)

        ev = evaluate(global_model, test_loader, device=device)
        # Built incrementally: this used to rebuild the schedule from round 0
        # every round, which is quadratic in rounds and, at full-corpus step
        # counts, the single most expensive line in the loop.
        if cfg.dp:
            client_eps = ledger.epsilon_by_client()
            cum_eps = max(client_eps.values())
            if cfg.target_epsilon is not None and cum_eps > cfg.target_epsilon + 1e-9:
                raise ValueError("Executed accounting exceeds target epsilon; refusing a successful result")
        else:
            client_eps = {}
            cum_eps = 0.0

        # RO4: did the defence exclude the Byzantine clients this round?
        sel_global = [selected[i] for i in agg_info.get("selected", range(len(selected)))]
        byz_selected = [c for c in sel_global if c in byz_round]
        rec = {
            "round": r + 1, "accuracy": ev["accuracy"], "macro_f1": ev["macro_f1"],
            "loss": ev["loss"], "sigma": float(sigma_r), "cum_epsilon": float(cum_eps),
            "time_s": time.time() - t0,
            "selected": [int(c) for c in sel_global],
            "byzantine_clients": sorted(int(c) for c in byz_round),
            "byz_selected": [int(c) for c in byz_selected],
            "scores": agg_info.get("scores"),
            "epsilon_by_client": client_eps,
            "diagnostics": diagnostics,
            "attack_measurement": attack_measurement,
        }
        timeline.append(rec)
        _emit(on_event, "aggregate_end", aggregator=cfg.aggregator,
              selected=[int(c) for c in sel_global],
              byz_selected=[int(c) for c in byz_selected],
              excluded=[int(c) for c in selected if c not in sel_global],
              f_est=f_est, scores=agg_info.get("scores"))
        _emit(on_event, "round_end", **rec, rounds=cfg.rounds)
        if verbose:
            print(f"    r{r + 1:02d}/{cfg.rounds}  acc={ev['accuracy']:.3f} "
                  f"f1={ev['macro_f1']:.3f}  eps={cum_eps:.2f}  "
                  # The denominator is the actual compromised coalition, not
                  # the operator's f estimate.  They intentionally differ in
                  # the f-sensitivity experiment.
                  f"byz_kept={len(byz_selected)}/{len(byz_round)}  ({rec['time_s']:.1f}s)")

    # ── final metrics, always on the COMPLETE held-out set ──
    # Per-round scoring may use the stratified eval subset for speed; the
    # numbers that get reported must not.
    # Streamed from parquet: the complete held-out set is ~2 GB as float32 at
    # full corpus scale, and nothing here needs it resident.
    final_loader = (root.stream_test(batch_size=4096, subset=False,
                                     mask_ran_features=cfg.mask_ran_features)
                    if getattr(cfg, "eval_on_subset", True) else test_loader)
    final = M.evaluate_full(global_model, final_loader, device, label_map=root.label_map())
    accs = [t["accuracy"] for t in timeline]
    conv = next((t["round"] for t in timeline if t["accuracy"] >= cfg.conv_threshold), None)

    # Maximum across all executed client histories, not a nominal average shard.
    eps_worst = max(ledger.epsilon_by_client().values()) if cfg.dp else 0.0

    result = {
        "config": cfg.to_dict(),
        "provenance": provenance,
        "timeline": timeline,
        "final": final,
        "convergence_round": conv,
        "final_accuracy": final["accuracy"],
        "final_macro_f1": final["macro_f1"],
        "final_fpr": final["fpr"],
        "final_epsilon": timeline[-1]["cum_epsilon"] if timeline else 0.0,
        "final_epsilon_worst_client": float(eps_worst),
        "avg_round_time_s": float(np.mean([t["time_s"] for t in timeline])) if timeline else 0.0,
        "byzantine_clients": sorted(int(c) for c in byz),
        "noise_schedule": [float(s) for s in sched],
        "privacy_accounting": {
            "version": "executed-opacus-v2",
            "scope": "conditional on fixed prepared client data; no end-to-end raw-record guarantee",
            "randomness": "seeded experimental randomness; not production secure mode",
            "client_selection_amplification": False,
            "history_by_client": ledger.histories if cfg.dp else {},
            "planned_mechanisms_by_client": profiles if cfg.dp else [],
            "epsilon_by_client": ledger.epsilon_by_client() if cfg.dp else {},
        },
        "dataset": {
            "root": root.name,
            "n_train": int(sum(sizes)),
            "client_sizes": [int(s) for s in sizes],
            "round_chunks": chunked,
            "rows_per_client_per_round": int(avg_n),
            "sample_rate_q": None if cfg.dp else float(q),
            "steps_per_round": None if cfg.dp else int(steps),
            "eval_rows": len(final_loader.dataset),
        },
    }
    _emit(on_event, "run_end", name=cfg.name,
          accuracy=result["final_accuracy"], macro_f1=result["final_macro_f1"],
          fpr=result["final_fpr"], epsilon=result["final_epsilon"],
          convergence_round=conv)

    if return_model:
        return result, global_model
    return result
