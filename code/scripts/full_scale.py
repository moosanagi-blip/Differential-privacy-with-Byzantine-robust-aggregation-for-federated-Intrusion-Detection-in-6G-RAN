#!/usr/bin/env python3
"""
Run the headline experiment set against a full-corpus dataset root.

This is the driver behind ``python main.py --campaign``, and the one that
produced the reported campaign, filed under ``results/Research Objective 1..4/``. A single run over the shards built by
``src.data.build_scaled`` is tens of minutes, so it adds the two things that
matter at that length:

* **Resumability.** Every run is written as soon as it finishes and skipped on a
  later invocation. A crash, an OOM, or a Ctrl-C costs the current run, not the
  campaign.
* **Deduplication.** Several blocks ask for the same configuration — the
  ablation's no-DP FedAvg cell is the Byzantine block's 0%-attackers cell, and
  its DP+Multi-Krum cell is the epsilon sweep's eps=5 point. Results are seeded
  and deterministic, so those are computed once.

Chunked rounds are on by default: each client's shard is split into `rounds`
disjoint slices and round r trains on slice r, so the run consumes every row of
the corpus exactly once.

    python3 scripts/full_scale.py --data-root data_full   --out results/full_scale
    python3 scripts/full_scale.py --data-root data_capped --out results/capped_scale
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message=".*Secure RNG turned off.*")
warnings.filterwarnings("ignore", message=".*Optimal order is the.*alpha.*")
warnings.filterwarnings("ignore", message=".*CUDA initialization.*")

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))
from src.data.config import PROJECT_ROOT  # noqa: E402  (data/ + results/ live above code/)

from src.experiment_config import RunConfig                      # noqa: E402
from src.fl.simulation import run_federated                      # noqa: E402
from src.utils import console, reporting                         # noqa: E402

DEFAULT_EPS = 5.0


def plan(eps_grid, byz_fracs, aggregators, attack) -> list[tuple[str, str, dict]]:
    """(block, run name, RunConfig overrides) for the whole headline set."""
    runs: list[tuple[str, str, dict]] = []
    runs += [
        ("ablation", "ablation_fl_only", dict(aggregator="fedavg", dp=False)),
        ("ablation", "ablation_fl_dp",
         dict(aggregator="fedavg", dp=True, target_epsilon=DEFAULT_EPS)),
        ("ablation", "ablation_fl_krum", dict(aggregator="multikrum", dp=False)),
        ("ablation", "ablation_fl_dp_krum",
         dict(aggregator="multikrum", dp=True, target_epsilon=DEFAULT_EPS)),
    ]
    for e in eps_grid:
        runs.append(("privacy_utility", f"pu_eps{e:g}",
                     dict(aggregator="multikrum", dp=True, target_epsilon=float(e))))
    for agg in aggregators:
        for f in byz_fracs:
            runs.append(("byzantine", f"byz_{agg}_f{int(f * 100)}",
                         dict(aggregator=agg, dp=False, attack=attack,
                              byzantine_fraction=float(f))))
    # RO2 — the same total budget spent in three different shapes.
    runs += [
        ("adaptive_eps", "eps_static",
         dict(aggregator="multikrum", dp=True, target_epsilon=DEFAULT_EPS,
              adaptive_eps=False)),
        ("adaptive_eps", "eps_decreasing",
         dict(aggregator="multikrum", dp=True, target_epsilon=DEFAULT_EPS,
              adaptive_eps=True, eps_schedule="decreasing_noise")),
        ("adaptive_eps", "eps_increasing",
         dict(aggregator="multikrum", dp=True, target_epsilon=DEFAULT_EPS,
              adaptive_eps=True, eps_schedule="increasing_noise")),
    ]
    # RO4b — the interaction question: privacy noise and an attack at once, so
    # Multi-Krum's exclusion can be compared against the no-DP run above.
    runs.append(("interaction", "byz_multikrum_dp_f30",
                 dict(aggregator="multikrum", dp=True, target_epsilon=DEFAULT_EPS,
                      attack=attack, byzantine_fraction=0.3)))
    return runs


def _key(cfg: RunConfig) -> str:
    d = {k: v for k, v in cfg.to_dict().items() if k != "name"}
    return json.dumps(d, sort_keys=True, default=str)


def run_campaign(*, data_root: str = "data_full", out: str | None = None,
                 rounds: int = 30, model: str = "mlp", local_epochs: int = 1,
                 batch_size: int = 64, device: str = "cpu",
                 optimizer: str = "sgd", lr: float | None = None,
                 seed: int = 42, dp_calibrate: str = "worst",
                 round_chunks: bool = True, blocks: list[str] | None = None,
                 attack: str = "sign_flip",
                 eps_grid=(1.0, 5.0, 10.0, 50.0), byz_fracs=(0.0, 0.3),
                 aggregators=("fedavg", "krum", "multikrum", "trimmedmean"),
                 dry_run: bool = False) -> int:
    """Run the campaign. Importable so ``main.py`` can be the only entry point."""
    args = argparse.Namespace(
        data_root=data_root, out=out, rounds=rounds, model=model,
        local_epochs=local_epochs, batch_size=batch_size, device=device,
        optimizer=optimizer, lr=lr, seed=seed, dp_calibrate=dp_calibrate,
        round_chunks=round_chunks, blocks=blocks, attack=attack,
        eps_grid=list(eps_grid), byz_fracs=list(byz_fracs),
        aggregators=list(aggregators), dry_run=dry_run)

    root = Path(args.data_root)
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    outdir = Path(args.out) if args.out else PROJECT_ROOT / "results" / f"{root.name}_scale"
    if not outdir.is_absolute():
        outdir = PROJECT_ROOT / outdir
    outdir.mkdir(parents=True, exist_ok=True)

    lr = args.lr if args.lr is not None else (0.1 if args.optimizer == "sgd" else 1e-3)
    base = dict(model=args.model, rounds=args.rounds, local_epochs=args.local_epochs,
                batch_size=args.batch_size, num_clients=10, device=args.device,
                seed=args.seed, optimizer=args.optimizer, lr=lr,
                data_root=str(root), round_chunks=args.round_chunks,
                dp_calibrate=args.dp_calibrate, eval_on_subset=True)

    runs = plan(args.eps_grid, args.byz_fracs, args.aggregators, args.attack)
    if args.blocks:
        runs = [r for r in runs if r[0] in args.blocks]

    console.banner(f"FULL-SCALE CAMPAIGN — {root.name}", color="magenta")
    console.kv({"dataset root": str(root), "output": str(outdir),
                "rounds": args.rounds, "model": args.model,
                "chunked rounds": args.round_chunks,
                "epsilon calibrated for": args.dp_calibrate + "-case client",
                "runs planned": len(runs)})

    if args.dry_run:
        console.table(["block", "run"], [[b, n] for b, n, _ in runs], aligns="<<")
        return 0

    memo: dict[str, dict] = {}
    rows: list[dict] = []
    t_start = time.time()
    done = skipped = failed = 0

    for i, (block, name, over) in enumerate(runs, start=1):
        path = outdir / f"{name}.json"
        cfg = RunConfig(name=name, **{**base, **over})

        if path.exists():
            try:
                res = json.loads(path.read_text(encoding="utf-8"))
                memo.setdefault(_key(cfg), res)
                rows.append(reporting.row_from_result(res))
                skipped += 1
                console.info(f"[{i}/{len(runs)}] {name}: already done, skipping")
                continue
            except Exception:
                console.warn(f"{path.name} unreadable — recomputing")

        k = _key(cfg)
        if k in memo:
            res = dict(memo[k])
            res["config"] = cfg.to_dict()
            reporting.save_run(res, outdir)
            rows.append(reporting.row_from_result(res))
            done += 1
            console.info(f"[{i}/{len(runs)}] {name}: identical to an earlier run, reused")
            continue

        console.ro({"ablation": 1, "adaptive_eps": 2, "privacy_utility": 3,
                    "byzantine": 4, "interaction": 4}[block],
                   f"[{i}/{len(runs)}] {name}")
        t0 = time.time()
        try:
            res = run_federated(cfg, verbose=True)
        except KeyboardInterrupt:
            console.warn("interrupted — finished runs are saved; rerun to resume")
            return 130
        except Exception as exc:
            failed += 1
            console.warn(f"{name} FAILED: {exc!r} — continuing")
            continue
        memo[k] = res
        reporting.save_run(res, outdir)
        rows.append(reporting.row_from_result(res))
        done += 1
        console.metric("accuracy", res["final_accuracy"])
        console.metric("macro-F1", res["final_macro_f1"])
        if res["config"]["dp"]:
            console.metric("epsilon (calibrated / worst client)",
                           f"{res['final_epsilon']:.2f} / "
                           f"{res.get('final_epsilon_worst_client', 0):.2f}")
        console.info(f"{name} took {(time.time() - t0) / 60:.1f} min "
                     f"({(time.time() - t_start) / 60:.0f} min elapsed)")

    # ── tables, figures, summary ──
    if rows:
        reporting.results_table(rows, outdir)
        reporting.write_summary(
            rows, outdir,
            notes=(f"Full-scale campaign on {root.name}. Chunked rounds="
                   f"{args.round_chunks} (each of the {args.rounds} rounds trains "
                   f"on a disjoint slice of every shard, so the run consumes the "
                   f"dataset exactly once). Epsilon calibrated for the "
                   f"{args.dp_calibrate}-case client. Per-round metrics use the "
                   f"stratified eval subset; final metrics use the complete "
                   f"held-out set."))
        # One figure per evaluation dimension, rebuilt from the run JSONs just
        # written — the same call anyone can repeat later without retraining.
        from scripts.make_figures import make_figures
        make_figures(outdir)

    console.banner(f"DONE — {done} run, {skipped} skipped, {failed} failed "
                   f"({(time.time() - t_start) / 60:.0f} min)",
                   color="green" if not failed else "yellow")
    console.kv({"artefacts": str(outdir)})
    return 1 if failed else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default="data_full",
                    help="dataset root built by src.data.build_scaled")
    ap.add_argument("--out", default=None,
                    help="output dir (default results/<data-root>_scale)")
    ap.add_argument("--rounds", type=int, default=30)
    ap.add_argument("--model", default="mlp", choices=["mlp", "lstm", "transformer"])
    ap.add_argument("--local-epochs", type=int, default=1, dest="local_epochs")
    ap.add_argument("--batch-size", type=int, default=64, dest="batch_size")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])
    ap.add_argument("--optimizer", default="sgd", choices=["sgd", "adam"])
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dp-calibrate", default="worst", choices=["mean", "worst"],
                    dest="dp_calibrate",
                    help="calibrate epsilon for the worst-case (smallest) shard "
                         "so the stated guarantee holds for every gNB")
    ap.add_argument("--no-chunks", action="store_false", dest="round_chunks",
                    help="train on the whole shard each round instead of one "
                         "disjoint slice (very slow at full scale)")
    ap.add_argument("--blocks", nargs="*", default=None,
                    choices=["ablation", "privacy_utility", "byzantine",
                             "adaptive_eps", "interaction"])
    ap.add_argument("--attack", default="sign_flip")
    ap.add_argument("--eps-grid", nargs="*", type=float, default=[1.0, 5.0, 10.0, 50.0])
    ap.add_argument("--byz-fracs", nargs="*", type=float, default=[0.0, 0.3])
    ap.add_argument("--aggregators", nargs="*",
                    default=["fedavg", "krum", "multikrum", "trimmedmean"])
    ap.add_argument("--dry-run", action="store_true",
                    help="list what would run, then exit")
    return run_campaign(**vars(ap.parse_args()))


if __name__ == "__main__":
    sys.exit(main())
