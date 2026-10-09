#!/usr/bin/env python3
"""
DP-FL-IDS experiment orchestrator — runs the RO2/RO3/RO4 campaign and writes
metrics, figures, and a summary that feed the thesis results chapter.

Blocks (each maps to Research Objectives):
  * ablation          — RO1/RO3: centralized, FL, FL+DP, FL+Krum, FL+DP+Krum
  * privacy_utility   — RO3: FL+DP+Multi-Krum across ε ∈ {1,5,10,50}
  * byzantine         — RO4: {FedAvg, Multi-Krum, Trimmed-Mean} × f ∈ {0,10,20,30%}
  * adaptive_eps      — RO2: static vs decreasing vs increasing noise at fixed ε
  * interaction       — RO4: DP × Multi-Krum exclusion + score-separation study

Usage:
  python run_experiment.py --all                 # full matrix
  python run_experiment.py --smoke               # fast reduced matrix
  python run_experiment.py --blocks privacy_utility byzantine
  python run_experiment.py --model lstm --rounds 15
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

# Opacus emits two benign UserWarnings on every private step (secure-RNG off for
# speed; RDP alpha-range note). We calibrate ε with a wide alpha grid already, so
# silence the noise to keep the RO-keyed progress log readable.
warnings.filterwarnings("ignore", message=".*Secure RNG turned off.*")
warnings.filterwarnings("ignore", message=".*Optimal order is the.*alpha.*")

from src.data.config import PROJECT_ROOT
from src.experiment_config import RunConfig
from src.fl.simulation import run_federated
from src.utils import console, reporting

RESULTS_DIR = PROJECT_ROOT / "results"
DEFAULT_EPSILON = 5.0   # nominal total privacy budget for DP blocks
DEFAULT_DATA_ROOT = "data_full"   # the complete 46.7M-flow corpus


# ── one grid of RunConfigs -> results + summary rows ───────────────────────
def _run_all(cfgs: list[RunConfig], outdir: Path) -> tuple[list[dict], list[dict]]:
    results, rows = [], []
    for i, cfg in enumerate(cfgs, 1):
        with console.step(f"[{i}/{len(cfgs)}] {cfg.name}"):
            res = run_federated(cfg, verbose=True)
        reporting.save_run(res, outdir)
        results.append(res)
        rows.append(reporting.row_from_result(res))
        console.metric("accuracy", res["final_accuracy"])
        console.metric("macro_f1", res["final_macro_f1"])
        if cfg.dp:
            console.metric("epsilon", res["final_epsilon"])
    return results, rows


def block_ablation(base: dict, outdir: Path) -> list[dict]:
    console.ro(1, "Framework ablation — DP + Multi-Krum components")
    eps = base.get("target_epsilon", DEFAULT_EPSILON)
    cfgs = [
        RunConfig(name="ablation_fl_only", aggregator="fedavg", dp=False, **base),
        RunConfig(name="ablation_fl_dp", aggregator="fedavg", dp=True, target_epsilon=eps, **base),
        RunConfig(name="ablation_fl_krum", aggregator="multikrum", dp=False, **base),
        RunConfig(name="ablation_fl_dp_krum", aggregator="multikrum", dp=True, target_epsilon=eps, **base),
    ]
    _, rows = _run_all(cfgs, outdir)
    return rows


def block_privacy_utility(base: dict, outdir: Path, eps_grid) -> list[dict]:
    console.ro(3, "Privacy–utility characterisation across ε")
    cfgs = [RunConfig(name=f"pu_eps{e}", aggregator="multikrum", dp=True,
                      target_epsilon=e, **base) for e in eps_grid]
    _, rows = _run_all(cfgs, outdir)
    return rows


def block_byzantine(base: dict, outdir: Path, fracs, attack: str) -> list[dict]:
    console.ro(4, f"Byzantine robustness under '{attack}' attack")
    cfgs = []
    for agg in ("fedavg", "multikrum", "trimmedmean"):
        for f in fracs:
            cfgs.append(RunConfig(name=f"byz_{agg}_f{int(f*100)}", aggregator=agg,
                                  attack=(attack if f > 0 else "none"),
                                  byzantine_fraction=f, dp=False, **base))
    _, rows = _run_all(cfgs, outdir)
    return rows


def block_adaptive_eps(base: dict, outdir: Path) -> list[dict]:
    console.ro(2, "Adaptive ε scheduling vs static (fixed total budget)")
    eps = base.get("target_epsilon", DEFAULT_EPSILON)
    b = {k: v for k, v in base.items() if k != "target_epsilon"}
    cfgs = [
        RunConfig(name="eps_static", aggregator="multikrum", dp=True,
                  target_epsilon=eps, adaptive_eps=False, **b),
        RunConfig(name="eps_decreasing", aggregator="multikrum", dp=True,
                  target_epsilon=eps, adaptive_eps=True, eps_schedule="decreasing_noise", **b),
        RunConfig(name="eps_increasing", aggregator="multikrum", dp=True,
                  target_epsilon=eps, adaptive_eps=True, eps_schedule="increasing_noise", **b),
    ]
    _, rows = _run_all(cfgs, outdir)
    return rows


def block_interaction(outdir: Path, attack: str, rounds: int,
                      base: dict | None = None) -> list[dict]:
    console.ro(4, "DP-noise × Multi-Krum interaction")
    from src.analysis import interaction
    base = base or {}
    with console.step("matched no-DP vs DP Multi-Krum runs"):
        comp = interaction.run_interaction(target_epsilon=5.0, attack=attack,
                                           byz_fraction=0.3, rounds=rounds,
                                           model=base.get("model", "mlp"),
                                           optimizer=base.get("optimizer", "sgd"),
                                           lr=base.get("lr"),
                                           device=base.get("device", "cpu"),
                                           seed=base.get("seed", 42),
                                           data_root=base.get("data_root"),
                                           round_chunks=base.get("round_chunks"))
    console.metric("exclusion no-DP", comp["no_dp"]["exclusion_rate"])
    console.metric("exclusion DP", comp["dp"]["exclusion_rate"])
    with console.step("Krum score-separation vs DP noise σ"):
        sep = interaction.score_separation(attack=attack, byz_fraction=0.3,
                                           data_root=base.get("data_root"))
    (outdir / "interaction.json").write_text(json.dumps({"comparison": comp, "separation": sep},
                                                        indent=2, default=float))
    # plot separation vs sigma
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(7, 4.2))
    plt.plot(sep["sigmas"], sep["separation"], "o-", label="Byz/honest score gap")
    plt.plot(sep["sigmas"], sep["exclusion"], "s--", label="Byz exclusion rate")
    plt.xlabel("DP-scale noise σ added to updates"); plt.ylabel("Value")
    plt.title("Does DP noise impair Multi-Krum?"); plt.grid(alpha=0.3); plt.legend()
    plt.tight_layout(); plt.savefig(outdir / "fig_interaction.png", dpi=150); plt.close()
    return [{"study": "interaction", **{f"no_dp_{k}": v for k, v in comp["no_dp"].items()},
             **{f"dp_{k}": v for k, v in comp["dp"].items()}}]


def main() -> None:
    ap = argparse.ArgumentParser(description="DP-FL-IDS experiment runner")
    ap.add_argument("--all", action="store_true", help="run every block (full matrix)")
    ap.add_argument("--smoke", action="store_true", help="fast reduced matrix")
    ap.add_argument("--blocks", nargs="*", default=None,
                    help="subset: ablation privacy_utility byzantine adaptive_eps interaction")
    ap.add_argument("--model", default="mlp", choices=["mlp", "lstm", "transformer"])
    ap.add_argument("--rounds", type=int, default=None)
    ap.add_argument("--local-epochs", type=int, default=1, dest="local_epochs",
                    help="local training epochs per round (LSTM benefits from >1)")
    ap.add_argument("--attack", default="sign_flip")
    ap.add_argument("--attack-scale", type=float, default=10.0,
                    help="attack magnitude; for adaptive_krum this is the maximum boundary-search scale")
    ap.add_argument("--n-byzantine-estimate", type=int, default=None,
                    help="operator estimate supplied to Krum; default uses the true count")
    ap.add_argument("--rotate-byzantine", action="store_true",
                    help="redraw the same-sized compromised coalition every round")
    ap.add_argument("--mask-ran-features", action="store_true",
                    help="zero the seven standardized synthetic RAN features for the ablation")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
                    help="compute device; 'auto' picks CUDA if available")
    ap.add_argument("--engine", default="sequential", choices=["sequential", "flower"],
                    help="FL engine: sequential (reference) or flower (proposal fidelity)")
    ap.add_argument("--optimizer", default="sgd", choices=["adam", "sgd"],
                    help="local optimizer; SGD is the canonical DP-SGD optimizer (default)")
    ap.add_argument("--lr", type=float, default=None,
                    help="learning rate (default: 0.1 for SGD, 1e-3 for Adam)")
    ap.add_argument("--data-root", default=DEFAULT_DATA_ROOT, dest="data_root",
                    help=f"prepared dataset root (default: {DEFAULT_DATA_ROOT}, "
                         "the complete corpus; build it with "
                         "`python main.py --build-data`)")
    ap.add_argument("--out", default=str(RESULTS_DIR))
    args = ap.parse_args()

    smoke = args.smoke
    rounds = args.rounds or (5 if smoke else 15)
    # NOTE: target_epsilon is intentionally NOT in `base` — blocks pass it
    # explicitly (via DEFAULT_EPSILON) so it never collides as a duplicate kwarg.
    lr = args.lr if args.lr is not None else (0.1 if args.optimizer == "sgd" else 1e-3)
    # Chunked rounds: each shard is cut into `rounds` disjoint slices and round
    # r trains on slice r, so a campaign consumes the corpus exactly once
    # instead of re-reading 37M rows every round.
    base = dict(model=args.model, rounds=rounds, local_epochs=args.local_epochs,
                batch_size=64, num_clients=10, device=args.device, seed=42,
                engine=args.engine, optimizer=args.optimizer, lr=lr,
                data_root=args.data_root, round_chunks=True,
                attack_scale=args.attack_scale,
                n_byzantine_estimate=args.n_byzantine_estimate,
                rotate_byzantine=args.rotate_byzantine,
                mask_ran_features=args.mask_ran_features)
    eps_grid = [1, 10] if smoke else [1, 5, 10, 50]
    fracs = [0.0, 0.2] if smoke else [0.0, 0.1, 0.2, 0.3]

    blocks = args.blocks or (["ablation", "privacy_utility", "byzantine",
                              "adaptive_eps", "interaction"] if (args.all or smoke) else
                             ["ablation"])
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    console.banner(f"DP-FL-IDS experiment campaign  (model={args.model}, rounds={rounds}, "
                   f"dataset={args.data_root}, {'SMOKE' if smoke else 'FULL'})")

    all_rows: list[dict] = []
    if "ablation" in blocks:
        all_rows += block_ablation(base, outdir)
    if "privacy_utility" in blocks:
        all_rows += block_privacy_utility(base, outdir, eps_grid)
    if "byzantine" in blocks:
        all_rows += block_byzantine(base, outdir, fracs, args.attack)
    if "adaptive_eps" in blocks:
        all_rows += block_adaptive_eps(base, outdir)
    if "interaction" in blocks:
        all_rows += block_interaction(outdir, args.attack, rounds, base)

    reporting.results_table(all_rows, outdir)
    reporting.write_summary(all_rows, outdir,
                            notes=f"Model={args.model}, rounds={rounds}, "
                                  f"mode={'smoke' if smoke else 'full'}.")
    # Rebuild the complete figure set from the JSONs this campaign just wrote,
    # so every evaluation dimension is plotted, not only the blocks that were run.
    from scripts.make_figures import make_figures
    make_figures(outdir)
    console.banner("Done — results, figures, and SUMMARY.md written to " + str(outdir),
                   color="green")


if __name__ == "__main__":
    main()
