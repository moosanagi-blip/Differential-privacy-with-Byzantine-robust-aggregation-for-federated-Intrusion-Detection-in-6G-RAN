#!/usr/bin/env python3
"""
DP-FL-IDS — Privacy-Preserving Federated Learning for Intrusion Detection
             in 6G Radio Access Networks (Byzantine-Robust DP Framework)

Single entry point. Runs the complete implementation end to end on the COMPLETE
46.7M-flow CICIoT2023 corpus — the dataset the thesis reports — and explains
every step in plain English while it happens: the data pipeline, the model, one
federated round dissected, then all four Research Objectives in turn.

    python3 main.py                  complete narrated run (all 17 steps),
                                     pausing for Enter between each
    python3 main.py --no-pause       run straight through without waiting
    python3 main.py --replay         narrate saved results, train nothing
    python3 main.py --profile quick  fewer rounds, same coverage
    python3 main.py --steps 12-13    run only some steps
    python3 main.py --model lstm     use the LSTM instead of the MLP
    python3 main.py --build-data     just prepare the corpus (~15 min), then stop
    python3 main.py --campaign       the full 16-experiment campaign (~8 h,
                                     resumable — rerun to continue)

The corpus is prepared automatically on the first run if data_full/ is not on
disk yet. Training on all 46.7M flows takes hours rather than minutes; --replay
narrates the finished campaign in seconds and trains nothing.

Artefacts, figures and a full transcript are written to results/demo_run/;
campaign results are filed under results/Research Objective 1..4/.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Opacus emits two benign UserWarnings on every private step, and torch's CUDA
# probe complains on machines where the driver is fine but uvm is wedged. None
# of them mean anything here and all of them break the narration's flow.
warnings.filterwarnings("ignore", message=".*Secure RNG turned off.*")
warnings.filterwarnings("ignore", message=".*Optimal order is the.*alpha.*")
warnings.filterwarnings("ignore", message=".*CUDA initialization.*")


FULL_ROOT = "data_full"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="main.py",
        description="Narrated end-to-end demonstration of the DP-FL-IDS framework.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Artefacts and a transcript are written to results/demo_run/.")
    p.add_argument("--profile", default="demo", choices=["quick", "demo", "full"],
                   help="how much to compute: quick (5 rounds), demo (default, "
                        "10 rounds, complete coverage), full (30 rounds, "
                        "reproduces the thesis grid)")
    p.add_argument("--replay", action="store_true",
                   help="narrate the saved thesis results instead of training")
    p.add_argument("--steps", default=None, metavar="SPEC",
                   help="subset of steps, e.g. '12', '7-13' or '1,4,12'")
    # Pausing is the default: this is meant to be presented, and a wall of
    # output that scrolls past unattended is no use to anyone explaining it.
    g = p.add_mutually_exclusive_group()
    g.add_argument("--pause", dest="pause", action="store_true",
                   help="wait for Enter between steps (the default)")
    g.add_argument("--no-pause", dest="pause", action="store_false",
                   help="run straight through without waiting")
    p.set_defaults(pause=True)
    p.add_argument("--data-root", default=FULL_ROOT, dest="data_root",
                   help=f"prepared dataset root to run on (default: {FULL_ROOT}, "
                        "the complete 46.7M-flow corpus, prepared automatically "
                        "if it is not on disk yet)")
    # The complete corpus is now the default, so --full asks for what already
    # happens. Kept accepted, and silent, so the older documented command and
    # anything scripted against it keep working.
    p.add_argument("--full", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--build-data", action="store_true", dest="build_data",
                   help="prepare the complete corpus (~15 min) and exit")
    p.add_argument("--campaign", action="store_true",
                   help="run the 16-experiment campaign on the complete corpus "
                        "instead of the narrated demo (~8 h, resumable)")
    p.add_argument("--rebuild", action="store_true",
                   help="with --build-data, rebuild even if data_full/ exists")
    p.add_argument("--model", default="mlp", choices=["mlp", "lstm", "transformer"],
                   help="mlp (default) carries the quantitative results; "
                        "lstm is the thesis primary architecture")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])
    p.add_argument("--engine", default="sequential", choices=["sequential", "flower"])
    p.add_argument("--optimizer", default="sgd", choices=["sgd", "adam"])
    p.add_argument("--lr", type=float, default=None,
                   help="learning rate (default 0.1 for SGD, 1e-3 for Adam)")
    p.add_argument("--seed", type=int, default=42)
    return p


def ensure_corpus(rebuild: bool = False) -> str:
    """Prepare the complete corpus if it is not already on disk.

    Keeps `main.py` the only command anyone needs to type: the dataset build is
    a prerequisite, not a separate step to remember.
    """
    from src.data.build_scaled import build
    from src.data.config import PROJECT_ROOT
    from src.utils import console

    # Datasets live beside code/, not inside it.
    root = PROJECT_ROOT / FULL_ROOT
    if root.exists() and not rebuild:
        console.info(f"{FULL_ROOT}/ already prepared — reusing it "
                     "(pass --rebuild to build it again)")
        return FULL_ROOT

    console.banner("PREPARING THE COMPLETE CICIoT2023 CORPUS", color="magenta")
    console.say("""Streaming all 169 raw CSV files into scaled, non-IID gNB
        shards. Roughly 15 minutes and about 1 GB of memory; the dataset itself
        is never held in RAM. This happens once — later runs reuse the result.""")
    stats = build(out=root)
    console.kv({"rows processed": f"{stats['n_rows']:,}", "written to": str(root)})
    return FULL_ROOT


def main() -> int:
    args = build_parser().parse_args()

    if args.build_data:
        ensure_corpus(rebuild=args.rebuild)
        return 0

    if args.campaign:
        root = ensure_corpus()
        from scripts.full_scale import run_campaign
        return run_campaign(data_root=root, model=args.model, device=args.device,
                            optimizer=args.optimizer, lr=args.lr, seed=args.seed)

    # Replay reads finished result files, so it must not trigger a dataset build.
    if args.replay:
        from src.demo.replay import replay
        return replay(args)

    if args.data_root == FULL_ROOT:
        ensure_corpus()

    from src.demo.runner import run_demo
    return run_demo(args)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)
