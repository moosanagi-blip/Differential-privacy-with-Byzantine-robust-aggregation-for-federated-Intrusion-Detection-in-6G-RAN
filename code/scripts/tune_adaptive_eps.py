#!/usr/bin/env python3
"""
RO2 bounded tuning sweep — does adaptive-ε scheduling beat static?

For each (target ε, rounds) cell we run the static baseline and every decreasing
schedule, all calibrated to the *same* total (ε, δ) via RDP bisection, then
compare final accuracy / macro-F1. The hypothesis is that decreasing-noise
schedules help most at small ε and many rounds (fragile late-round signal).

Writes a table to stdout + JSON to "results/Research Objective 2/schedule_sweep.json". This is analysis, not
part of the main matrix; it informs whether to change the default schedule.

Run:  python scripts/tune_adaptive_eps.py            # default bounded grid
      python scripts/tune_adaptive_eps.py --quick     # tiny grid for a smoke check
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # project root importable

from src.data.config import PROJECT_ROOT          # noqa: E402
from src.experiment_config import RunConfig        # noqa: E402
from src.fl.simulation import run_federated         # noqa: E402

SCHEDULES = ["decreasing_noise", "decreasing_steep", "cosine_decreasing",
             "step_decreasing", "increasing_noise"]


def one(model: str, eps: float, rounds: int, schedule: str | None) -> dict:
    """Run static (schedule=None) or an adaptive schedule at fixed total ε."""
    common = dict(model=model, rounds=rounds, aggregator="multikrum", dp=True,
                  target_epsilon=eps, local_epochs=1, batch_size=64, device="cpu",
                  seed=42)
    if schedule is None:
        cfg = RunConfig(name=f"static_e{eps}_r{rounds}", adaptive_eps=False, **common)
    else:
        cfg = RunConfig(name=f"{schedule}_e{eps}_r{rounds}", adaptive_eps=True,
                        eps_schedule=schedule, **common)
    r = run_federated(cfg, verbose=False)
    return {"acc": r["final_accuracy"], "f1": r["final_macro_f1"],
            "eps": r["final_epsilon"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlp", choices=["mlp", "lstm"])
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    if args.quick:
        eps_grid, rounds_grid = [1.0, 5.0], [20]
    else:
        eps_grid, rounds_grid = [1.0, 5.0, 10.0], [20, 40]

    results: list[dict] = []
    print(f"\nRO2 adaptive-ε tuning  (model={args.model})")
    print(f"{'eps':>5} {'rnds':>5} {'schedule':>18} {'ε_got':>7} "
          f"{'acc':>7} {'f1':>7} {'Δacc_vs_static':>15}")
    for eps in eps_grid:
        for rounds in rounds_grid:
            base = one(args.model, eps, rounds, None)
            row = {"eps": eps, "rounds": rounds, "schedule": "static", **base,
                   "delta_acc": 0.0}
            results.append(row)
            # tolerance check: achieved ε within 2% of target
            ok = abs(base["eps"] - eps) <= 0.02 * eps + 1e-6
            print(f"{eps:>5} {rounds:>5} {'static':>18} {base['eps']:>7.3f} "
                  f"{base['acc']:>7.3f} {base['f1']:>7.3f} {'—':>15}"
                  + ("" if ok else "  !!ε off target"))
            for sched in SCHEDULES:
                r = one(args.model, eps, rounds, sched)
                d = r["acc"] - base["acc"]
                results.append({"eps": eps, "rounds": rounds, "schedule": sched,
                                **r, "delta_acc": d})
                flag = "  <== beats static" if d > 0.005 else ""
                print(f"{eps:>5} {rounds:>5} {sched:>18} {r['eps']:>7.3f} "
                      f"{r['acc']:>7.3f} {r['f1']:>7.3f} {d:>+15.3f}{flag}")

    out = PROJECT_ROOT / "results" / "Research Objective 2" / "schedule_sweep.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")

    # verdict: best schedule averaged over all cells
    from collections import defaultdict
    agg = defaultdict(list)
    for r in results:
        if r["schedule"] != "static":
            agg[r["schedule"]].append(r["delta_acc"])
    print("\nMean Δacc vs static (higher = better):")
    best, best_val = None, -1e9
    for sched, ds in sorted(agg.items(), key=lambda kv: -sum(kv[1]) / len(kv[1])):
        m = sum(ds) / len(ds)
        print(f"  {sched:>18}: {m:+.4f}")
        if m > best_val:
            best, best_val = sched, m
    verdict = (f"WINNER: '{best}' (+{best_val:.4f} mean acc)" if best_val > 0.005
               else f"NO ADAPTIVE GAIN — best '{best}' only {best_val:+.4f}; "
                    "report adaptive ≈ static (honest negative result)")
    print(f"\n{verdict}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
