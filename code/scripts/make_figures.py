#!/usr/bin/env python3
"""
Rebuild every figure from the saved per-run JSON files.

Figures used to be produced only *during* a campaign, which meant a new plot
cost a re-run — hours at full scale. Everything the thesis plots is already
stored in each run's JSON (per-round accuracy, macro-F1, loss, sigma, cumulative
epsilon, wall-clock, Multi-Krum scores, which updates were admitted, plus the
final per-category breakdown), so this script reads those files and writes the
complete figure set in seconds, without touching the dataset or the model.

**One measure per image.** A chart printed at full text width stays readable in
the PDF and can be lifted into a slide on its own; the same content packed into
a multi-panel strip shrinks every label until a reader has to zoom.

``results/`` is organised by Research Objective, and the figures follow the runs
into those folders:

    Research Objective 1/   ablation runs    accuracy · macro-F1 · FPR ·
                                             convergence in accuracy / macro-F1 / loss
    Research Objective 2/   eps_* schedules  noise schedule · privacy spend ·
                                             accuracy · macro-F1 per round
    Research Objective 3/   pu_eps* sweep    privacy–utility · FPR vs ε · σ vs ε ·
                                             convergence · per-family recall
    Research Objective 4/   byz_* runs       accuracy · macro-F1 · FPR ·
                                             updates admitted · score separation
    (results root)          every run        round cost, and cost per round

Run:  python scripts/make_figures.py results          # the whole RO layout
      python scripts/make_figures.py results/some_dir # a flat directory of runs
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.config import PROJECT_ROOT          # noqa: E402
from src.utils import console, reporting          # noqa: E402

RO_DIRS = {1: "Research Objective 1", 2: "Research Objective 2",
           3: "Research Objective 3", 4: "Research Objective 4"}
RUN_HOME = [("ablation", 1), ("eps_", 2), ("pu_eps", 3), ("byz_", 4)]
ABLATION_ORDER = ["ablation_fl_only", "ablation_fl_dp", "ablation_fl_krum",
                  "ablation_fl_dp_krum"]
SCHEDULE_ORDER = ["eps_static", "eps_decreasing", "eps_increasing"]


def home_of(run_name: str) -> int | None:
    """Which Research Objective a run belongs to, by its name."""
    for prefix, ro in RUN_HOME:
        if run_name.startswith(prefix):
            return ro
    return None


def load_runs(d: Path) -> list[dict]:
    """Every per-run result under a directory, including RO subfolders."""
    runs = []
    for p in sorted(d.glob("**/*.json")):
        try:
            r = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if isinstance(r, dict) and "config" in r and "timeline" in r:
            runs.append(r)
    return runs


def _named(runs: list[dict], prefix: str, order: list[str] | None = None) -> list[dict]:
    hits = [r for r in runs if r["config"]["name"].startswith(prefix)]
    if order:  # read as a ladder (each row adds a component), not alphabetically
        hits.sort(key=lambda r: (order.index(r["config"]["name"])
                                 if r["config"]["name"] in order else len(order)))
    return hits


def make_figures(d: Path) -> list[Path]:
    """Write every figure the runs under ``d`` support, into their RO folders."""
    runs = load_runs(d)
    names = [r["config"]["name"] for r in runs]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate run names across campaigns/seeds: select one explicit campaign before plotting")
    if not runs:
        console.warn(f"{d}: no per-run result JSONs found — nothing to plot")
        return []
    ro_layout = any((d / name).is_dir() for name in RO_DIRS.values())
    stamp = reporting.scale_stamp(runs, source="results" if ro_layout else f"results/{d.name}")
    rows = {r["config"]["name"]: reporting.row_from_result(r) for r in runs}
    written: list[Path] = []

    def out(fig: str, ro: int | None) -> Path:
        return d / RO_DIRS[ro] / fig if (ro_layout and ro) else d / fig

    def did(p: Path | None) -> None:
        if p and Path(p).exists():
            written.append(Path(p))

    # ── RO1 — what each component contributes, and how it converges ────────
    abl = _named(runs, "ablation", ABLATION_ORDER)
    if abl:
        did(reporting.plot_ablation_metric(
            abl, "accuracy", out("fig_ablation_accuracy.png", 1),
            title="RO1 — accuracy by configuration", ylabel="Global test accuracy",
            stamp=stamp, baseline_delta=True))
        did(reporting.plot_ablation_metric(
            abl, "macro_f1", out("fig_ablation_macro_f1.png", 1),
            title="RO1 — macro-F1 by configuration", ylabel="Macro-F1",
            stamp=stamp, baseline_delta=True))
        did(reporting.plot_ablation_metric(
            abl, "fpr", out("fig_ablation_fpr.png", 1),
            title="RO1 — false-alarm rate by configuration",
            ylabel="False-positive rate", stamp=stamp, lower_is_better=True))
        for key, fig, title, ylab, mark in [
            ("accuracy", "fig_convergence_accuracy.png", "Accuracy per round",
             "Global test accuracy", True),
            ("macro_f1", "fig_convergence_macro_f1.png", "Macro-F1 per round",
             "Macro-F1", False),
            ("loss", "fig_convergence_loss.png", "Training loss per round",
             "Cross-entropy loss", False)]:
            did(reporting.plot_runs_over_rounds(
                abl, key, out(fig, 1), title=title, ylabel=ylab, stamp=stamp,
                mark_convergence=mark))

    # ── RO3 — the ε sweep ──────────────────────────────────────────────────
    pu = _named(runs, "pu_eps")
    if len(pu) > 1:
        pu_rows = sorted((rows[r["config"]["name"]] for r in pu), key=lambda r: r["epsilon"])
        did(reporting.plot_privacy_utility(pu_rows, out("fig_privacy_utility.png", 3),
                                           results=pu, stamp=stamp))
        did(reporting.plot_vs_epsilon(
            pu_rows, "fpr", out("fig_fpr_vs_epsilon.png", 3),
            title="False-alarm rate against the privacy budget",
            ylabel="False-positive rate  (lower is better)", results=pu, stamp=stamp))
        did(reporting.plot_vs_epsilon(
            pu_rows, "sigma", out("fig_sigma_vs_epsilon.png", 3),
            title="Noise the guarantee requires", ylabel="Noise multiplier σ",
            colour=reporting.C["violet"], results=pu, stamp=stamp,
            annotate="σ={:.2f}"))
        eps_sorted = sorted(pu, key=lambda r: r.get("final_epsilon", 0))
        did(reporting.plot_runs_over_rounds(
            eps_sorted, "accuracy", out("fig_eps_convergence_accuracy.png", 3),
            title="Accuracy per round, at each privacy budget",
            ylabel="Global test accuracy", stamp=stamp, mark_convergence=True))
        did(reporting.plot_runs_over_rounds(
            eps_sorted, "macro_f1", out("fig_eps_convergence_macro_f1.png", 3),
            title="Macro-F1 per round, at each privacy budget", ylabel="Macro-F1",
            stamp=stamp))

    # ── RO2 — schedule shapes at a matched budget ──────────────────────────
    sched = _named(runs, "eps_", SCHEDULE_ORDER)
    spend = sched or pu or [r for r in runs if r["config"].get("dp")]
    if spend:
        did(reporting.plot_noise_schedule(spend, out("fig_noise_schedule.png", 2), stamp))
        did(reporting.plot_privacy_spend(spend, out("fig_privacy_spend.png", 2), stamp))
    if sched:
        did(reporting.plot_runs_over_rounds(
            sched, "accuracy", out("fig_schedule_accuracy.png", 2),
            title="Accuracy per round, by noise schedule",
            ylabel="Global test accuracy", stamp=stamp, mark_convergence=True))
        did(reporting.plot_runs_over_rounds(
            sched, "macro_f1", out("fig_schedule_macro_f1.png", 2),
            title="Macro-F1 per round, by noise schedule", ylabel="Macro-F1",
            stamp=stamp))

    # ── RO4 — robustness, and the geometry the defence rests on ────────────
    byz = _named(runs, "byz_")
    if byz:
        clean = [r for r in byz if not r["config"].get("dp")]
        byz_rows = [rows[r["config"]["name"]] for r in clean]
        for key, fig, title, ylab in [
            ("accuracy", "fig_byz_accuracy.png", "RO4 — accuracy under attack",
             "Global test accuracy"),
            ("macro_f1", "fig_byz_macro_f1.png", "RO4 — macro-F1 under attack",
             "Macro-F1"),
            ("fpr", "fig_byz_fpr.png", "RO4 — false-alarm rate under attack",
             "False-positive rate  (lower is better)")]:
            did(reporting.plot_byzantine_metric(byz_rows, key, out(fig, 4),
                                                title=title, ylabel=ylab, stamp=stamp))
        did(reporting.plot_byzantine_admitted(clean, out("fig_byz_admitted.png", 4), stamp))
        # the interaction test: the same attack with privacy off, then on
        heaviest = max((r["config"].get("byzantine_fraction") or 0) for r in byz)
        for r in byz:
            cfg = r["config"]
            if cfg["aggregator"] == "multikrum" and cfg.get("byzantine_fraction") == heaviest:
                name = ("fig_krum_scores_privacy_on.png" if cfg.get("dp")
                        else "fig_krum_scores_privacy_off.png")
                did(reporting.plot_krum_scores(r, out(name, 4), stamp))

    # ── cross-cutting: what a round costs ──────────────────────────────────
    did(reporting.plot_round_time(runs, out("fig_round_time.png", None), stamp))
    did(reporting.plot_round_time_series(runs, out("fig_round_time_per_round.png", None), stamp))

    # ── per-family detection: no attack, no DP, robust aggregator ──────────
    for name in ("ablation_fl_krum", "byz_multikrum_f0", "ablation_fl_only",
                 "ablation_fl_dp_krum"):
        hit = [r for r in runs if r["config"]["name"] == name]
        if hit and (hit[0].get("final") or {}).get("per_category"):
            did(reporting.plot_per_category(hit[0], out("fig_per_category.png", 3), stamp))
            break

    console.info(f"{d.name}: {len(written)} figures from {len(runs)} runs")
    for p in written:
        console.info(f"  {p.relative_to(d)}")
    return written


def main() -> int:
    ap = argparse.ArgumentParser(description="Rebuild figures from saved results")
    ap.add_argument("dirs", nargs="*", help="results directories (default: results/)")
    args = ap.parse_args()

    dirs = [Path(x) for x in args.dirs] or [Path("results")]
    dirs = [d if d.is_absolute() else (PROJECT_ROOT / d) for d in dirs]

    console.banner("REBUILDING FIGURES FROM SAVED RESULTS", color="magenta")
    total = 0
    for d in dirs:
        if not d.exists():
            console.warn(f"{d} does not exist — skipped")
            continue
        total += len(make_figures(d))
    console.banner(f"DONE — {total} figures written", color="green")
    return 0


if __name__ == "__main__":
    sys.exit(main())
