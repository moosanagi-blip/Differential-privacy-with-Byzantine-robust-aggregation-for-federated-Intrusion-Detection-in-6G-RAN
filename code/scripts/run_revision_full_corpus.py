#!/usr/bin/env python3
"""Full-corpus, multi-seed evidence for the reviewer-requested revisions.

The script deliberately focuses on the three new reviewer experiments rather
than re-running the whole original campaign:

* adaptive/rotating white-box poisoning against Multi-Krum;
* sensitivity to the deployed estimate of the Byzantine client count ``f``;
* ablation of the seven synthetic RAN features.

Each cell is run on every supplied seed.  A completed seed/cell JSON is kept,
so an interrupted campaign resumes without discarding completed evidence.

Example (three independent full-corpus replications):

    python scripts/run_revision_full_corpus.py --seeds 42 43 44 --rounds 30
"""
from __future__ import annotations

import argparse
import json
import statistics as stats
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CODE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = CODE_ROOT.parent
# Keep publication artefacts next to the LaTeX figure sources as well as in
# the reproducibility/results directory.
PAPER_FIGURES = PROJECT_ROOT.parent / "paper" / "figures" / "revision_full_corpus"
sys.path.insert(0, str(CODE_ROOT))

from src.experiment_config import RunConfig
from src.fl.simulation import run_federated
from src.utils import reporting
from audit_revision_numerics import issues_for_run


def experiment_grid() -> list[dict]:
    """The nine cells needed for the reviewer response."""
    common_attack = dict(aggregator="multikrum", dp=False,
                         attack="adaptive_krum", byzantine_fraction=0.30,
                         attack_scale=10.0)
    return [
        dict(group="adaptive_attack", name="adaptive_static", **common_attack),
        dict(group="adaptive_attack", name="adaptive_rotating", rotate_byzantine=True,
             **common_attack),
        *[dict(group="f_sensitivity", name=f"f_estimate_{f}",
               n_byzantine_estimate=f, **common_attack) for f in range(5)],
        dict(group="ran_ablation", name="ran_features_retained", aggregator="multikrum", dp=False),
        dict(group="ran_ablation", name="ran_features_masked", aggregator="multikrum", dp=False,
             mask_ran_features=True),
    ]


def _mean_std(values: list[float]) -> tuple[float, float]:
    return float(stats.mean(values)), float(stats.stdev(values)) if len(values) > 1 else 0.0


def run(args: argparse.Namespace) -> list[dict]:
    out = Path(args.out)
    runs_dir = out / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    base = dict(model=args.model, hidden=64, rounds=args.rounds, local_epochs=1,
                batch_size=64, num_clients=10, device=args.device, optimizer="sgd",
                lr=0.1, data_root=args.data_root, round_chunks=True,
                eval_on_subset=True, dp_calibrate="worst")
    grid = experiment_grid()
    total = len(grid) * len(args.seeds)
    completed = 0
    started = time.monotonic()

    for seed in args.seeds:
        for cell in grid:
            filename = f"{cell['name']}__seed{seed}.json"
            path = runs_dir / filename
            if path.exists():
                completed += 1
                print(f"[{completed}/{total}] {filename}: already completed")
                continue
            cfg = RunConfig(name=cell["name"], seed=seed,
                            **base, **{k: v for k, v in cell.items() if k not in ("group", "name")})
            print(f"[{completed + 1}/{total}] {cell['group']}/{cell['name']} seed={seed}", flush=True)
            result = run_federated(cfg, verbose=True)
            result["revision_evidence"] = {
                "group": cell["group"], "seed": seed,
                "corpus": "data_full",
                "rounds": args.rounds,
                "completed_utc": datetime.now(timezone.utc).isoformat(),
            }
            reporting.save_run(result, runs_dir)
            # reporting names from RunConfig; normalize to an explicit seed filename.
            generated = runs_dir / f"{cell['name']}.json"
            if generated.exists() and generated != path:
                generated.replace(path)
            completed += 1
            elapsed = (time.monotonic() - started) / 60
            print(f"  completed in {elapsed:.1f} min elapsed", flush=True)
    return summarise(out, args.seeds)


def summarise(out: Path, seeds: list[int]) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for path in sorted((out / "runs").glob("*.json")):
        item = json.loads(path.read_text(encoding="utf-8"))
        meta = item.get("revision_evidence", {})
        if meta.get("seed") not in seeds:
            continue
        grouped.setdefault(item["config"]["name"], []).append(item)

    expected = {c["name"] for c in experiment_grid()}
    missing = expected - set(grouped)
    if missing:
        raise RuntimeError(f"Cannot aggregate: missing completed cells {sorted(missing)}")
    rows: list[dict] = []
    for cell in experiment_grid():
        name = cell["name"]
        items = grouped[name]
        if len(items) != len(seeds):
            raise RuntimeError(f"Cannot aggregate {name}: expected {len(seeds)} seeds, found {len(items)}")
        # Older saved top-level aliases contain subset metrics. Always use
        # the independently computed complete held-out evaluation.
        for x in items:
            if issues_for_run(x):
                raise ValueError(f"Numerically invalid run {name}, seed {x['revision_evidence']['seed']}; see numerical integrity audit")
            if x["dataset"]["eval_rows"] != 4673255:
                raise ValueError("Unexpected held-out corpus size")
            if any(not np.isfinite(x["final"][k]) for k in ("accuracy", "macro_f1", "fpr")):
                raise ValueError("Non-finite full-test metric")
        acc_m, acc_s = _mean_std([x["final"]["accuracy"] for x in items])
        f1_m, f1_s = _mean_std([x["final"]["macro_f1"] for x in items])
        fpr_m, fpr_s = _mean_std([x["final"]["fpr"] for x in items])
        admitted, possible = [], []
        for x in items:
            timeline = x.get("timeline", [])
            admitted.append(sum(len(t.get("byz_selected", [])) for t in timeline))
            possible.append(sum(len(t.get("byzantine_clients", [])) for t in timeline))
        adm_m, adm_s = _mean_std(admitted)
        poss_m, _ = _mean_std(possible)
        rows.append(dict(name=name, group=cell["group"], n_seeds=len(items), seeds=seeds,
                         metric_source="final: complete held-out evaluation", eval_rows=4673255,
                         accuracy_mean=acc_m, accuracy_std=acc_s,
                         macro_f1_mean=f1_m, macro_f1_std=f1_s,
                         fpr_mean=fpr_m, fpr_std=fpr_s,
                         malicious_admitted_mean=adm_m, malicious_admitted_std=adm_s,
                         malicious_possible_mean=poss_m,
                         n_byzantine_estimate=cell.get("n_byzantine_estimate"),
                         rotate_byzantine=cell.get("rotate_byzantine", False),
                         mask_ran_features=cell.get("mask_ran_features", False)))
    (out / "full_corpus_summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    write_table(rows, out)
    export_figures(rows, out)
    return rows


def write_table(rows: list[dict], out: Path) -> None:
    lines = ["# Full-corpus reviewer evidence", "",
             "All values are mean ± sample standard deviation across independent seeds.", "",
             "| Experiment | Accuracy | Macro-F1 | FPR | Malicious updates admitted |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for r in rows:
        lines.append(f"| {r['name']} | {r['accuracy_mean']:.4f} ± {r['accuracy_std']:.4f} | "
                     f"{r['macro_f1_mean']:.4f} ± {r['macro_f1_std']:.4f} | "
                     f"{r['fpr_mean']:.4f} ± {r['fpr_std']:.4f} | "
                     f"{r['malicious_admitted_mean']:.1f} ± {r['malicious_admitted_std']:.1f} / "
                     f"{r['malicious_possible_mean']:.1f} |")
    (out / "FULL_CORPUS_EVIDENCE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _save(fig, out: Path, stem: str) -> None:
    figures = out / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    PAPER_FIGURES.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(figures / f"{stem}.png", dpi=400, bbox_inches="tight")
    fig.savefig(figures / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(PAPER_FIGURES / f"{stem}.png", dpi=400, bbox_inches="tight")
    fig.savefig(PAPER_FIGURES / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def export_figures(rows: list[dict], out: Path) -> None:
    # Adaptive attacker admission (use a rate, so this is valid across round counts).
    attacks = [r for r in rows if r["group"] == "adaptive_attack"]
    labels = ["Static\ncoalition", "Rotating\ncoalition"]
    rates = [r["malicious_admitted_mean"] / r["malicious_possible_mean"] for r in attacks]
    rate_err = [r["malicious_admitted_std"] / r["malicious_possible_mean"] for r in attacks]
    fig, ax = plt.subplots(figsize=(6.5, 4.1))
    bars = ax.bar(labels, rates, yerr=rate_err, capsize=5, color=["#b2182b", "#ef8a62"], width=.58)
    ax.set_ylim(0, 1.08); ax.set_ylabel("Malicious-update admission rate")
    ax.set_title("Adaptive poisoning against Multi-Krum")
    ax.grid(axis="y", alpha=.25)
    for bar, rate, err in zip(bars, rates, rate_err):
        # Put the annotation above its uncertainty bar, not over it.
        ax.text(bar.get_x() + bar.get_width()/2, min(rate + err + .025, 1.045),
                f"{rate:.1%}", ha="center", weight="bold")
    _save(fig, out, "fig_full_adaptive_admission")

    fs = sorted([r for r in rows if r["group"] == "f_sensitivity"], key=lambda r: r["n_byzantine_estimate"])
    x = np.array([r["n_byzantine_estimate"] for r in fs])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.4, 3.8))
    ax1.errorbar(x, [r["accuracy_mean"] for r in fs], yerr=[r["accuracy_std"] for r in fs],
                 marker="o", capsize=4, color="#2166ac", linewidth=2)
    ax1.set(xlabel="Configured attacker count $f$", ylabel="Final accuracy", ylim=(0, 1), xticks=x)
    ax1.set_title("Utility"); ax1.grid(alpha=.25)
    rates = [r["malicious_admitted_mean"] / r["malicious_possible_mean"] for r in fs]
    errs = [r["malicious_admitted_std"] / r["malicious_possible_mean"] for r in fs]
    ax2.errorbar(x, rates, yerr=errs, marker="o", capsize=4, color="#b2182b", linewidth=2)
    ax2.set(xlabel="Configured attacker count $f$", ylabel="Malicious-update admission rate", ylim=(0, 1.08), xticks=x)
    ax2.set_title("Attack admission"); ax2.grid(alpha=.25)
    fig.suptitle("Sensitivity to estimated attacker count", y=1.03)
    _save(fig, out, "fig_full_f_sensitivity")

    ablation = [r for r in rows if r["group"] == "ran_ablation"]
    labels = ["All features", "Synthetic RAN\nfeatures masked"]
    metrics = [("Accuracy", "accuracy_mean", "accuracy_std"), ("Macro-F1", "macro_f1_mean", "macro_f1_std"),
               ("False-positive rate", "fpr_mean", "fpr_std")]
    fig, axes = plt.subplots(1, 3, figsize=(10.3, 3.7))
    for ax, (title, mean, std) in zip(axes, metrics):
        vals = [r[mean] for r in ablation]; errs = [r[std] for r in ablation]
        bars = ax.bar(labels, vals, yerr=errs, capsize=4, color=["#2166ac", "#b2182b"], width=.62)
        ax.set_title(title); ax.set_ylim(0, 1); ax.grid(axis="y", alpha=.25)
        for bar, value, err in zip(bars, vals, errs):
            ax.text(bar.get_x()+bar.get_width()/2, min(value + err + .025, .97),
                    f"{value:.3f}", ha="center", fontsize=9)
    fig.suptitle("Dependence on synthetic radio features", y=1.04)
    _save(fig, out, "fig_full_radio_ablation")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--model", default="mlp", choices=["mlp", "lstm", "transformer"])
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--data-root", default="data_full")
    parser.add_argument("--out", default=str(PROJECT_ROOT / "results" / "revision_full_corpus"))
    args = parser.parse_args()
    run(args)
    print(f"Full-corpus evidence and figures written to {args.out}")


if __name__ == "__main__":
    main()
