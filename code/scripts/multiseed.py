#!/usr/bin/env python3
"""
Multi-seed replication with mean ± std — statistical hardening for the defense.

Re-runs the key RO blocks (ablation, privacy-utility, Byzantine, adaptive-ε) over
several seeds and reports mean ± standard deviation per configuration, plus
error-bar figures. This answers the examiner question "are these differences
significant / did you repeat the runs?".

Run:  python scripts/multiseed.py                       # 5 seeds, MLP, DP-SGD
      python scripts/multiseed.py --seeds 42 43 44 --model mlp
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib                                       # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                          # noqa: E402

from src.data.config import PROJECT_ROOT                # noqa: E402
from src.experiment_config import RunConfig              # noqa: E402
from src.fl.simulation import run_federated              # noqa: E402


def config_grid(base: dict) -> list[dict]:
    """Flat list of the key experiment cells (each maps to an RO)."""
    g: list[dict] = []
    # RO1 ablation
    g += [dict(block="ablation", name="fl_only", aggregator="fedavg", dp=False),
          dict(block="ablation", name="fl_dp", aggregator="fedavg", dp=True, target_epsilon=5.0),
          dict(block="ablation", name="fl_krum", aggregator="multikrum", dp=False),
          dict(block="ablation", name="fl_dp_krum", aggregator="multikrum", dp=True, target_epsilon=5.0)]
    # RO3 privacy-utility
    for e in (1.0, 5.0, 10.0, 50.0):
        g.append(dict(block="privacy_utility", name=f"eps{int(e)}", aggregator="multikrum",
                      dp=True, target_epsilon=e))
    # RO4 Byzantine
    for agg in ("fedavg", "multikrum", "trimmedmean"):
        for f in (0.0, 0.1, 0.2, 0.3):
            g.append(dict(block="byzantine", name=f"{agg}_f{int(f*100)}", aggregator=agg, dp=False,
                          attack=("sign_flip" if f > 0 else "none"), byzantine_fraction=f))
    # RO2 adaptive-eps
    g += [dict(block="adaptive", name="static", aggregator="multikrum", dp=True,
               target_epsilon=5.0, adaptive_eps=False),
          dict(block="adaptive", name="decreasing", aggregator="multikrum", dp=True,
               target_epsilon=5.0, adaptive_eps=True, eps_schedule="decreasing_noise"),
          dict(block="adaptive", name="increasing", aggregator="multikrum", dp=True,
               target_epsilon=5.0, adaptive_eps=True, eps_schedule="increasing_noise")]
    return [{**base, **c} for c in g]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    ap.add_argument("--model", default="mlp")
    ap.add_argument("--rounds", type=int, default=30)
    ap.add_argument("--optimizer", default="sgd")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--data-root", default="data_full", dest="data_root",
                    help="prepared dataset root (default: data_full)")
    ap.add_argument("--out", default=str(PROJECT_ROOT / "results" / "multiseed"))
    args = ap.parse_args()
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    lr = 0.1 if args.optimizer == "sgd" else 1e-3
    base = dict(model=args.model, rounds=args.rounds, local_epochs=1, batch_size=64,
                num_clients=10, device=args.device, optimizer=args.optimizer, lr=lr,
                data_root=args.data_root, round_chunks=True)

    grid = config_grid(base)
    # acc[key] = list of accuracies across seeds
    agg: dict[str, dict] = {}
    for si, seed in enumerate(args.seeds, 1):
        print(f"\n=== seed {seed} ({si}/{len(args.seeds)}) ===")
        for cfg_kw in grid:
            key = f"{cfg_kw['block']}/{cfg_kw['name']}"
            cfg = RunConfig(name=key.replace("/", "_"), seed=seed,
                            **{k: v for k, v in cfg_kw.items() if k not in ("block", "name")})
            r = run_federated(cfg, verbose=False)
            d = agg.setdefault(key, {"block": cfg_kw["block"], "acc": [], "f1": [],
                                     "fpr": [], "eps": [],
                                     "aggregator": cfg_kw["aggregator"],
                                     "byz_fraction": cfg_kw.get("byzantine_fraction", 0.0),
                                     "target_epsilon": cfg_kw.get("target_epsilon")})
            d["acc"].append(r["final_accuracy"]); d["f1"].append(r["final_macro_f1"])
            d["fpr"].append(r["final_fpr"]); d["eps"].append(r["final_epsilon"])
            print(f"  {key:<26} acc={r['final_accuracy']:.3f}")

    def ms(xs):  # mean, std
        return (st.mean(xs), (st.stdev(xs) if len(xs) > 1 else 0.0))

    rows = []
    for key, d in agg.items():
        am, asd = ms(d["acc"]); fm, fsd = ms(d["f1"]); pm, psd = ms(d["fpr"])
        rows.append({"config": key, "block": d["block"], "aggregator": d["aggregator"],
                     "byz_fraction": d["byz_fraction"], "target_epsilon": d["target_epsilon"],
                     "acc_mean": round(am, 4), "acc_std": round(asd, 4),
                     "f1_mean": round(fm, 4), "f1_std": round(fsd, 4),
                     "fpr_mean": round(pm, 4), "fpr_std": round(psd, 4),
                     "n_seeds": len(d["acc"])})
    (outdir / "multiseed.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    cols = list(rows[0].keys())
    (outdir / "multiseed_table.csv").write_text(
        "\n".join([",".join(cols)] + [",".join(str(r[c]) for c in cols) for r in rows]))

    # markdown summary
    md = [f"# Multi-seed results (mean ± std over {len(args.seeds)} seeds: "
          f"{args.seeds}, {args.model}, DP-{args.optimizer.upper()})\n",
          "| config | acc mean±std | macro-F1 mean±std | FPR mean±std |",
          "| --- | --- | --- | --- |"]
    for r in rows:
        md.append(f"| {r['config']} | {r['acc_mean']:.3f}±{r['acc_std']:.3f} "
                  f"| {r['f1_mean']:.3f}±{r['f1_std']:.3f} | {r['fpr_mean']:.3f}±{r['fpr_std']:.3f} |")
    (outdir / "MULTISEED.md").write_text("\n".join(md), encoding="utf-8")

    # error-bar figures
    _fig_privacy_utility(rows, outdir / "fig_pu_errorbars.png")
    _fig_byzantine(rows, outdir / "fig_byz_errorbars.png")
    print(f"\nwrote MULTISEED.md, multiseed_table.csv, 2 figures -> {outdir}")


def _fig_privacy_utility(rows, path):
    pu = sorted([r for r in rows if r["block"] == "privacy_utility"],
                key=lambda r: r["target_epsilon"])
    if not pu:
        return
    eps = [r["target_epsilon"] for r in pu]
    plt.figure(figsize=(7, 4.5))
    plt.errorbar(eps, [r["acc_mean"] for r in pu], yerr=[r["acc_std"] for r in pu],
                 marker="o", capsize=4, label="Accuracy")
    plt.errorbar(eps, [r["f1_mean"] for r in pu], yerr=[r["f1_std"] for r in pu],
                 marker="s", capsize=4, label="Macro-F1")
    plt.xscale("log"); plt.xlabel("Privacy budget ε (log)"); plt.ylabel("Score")
    plt.title("Privacy–utility (mean ± std over seeds)"); plt.grid(alpha=0.3); plt.legend()
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def _fig_byzantine(rows, path):
    byz = [r for r in rows if r["block"] == "byzantine"]
    if not byz:
        return
    plt.figure(figsize=(7, 4.5))
    for agg in sorted({r["aggregator"] for r in byz}):
        pts = sorted([r for r in byz if r["aggregator"] == agg], key=lambda r: r["byz_fraction"])
        plt.errorbar([p["byz_fraction"] * 100 for p in pts], [p["acc_mean"] for p in pts],
                     yerr=[p["acc_std"] for p in pts], marker="o", capsize=4, label=agg)
    plt.xlabel("Byzantine clients (%)"); plt.ylabel("Global test accuracy")
    plt.title("Byzantine robustness (mean ± std over seeds)"); plt.grid(alpha=0.3); plt.legend()
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


if __name__ == "__main__":
    main()
