"""
Persistence + plotting + summary reporting for experiment results.

Turns the result dicts produced by the FL engine into the artefacts the thesis
needs: per-run JSON + timeline CSV, one figure per evaluation dimension promised
in the proposal defence (detection quality, privacy spend, Byzantine robustness,
convergence speed, ablation contribution, compute overhead), a machine-readable
``results_table.csv``, and a human-readable ``SUMMARY.md``. All figures are
written with a non-interactive matplotlib backend so it works head-less.

Every plot here reads only what is already stored in a finished run, so
``scripts/make_figures.py`` can rebuild the whole figure set from saved JSON
without retraining anything.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


# ── house style ────────────────────────────────────────────────────────────
# Fixed categorical palette (validated for colour-vision deficiency). Hues are
# assigned to *entities* — an aggregator keeps its colour in every figure it
# appears in — never to a series' rank within one chart.
C = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100",
     "magenta": "#e87ba4", "green": "#008300", "violet": "#4a3aa7", "red": "#e34948"}
INK, INK2, GRIDC = "#0b0b0b", "#52514e", "#d8d7d2"

AGG_COLOR = {"fedavg": C["orange"], "multikrum": C["blue"],
             "trimmedmean": C["aqua"], "krum": C["violet"]}
AGG_LABEL = {"fedavg": "FedAvg", "multikrum": "Multi-Krum",
             "trimmedmean": "Trimmed-mean", "krum": "Krum"}
RUN_COLOR = {"ablation_fl_only": C["orange"], "ablation_fl_dp": C["yellow"],
             "ablation_fl_krum": C["blue"], "ablation_fl_dp_krum": C["violet"],
             "eps_static": C["blue"], "eps_decreasing": C["orange"],
             "eps_increasing": C["aqua"]}
RUN_LABEL = {"ablation_fl_only": "FL only", "ablation_fl_dp": "FL + DP",
             "ablation_fl_krum": "FL + Multi-Krum",
             "ablation_fl_dp_krum": "FL + DP + Multi-Krum",
             "eps_static": "static (uniform noise)",
             "eps_decreasing": "decreasing noise", "eps_increasing": "increasing noise"}
EPS_COLORS = [C["blue"], C["orange"], C["aqua"], C["violet"], C["magenta"]]

_RC = {
    "figure.dpi": 200, "savefig.dpi": 200, "figure.facecolor": "white",
    "savefig.facecolor": "white", "font.size": 9,
    "axes.titlesize": 10, "axes.labelsize": 9, "axes.labelcolor": INK2,
    "axes.edgecolor": GRIDC, "axes.titlecolor": INK, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRIDC,
    "grid.linewidth": 0.6, "grid.alpha": 0.7, "xtick.color": INK2,
    "ytick.color": INK2, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "legend.frameon": False, "lines.linewidth": 2.0,
    "lines.markersize": 5,
}


def _label(name: str) -> str:
    return RUN_LABEL.get(name, name)


def _run_color(name: str, fallback: str = C["blue"]) -> str:
    return RUN_COLOR.get(name, fallback)


# These figures are read in the PDF, where anything wider than the text block is
# scaled down and its labels shrink with it. Sizing panels so the whole figure
# lands within one text width keeps type at its designed size on the page.
PAGE_W = 7.2  # inches of text width at the document's 0.85in margins


def _fig(ncols: int = 1, nrows: int = 1, w: float = 5.2, h: float = 3.8):
    plt.rcParams.update(_RC)
    # Narrow the panels to fit the page; keep their height, since squashing that
    # too is what makes legends and titles collide.
    w = min(w, PAGE_W / ncols)
    fig, ax = plt.subplots(nrows, ncols, figsize=(w * ncols, h * nrows))
    return fig, (np.atleast_1d(ax).ravel() if (ncols * nrows) > 1 else ax)


def _save(fig, outpath: Path, stamp: str = "", title: str | None = None) -> Path:
    """Finish a figure: super-title, provenance stamp, tight layout, write."""
    if title:
        fig.suptitle(title, fontsize=11.5, color=INK, y=0.99)
    fig.tight_layout(rect=(0, 0.035 if stamp else 0, 1, 0.965 if title else 1))
    if stamp:
        # Every figure carries the scale it was computed at, so one lifted into
        # a slide deck can never be mistaken for a different campaign.
        fig.text(0.5, 0.012, stamp, ha="center", fontsize=7, color=INK2)
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath)
    plt.close(fig)
    return outpath


def _eps_axis(ax, eps: list[float], labels: list[str] | None = None) -> None:
    """Log-x, ticked at the ε values measured — not at 10^n."""
    ax.set_xscale("log")
    ax.set_xticks(list(eps))
    # Ticks read as the budget that was *requested* (1, 5, 10, 50); the achieved
    # value it calibrated to (0.99, 5.00, …) belongs in the table, not the axis.
    ax.set_xticklabels(labels or [f"{e:g}" for e in eps])
    ax.set_xticks([], minor=True)
    ax.set_xlabel("Privacy budget ε (log scale)")


def _bar_labels(ax, bars, fmt="{:.3f}", fontsize=7.5) -> None:
    for b in bars:
        v = b.get_height()
        ax.annotate(fmt.format(v), (b.get_x() + b.get_width() / 2, v),
                    ha="center", va="bottom", fontsize=fontsize, color=INK2,
                    xytext=(0, 2), textcoords="offset points")


def scale_stamp(results: list[dict] | dict, source: str = "") -> str:
    """One-line provenance for a figure: model, corpus size, rounds, seed, source.

    Runs from the earlier campaign predate the ``dataset`` block, so the row
    count is simply omitted for those rather than guessed at — the source
    directory still identifies which campaign a figure came from.
    """
    r = results[0] if isinstance(results, list) else results
    if not r:
        return ""
    cfg = r.get("config", {})
    n_train = (r.get("dataset") or {}).get("n_train")
    bits = [str(cfg.get("model", "")).upper(),
            f"{n_train:,} training rows" if n_train else "",
            f"{cfg.get('rounds', '?')} rounds",
            str(cfg.get("optimizer", "")).upper(),
            f"seed {cfg.get('seed', '?')}",
            source]
    return "  ·  ".join(b for b in bits if b)


def rounds_to_fraction_of_final(timeline: list[dict], frac: float = 0.9) -> int | None:
    """First round reaching ``frac`` of the run's own final accuracy.

    The campaign's ``convergence_round`` uses an *absolute* 0.9 accuracy
    threshold, which the 34-class task never reaches — it is ``None`` in every
    run and reports nothing. This is the usable convergence-speed measure: how
    fast a run gets to within 10 % of where it ends up.
    """
    if not timeline:
        return None
    acc = [t.get("accuracy") or 0.0 for t in timeline]
    target = frac * acc[-1]
    for t, a in zip(timeline, acc):
        if a >= target:
            return int(t.get("round", 0))
    return None


def _ensure(d: Path) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_run(result: dict, outdir: Path) -> Path:
    """Write ``<name>.json`` + ``<name>_timeline.csv`` for one run."""
    outdir = _ensure(outdir)
    name = result["config"]["name"]
    (outdir / f"{name}.json").write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
    tl = result["timeline"]
    if tl:
        cols = ["round", "accuracy", "macro_f1", "loss", "sigma", "cum_epsilon",
                "time_s", "byz_selected"]
        lines = [",".join(cols)]
        for t in tl:
            lines.append(",".join(str(t.get(c) if c != "byz_selected"
                                       else len(t.get("byz_selected", []))) for c in cols))
        (outdir / f"{name}_timeline.csv").write_text("\n".join(lines), encoding="utf-8")
    return outdir / f"{name}.json"


# ── figures ────────────────────────────────────────────────────────────────
# One measure per image. A single chart printed at full text width stays
# readable in the PDF and can be lifted into a slide on its own; the same
# content packed into a 3-panel strip shrinks every label to the point where a
# supervisor has to zoom. Where two measures genuinely belong on one axis —
# accuracy against macro-F1, both scores in [0, 1] — they share a chart.

def _timeline_xy(r: dict, key: str) -> tuple[list, list]:
    tl = r.get("timeline") or []
    return ([t.get("round", k + 1) for k, t in enumerate(tl)],
            [t.get(key) for t in tl])


def plot_runs_over_rounds(results: list[dict], key: str, outpath: Path, *,
                          title: str, ylabel: str, stamp: str = "",
                          mark_convergence: bool = False,
                          logy: bool = False) -> Path:
    """One metric, per round, one line per run."""
    if not results:
        return Path(outpath)
    fig, ax = _fig(w=7.0, h=4.3)
    for j, r in enumerate(results):
        name = r["config"]["name"]
        xs, ys = _timeline_xy(r, key)
        colour = _run_color(name, EPS_COLORS[j % len(EPS_COLORS)])
        label = _label(name) if name in RUN_LABEL else _eps_label(r)
        ax.plot(xs, ys, color=colour, label=label)
        if mark_convergence:
            rc = rounds_to_fraction_of_final(r.get("timeline") or [])
            if rc:
                ax.plot([rc], [(r["timeline"][rc - 1]).get(key)], marker="o",
                        ms=9, mfc="none", mec=colour, mew=1.8, ls="none")
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("Communication round")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best")
    return _save(fig, outpath, stamp)


def _eps_label(r: dict) -> str:
    t = (r.get("config") or {}).get("target_epsilon")
    return f"ε = {t:g}" if t else r["config"]["name"]


def plot_ablation_metric(results: list[dict], key: str, outpath: Path, *,
                         title: str, ylabel: str, stamp: str = "",
                         baseline_delta: bool = False,
                         lower_is_better: bool = False) -> Path:
    """One metric across the ablation ladder, as labelled bars."""
    if not results:
        return Path(outpath)
    names = [r["config"]["name"] for r in results]
    vals = [(r.get("final") or {}).get(key, r.get(f"final_{key}", 0.0)) for r in results]
    fig, ax = _fig(w=7.0, h=4.3)
    x = np.arange(len(names))
    colour = C["red"] if lower_is_better else C["blue"]
    bars = ax.bar(x, vals, 0.55, color=colour)
    _bar_labels(ax, bars, fontsize=9)
    if baseline_delta and len(vals) > 1:
        for i in range(1, len(vals)):
            ax.annotate(f"Δ {vals[i] - vals[0]:+.3f}",
                        (x[i], vals[i]), ha="center", va="bottom", fontsize=8.5,
                        color=INK, xytext=(0, 15), textcoords="offset points")
    ax.set_xticks(x)
    ax.set_xticklabels([_label(n) for n in names], rotation=12, ha="right")
    ax.set_ylim(0, max(vals) * (1.30 if baseline_delta else 1.18) or 1)
    ax.set_ylabel(ylabel + ("  (lower is better)" if lower_is_better else ""))
    ax.set_title(title)
    return _save(fig, outpath, stamp)


def plot_privacy_utility(rows: list[dict], outpath: Path,
                         results: list[dict] | None = None, stamp: str = "") -> Path:
    """RO3 — the four detection measures against the privacy budget."""
    rows = sorted(rows, key=lambda r: r["epsilon"])
    eps = [r["epsilon"] for r in rows]
    by_eps = {round(r.get("final_epsilon", 0.0), 3): r for r in (results or [])}
    targets = None
    if results:
        t = [(by_eps.get(round(r["epsilon"], 3), {}).get("config") or {}).get("target_epsilon")
             for r in rows]
        if all(v is not None for v in t):
            targets = [f"{v:g}" for v in t]

    def _pick(key):
        out = [by_eps.get(round(r["epsilon"], 3), {}).get("final", {}).get(key) for r in rows]
        return out if all(v is not None for v in out) else None

    fig, ax = _fig(w=7.0, h=4.6)
    ax.plot(eps, [r["accuracy"] for r in rows], "o-", color=C["blue"], label="Accuracy")
    ax.plot(eps, [r["macro_f1"] for r in rows], "s-", color=C["orange"], label="Macro-F1")
    prec, rec = _pick("macro_precision"), _pick("macro_recall")
    if prec and rec:
        ax.plot(eps, prec, "^--", color=C["aqua"], label="Macro-precision")
        ax.plot(eps, rec, "v--", color=C["violet"], label="Macro-recall")
    _eps_axis(ax, eps, targets)
    ax.set_ylabel("Score  (0 = useless, 1 = perfect)")
    ax.set_title("Detection quality against the privacy budget")
    ax.set_ylim(top=max(r["accuracy"] for r in rows) * 1.12)
    ax.legend(loc="upper left", ncol=2)
    return _save(fig, outpath, stamp)


def plot_vs_epsilon(rows: list[dict], key: str, outpath: Path, *, title: str,
                    ylabel: str, colour: str = C["red"], stamp: str = "",
                    results: list[dict] | None = None, annotate: str = "") -> Path:
    """One quantity against the privacy budget."""
    rows = sorted(rows, key=lambda r: r["epsilon"])
    eps = [r["epsilon"] for r in rows]
    by_eps = {round(r.get("final_epsilon", 0.0), 3): r for r in (results or [])}
    targets = None
    if results:
        t = [(by_eps.get(round(r["epsilon"], 3), {}).get("config") or {}).get("target_epsilon")
             for r in rows]
        if all(v is not None for v in t):
            targets = [f"{v:g}" for v in t]
    if key == "sigma":
        vals = [((by_eps.get(round(r["epsilon"], 3), {}).get("noise_schedule") or [np.nan])[0])
                for r in rows]
    else:
        vals = [r[key] for r in rows]
    fig, ax = _fig(w=7.0, h=4.3)
    ax.plot(eps, vals, "d-", color=colour)
    for e, v in zip(eps, vals):
        if np.isfinite(v):
            ax.annotate(annotate.format(v) if annotate else f"{v:.3f}", (e, v),
                        fontsize=8.5, color=INK2, ha="center",
                        xytext=(0, 9), textcoords="offset points")
    _eps_axis(ax, eps, targets)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0, max(v for v in vals if np.isfinite(v)) * 1.25)
    return _save(fig, outpath, stamp)


def plot_privacy_spend(results: list[dict], outpath: Path, stamp: str = "") -> Path:
    """Cumulative ε per round — what the RDP accountant says has been spent."""
    dp = [r for r in results if r.get("config", {}).get("dp")]
    if not dp:
        return Path(outpath)
    if len({round(r.get("final_epsilon", 0.0), 1) for r in dp}) > 1:
        dp = sorted(dp, key=lambda r: r.get("final_epsilon", 0.0))
    fig, ax = _fig(w=7.0, h=4.3)
    for j, r in enumerate(dp):
        name = r["config"]["name"]
        xs, ys = _timeline_xy(r, "cum_epsilon")
        ax.plot(xs, ys, color=_run_color(name, EPS_COLORS[j % len(EPS_COLORS)]),
                label=_label(name) if name in RUN_LABEL else _eps_label(r))
    ax.set_yscale("log")
    ax.set_xlabel("Communication round")
    ax.set_ylabel("Cumulative ε spent  (log scale)")
    ax.set_title("Privacy spent so far, accounted by RDP")
    ax.legend(loc="lower right")
    return _save(fig, outpath, stamp)


def plot_noise_schedule(results: list[dict], outpath: Path, stamp: str = "") -> Path:
    """The σ actually applied in each round — the schedule shape itself."""
    dp = [r for r in results if r.get("config", {}).get("dp")]
    if not dp:
        return Path(outpath)
    fig, ax = _fig(w=7.0, h=4.3)
    for j, r in enumerate(dp):
        name = r["config"]["name"]
        ns = r.get("noise_schedule") or [t.get("sigma") for t in (r.get("timeline") or [])]
        ax.plot(range(1, len(ns) + 1), ns,
                color=_run_color(name, EPS_COLORS[j % len(EPS_COLORS)]),
                label=_label(name) if name in RUN_LABEL else _eps_label(r))
    ax.set_xlabel("Communication round")
    ax.set_ylabel("Noise multiplier σ")
    ax.set_title("How the budget is spent across rounds")
    ax.legend(loc="best")
    return _save(fig, outpath, stamp)


def _byz_admitted(result: dict) -> tuple[int, int]:
    """(poisoned updates admitted, poisoned updates offered) across all rounds."""
    tl = result.get("timeline") or []
    n_byz = len(result.get("byzantine_clients") or [])
    return (sum(len(t.get("byz_selected") or []) for t in tl), n_byz * len(tl))


def plot_byzantine_metric(rows: list[dict], key: str, outpath: Path, *, title: str,
                          ylabel: str, stamp: str = "") -> Path:
    """One metric against the malicious fraction, one line per aggregation rule."""
    if not rows:
        return Path(outpath)
    aggs = [a for a in ("fedavg", "krum", "multikrum", "trimmedmean")
            if a in {r["aggregator"] for r in rows}]
    aggs += sorted({r["aggregator"] for r in rows} - set(aggs))
    fracs = sorted({r["byz_fraction"] for r in rows})
    fig, ax = _fig(w=7.0, h=4.3)
    if len(fracs) > 2:
        for a in aggs:
            pts = sorted([r for r in rows if r["aggregator"] == a],
                         key=lambda r: r["byz_fraction"])
            ax.plot([p["byz_fraction"] * 100 for p in pts], [p[key] for p in pts],
                    marker="o", color=AGG_COLOR.get(a, C["magenta"]),
                    label=AGG_LABEL.get(a, a))
        ax.set_xlabel("Malicious gNBs (%)")
    else:
        # Two measured points would draw a degradation curve nobody observed.
        x = np.arange(len(fracs))
        wdt = 0.8 / max(len(aggs), 1)
        for k, a in enumerate(aggs):
            vals = [next((r[key] for r in rows if r["aggregator"] == a
                          and r["byz_fraction"] == f), np.nan) for f in fracs]
            bars = ax.bar(x + (k - (len(aggs) - 1) / 2) * wdt, vals, wdt * 0.9,
                          color=AGG_COLOR.get(a, C["magenta"]), label=AGG_LABEL.get(a, a))
            _bar_labels(ax, bars, fmt="{:.2f}", fontsize=7.5)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{f:.0%} malicious" for f in fracs])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best")
    return _save(fig, outpath, stamp)


def plot_byzantine_admitted(results: list[dict], outpath: Path, stamp: str = "") -> Path:
    """How many poisoned updates each rule let into the average."""
    by_agg: dict[str, tuple[int, int]] = {}
    for r in results:
        cfg = r.get("config", {})
        if not cfg.get("byzantine_fraction") or cfg.get("dp"):
            continue
        adm, off = _byz_admitted(r)
        if off > by_agg.get(cfg["aggregator"], (0, 0))[1]:
            by_agg[cfg["aggregator"]] = (adm, off)
    names = [a for a in ("fedavg", "krum", "multikrum", "trimmedmean") if a in by_agg]
    if not names:
        return Path(outpath)
    offered = by_agg[names[0]][1]
    vals = [by_agg[a][0] for a in names]
    fig, ax = _fig(w=7.0, h=4.3)
    x = np.arange(len(names))
    bars = ax.bar(x, vals, 0.55, color=[AGG_COLOR.get(a, C["magenta"]) for a in names])
    for b, v in zip(bars, vals):
        ax.annotate(f"{v} / {offered}", (b.get_x() + b.get_width() / 2, v), ha="center",
                    va="bottom", fontsize=10, color=INK,
                    xytext=(0, 4), textcoords="offset points")
    ax.set_xticks(x)
    ax.set_xticklabels([AGG_LABEL.get(a, a) for a in names])
    ax.set_ylim(0, offered * 1.2 or 1)
    ax.set_ylabel("Poisoned updates admitted")
    ax.set_title(f"Poisoned updates that reached the average, of {offered} offered")
    return _save(fig, outpath, stamp)


def plot_krum_scores(result: dict, outpath: Path, stamp: str = "") -> Path:
    """The distances Multi-Krum decides on: honest gNBs against malicious ones."""
    tl = result.get("timeline") or []
    byz = set(result.get("byzantine_clients") or [])
    if not byz or not tl or not tl[0].get("scores"):
        return Path(outpath)
    xs, hon, mal = [], [], []
    for k, t in enumerate(tl):
        sc = t.get("scores") or []
        if not sc:
            continue
        xs.append(t.get("round", k + 1))
        hon.append(np.mean([s for j, s in enumerate(sc) if j not in byz]))
        mal.append(np.mean([s for j, s in enumerate(sc) if j in byz]))
    fig, ax = _fig(w=7.0, h=4.3)
    ax.plot(xs, hon, color=C["blue"], label="honest gNBs")
    ax.plot(xs, mal, color=C["red"], label="malicious gNBs")
    ax.set_yscale("log")
    ax.set_xlabel("Communication round")
    ax.set_ylabel("Mean distance score  (log scale)")
    cfg = result["config"]
    privacy = (f"privacy ON, ε={result.get('final_epsilon', 0):.0f}"
               if cfg.get("dp") else "privacy OFF")
    gap = float(np.mean([m / h for m, h in zip(mal, hon)]))
    ax.set_title(f"{AGG_LABEL.get(cfg['aggregator'], cfg['aggregator'])}, "
                 f"{cfg['byzantine_fraction']:.0%} malicious — {privacy}\n"
                 f"malicious updates score {gap:.0f}× further from the crowd")
    ax.legend(loc="center right")
    return _save(fig, outpath, stamp)


def plot_round_time(results: list[dict], outpath: Path, stamp: str = "") -> Path:
    """Wall-clock per round, DP against no DP.

    Wall-clock is measured on a shared CPU and identical configurations run at
    different times differ by up to 3×, so this shows the distribution — median
    with every run as a point — for the one comparison the data supports.
    """
    runs = [r for r in results if r.get("avg_round_time_s")]
    if not runs:
        return Path(outpath)
    groups = [("no DP", [r["avg_round_time_s"] for r in runs if not r["config"].get("dp")],
               C["aqua"]),
              ("with DP", [r["avg_round_time_s"] for r in runs if r["config"].get("dp")],
               C["violet"])]
    groups = [g for g in groups if g[1]]
    fig, ax = _fig(w=7.0, h=4.3)
    x = np.arange(len(groups))
    meds = [float(np.median(g[1])) for g in groups]
    ax.bar(x, meds, 0.45, color=[g[2] for g in groups])
    rng = np.random.default_rng(0)
    for i, (_, vals, _c) in enumerate(groups):
        ax.plot(np.full(len(vals), x[i]) + rng.uniform(-0.1, 0.1, size=len(vals)), vals,
                "o", ms=6, mfc="white", mec=INK2, mew=1.2, ls="none",
                label="individual runs" if i == 0 else None, zorder=3)
    for xi, m in zip(x, meds):
        ax.annotate(f"median {m:.0f}s", (xi, m * 0.06), ha="center", va="bottom",
                    fontsize=9.5, color="white")
    ax.set_xticks(x)
    ax.set_xticklabels([g[0] for g in groups])
    ax.set_ylabel("Wall-clock seconds per round")
    ax.set_ylim(0, max(max(g[1]) for g in groups) * 1.25)
    ratio = f" — privacy costs ×{meds[1] / meds[0]:.1f}" if len(groups) == 2 else ""
    ax.set_title(f"Cost of one federated round{ratio}")
    ax.legend(loc="upper right")
    return _save(fig, outpath, stamp)


def plot_round_time_series(results: list[dict], outpath: Path, stamp: str = "") -> Path:
    """Per-round wall-clock over a run, for the four framework configurations."""
    shown = [r for r in results if r["config"]["name"].startswith("ablation")]
    if not shown:
        return Path(outpath)
    fig, ax = _fig(w=7.0, h=4.3)
    for j, r in enumerate(shown):
        xs, ys = _timeline_xy(r, "time_s")
        ax.plot(xs, ys, color=_run_color(r["config"]["name"],
                                         EPS_COLORS[j % len(EPS_COLORS)]),
                label=_label(r["config"]["name"]))
    ax.set_xlabel("Communication round")
    ax.set_ylabel("Wall-clock seconds")
    ax.set_title("Per-round cost over a run\n(spikes are contention on a shared CPU)")
    ax.legend(loc="best")
    return _save(fig, outpath, stamp)


def plot_per_category(result: dict, outpath: Path, stamp: str = "",
                      title: str | None = None) -> Path:
    """Per-family recall with support — the per-class detection dimension."""
    per_cat = (result.get("final") or {}).get("per_category") or {}
    if not per_cat:
        return Path(outpath)
    items = sorted(per_cat.items(), key=lambda kv: kv[1].get("support", 0))
    names = [k for k, _ in items]
    rec = [v.get("recall", 0.0) for _, v in items]
    sup = [v.get("support", 0) for _, v in items]
    fig, ax = _fig(w=7.2, h=4.6)
    y = np.arange(len(names))
    ax.grid(axis="y", visible=False)
    ax.barh(y, rec, 0.68, color=C["blue"])
    ax.axvline(0.5, color=INK2, ls="--", lw=1)
    for i, (r, s) in enumerate(zip(rec, sup)):
        ax.annotate(f"{r:.3f}   ({s:,} rows)", (r, i), va="center", fontsize=8.5,
                    color=INK2, xytext=(4, 0), textcoords="offset points")
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlim(0, 1.30)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel("Recall on the held-out set")
    ax.set_ylabel("Attack family (ordered by size)")
    cfg = result.get("config", {})
    agg = AGG_LABEL.get(cfg.get("aggregator"), str(cfg.get("aggregator", "")))
    privacy = f"DP ε={result.get('final_epsilon', 0):.0f}" if cfg.get("dp") else "no DP"
    return _save(fig, outpath, stamp,
                 title or f"Detection by attack family — {agg}, {privacy}")


# ── tables ─────────────────────────────────────────────────────────────────
def results_table(rows: list[dict], outdir: Path) -> Path:
    outdir = _ensure(outdir)
    if not rows:
        return outdir / "results_table.csv"
    cols = list(rows[0].keys())
    lines = [",".join(cols)] + [",".join(str(r.get(c, "")) for c in cols) for r in rows]
    p = outdir / "results_table.csv"
    p.write_text("\n".join(lines), encoding="utf-8")
    return p


def write_summary(rows: list[dict], outdir: Path, notes: str = "") -> Path:
    outdir = _ensure(outdir)
    lines = ["# DP-FL-IDS — Experiment Summary\n"]
    if notes:
        lines += [notes, ""]
    if rows:
        cols = list(rows[0].keys())
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("| " + " | ".join("---" for _ in cols) + " |")
        for r in rows:
            def fmt(v):
                return f"{v:.4f}" if isinstance(v, float) else str(v)
            lines.append("| " + " | ".join(fmt(r.get(c, "")) for c in cols) + " |")
    p = outdir / "SUMMARY.md"
    p.write_text("\n".join(lines), encoding="utf-8")
    return p


def row_from_result(result: dict) -> dict:
    """Flatten one run into a summary-table row."""
    for t in result.get("timeline", []):
        values = [t[k] for k in ("loss", "accuracy", "macro_f1") if k in t]
        values += t.get("scores") or []
        if not all(np.isfinite(v) for v in values):
            raise ValueError("Numerically failed run cannot be summarized as a successful evaluation")
    c = result["config"]
    final = result.get("final") or {}
    if not all(k in final and np.isfinite(final[k]) for k in ("accuracy", "macro_f1", "fpr")):
        raise ValueError("Full-test final metrics are required; subset aliases are not a fallback")
    return {
        "name": c["name"],
        "aggregator": c["aggregator"],
        "dp": c["dp"],
        "epsilon": round(result.get("final_epsilon", 0.0), 3),
        "byz_fraction": c["byzantine_fraction"],
        "attack": c["attack"],
        "accuracy": round(final["accuracy"], 4),
        "macro_f1": round(final["macro_f1"], 4),
        "fpr": round(final["fpr"], 4),
        "conv_round": result.get("convergence_round"),
        "conv_round_90pct_final": rounds_to_fraction_of_final(result.get("timeline") or []),
        "round_time_s": round(result.get("avg_round_time_s", 0.0), 2),
        "macro_precision": round(final.get("macro_precision", 0.0), 4),
        "macro_recall": round(final.get("macro_recall", 0.0), 4),
    }
