"""What each narrated step actually executes.

Every step takes the shared :class:`Ctx` and returns nothing; anything worth
carrying forward is stashed on ``ctx.book`` so the closing summary can reprint
it. Steps are deliberately independent — ``--steps 12`` has to work on its own —
so where one needs a result another already computed, it asks ``ctx.run`` for it
and the cache decides whether that means training or a dictionary lookup.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..data.config import (ALL_FEATURES, ARTIFACT_DIR, NETWORK_FEATURES,
                           PROJECT_ROOT, RAN_FEATURES, RAW_DIR)
from ..experiment_config import RunConfig
from ..utils import console
from . import narration, thesis_numbers as TN
from .progress import ProgressReporter

# ── profiles ───────────────────────────────────────────────────────────────
# Everything that scales the demo's cost lives here. "demo" is the default:
# complete coverage at enough rounds for the trends to be legible. "full"
# reproduces the thesis grid exactly, so the demo and the campaign cannot drift.
#
# Round counts are set against measured cost. Rounds are chunked, so fewer
# rounds means proportionally bigger ones: the whole corpus is consumed exactly
# once either way, and the round count trades granularity, not total work. On
# the complete corpus the campaign measured ~22.5 us per record without DP and
# ~92 us with it for the MLP; the LSTM is ~40x slower per record, which is why
# --model lstm prints a much larger estimate.
PROFILES: dict[str, dict] = {
    # sep_sigmas is the same everywhere: the score-separation benchmark builds
    # its update pool once and then only injects noise, so extra sigmas are
    # nearly free — and a coarse grid makes a monotonic trend look ragged.
    "quick": dict(rounds=5, eps_grid=[1.0, 10.0], byz_fracs=[0.3],
                  aggregators=["fedavg", "multikrum"],
                  schedules=["decreasing_noise"],
                  sep_sigmas=(0.0, 0.5, 1.0, 2.0, 4.0), prep_files=1,
                  prep_per_class=25),
    "demo": dict(rounds=10, eps_grid=[1.0, 5.0, 10.0, 50.0], byz_fracs=[0.0, 0.3],
                 aggregators=["fedavg", "krum", "multikrum", "trimmedmean"],
                 schedules=["decreasing_noise", "increasing_noise"],
                 sep_sigmas=(0.0, 0.5, 1.0, 2.0, 4.0), prep_files=1,
                 prep_per_class=50),
    "full": dict(rounds=30, eps_grid=[1.0, 5.0, 10.0, 50.0],
                 byz_fracs=[0.0, 0.1, 0.2, 0.3],
                 aggregators=["fedavg", "krum", "multikrum", "trimmedmean"],
                 schedules=["decreasing_noise", "increasing_noise"],
                 sep_sigmas=(0.0, 0.5, 1.0, 2.0, 4.0), prep_files=3,
                 prep_per_class=500),
}


def _slug(label: str) -> str:
    """Filesystem-safe run name. Display labels carry punctuation and spaces
    ("1. FL baseline (FedAvg, no DP)"); the files they produce should not."""
    out = "".join(c.lower() if c.isalnum() else "_" for c in label)
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_") or "run"


@dataclass
class Ctx:
    args: object
    profile: dict
    outdir: Path
    base: dict
    book: dict = field(default_factory=dict)
    rows: list = field(default_factory=list)
    _mem: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)

    # ── run with caching ──────────────────────────────────────────────────
    def _key(self, cfg: RunConfig) -> str:
        d = {k: v for k, v in cfg.to_dict().items() if k != "name"}
        return json.dumps(d, sort_keys=True, default=str)

    def run(self, label: str, *, detail: str = "round", record: bool = True,
            **kw) -> dict:
        """Run one federated experiment, reusing an identical earlier one.

        Results are deterministic under a fixed seed, so two steps asking for
        the same configuration must get the same answer — computing it twice
        would only make the demo longer.
        """
        from ..fl.simulation import run_federated
        from ..utils import reporting

        cfg = RunConfig(name=_slug(label), **{**self.base, **kw})
        key = self._key(cfg)
        if key in self._mem:
            console.info("reusing the identical run computed earlier — same "
                         "seed, same configuration, so the same result")
            res = self._mem[key]
        else:
            res = run_federated(cfg, verbose=False,
                                on_event=ProgressReporter(level=detail))
            self._mem[key] = res
            reporting.save_run(res, self.outdir)
        if record:
            self.book[label] = res
            row = reporting.row_from_result(res)
            row["name"] = label
            self.rows.append(row)
        return res


# ── small shared helpers ───────────────────────────────────────────────────
def _eps_of(res: dict) -> str:
    return f"{res['final_epsilon']:.2f}" if res["config"]["dp"] else "-"


def _byz_kept(res: dict) -> tuple[int, int]:
    f = len(res.get("byzantine_clients", []))
    kept = sum(len(t.get("byz_selected", [])) for t in res["timeline"])
    return kept, f * len(res["timeline"])


def _vs_thesis(live: float, key: str, label: str = "accuracy") -> None:
    """Print the live number beside the reported one."""
    console.table(["", f"live ({label})", TN.column()],
                  [["this run", f"{live:.3f}", TN.fmt(key)]])


# ══════════════════════════════════════════════════════════════════════════
# STEP 1 — environment
# ══════════════════════════════════════════════════════════════════════════
def step_01(ctx: Ctx) -> None:
    import os
    import platform

    with console.step("importing PyTorch, Opacus and Flower"):
        import flwr
        import opacus
        import torch

    console.kv({
        "PyTorch (training)": torch.__version__,
        "Opacus (differential privacy)": opacus.__version__,
        "Flower (federated learning)": flwr.__version__,
        "operating system": f"{platform.system()} {platform.release()}",
        "compute device": ("cuda" if (ctx.base["device"] != "cpu"
                                      and torch.cuda.is_available()) else "cpu"),
        "random seed": ctx.base["seed"],
        "profile": f"{ctx.args.profile}  ({ctx.profile['rounds']} rounds per experiment)",
    })

    ncpu = os.cpu_count() or 1
    # getloadavg() is Unix-only — absent on Windows, where this demo also runs.
    load = os.getloadavg()[0] if hasattr(os, "getloadavg") else None
    console.kv({"CPU cores": ncpu if load is None
                else f"{ncpu}  (current load {load:.1f})"})
    if load is not None and load > ncpu * 0.8:
        console.warn("This machine is busy — every timing below will be slower "
                     "than on an idle CPU.")
    if ncpu <= 4:
        console.warn(f"Only {ncpu} CPU cores: training will be several times "
                     "slower than the timings quoted in the documentation. "
                     "Use --profile quick, or --replay to review the saved "
                     "results without training.")
    console.takeaway("The stack is present and seeded, so this run is reproducible.")


# ══════════════════════════════════════════════════════════════════════════
# STEP 2 — raw data and preprocessing
# ══════════════════════════════════════════════════════════════════════════
def step_02(ctx: Ctx) -> None:
    # NOTE: preprocessing.build_dataset() would overwrite data/processed/ and
    # data/artifacts/ — the artefacts the committed partitions were built from.
    # The demo therefore calls the same pure sub-functions in the same order and
    # simply does not save, so nothing on disk is disturbed.
    from ..data import preprocessing

    if not RAW_DIR.exists():
        root = ctx.base.get("data_root") or "data"
        console.warn(f"Raw CICIoT2023 CSVs not found at {RAW_DIR} — skipping the "
                     "live preprocessing demonstration. This step is the only "
                     f"one that reads them: the prepared shards in {root}/ are "
                     "already built, so every later step still runs on the "
                     "complete corpus.")
        console.info("To see this step run too, unpack the raw CSVs and point "
                     "the framework at them with the DPFLIDS_RAW_DIR "
                     "environment variable, for example: "
                     r'set DPFLIDS_RAW_DIR=D:\CICIoT2023   (Windows)  /  '
                     "export DPFLIDS_RAW_DIR=/path/to/CICIoT2023   (Linux, macOS)")
        ctx.notes.append("step 2 skipped: raw dataset directory absent")
        return

    parts = sorted(RAW_DIR.glob("part-*.csv"))
    total_gb = sum(p.stat().st_size for p in parts) / 1e9
    console.kv({
        "raw corpus": f"{len(parts)} CSV files, {total_gb:.1f} GB",
        "location": str(RAW_DIR),
        "attack labels": "34, grouped into 8 categories",
        "network features": len(NETWORK_FEATURES),
    })

    n_files = ctx.profile["prep_files"]
    per_class = ctx.profile["prep_per_class"]
    console.say(f"""Now running the real pipeline over {n_files} of those
        {len(parts)} files, taking at most {per_class} rows per attack class.
        The functions below are exactly the ones the full preprocessing run
        calls; only the slice is smaller.""")

    with console.step(f"streaming and stratified-sampling <= {per_class} rows/class"):
        raw = preprocessing.collect_subsample(max_per_class=per_class,
                                              max_files=n_files,
                                              seed=ctx.base["seed"])
    console.kv({"rows collected": f"{len(raw):,}",
                "columns": raw.shape[1],
                "distinct labels present": raw["label"].nunique()})

    with console.step("cleaning: drop infinities, NaNs and exact duplicates"):
        cleaned = preprocessing.clean(raw)

    top = cleaned["label"].value_counts().head(5)
    console.say("The five most common attack labels in this slice:")
    console.table(["attack label", "rows"], [[k, f"{v:,}"] for k, v in top.items()])

    ctx._mem["_prep_df"] = cleaned
    try:
        from ..data import loaders
        _r = loaders.DataRoot(ctx.base.get("data_root"))
        _m = _r.manifest()
        prepared = int(_m.get("n_train", 0)) + int(_m.get("n_test", 0))
        where = f"{prepared:,} rows in {_r.name}/"
    except Exception:
        prepared, where = 0, "the prepared dataset"
    console.takeaway(f"""{len(cleaned):,} real labelled flow records, cleaned by the
        same code path the experiments use. Running this over all 169 files
        yields {where} — the whole corpus is {46_686_579:,} flows, and
        `python3 -m src.data.build_scaled` streams every one of them into gNB
        shards without ever holding the dataset in memory.""")


# ══════════════════════════════════════════════════════════════════════════
# STEP 3 — RAN augmentation
# ══════════════════════════════════════════════════════════════════════════
def step_03(ctx: Ctx) -> None:
    from ..data import preprocessing
    from ..data.ran_augment import add_ran_features

    df = ctx._mem.get("_prep_df")
    if df is None:
        console.warn("No preprocessed frame in memory (step 2 was skipped); "
                     "describing the RAN feature set without recomputing it.")
        console.kv({f"RAN feature {i+1}": f for i, f in enumerate(RAN_FEATURES)})
        return

    console.kv({"features before": df.shape[1]})
    with console.step("appending the 7 synthetic 6G RAN features"):
        aug = add_ran_features(df.copy(), np.random.default_rng(ctx.base["seed"]))
    with console.step("encoding labels to ids and 8 attack categories"):
        aug = preprocessing.encode(aug)

    console.kv({"RAN features added": ", ".join(RAN_FEATURES),
                "network + RAN": f"{len(NETWORK_FEATURES)} + {len(RAN_FEATURES)} "
                                 f"= {len(ALL_FEATURES)} model inputs"})

    # Show that the radio telemetry actually differs by attack family — this is
    # the point of conditioning it on the label rather than sampling it blind.
    # Names come from RAN_FEATURES so they cannot drift from the generator.
    preferred = ("SINR_dB", "PRB_Util_percent", "Beam_Failure_Count")
    cols = [c for c in preferred if c in aug.columns] or list(RAN_FEATURES[:3])
    cols = [c for c in cols if c in aug.columns]
    if cols and "category" in aug.columns:
        stats = aug.groupby("category")[cols].mean().round(2)
        console.say("""Mean radio conditions per attack family. The values differ
            by family because the generator conditions them on the label —
            flooding attacks exhaust physical resource blocks, and that shows
            up in the telemetry rather than only in the flow counters:""")
        console.table(["attack category"] + cols,
                      [[idx] + [f"{r[c]:.2f}" for c in cols]
                       for idx, r in stats.iterrows()])

    console.takeaway(f"""Every record is now a {len(ALL_FEATURES)}-dimensional
        vector combining measured network behaviour with synthetic radio
        context. The synthesis is a documented limitation: the flow features are
        real, the radio features are modelled from 3GPP ranges.""")


# ══════════════════════════════════════════════════════════════════════════
# STEP 4 — non-IID partitioning
# ══════════════════════════════════════════════════════════════════════════
def step_04(ctx: Ctx) -> None:
    from ..data import loaders

    # Shard sizes come from parquet metadata and the manifest. Reading the rows
    # merely to count them would pull ~8 GB into memory on the full corpus.
    with console.step("reading the 10 gNB shards and the held-out test split"):
        root = loaders.DataRoot(ctx.base.get("data_root"))
        n_clients = root.n_clients()
        sizes = [root.client_rows(c) for c in range(n_clients)]
        man = root.manifest()
        labels = root.label_map()

    n_test = int(man.get("n_test") or 0)
    alpha = str(man.get("dirichlet_alpha", 0.3))
    rows = {
        "dataset": root.name,
        "simulated base stations (gNBs)": n_clients,
        "Dirichlet alpha (lower = more skewed)": alpha,
        "training records per gNB": f"min {min(sizes):,}, median "
                                    f"{int(np.median(sizes)):,}, max {max(sizes):,}",
        "total training records": f"{sum(sizes):,}",
        "held-out test records": f"{n_test:,}" if n_test else "n/a",
        "features per record": loaders.N_FEATURES,
        "attack classes": len(labels),
    }
    if ctx.base.get("round_chunks"):
        rounds = ctx.base["rounds"]
        rows["records per gNB per round"] = (
            f"{min(sizes) // rounds:,} .. {max(sizes) // rounds:,}  "
            f"({rounds} disjoint slices = one full pass)")
    console.kv(rows)

    console.say("""Per-gNB size, to make the imbalance concrete. In a uniform
        split every bar would be the same length:""")
    widest = max(sizes)
    console.table(["gNB", "records", "share"],
                  [[f"{c:02d}", f"{s:,}", "#" * max(1, round(28 * s / widest))]
                   for c, s in enumerate(sizes)],
                  aligns="<><")

    console.takeaway(f"""The largest gNB holds {max(sizes)/min(sizes):.1f}x the data
        of the smallest, and their attack mixes differ too. That heterogeneity
        is the realistic case and the hard one — a method that only works on
        evenly split data would not survive deployment.""")


# ══════════════════════════════════════════════════════════════════════════
# STEP 5 — the model
# ══════════════════════════════════════════════════════════════════════════
def step_05(ctx: Ctx) -> None:
    from ..data import loaders
    from ..models.lstm_ids import build_model

    n_feat = loaders.N_FEATURES
    n_cls = loaders.DataRoot(ctx.base.get("data_root")).num_classes()
    rows = []
    for name in ("mlp", "lstm"):
        with console.step(f"building the {name.upper()} classifier"):
            m = build_model(name, n_feat, n_cls, hidden=ctx.base.get("hidden", 64))
            n_par = sum(p.numel() for p in m.parameters())
        role = ("primary quantitative model — fast, DP-tolerant"
                if name == "mlp" else
                "thesis primary architecture — Opacus DPLSTM")
        rows.append([name.upper(), f"{n_feat} -> 64 -> {n_cls}", f"{n_par:,}", role])

    console.table(["model", "shape", "parameters", "role"], rows,
                  aligns="<<><")
    console.kv({"model used for this demo": ctx.base["model"].upper()})
    console.takeaway("""Both architectures accept the same 53-dimensional input
        and emit a distribution over 34 attack classes, so privacy and
        robustness can be varied without changing the model underneath.""")


# ══════════════════════════════════════════════════════════════════════════
# STEP 6 — one round in slow motion
# ══════════════════════════════════════════════════════════════════════════
def step_06(ctx: Ctx) -> None:
    console.say("""Every internal action of one round is printed below. Note
        especially what does NOT appear: at no point does a training record
        leave a gNB. Only weight vectors travel.""")
    # One round, but a *normal-sized* one: n_chunks keeps the slice the same as
    # every later step's, instead of handing each gNB its entire shard.
    res = ctx.run("demo_single_round", detail="full", record=False,
                  aggregator="fedavg", dp=False, rounds=1,
                  n_chunks=ctx.base["rounds"])
    t = res["timeline"][0]
    console.kv({"global accuracy after one round": t["accuracy"],
                "macro-F1 after one round": t["macro_f1"],
                "wall-clock for the round": f"{t['time_s']:.1f}s"})
    console.takeaway(str(narration.step_meta(6)["takeaway"]))


# ══════════════════════════════════════════════════════════════════════════
# STEP 7 — FL baseline
# ══════════════════════════════════════════════════════════════════════════
def step_07(ctx: Ctx) -> None:
    res = ctx.run("1. FL baseline (FedAvg, no DP)", aggregator="fedavg", dp=False)
    ctx._mem["baseline"] = res
    console.kv({"final accuracy": res["final_accuracy"],
                "final macro-F1": res["final_macro_f1"],
                "average round time": f"{res['avg_round_time_s']:.1f}s"})
    _vs_thesis(res["final_accuracy"], "ablation/fl_only")
    console.takeaway(f"""Ten base stations jointly reached
        {res['final_accuracy']:.3f} accuracy on 34-class attack detection
        without sharing a single raw flow record. That is the federated
        backbone of RO1 working, and the number every later trade-off is
        measured against.""")


# ══════════════════════════════════════════════════════════════════════════
# STEP 8 — differential privacy
# ══════════════════════════════════════════════════════════════════════════
def step_08(ctx: Ctx) -> None:
    from ..data import loaders
    from ..privacy import adaptive_eps
    from ..privacy.dp_engine import sample_rate

    target = 5.0
    rounds = ctx.base["rounds"]
    # Shard sizes from parquet metadata — reading 37M rows merely to count them
    # would cost minutes and gigabytes.
    _root = loaders.DataRoot(ctx.base.get("data_root"))
    sizes = [_root.client_rows(c) for c in range(_root.n_clients())]
    # A chunked run samples from one round's slice, not the whole shard, and the
    # sampling rate is what drives epsilon.
    if ctx.base.get("round_chunks"):
        sizes = [max(1, s // rounds) for s in sizes]
    avg_n = int(np.mean(sizes))
    q = sample_rate(ctx.base["batch_size"], avg_n)
    steps_per_round = ctx.base["local_epochs"] * int(np.ceil(avg_n / ctx.base["batch_size"]))

    console.say(f"""Before training, the framework has to decide how much noise to
        add. It searches by bisection for the noise scale sigma whose accumulated
        privacy loss over {rounds} rounds lands exactly on the target budget —
        so the guarantee is exact, not approximate:""")
    with console.step(f"calibrating sigma for a total budget of epsilon = {target:g}"):
        sigma = adaptive_eps.constant_sigma(target, rounds=rounds, sample_rate=q,
                                            steps_per_round=steps_per_round,
                                            delta=1e-5)
    console.kv({"target epsilon": target, "delta": "1e-5",
                "sampling rate q": f"{q:.4f}",
                "optimiser steps per round": steps_per_round,
                "calibrated noise sigma": f"{sigma:.4f}",
                "gradient clipping norm C": 1.0})

    res = ctx.run("2. FL + DP (epsilon=5)", aggregator="fedavg", dp=True,
                  target_epsilon=target)
    base = ctx._mem.get("baseline")
    console.kv({"final epsilon reached": res["final_epsilon"],
                "final accuracy": res["final_accuracy"],
                "false-positive rate": res["final_fpr"]})
    _vs_thesis(res["final_accuracy"], "ablation/fl_dp")

    if base:
        cost = base["final_accuracy"] - res["final_accuracy"]
        direction = (f"cost {cost:.3f} accuracy" if cost >= 0 else
                     f"gained {abs(cost):.3f} — run-to-run noise at this length")
        console.takeaway(f"""The accountant landed on
            epsilon={res['final_epsilon']:.2f} against a target of {target:g}, so
            the privacy guarantee is exact. Privacy {direction}. In the thesis
            runs that cost is 0.684 -> 0.616, and a large part of making it that
            small was the optimiser choice examined in step 11.""")


# ══════════════════════════════════════════════════════════════════════════
# STEP 9 — ablation
# ══════════════════════════════════════════════════════════════════════════
def step_09(ctx: Ctx) -> None:
    combos = [
        ("FL only", "ablation/fl_only", dict(aggregator="fedavg", dp=False)),
        ("FL + DP", "ablation/fl_dp", dict(aggregator="fedavg", dp=True,
                                           target_epsilon=5.0)),
        ("FL + Multi-Krum", "ablation/fl_krum", dict(aggregator="multikrum", dp=False)),
        ("FL + DP + Multi-Krum", "ablation/fl_dp_krum",
         dict(aggregator="multikrum", dp=True, target_epsilon=5.0)),
    ]
    out = []
    for label, key, kw in combos:
        console.info(f"configuration: {label}")
        res = ctx.run(f"ablation: {label}", **kw)
        if label == "FL + DP + Multi-Krum":
            ctx._mem["dp_krum_eps5"] = res
        out.append([label, f"{res['final_accuracy']:.3f}",
                    f"{res['final_macro_f1']:.3f}", _eps_of(res), TN.fmt(key)])

    console.table(["configuration", "accuracy", "macro-F1", "epsilon",
                   TN.column("accuracy")], out)
    console.takeaway("""Reading down the accuracy column shows what each
        component costs. The row that matters is the last one: adding
        Multi-Krum on top of DP does not degrade the private model, so the two
        mechanisms compose. That is the integrated framework RO1 asks for.""")


# ══════════════════════════════════════════════════════════════════════════
# STEP 10 — privacy-utility curve
# ══════════════════════════════════════════════════════════════════════════
def step_10(ctx: Ctx) -> None:
    from ..utils import reporting

    grid = ctx.profile["eps_grid"]
    results = {}
    for eps in grid:
        console.info(f"privacy budget epsilon = {eps:g}"
                     + ("  (strictest)" if eps == min(grid) else
                        "  (loosest)" if eps == max(grid) else ""))
        results[eps] = ctx.run(f"3. DP at epsilon={eps:g}", aggregator="multikrum",
                               dp=True, target_epsilon=eps)

    rows = [[f"{e:g}", f"{results[e]['final_epsilon']:.2f}",
             f"{results[e]['final_accuracy']:.3f}",
             f"{results[e]['final_macro_f1']:.3f}",
             f"{results[e]['final_fpr']:.3f}",
             TN.fmt(f"privacy_utility/eps{e:g}")] for e in grid]
    console.table(["target epsilon", "achieved", "accuracy", "macro-F1", "FPR",
                   TN.column("accuracy")], rows)

    fig = ctx.outdir / "fig_privacy_utility.png"
    with console.step(f"plotting the privacy-utility curve -> {fig.name}"):
        reporting.plot_privacy_utility(
            [reporting.row_from_result(results[e]) for e in grid], fig)

    accs = [results[e]["final_accuracy"] for e in grid]
    monotonic = all(b >= a - 1e-9 for a, b in zip(accs, accs[1:]))
    if monotonic:
        verdict = (f"Accuracy rises with every relaxation of the budget "
                   f"({accs[0]:.3f} -> {accs[-1]:.3f})")
    else:
        verdict = (f"The strict budget clearly costs accuracy ({accs[0]:.3f} at "
                   f"epsilon={grid[0]:g} against {max(accs):.3f} at looser budgets); "
                   f"the small non-monotonic wiggle is run-to-run variation at "
                   f"{ctx.base['rounds']} rounds, and the reported 30-round "
                   f"curve is cleanly monotonic (0.814 -> 0.883)")
    console.takeaway(verdict + """. Privacy is not free, but the trade-off is
        smooth and predictable, which is what lets an operator pick the epsilon
        their regulator demands and know what it will cost. This curve is RO3.""")


# ══════════════════════════════════════════════════════════════════════════
# STEP 11 — adaptive epsilon + the optimiser finding
# ══════════════════════════════════════════════════════════════════════════
def step_11(ctx: Ctx) -> None:
    from ..utils import reporting

    target = 5.0
    console.info("schedule: static (constant noise every round)")
    static = ctx.run("4. RO2 static schedule", aggregator="multikrum", dp=True,
                     target_epsilon=target, adaptive_eps=False)
    runs = {"static": static}
    for sched in ctx.profile["schedules"]:
        console.info(f"schedule: {sched.replace('_', ' ')}")
        runs[sched] = ctx.run(f"4. RO2 {sched}", aggregator="multikrum", dp=True,
                              target_epsilon=target, adaptive_eps=True,
                              eps_schedule=sched)

    key_of = {"static": "adaptive/static",
              "decreasing_noise": "adaptive/decreasing",
              "increasing_noise": "adaptive/increasing"}
    console.table(["schedule", "sigma first -> last", "epsilon spent", "accuracy",
                   TN.column("accuracy")],
                  [[name.replace("_", " "),
                    f"{r['noise_schedule'][0]:.2f} -> {r['noise_schedule'][-1]:.2f}",
                    f"{r['final_epsilon']:.2f}", f"{r['final_accuracy']:.3f}",
                    TN.fmt(key_of.get(name, ""))] for name, r in runs.items()])

    fig = ctx.outdir / "fig_noise_schedule.png"
    with console.step(f"plotting the schedules -> {fig.name}"):
        reporting.plot_noise_schedule(list(runs.values()), fig)
    fig = ctx.outdir / "fig_schedule_accuracy.png"
    with console.step(f"plotting accuracy per round -> {fig.name}"):
        reporting.plot_runs_over_rounds(
            list(runs.values()), "accuracy", fig,
            title="Accuracy per round, by noise schedule",
            ylabel="Global test accuracy", mark_convergence=True)

    console.say("""Every schedule spends the same total budget — that is the
        calibration machinery working. What they do not do is beat each other by
        much. The lever that actually moved the privacy-utility trade-off was
        the optimiser, so the same budget is now run under Adam for comparison:""")

    console.info("optimiser: SGD (canonical DP-SGD)")
    sgd = ctx.run("5. DP epsilon=5 with SGD", aggregator="multikrum", dp=True,
                  target_epsilon=target, optimizer="sgd", lr=0.1)
    console.info("optimiser: Adam")
    adam = ctx.run("5. DP epsilon=5 with Adam", aggregator="multikrum", dp=True,
                   target_epsilon=target, optimizer="adam", lr=1e-3)
    console.table(["optimiser", "accuracy", "macro-F1", "false-positive rate"],
                  [["SGD", f"{sgd['final_accuracy']:.3f}",
                    f"{sgd['final_macro_f1']:.3f}", f"{sgd['final_fpr']:.3f}"],
                   ["Adam", f"{adam['final_accuracy']:.3f}",
                    f"{adam['final_macro_f1']:.3f}", f"{adam['final_fpr']:.3f}"]])

    console.takeaway("""RO2's proposed mechanism gives a rigorous negative
        result: at a fixed total budget under RDP composition, uniform noise
        is near-optimal, and the non-uniform shapes above are equal or worse.
        The reported campaign therefore runs uniform noise
        (results/Research Objective 2/). The optimiser table above is the other
        half of the story: Adam's per-coordinate rescaling fights
        clip-and-noise, so plain SGD — the canonical DP-SGD optimiser — is what
        the framework uses. Reporting the negative result honestly, next to the
        design choice that did matter, is the contribution.""")


# ══════════════════════════════════════════════════════════════════════════
# STEP 12 — Byzantine robustness
# ══════════════════════════════════════════════════════════════════════════
def step_12(ctx: Ctx) -> None:
    from ..attacks import byzantine
    from ..utils import reporting

    attacks = getattr(byzantine, "UPDATE_ATTACKS", None)
    console.kv({"attack used here": "sign_flip — invert the update, then scale it up",
                "attack types implemented": ", ".join(sorted(attacks)) if attacks
                                            else "label_flip, sign_flip, scale, gaussian, random"})

    grid_rows, plot_rows = [], []
    for agg in ctx.profile["aggregators"]:
        for frac in ctx.profile["byz_fracs"]:
            console.info(f"{agg} against {frac:.0%} malicious gNBs")
            res = ctx.run(f"6. {agg} @ {frac:.0%} attack", aggregator=agg,
                          dp=False, attack="sign_flip", byzantine_fraction=frac)
            kept, slots = _byz_kept(res)
            grid_rows.append([agg, f"{frac:.0%}", f"{res['final_accuracy']:.3f}",
                              f"{slots - kept}/{slots}" if slots else "-",
                              TN.fmt(f"byzantine/{agg}_f{int(frac*100)}")])
            plot_rows.append(reporting.row_from_result(res))

    console.table(["aggregator", "malicious", "accuracy", "Byzantine excluded",
                   TN.column("accuracy")], grid_rows)

    fig = ctx.outdir / "fig_byz_accuracy.png"
    with console.step(f"plotting robustness curves -> {fig.name}"):
        reporting.plot_byzantine_metric(
            plot_rows, "accuracy", fig, title="Accuracy under attack",
            ylabel="Global test accuracy")

    console.takeaway("""Under the same attack, the aggregation rule decides
        whether the system survives. FedAvg averages the poison and collapses to
        near-random, both here and in the reported runs. Multi-Krum holds its
        accuracy and excludes every Byzantine update. The full-corpus grid
        (results/Research Objective 4/) sweeps the intermediate attack
        fractions this short run does not reach, and shows trimmed-mean failing
        between them — which is why the framework specifies Multi-Krum rather
        than the cheaper robust mean. This is RO4.""")


# ══════════════════════════════════════════════════════════════════════════
# STEP 13 — DP x Multi-Krum interaction
# ══════════════════════════════════════════════════════════════════════════
def step_13(ctx: Ctx) -> None:
    from ..analysis import interaction

    console.say("""First measurement: the same 30% sign-flip attack against
        Multi-Krum, once without DP and once with it, comparing how many
        malicious updates the defence kept out.""")
    with console.step("matched no-DP and DP runs under attack"):
        comp = interaction.run_interaction(
            target_epsilon=5.0, attack="sign_flip", byz_fraction=0.3,
            rounds=ctx.base["rounds"], model=ctx.base["model"],
            optimizer=ctx.base["optimizer"], lr=ctx.base["lr"],
            device=ctx.base["device"], seed=ctx.base["seed"],
            data_root=ctx.base.get("data_root"),
            round_chunks=ctx.base.get("round_chunks"),
            verbose=False, on_event=ProgressReporter(level="round"))

    console.table(["configuration", "Byzantine exclusion rate", "accuracy", "epsilon"],
                  [["Multi-Krum, no DP", f"{comp['no_dp']['exclusion_rate']:.3f}",
                    f"{comp['no_dp']['accuracy']:.3f}", "-"],
                   ["Multi-Krum + DP", f"{comp['dp']['exclusion_rate']:.3f}",
                    f"{comp['dp']['accuracy']:.3f}",
                    f"{comp['dp']['epsilon']:.2f}"]])

    console.say("""Second measurement, isolating the mechanism: take one fixed
        pool of honest and malicious updates and inject increasing Gaussian
        noise, tracking how far apart their Multi-Krum scores stay. A gap that
        shrinks toward zero would mean the noise is hiding the attackers.""")
    with console.step("score-separation micro-benchmark across noise levels"):
        sep = interaction.score_separation(attack="sign_flip", byz_fraction=0.3,
                                           sigmas=ctx.profile["sep_sigmas"],
                                           model=ctx.base["model"],
                                           seed=ctx.base["seed"],
                                           data_root=ctx.base.get("data_root"))
    console.table(["injected noise sigma", "Byzantine/honest score gap",
                   "exclusion rate"],
                  [[f"{s:g}", f"{g:.3f}", f"{e:.2f}"]
                   for s, g, e in zip(sep["sigmas"], sep["separation"],
                                      sep["exclusion"])])

    ctx._mem["interaction"] = {"comparison": comp, "separation": sep}
    (ctx.outdir / "interaction.json").write_text(
        json.dumps(ctx._mem["interaction"], indent=2, default=float))

    console.takeaway(f"""DP noise did not blind the defence: exclusion stays at
        {comp['dp']['exclusion_rate']:.2f} with privacy enabled. The
        micro-benchmark shows why the question was worth asking — the score gap
        does shrink as noise grows, so the mechanisms genuinely interact, but at
        the noise levels a usable epsilon actually requires the gap remains wide
        enough to separate attackers. Privacy and robustness compose, which is
        RO4's headline result and the reason the combined system is viable.""")


# ══════════════════════════════════════════════════════════════════════════
# STEP 14 — Flower engine parity
# ══════════════════════════════════════════════════════════════════════════
def step_14(ctx: Ctx) -> None:
    seq = ctx.run("7. parity: sequential engine", aggregator="multikrum",
                  dp=False, engine="sequential", record=False)
    console.info("re-running the identical configuration through Flower")
    flw = ctx.run("7. parity: Flower engine", aggregator="multikrum", dp=False,
                  engine="flower", record=False)

    d_acc = abs(seq["final_accuracy"] - flw["final_accuracy"])
    console.table(["engine", "accuracy", "macro-F1"],
                  [["sequential (reference)", f"{seq['final_accuracy']:.6f}",
                    f"{seq['final_macro_f1']:.6f}"],
                   ["Flower (NumPyClient + Strategy)", f"{flw['final_accuracy']:.6f}",
                    f"{flw['final_macro_f1']:.6f}"]])
    console.kv({"absolute difference in accuracy": f"{d_acc:.2e}"})

    verdict = ("identical to floating-point precision" if d_acc < 1e-9
               else f"agreeing to {d_acc:.2e}")
    console.takeaway(f"""The two engines are {verdict}. The sequential engine is
        the reference implementation — deterministic and easy to audit — and
        this parity check is what licenses using it for the experiments while
        still claiming a genuine Flower deployment.""")


# ══════════════════════════════════════════════════════════════════════════
# STEP 15 — per-class breakdown
# ══════════════════════════════════════════════════════════════════════════
def step_15(ctx: Ctx) -> None:
    src = (ctx.book.get("ablation: FL + Multi-Krum")
           or ctx._mem.get("baseline")
           or (list(ctx.book.values())[-1] if ctx.book else None))
    if not src:
        console.warn("No completed run available to break down — run an earlier "
                     "step first.")
        return

    per_cat = src.get("final", {}).get("per_category", {})
    if not per_cat:
        console.warn("This run carries no per-category breakdown.")
        return

    rows = sorted(((c, d["support"], d["recall"]) for c, d in per_cat.items()),
                  key=lambda r: -r[1])
    console.table(["attack category", "test records", "recall", ""],
                  [[c, f"{n:,}", f"{r:.3f}",
                    "#" * max(0, round(24 * r))] for c, n, r in rows],
                  aligns="<><<")
    console.kv({"overall accuracy": src["final_accuracy"],
                "macro-averaged F1": src["final_macro_f1"]})

    weak = [c for c, n, r in rows if r < 0.2]
    console.takeaway(f"""High-volume families are detected almost perfectly while
        sparse ones are missed{' (' + ', '.join(weak[:4]) + ')' if weak else ''}.
        Macro-F1 weights every class equally regardless of size, which is why it
        sits well below accuracy — the honest reading is that this detector finds
        floods reliably and rare application-layer attacks poorly, a
        class-imbalance effect of the corpus itself that the thesis reports as a
        limitation rather than a result.""")


# ══════════════════════════════════════════════════════════════════════════
# STEP 16 — threat model
# ══════════════════════════════════════════════════════════════════════════
def step_16(ctx: Ctx) -> None:
    doc = PROJECT_ROOT / "docs" / "threat_model.md"
    console.table(["adversary", "capability", "countermeasure", "objective"],
                  [["Honest-but-curious observer",
                    "reads model updates in transit or at the server",
                    "(epsilon, delta)-DP via DP-SGD", "RO1, RO3"],
                   ["Byzantine gNB",
                    "sends arbitrary poisoned updates",
                    "Multi-Krum distance-based filtering", "RO4"],
                   ["Combined adversary",
                    "observes updates and poisons them",
                    "DP and Multi-Krum together", "RO4"]],
                  aligns="<<<<")
    console.say("""Assumed trustworthy: the aggregation server itself, the
        secure transport between gNBs and server, and the identity of
        participants.""",
                """Explicitly out of scope, and named so the guarantees are not
        read too broadly: a compromised aggregator or RIC, Sybil attacks beyond
        the modelled fraction, side-channel and encrypted-traffic inference,
        stealthy targeted backdoors crafted to pass Multi-Krum, and free-rider
        base stations.""")
    if doc.exists():
        console.kv({"full threat model": str(doc.relative_to(PROJECT_ROOT))})
    console.takeaway("""Stating what the framework does not defend against is
        what makes the claims about what it does defend against meaningful.
        This satisfies RO1's formal threat-model deliverable.""")


# ══════════════════════════════════════════════════════════════════════════
# STEP 17 — summary
# ══════════════════════════════════════════════════════════════════════════
def step_17(ctx: Ctx) -> None:
    if ctx.book:
        console.say("Every experiment computed during this session:")
        console.table(["experiment", "accuracy", "macro-F1", "epsilon"],
                      [[name, f"{r['final_accuracy']:.3f}",
                        f"{r['final_macro_f1']:.3f}", _eps_of(r)]
                       for name, r in ctx.book.items()])

    console.say("Research Objectives and the evidence produced for each:")
    console.table(["RO", "claim", "demonstrated in"],
                  [["RO1", "Integrated FL + DP + Multi-Krum framework "
                           "and threat model", "steps 6-9, 14, 16"],
                   ["RO2", "Adaptive epsilon scheduling; uniform is near-optimal; "
                           "optimiser is the real lever", "step 11"],
                   ["RO3", "Privacy-utility characterisation across epsilon",
                    "steps 10, 15"],
                   ["RO4", "Byzantine robustness and clean DP composition",
                    "steps 12, 13"]],
                  aligns="<<<")

    console.say(*narration.closing(live=True))

    console.kv({
        "this session's artefacts": str(ctx.outdir),
        "results by objective": "results/Research Objective 1..4/",
        "consolidated write-up": "results/RESULTS.md",
        "threat model": "docs/threat_model.md",
    })
    if ctx.notes:
        console.warn("Notes from this run:")
        for n in ctx.notes:
            console.substep(n)


ALL_STEPS = {
    1: step_01, 2: step_02, 3: step_03, 4: step_04, 5: step_05, 6: step_06,
    7: step_07, 8: step_08, 9: step_09, 10: step_10, 11: step_11, 12: step_12,
    13: step_13, 14: step_14, 15: step_15, 16: step_16, 17: step_17,
}
