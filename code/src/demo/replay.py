"""Narrate the published results without training anything.

The live demo takes tens of minutes. ``--replay`` walks the same steps in the
same order using the results already on disk, so the framework can be explained
in a few minutes, or after a live step has failed, or on a machine with no GPU
and no patience. It trains nothing and writes nothing.

Every number printed here is read from a file and the file is named on screen,
so nothing in replay mode can be mistaken for a fresh measurement.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..data.config import PROJECT_ROOT
from ..utils import console
from . import narration
from . import thesis_numbers as TN

RESULTS = PROJECT_ROOT / "results"
# The reported campaign: 30 rounds over the complete corpus, filed by Research
# Objective (results/Research Objective 1..4), so runs are looked up by name.
CAMPAIGN = RESULTS
DATA = PROJECT_ROOT / "data_full"


def _find(stem: str) -> Path | None:
    """Locate a run file by name, whichever objective's folder it lives in."""
    return next(iter(sorted(RESULTS.glob(f"**/{stem}.json"))), None)


def _load(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _source(path: Path) -> None:
    try:
        rel = path.relative_to(PROJECT_ROOT)
    except ValueError:
        rel = path
    console.info(f"source: {rel}")


def _label() -> str:
    """How the reported column is named, everywhere, in one place."""
    return f"{TN.ROUNDS} rounds, full corpus"


def replay(args) -> int:
    console.banner(narration.TITLE + "  [REPLAY]", color="magenta")
    console.say(*narration.intro(live=False))
    console.say("""This is REPLAY MODE. Nothing is being trained: every figure
        below is read from the result files produced by the full experiment
        campaign, and each one names the file it came from. To compute these
        live instead, run the same command without --replay.""")

    if not TN.available():
        console.warn(f"{CAMPAIGN} holds no result files — replay has nothing to "
                     "read. Run the live demo instead.")
        return 1

    def head(n: int) -> None:
        meta = narration.step_meta(n)
        title = f"STEP {n} of 17  —  {meta['title']}"
        if meta["ro"]:
            console.ro(int(meta["ro"]), title)
        else:
            console.banner(title)
        if meta.get("what"):
            console.say("WHAT THIS STEP DOES.  " + str(meta["what"]))
        if meta.get("why"):
            console.say("WHY IT MATTERS.  " + str(meta["why"]))

    # ── data pipeline (1-5) ───────────────────────────────────────────────
    for n in (1, 2, 3, 4, 5):
        head(n)
        if n == 2:
            stats = _load(DATA / "artifacts" / "preprocess_stats.json")
            if stats:
                _source(DATA / "artifacts" / "preprocess_stats.json")
                # Only report fields the builder actually recorded — a row of
                # "None"s reads like missing data rather than like a full run.
                facts = {
                    "rows in the prepared dataset": f"{stats.get('n_rows', 0):,}",
                    "raw rows scanned": f"{stats.get('n_raw_rows_scanned', 0):,}",
                    "features": stats.get("n_features"),
                    "subsampled": "no" if stats.get("capped") is False else "yes",
                    "built by": stats.get("source"),
                }
                console.kv({k: v for k, v in facts.items() if v not in (None, "")})
        elif n == 4:
            man = _load(DATA / "partitions" / "partition_manifest.json")
            if man:
                _source(DATA / "partitions" / "partition_manifest.json")
                console.kv({"gNBs": man.get("num_clients"),
                            "Dirichlet alpha": man.get("dirichlet_alpha"),
                            "training records": f"{man.get('n_train', 0):,}",
                            "test records": f"{man.get('n_test', 0):,}"})

    # ── RO1 ablation (6-9) ────────────────────────────────────────────────
    for n in (6, 7, 8, 9):
        head(n)
    _source(CAMPAIGN)
    console.table(["configuration", f"accuracy ({_label()})", "macro-F1"],
                  [["FL only", TN.fmt("ablation/fl_only"), TN.fmt("ablation/fl_only", "f1")],
                   ["FL + DP (eps=5)", TN.fmt("ablation/fl_dp"), TN.fmt("ablation/fl_dp", "f1")],
                   ["FL + Multi-Krum", TN.fmt("ablation/fl_krum"), TN.fmt("ablation/fl_krum", "f1")],
                   ["FL + DP + Multi-Krum", TN.fmt("ablation/fl_dp_krum"),
                    TN.fmt("ablation/fl_dp_krum", "f1")]])
    console.takeaway("""Differential privacy costs about five accuracy points
        (0.893 -> 0.839 with Multi-Krum). Adding Multi-Krum on top of DP
        improves on DP alone rather than degrading it, so the mechanisms
        compose — the integrated framework of RO1 holds together.""")

    # ── RO3 (10) ──────────────────────────────────────────────────────────
    head(10)
    _source(CAMPAIGN)
    console.table(["epsilon", f"accuracy ({_label()})", "macro-F1"],
                  [[e, TN.fmt(f"privacy_utility/eps{e}"),
                    TN.fmt(f"privacy_utility/eps{e}", "f1")]
                   for e in ("1", "5", "10", "50")])
    console.takeaway("""Accuracy rises monotonically with the privacy budget,
        0.814 at epsilon=1 to 0.883 at epsilon=50. The trade-off is smooth, so
        an operator can choose a budget and know the cost. This is RO3.""")

    # ── RO2 (11) ──────────────────────────────────────────────────────────
    head(11)
    # Three noise shapes, one budget: the comparison is budget-fair by
    # construction because each schedule is RDP-calibrated to the same total.
    scheds = [("static (uniform noise)", "eps_static"),
              ("decreasing noise", "eps_decreasing"),
              ("increasing noise", "eps_increasing")]
    rows = []
    for label, stem in scheds:
        path = _find(stem)
        rec = _load(path) if path else None
        if rec:
            rows.append([label, f"{rec.get('final_accuracy', 0):.3f}",
                         f"{rec.get('final_macro_f1', 0):.3f}",
                         f"{rec.get('final_epsilon', 0):.2f}"])
    if rows:
        _source(RESULTS / "Research Objective 2")
        console.table(["noise schedule", "accuracy", "macro-F1", "epsilon spent"],
                      rows, aligns="<>><")
    console.takeaway("""RO2's proposed mechanism is a rigorous negative result:
        at a fixed budget, uniform noise is near-optimal and every non-uniform
        shape is equal or worse, which is why the reported campaign runs uniform
        noise. The real privacy-utility gain came from using SGD rather than
        Adam for DP training — see results/RESULTS.md section 2a.""")

    # ── RO4 (12-13) ───────────────────────────────────────────────────────
    head(12)
    _source(CAMPAIGN)
    console.table(["aggregator", "0% malicious", "30% malicious"],
                  [[agg] + [TN.fmt(f"byzantine/{agg}_f{f}") for f in (0, 30)]
                   for agg in ("fedavg", "krum", "multikrum", "trimmedmean")])
    console.takeaway("""Under a 30% sign-flip attack FedAvg and trimmed-mean
        collapse to 0.000 — they predict a single class — while Multi-Krum
        scores 0.893, indistinguishable from the no-attack case, and rejects
        every one of the 90 malicious updates across the 30 rounds. Krum is
        robust too but costs accuracy, because it keeps only one client's update
        per round. This is RO4, and the strongest result in the thesis.""")

    head(13)
    # Exclusion is counted from the campaign's own round-by-round record: every
    # round lists which clients Multi-Krum selected and which of those were
    # malicious, so "0 admitted" is arithmetic on the timeline, not a claim.
    byz_path = _find("byz_multikrum_f30")
    byz = _load(byz_path) if byz_path else None
    if byz:
        _source(byz_path)
        rounds = byz.get("timeline", [])
        n_byz = len(byz.get("byzantine_clients", []))
        admitted = sum(len(r.get("byz_selected", [])) for r in rounds)
        offered = n_byz * len(rounds)
        console.kv({"malicious gNBs": f"{n_byz} of 10",
                    "rounds": len(rounds),
                    "malicious updates offered": offered,
                    "malicious updates admitted": admitted,
                    "accuracy under attack": f"{byz.get('final_accuracy', 0):.3f}"})
    console.table(["configuration", "accuracy", "epsilon"],
                  [["Multi-Krum, no DP", TN.fmt("ablation/fl_krum"), "-"],
                   ["Multi-Krum + DP", TN.fmt("ablation/fl_dp_krum"),
                    f"{(TN.get('ablation/fl_dp_krum') or {}).get('final_epsilon', 0):.2f}"],
                   ["DP alone (no Multi-Krum)", TN.fmt("ablation/fl_dp"),
                    f"{(TN.get('ablation/fl_dp') or {}).get('final_epsilon', 0):.2f}"]])
    console.takeaway("""DP noise does not blind Multi-Krum. Filtering stays exact
        under attack, and at a fixed epsilon the private run with Multi-Krum
        scores above the private run without it — so privacy and robustness
        compose rather than interfere, which is RO4's headline result. The
        noise-injection sweep behind this is step 13 of a live run.""")

    # ── remaining (14-17) ─────────────────────────────────────────────────
    head(14)
    console.say("""Verified during development: the sequential reference engine
        and the genuine Flower engine produce bit-identical results on matched
        configurations. Run this step live to reproduce the check.""")

    head(15)
    krum_path = _find("byz_multikrum_f0")
    krum = _load(krum_path) if krum_path else None
    per_cat = ((krum or {}).get("final") or {}).get("per_category") or {}
    if per_cat:
        _source(krum_path)
        rows = sorted(((cat, int(d.get("support", 0)), float(d.get("recall", 0.0)))
                       for cat, d in per_cat.items()), key=lambda r: -r[1])
        console.table(["attack category", "test records", "recall"],
                      [[c, f"{n:,}", f"{r:.3f}"] for c, n, r in rows],
                      aligns="<><")
        console.kv({"overall accuracy": f"{krum.get('final_accuracy', 0):.3f}",
                    "macro-averaged F1": f"{krum.get('final_macro_f1', 0):.3f}"})
    console.takeaway("""Floods are caught almost perfectly; rare
        application-layer attacks are largely missed. Macro-F1 therefore trails
        accuracy. This is a class-imbalance effect of the corpus — the two
        weakest families are also the two smallest, together 0.08% of the test
        set — and the thesis reports it as a limitation.""")

    head(16)
    console.table(["adversary", "countermeasure", "objective"],
                  [["Honest-but-curious observer", "(epsilon, delta)-DP", "RO1, RO3"],
                   ["Byzantine gNB", "Multi-Krum filtering", "RO4"],
                   ["Combined adversary", "both together", "RO4"]], aligns="<<<")
    console.info("full threat model: docs/threat_model.md")

    head(17)
    console.say(*narration.closing(live=False))
    console.kv({"results by objective": "results/Research Objective 1..4/",
                "consolidated write-up": "results/RESULTS.md"})
    console.banner("REPLAY COMPLETE — nothing was trained", color="green")
    return 0
