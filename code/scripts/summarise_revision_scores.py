"""Descriptive Krum-score quantiles from existing non-private full-corpus runs.

No training. Normalize each round by its honest median score, compute pooled
client-round quantiles within each seed, then summarize across three seeds.
Round/client observations are dependent; quantile ranges are not confidence
intervals, and the sample standard deviation uses only the three seed summaries.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from audit_revision_numerics import issues_for_run

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "revision_full_corpus"
PAPER = ROOT.parent / "paper"
SEEDS = (42, 43, 44)
CONDITIONS = ("adaptive_static", "adaptive_rotating", *(f"f_estimate_{f}" for f in range(5)))
LABELS = ("Static", "Rotating", *(f"f={f}" for f in range(5)))


def analyse(run):
    cfg = run["config"]
    assert cfg["dp"] is False and cfg["num_clients"] == 10
    assert run["dataset"]["eval_rows"] == 4673255
    assert len(run["timeline"]) == 30
    groups = {"honest": [], "malicious": []}
    admitted = 0
    for expected_round, t in enumerate(run["timeline"], 1):
        assert t["round"] == expected_round
        # The verified campaign uses all clients in ascending ID order.
        scores = np.asarray(t["scores"], dtype=float)
        assert scores.shape == (10,) and np.all(np.isfinite(scores)) and np.all(scores >= 0)
        bad = set(t["byzantine_clients"])
        assert len(bad) == 3 and bad <= set(range(10))
        chosen = set(t["selected"])
        assert len(chosen) == len(t["selected"]) and chosen <= set(range(10))
        assert set(t["byz_selected"]) == chosen & bad
        # Check selected clients really are among the lowest scores; allow ties.
        rejected = set(range(10)) - chosen
        if rejected:
            assert max(scores[list(chosen)]) <= min(scores[list(rejected)]) + 1e-9
        h = scores[[i for i in range(10) if i not in bad]]
        b = scores[sorted(bad)]
        denominator = float(np.median(h))
        assert denominator > 0, "Cannot normalize a zero honest median"
        groups["honest"].extend((h / denominator).tolist())
        groups["malicious"].extend((b / denominator).tolist())
        admitted += len(chosen & bad)
    assert len(groups["honest"]) == 210 and len(groups["malicious"]) == 90
    return dict(seed=run["revision_evidence"]["seed"], admitted=admitted,
                normalized_scores=groups,
                quantiles={g: np.quantile(v, [.1, .5, .9], method="linear").tolist()
                           for g, v in groups.items()})


def main():
    failures = []
    for condition in CONDITIONS:
        for seed in SEEDS:
            path = OUT / "runs" / f"{condition}__seed{seed}.json"
            issues = issues_for_run(json.loads(path.read_text(encoding="utf-8")))
            if issues:
                failures.append(f"{path.name}: first non-finite round {issues[0]['round']}")
    if failures:
        raise RuntimeError("Score export blocked by numerical failures; run audit_revision_numerics.py.\n" + "\n".join(failures))
    rows = []
    for condition in CONDITIONS:
        seeds = []
        for seed in SEEDS:
            run = json.loads((OUT / "runs" / f"{condition}__seed{seed}.json").read_text(encoding="utf-8"))
            assert run["config"]["name"] == condition and run["revision_evidence"]["seed"] == seed
            seeds.append(analyse(run))
        summary = {}
        for group in ("honest", "malicious"):
            matrix = np.array([r["quantiles"][group] for r in seeds])
            summary[group] = dict(mean=matrix.mean(axis=0).tolist(), sample_sd=matrix.std(axis=0, ddof=1).tolist())
        rows.append(dict(condition=condition, per_seed=seeds, summary=summary))
    payload = dict(normalization="score / median honest score in the same round",
                   quantiles=[.1, .5, .9], quantile_method="linear",
                   scope="21 non-private runs; 30 rounds; 10 clients; fixed data partition",
                   interpretation="Descriptive client-round distributions, not confidence intervals or independent observations",
                   limitation="No saved update vectors: no displacement-to-honest-spread measurement",
                   rows=rows)
    (OUT / "full_corpus_score_quantiles.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tex = [r"% Generated from saved per-round scores by summarise_revision_scores.py.",
           r"\begin{table}[H]", r"\centering", r"\small",
           r"\caption{\rev{Normalized Krum-score quantiles (10th / 50th / 90th percentile). Each score is divided by its round's honest median. Quantiles are computed within each seed over 30 rounds, then averaged across three seeds; ranges are descriptive, not confidence intervals.}}",
           r"\label{tab:revision-score-quantiles}", r"{\color{blue}",
           r"\begin{tabular}{@{}lll@{}}", r"\toprule",
           r"Condition & Honest & Malicious \\", r"\midrule"]
    for label, row in zip(LABELS, rows):
        triplets = [" / ".join(f"{v:.3f}" for v in row["summary"][g]["mean"]) for g in ("honest", "malicious")]
        tex.append(f"{label} & {triplets[0]} & {triplets[1]} " + r"\\")
    tex += [r"\bottomrule", r"\end{tabular}", "}", r"\end{table}"]
    (PAPER / "generated").mkdir(exist_ok=True)
    (PAPER / "generated" / "revision_score_quantiles.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    for index, row in enumerate(rows):
        for group, offset, color in (("honest", -.13, "#2166ac"), ("malicious", .13, "#b2182b")):
            low, middle, high = row["summary"][group]["mean"]
            ax.plot([low, high], [index+offset]*2, color=color, linewidth=2)
            ax.plot(middle, index+offset, "o", color=color, label=group.capitalize() if index == 0 else None)
    ax.set_yticks(range(len(rows)), LABELS)
    ax.invert_yaxis()
    ax.axvline(1, color="gray", linestyle="--", alpha=.6)
    ax.set_xlabel("Score / round's honest median (lower scores are selected first)")
    ax.set_title("Krum-score overlap under adaptive poisoning")
    ax.grid(axis="x", alpha=.2)
    ax.legend(loc="best")
    fig.text(.5, .015, "Dots: mean seed median. Lines: mean seed 10th-90th percentiles (not confidence intervals).", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .04, 1, 1))
    for folder in (OUT / "figures", PAPER / "figures" / "revision_full_corpus"):
        folder.mkdir(parents=True, exist_ok=True)
        fig.savefig(folder / "fig_full_score_quantiles.pdf", bbox_inches="tight")
        fig.savefig(folder / "fig_full_score_quantiles.png", dpi=400, bbox_inches="tight")
    plt.close(fig)
    print("Verified 21 runs, 630 rounds and 6,300 score observations.")
    print("\n".join(tex))


if __name__ == "__main__":
    main()
