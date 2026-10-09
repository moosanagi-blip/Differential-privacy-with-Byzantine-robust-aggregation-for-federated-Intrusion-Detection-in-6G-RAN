"""Read-only audit of raw runs; writes a separate numerical-integrity report."""
import json
import math
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "results" / "revision_full_corpus"


def issues_for_run(run):
    issues = []
    for t in run.get("timeline", []):
        scores = t.get("scores")
        fields = []
        if scores is not None and any(not math.isfinite(s) for s in scores):
            fields.append("scores")
        for key in ("loss", "accuracy", "macro_f1"):
            if key in t and not math.isfinite(t[key]):
                fields.append(key)
        if fields:
            issues.append(dict(round=t["round"], nonfinite_fields=fields))
    return issues


def main():
    rows = []
    for path in sorted((OUT / "runs").glob("*.json")):
        run = json.loads(path.read_text(encoding="utf-8"))
        rows.append(dict(file=path.name, issues=issues_for_run(run)))
    affected = [r for r in rows if r["issues"]]
    report = dict(runs_checked=len(rows), affected_runs=len(affected), rows=rows,
                  caveat="Finite final scalar metrics do not prove finite logits or valid model parameters. Raw results unchanged.")
    (OUT / "NUMERICAL_INTEGRITY_AUDIT.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    lines = ["# Numerical integrity audit", "",
             f"Checked {len(rows)} saved runs; {len(affected)} contain non-finite timeline values.",
             "These runs require failure-aware re-evaluation. Do not interpret their final scalar metrics or post-failure admissions as valid model/geometry measurements.",
             "Raw files are preserved. No retraining was performed. Absence of a recorded failure is not a check of unsaved logits or model tensors.", "",
             "| Run | First affected round | Affected round count | Fields |", "|---|---:|---:|---|"]
    for r in affected:
        fields = sorted({f for i in r["issues"] for f in i["nonfinite_fields"]})
        lines.append(f"| {r['file']} | {r['issues'][0]['round']} | {len(r['issues'])} | {', '.join(fields)} |")
    (OUT / "NUMERICAL_INTEGRITY_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    lookup = {r["file"]: r["issues"] for r in rows}
    conditions = [("Adaptive static", "adaptive_static"), ("Adaptive rotating", "adaptive_rotating")]
    conditions += [(f"Configured $f={f}$", f"f_estimate_{f}") for f in range(5)]
    conditions += [("Radio retained", "ran_features_retained"), ("Radio masked", "ran_features_masked")]
    tex = [r"% Generated from raw timelines; failure rounds are first recorded non-finite values.",
           r"\begin{table}[H]", r"\centering", r"\small",
           r"\caption{\rev{Numerical-integrity audit of the 27 saved runs. Entries give the first recorded non-finite loss or score; `none' means none was recorded in 30 rounds, not a retrospective check of unsaved tensors.}}",
           r"\label{tab:revision-numerical-audit}", r"{\color{blue}",
           r"\begin{tabular}{@{}lrrrr@{}}", r"\toprule",
           r"Condition & Seed 42 & Seed 43 & Seed 44 & Affected runs \\", r"\midrule"]
    for label, name in conditions:
        cells = [lookup[f"{name}__seed{s}.json"] for s in (42, 43, 44)]
        values = [str(c[0]["round"]) if c else "none" for c in cells]
        tex.append(label + " & " + " & ".join(values) + f" & {sum(bool(c) for c in cells)}/3 " + r"\\")
    tex += [r"\bottomrule", r"\end{tabular}", "}", r"\end{table}"]
    generated = OUT.parents[2] / "paper" / "generated"
    generated.mkdir(parents=True, exist_ok=True)
    (generated / "revision_numerical_audit.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
