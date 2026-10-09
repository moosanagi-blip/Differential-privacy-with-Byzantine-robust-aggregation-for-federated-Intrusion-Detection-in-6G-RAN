#!/usr/bin/env python3
"""
Rebuild results_table.csv + SUMMARY.md for a results directory from its per-run
JSON files. Useful when a campaign is assembled from multiple partial runs (or
crashed before the final reporting step) — the per-run JSONs are the source of
truth and always written as each run finishes.

Run:  python scripts/rebuild_summary.py results --notes "full-corpus campaign"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.config import PROJECT_ROOT  # noqa: E402
from src.utils import reporting  # noqa: E402

# stable block ordering for a tidy table
_ORDER = ["ablation", "pu_", "byz_", "eps_"]


def _rank(name: str) -> tuple[int, str]:
    for i, pre in enumerate(_ORDER):
        if name.startswith(pre):
            return i, name
    return len(_ORDER), name


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--notes", default="")
    args = ap.parse_args()
    # results/ lives beside code/, so a relative path is resolved against the
    # project root — otherwise running from code/ silently writes an empty
    # summary into a new code/results/ directory.
    d = Path(args.dir)
    if not d.is_absolute():
        d = PROJECT_ROOT / d

    rows = []
    for jf in sorted(d.glob("**/*.json")):   # runs live in per-objective folders
        if jf.name in ("interaction.json",):
            continue
        res = json.loads(jf.read_text(encoding="utf-8"))
        if "config" in res and "final_accuracy" in res:
            rows.append(reporting.row_from_result(res))
    rows.sort(key=lambda r: _rank(r["name"]))

    reporting.results_table(rows, d)
    reporting.write_summary(rows, d, notes=args.notes)
    print(f"rebuilt SUMMARY.md + results_table.csv from {len(rows)} runs in {d}")


if __name__ == "__main__":
    main()
