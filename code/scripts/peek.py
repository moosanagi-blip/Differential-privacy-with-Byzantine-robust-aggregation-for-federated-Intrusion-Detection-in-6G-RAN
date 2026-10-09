#!/usr/bin/env python3
"""
Look inside a parquet file, or export part of one to CSV.

Parquet is a binary columnar format — Excel and text editors cannot open it.
That is deliberate: as CSV the prepared corpus would be roughly 20 GB instead of
2.5 GB, and every read would be far slower. This is the tool for looking at it.

    # what is in this file?
    python3 scripts/peek.py ../data_full/partitions/client_00.parquet

    # show the radio columns in original units, not scaled
    python3 scripts/peek.py ../data_full/partitions/client_00.parquet --raw --ran

    # export the first 500 rows to CSV so it opens in Excel
    python3 scripts/peek.py ../data_full/partitions/client_00.parquet --csv out.csv -n 500

    # list every prepared dataset
    python3 scripts/peek.py --list
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))

from src.data.config import (ALL_FEATURES, NETWORK_FEATURES,  # noqa: E402
                             PROJECT_ROOT, RAN_FEATURES)


def list_datasets() -> None:
    print("Prepared datasets:\n")
    for root in sorted(PROJECT_ROOT.glob("data*")):
        parts = root / "partitions"
        if not parts.is_dir():
            continue
        files = sorted(parts.glob("*.parquet"))
        total = sum(pq.ParquetFile(f).metadata.num_rows for f in files)
        print(f"  {root.name}/  —  {len(files)} files, {total:,} rows")
        for f in files:
            n = pq.ParquetFile(f).metadata.num_rows
            print(f"      {f.name:<26} {n:>12,} rows  "
                  f"{f.stat().st_size / 1e6:>8.1f} MB")
        print()


def unscale(df: pd.DataFrame, root: Path) -> pd.DataFrame:
    """Undo the StandardScaler so values read in their real units.

    Stored features are standardised (mean 0, sd 1) because that is what the
    model consumes. For a human reading a sample, RSRP should look like -78 dBm,
    not 1.07.
    """
    import joblib
    scaler_path = root / "artifacts" / "scaler.joblib"
    if not scaler_path.exists():
        print(f"  ! no scaler at {scaler_path}; showing stored (scaled) values")
        return df
    sc = joblib.load(scaler_path)
    cols = [c for c in ALL_FEATURES if c in df.columns]
    out = df.copy()
    out[cols] = pd.DataFrame(
        sc.inverse_transform(df[ALL_FEATURES].to_numpy(dtype=np.float64)),
        columns=ALL_FEATURES, index=df.index)[cols].round(2)
    return out


def add_labels(df: pd.DataFrame, root: Path) -> pd.DataFrame:
    path = root / "artifacts" / "label_map.json"
    if "label_id" not in df.columns or not path.exists():
        return df
    id2l = json.loads(path.read_text(encoding="utf-8"))["id_to_label"]
    df.insert(0, "attack_label", [id2l.get(str(int(i)), "?") for i in df["label_id"]])
    return df


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file", nargs="?", help="path to a .parquet file")
    ap.add_argument("--list", action="store_true",
                    help="list the prepared datasets and their row counts")
    ap.add_argument("-n", type=int, default=10, help="how many rows (default 10)")
    ap.add_argument("--raw", action="store_true",
                    help="undo scaling so values are in original units")
    ap.add_argument("--ran", action="store_true",
                    help="show only the 7 radio columns")
    ap.add_argument("--all-columns", action="store_true",
                    help="show all 53 features instead of a readable subset")
    ap.add_argument("--csv", default=None, metavar="PATH",
                    help="write the rows to CSV instead of printing")
    args = ap.parse_args()

    if args.list or not args.file:
        list_datasets()
        if not args.file:
            print("Pass a .parquet path to look inside one. --help for options.")
        return 0

    path = Path(args.file)
    if not path.exists():
        path = PROJECT_ROOT / args.file
    if not path.exists():
        print(f"not found: {args.file}")
        return 1

    pf = pq.ParquetFile(path)
    md = pf.metadata
    print(f"\n{path.name}")
    print(f"  rows        {md.num_rows:,}")
    print(f"  columns     {md.num_columns}")
    print(f"  row groups  {md.num_row_groups}")
    print(f"  on disk     {path.stat().st_size / 1e6:.1f} MB\n")

    df = next(pf.iter_batches(batch_size=max(args.n, 1))).to_pandas()
    root = path.parent.parent            # <root>/partitions/file.parquet
    if args.raw:
        df = unscale(df, root)
    df = add_labels(df, root)

    if args.ran:
        cols = ["attack_label"] + [c for c in RAN_FEATURES if c in df.columns]
    elif args.all_columns:
        cols = list(df.columns)
    else:
        cols = (["attack_label"] + NETWORK_FEATURES[:5]
                + [c for c in RAN_FEATURES if c in df.columns])
    cols = [c for c in cols if c in df.columns]
    view = df[cols].head(args.n)

    if args.csv:
        out = Path(args.csv)
        view.to_csv(out, index=False)
        print(f"  wrote {len(view)} rows -> {out}")
        print("  (opens in Excel)")
    else:
        with pd.option_context("display.width", 200,
                               "display.max_columns", 60):
            print(view.to_string(index=False))
        print(f"\n  showing {len(view)} of {md.num_rows:,} rows"
              + ("" if args.raw else "   — values are standardised; add --raw "
                                     "for original units"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
