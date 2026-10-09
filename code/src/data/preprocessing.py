"""
Stage 1 of the data pipeline: build a clean, RAN-augmented, label-encoded
subsample of CICIoT2023.

Steps
  1. Stream the 169 raw part files in chunks.
  2. Stratified subsample: cap each of the 34 fine labels at MAX_PER_CLASS,
     keep every available row of the rare classes.
  3. Clean: replace +/-inf with NaN, drop NaN/duplicate rows.
  4. Augment each flow with the 7 synthetic RAN telemetry features.
  5. Encode labels: 34-class id (label_id) and 8-category id (category_id).
  6. Save the UNSCALED dataset (scaling is fit on the train split later, in
     partition.py, to avoid train/test leakage) plus label map + stats.

Run:
    python -m src.data.preprocessing                 # full defaults
    python -m src.data.preprocessing --max-per-class 2000 --max-files 4
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from .config import (
    ARTIFACT_DIR, ATTACK_LABELS, CATEGORIES, CHUNK_SIZE, LABEL_COLUMN,
    MAX_PER_CLASS, NETWORK_FEATURES, PROCESSED_DIR, RAW_DIR, SEED,
    label_to_category,
)
from .ran_augment import add_ran_features

LABEL_TO_ID: dict[str, int] = {lbl: i for i, lbl in enumerate(ATTACK_LABELS)}
CATEGORY_TO_ID: dict[str, int] = {cat: i for i, cat in enumerate(CATEGORIES)}
_USECOLS = NETWORK_FEATURES + [LABEL_COLUMN]


def collect_subsample(max_per_class: int, max_files: int | None,
                      seed: int) -> pd.DataFrame:
    """Stream raw files and return a stratified per-label subsample."""
    rng_state = seed
    remaining = {lbl: max_per_class for lbl in ATTACK_LABELS}
    parts = sorted(RAW_DIR.glob("part-*.csv"))
    if max_files is not None:
        parts = parts[:max_files]
    if not parts:
        raise FileNotFoundError(f"No part-*.csv files found in {RAW_DIR}")

    collected: list[pd.DataFrame] = []
    for pf in parts:
        if all(v <= 0 for v in remaining.values()):
            break
        for chunk in pd.read_csv(pf, usecols=_USECOLS, chunksize=CHUNK_SIZE):
            if all(v <= 0 for v in remaining.values()):
                break
            for lbl, grp in chunk.groupby(LABEL_COLUMN):
                need = remaining.get(str(lbl), 0)
                if need <= 0:
                    continue
                take = grp if len(grp) <= need else grp.sample(
                    n=need, random_state=rng_state)
                collected.append(take)
                remaining[str(lbl)] -= len(take)
                rng_state += 1

    df = pd.concat(collected, ignore_index=True)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Remove inf/NaN rows and exact duplicates."""
    before = len(df)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"  cleaned: {before} -> {len(df)} rows "
          f"({before - len(df)} removed)")
    return df


def encode(df: pd.DataFrame) -> pd.DataFrame:
    """Add label_id (0-33), category (str) and category_id (0-7)."""
    df["category"] = df[LABEL_COLUMN].map(label_to_category)
    df["label_id"] = df[LABEL_COLUMN].map(LABEL_TO_ID).astype("int16")
    df["category_id"] = df["category"].map(CATEGORY_TO_ID).astype("int8")
    return df


def build_dataset(max_per_class: int = MAX_PER_CLASS,
                  max_files: int | None = None,
                  seed: int = SEED) -> pd.DataFrame:
    t0 = time.time()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Collecting subsample (<= {max_per_class}/class, "
          f"files={max_files or 'all'}) ...")
    df = collect_subsample(max_per_class, max_files, seed)
    print(f"      raw subsample: {len(df)} rows")

    print("[2/4] Cleaning ...")
    df = clean(df)

    print("[3/4] Adding 7 RAN features + encoding labels ...")
    df = add_ran_features(df, np.random.default_rng(seed))
    df = encode(df)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)  # shuffle

    print("[4/4] Saving ...")
    out_path = PROCESSED_DIR / "dataset.parquet"
    df.to_parquet(out_path, index=False)

    label_map = {
        "label_to_id": LABEL_TO_ID,
        "id_to_label": {i: l for l, i in LABEL_TO_ID.items()},
        "category_to_id": CATEGORY_TO_ID,
    }
    (ARTIFACT_DIR / "label_map.json").write_text(json.dumps(label_map, indent=2), encoding="utf-8")

    stats = {
        "n_rows": int(len(df)),
        "n_features": 53,
        "max_per_class": max_per_class,
        "max_files": max_files,
        "seed": seed,
        "label_counts": {k: int(v) for k, v in
                         df[LABEL_COLUMN].value_counts().items()},
        "category_counts": {k: int(v) for k, v in
                            df["category"].value_counts().items()},
        "elapsed_sec": round(time.time() - t0, 1),
    }
    (ARTIFACT_DIR / "preprocess_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(f"\nDone in {stats['elapsed_sec']}s -> {out_path}")
    print(f"  rows={stats['n_rows']}  labels={df[LABEL_COLUMN].nunique()}  "
          f"categories={df['category'].nunique()}")
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description="Build CICIoT2023 + RAN dataset")
    ap.add_argument("--max-per-class", type=int, default=MAX_PER_CLASS)
    ap.add_argument("--max-files", type=int, default=None,
                    help="limit number of raw part files (smoke testing)")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    build_dataset(args.max_per_class, args.max_files, args.seed)


if __name__ == "__main__":
    main()
