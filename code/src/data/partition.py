"""
Stage 2 of the data pipeline: split the processed dataset into a held-out
global test set and 10 non-IID gNB client shards.

Steps
  1. Stratified train/test split (TEST_FRACTION held out globally).
  2. Fit StandardScaler on the TRAIN features only (avoids leakage), then
     transform train + test. Save the fitted scaler as an artifact.
  3. Dirichlet(alpha) label-skew partition of the train set across the 10
     gNBs: per class, draw a client-proportion vector ~ Dir(alpha) and
     allocate that class's rows accordingly. Lower alpha => more non-IID.
  4. Save client_00..client_09 shards, the global test set, scaler, and a
     partition manifest with per-client label distributions.

Run:
    python -m src.data.partition
"""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .config import (
    ALL_FEATURES, ARTIFACT_DIR, DIRICHLET_ALPHA, NUM_CLIENTS, PARTITION_DIR,
    PROCESSED_DIR, SEED, TEST_FRACTION,
)

META_COLS = ["label", "category", "label_id", "category_id"]


def dirichlet_partition(train: pd.DataFrame, alpha: float, n_clients: int,
                        seed: int) -> list[np.ndarray]:
    """Return a list of row-index arrays, one per client (label-skew)."""
    rng = np.random.default_rng(seed)
    client_indices: list[list[int]] = [[] for _ in range(n_clients)]
    for label_id, grp in train.groupby("label_id"):
        idx = grp.index.to_numpy()
        rng.shuffle(idx)
        proportions = rng.dirichlet(alpha=np.full(n_clients, alpha))
        # cut points for this class across clients
        cuts = (np.cumsum(proportions) * len(idx)).astype(int)[:-1]
        for c, part in enumerate(np.split(idx, cuts)):
            client_indices[c].extend(part.tolist())
    return [np.array(sorted(ci)) for ci in client_indices]


def run(seed: int = SEED) -> None:
    PARTITION_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    src = PROCESSED_DIR / "dataset.parquet"
    if not src.exists():
        raise FileNotFoundError(
            f"{src} not found - run `python -m src.data.preprocessing` first")
    df = pd.read_parquet(src)
    print(f"Loaded {len(df)} rows, {len(ALL_FEATURES)} features")

    # 1. Stratified global train/test split.
    train, test = train_test_split(
        df, test_size=TEST_FRACTION, random_state=seed,
        stratify=df["label_id"])
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
    print(f"  train={len(train)}  test={len(test)}")

    # 2. Scale: fit on train only, apply to both.
    scaler = StandardScaler().fit(train[ALL_FEATURES].to_numpy())
    train[ALL_FEATURES] = scaler.transform(train[ALL_FEATURES].to_numpy())
    test[ALL_FEATURES] = scaler.transform(test[ALL_FEATURES].to_numpy())
    joblib.dump(scaler, ARTIFACT_DIR / "scaler.joblib")
    test.to_parquet(PARTITION_DIR / "global_test.parquet", index=False)

    # 3. Dirichlet non-IID partition of the train set.
    parts = dirichlet_partition(train, DIRICHLET_ALPHA, NUM_CLIENTS, seed)

    # 4. Save shards + manifest.
    manifest: dict = {
        "num_clients": NUM_CLIENTS,
        "dirichlet_alpha": DIRICHLET_ALPHA,
        "test_fraction": TEST_FRACTION,
        "seed": seed,
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "clients": {},
    }
    for c, idx in enumerate(parts):
        shard = train.iloc[idx].reset_index(drop=True)
        shard.to_parquet(PARTITION_DIR / f"client_{c:02d}.parquet", index=False)
        manifest["clients"][f"client_{c:02d}"] = {
            "n_rows": int(len(shard)),
            "n_labels_present": int(shard["label_id"].nunique()),
            "category_counts": {k: int(v) for k, v in
                                shard["category"].value_counts().items()},
        }
        print(f"  client_{c:02d}: {len(shard):6d} rows, "
              f"{shard['label_id'].nunique():2d} labels, "
              f"top cats: {dict(shard['category'].value_counts().head(3))}")

    (PARTITION_DIR / "partition_manifest.json").write_text(
        json.dumps(manifest, indent=2))
    print(f"\nSaved {NUM_CLIENTS} client shards + global test -> {PARTITION_DIR}")


if __name__ == "__main__":
    run()
