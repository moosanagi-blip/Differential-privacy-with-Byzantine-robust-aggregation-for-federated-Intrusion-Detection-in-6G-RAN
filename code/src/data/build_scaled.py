"""
Stage 1+2 at full corpus scale: stream all 46.7M CICIoT2023 flows straight into
scaled, non-IID, per-gNB parquet shards.

Why this exists alongside ``preprocessing.py`` + ``partition.py``: those two
build the whole dataset as one in-memory DataFrame, which at 46.7M rows means
the shuffle in ``preprocessing.build_dataset`` and the ``read_parquet`` +
``train_test_split`` + ``scaler.transform`` sequence in ``partition.run`` need
40-60 GB of RAM. This module does the same transformations in bounded memory,
streaming throughout, which is what makes the complete corpus usable at all.

Two passes, both streaming:

  Pass 1  CSV -> interim parquet.  Per 200k-row chunk: clean, add the 7 RAN
          features, encode labels. Accumulate exact per-class counts and fit the
          StandardScaler on training rows only by ``partial_fit``. The same
          deterministic per-class holdout is reused in both passes.
          Features retain float64 precision until scaling and are
          written incrementally, so the interim file can be re-read cheaply
          instead of re-parsing 13 GB of CSV.

  Pass 2  interim -> shards.  Apply the fitted scaler, hold out a stratified
          test split, and route each training row to one of the 10 gNBs by the
          per-class Dirichlet(alpha) proportions drawn in pass 1.

Output goes to ``data_full/``: every row, natural class distribution, nothing
subsampled — the research objectives specify the complete dataset.

Differences from the small pipeline, both deliberate and both reported:
  * Duplicate removal happens within each 200k-row streaming window rather than
    globally. A global exact-duplicate pass over 46.7M rows needs a hash of
    every row resident at once, and dropping cross-file duplicates would alter
    the published class distribution.
  * Rows are assigned to clients by a multinomial draw with the class's
    Dirichlet proportions rather than by cutting a shuffled index array. The
    expected proportions are identical; realised counts differ by O(sqrt(n)),
    which at these sizes is under 0.1%.

Run:
    python -m src.data.build_scaled                     # the whole corpus
    python -m src.data.build_scaled --max-files 2 --out /tmp/smoke   # smoke test
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.preprocessing import StandardScaler

from .config import (ALL_FEATURES, ATTACK_LABELS, CATEGORIES, CHUNK_SIZE,
                     DIRICHLET_ALPHA, LABEL_COLUMN, NUM_CLIENTS, PROJECT_ROOT,
                     RAW_DIR, SEED, TEST_FRACTION, label_to_category)
from .preprocessing import clean
from .ran_augment import add_ran_features

# Row group size for the shard files. Chunked training reads whole row groups,
# so this sets the granularity at which a round can slice a client's shard.
ROW_GROUP = 65_536

# Target size of the per-round evaluation subset. The full held-out set is also
# written and is what final reported metrics are computed on; scoring 9M rows
# after every one of 30 rounds would dominate the run for no added information.
EVAL_SUBSET_TARGET = 200_000
PREPROCESSING_VERSION = "train-only-float64-interim-v3"

LABEL_TO_ID = {lbl: i for i, lbl in enumerate(ATTACK_LABELS)}
CATEGORY_TO_ID = {c: i for i, c in enumerate(CATEGORIES)}
# 34-element lookup so category resolution is an array index, not a string test.
CAT_OF_LABEL_ID = np.array(
    [CATEGORY_TO_ID[label_to_category(lbl)] for lbl in ATTACK_LABELS],
    dtype=np.int8)

_FEATURE_FIELDS = [pa.field(c, pa.float32()) for c in ALL_FEATURES]
SHARD_SCHEMA = pa.schema(_FEATURE_FIELDS
                         + [pa.field("label_id", pa.int16()),
                            pa.field("category_id", pa.int8())])
INTERIM_SCHEMA = pa.schema([pa.field(c, pa.float64()) for c in ALL_FEATURES]
                          + [pa.field("label_id", pa.int16()),
                             pa.field("category_id", pa.int8())])


def _chunks(max_files: int | None, chunk_rows: int):
    """Yield ``(global_chunk_index, raw_dataframe)`` across the whole corpus."""
    parts = sorted(RAW_DIR.glob("part-*.csv"))
    if max_files is not None:
        parts = parts[:max_files]
    if not parts:
        raise FileNotFoundError(f"No part-*.csv files found in {RAW_DIR}")
    usecols = None  # schema varies little; let pandas take all, then select
    k = 0
    for pf in parts:
        for chunk in pd.read_csv(pf, chunksize=chunk_rows, usecols=usecols):
            yield k, chunk
            k += 1


def _prepare(chunk: pd.DataFrame, chunk_idx: int, seed: int) -> pd.DataFrame:
    """clean -> RAN augment -> encode, deterministically per chunk.

    The RNG is derived from the chunk index so that pass 1 and any re-run
    produce byte-identical RAN features for the same chunk.
    """
    chunk = chunk[[c for c in chunk.columns if c in set(ALL_FEATURES) | {LABEL_COLUMN}]]
    chunk = clean(chunk)
    if chunk.empty:
        return chunk
    chunk = add_ran_features(chunk, np.random.default_rng(seed + 7919 * chunk_idx))
    chunk = chunk[chunk[LABEL_COLUMN].isin(LABEL_TO_ID)]
    if chunk.empty:
        return chunk
    label_id = chunk[LABEL_COLUMN].map(LABEL_TO_ID).to_numpy(dtype=np.int16)
    # Casting large, closely spaced raw values to float32 before centring can
    # make distinct observations identical. Only final scaled shards use f32.
    out = chunk[ALL_FEATURES].astype(np.float64)
    out["label_id"] = label_id
    out["category_id"] = CAT_OF_LABEL_ID[label_id]
    return out


def _to_table(df: pd.DataFrame) -> pa.Table:
    return pa.Table.from_pandas(df, schema=SHARD_SCHEMA, preserve_index=False)


def _test_mask(label_ids: np.ndarray, seen: np.ndarray) -> np.ndarray:
    """Per-class holdout, invariant to streaming batch boundaries; updates seen."""
    k = max(2, int(round(1.0 / TEST_FRACTION)))
    mask = np.zeros(len(label_ids), dtype=bool)
    for lid in np.unique(label_ids):
        positions = np.flatnonzero(label_ids == lid)
        mask[positions] = ((seen[lid] + np.arange(len(positions))) % k) == 0
        seen[lid] += len(positions)
    return mask


# ══════════════════════════════════════════════════════════════════════════
# Pass 1 — statistics + interim parquet
# ══════════════════════════════════════════════════════════════════════════
def pass1(interim: Path, max_files: int | None, chunk_rows: int,
          seed: int, quiet: bool = False) -> dict:
    interim.parent.mkdir(parents=True, exist_ok=True)
    scaler = StandardScaler()
    counts = np.zeros(len(ATTACK_LABELS), dtype=np.int64)
    split_seen = np.zeros(len(ATTACK_LABELS), dtype=np.int64)
    n_raw = n_kept = 0
    t0 = time.time()

    writer = pq.ParquetWriter(interim, INTERIM_SCHEMA, compression="snappy")
    try:
        for k, raw in _chunks(max_files, chunk_rows):
            n_raw += len(raw)
            df = _prepare(raw, k, seed)
            if df.empty:
                continue
            feats = df[ALL_FEATURES].to_numpy(dtype=np.float64, copy=False)
            is_test = _test_mask(df["label_id"].to_numpy(), split_seen)
            if (~is_test).any():
                scaler.partial_fit(feats[~is_test])
            np.add.at(counts, df["label_id"].to_numpy(), 1)
            n_kept += len(df)
            writer.write_table(pa.Table.from_pandas(
                df, schema=INTERIM_SCHEMA, preserve_index=False),
                row_group_size=ROW_GROUP)
            if not quiet and k % 25 == 0:
                print(f"  pass1 chunk {k:>5}  kept {n_kept:>12,}  "
                      f"({time.time() - t0:.0f}s)", flush=True)
    finally:
        writer.close()

    if not hasattr(scaler, "mean_"):
        raise ValueError("No training rows remain after the holdout split")
    stats = {
        "preprocessing_version": PREPROCESSING_VERSION,
        "scaler_fit_scope": "training rows only",
        "interim_feature_dtype": "float64",
        "output_precision_scope": "chosen at pass2; see output partition manifest",
        "n_raw_rows": int(n_raw),
        "n_rows": int(n_kept),
        "n_features": len(ALL_FEATURES),
        "seed": seed,
        "max_files": max_files,
        "chunk_rows": chunk_rows,
        "label_counts": {ATTACK_LABELS[i]: int(counts[i])
                         for i in range(len(ATTACK_LABELS))},
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_var": scaler.var_.tolist(),
        "scaler_n": int(scaler.n_samples_seen_ if np.isscalar(scaler.n_samples_seen_)
                        else scaler.n_samples_seen_[0]),
        "elapsed_sec": round(time.time() - t0, 1),
    }
    if not quiet:
        print(f"  pass1 done: {n_kept:,} rows kept of {n_raw:,} raw "
              f"({time.time() - t0:.0f}s)")
    return stats


def _scaler_from_stats(stats: dict) -> StandardScaler:
    sc = StandardScaler()
    sc.mean_ = np.asarray(stats["scaler_mean"], dtype=np.float64)
    sc.var_ = np.asarray(stats["scaler_var"], dtype=np.float64)
    sc.scale_ = np.sqrt(sc.var_)
    sc.scale_[sc.scale_ == 0] = 1.0          # constant column -> leave centred
    sc.n_features_in_ = len(sc.mean_)
    sc.n_samples_seen_ = stats["scaler_n"]
    return sc


# ══════════════════════════════════════════════════════════════════════════
# Pass 2 — scale, split, partition, write shards
# ══════════════════════════════════════════════════════════════════════════
class _Root:
    """The output dataset root: 10 client shards + test sets + artifacts."""

    def __init__(self, root: Path, feature_dtype: str = 'float32'):
        if feature_dtype not in ('float32', 'float64'):
            raise ValueError('Expected float32 or float64 feature dtype')
        self.feature_dtype = feature_dtype
        self.schema = SHARD_SCHEMA if feature_dtype == 'float32' else INTERIM_SCHEMA
        if root.exists() and any(root.iterdir()):
            raise FileExistsError(f"Refusing to overwrite dataset artifacts: {root}")
        self.root = root
        self.parts = root / "partitions"
        self.arts = root / "artifacts"
        self.parts.mkdir(parents=True, exist_ok=True)
        self.arts.mkdir(parents=True, exist_ok=True)
        self.used = np.zeros(len(ATTACK_LABELS), dtype=np.int64)
        self.w_client = [
            pq.ParquetWriter(self.parts / f"client_{c:02d}.parquet",
                             self.schema, compression="snappy")
            for c in range(NUM_CLIENTS)]
        self.w_test = pq.ParquetWriter(self.parts / "global_test.parquet",
                                       self.schema, compression="snappy")
        self.w_eval = pq.ParquetWriter(self.parts / "eval_subset.parquet",
                                       self.schema, compression="snappy")
        self.n_client = np.zeros(NUM_CLIENTS, dtype=np.int64)
        self.cat_client = np.zeros((NUM_CLIENTS, len(CATEGORIES)), dtype=np.int64)
        self.n_test = 0
        self.n_eval = 0

    def count(self, label_ids: np.ndarray) -> None:
        self.used += np.bincount(label_ids, minlength=len(ATTACK_LABELS))

    def write(self, df: pd.DataFrame, is_test: np.ndarray, client: np.ndarray,
              eval_mask: np.ndarray) -> None:
        test_df = df[is_test]
        if len(test_df):
            self.w_test.write_table(pa.Table.from_pandas(test_df, schema=self.schema, preserve_index=False), row_group_size=ROW_GROUP)
            self.n_test += len(test_df)
        ev = df[eval_mask]
        if len(ev):
            self.w_eval.write_table(pa.Table.from_pandas(ev, schema=self.schema, preserve_index=False), row_group_size=ROW_GROUP)
            self.n_eval += len(ev)
        tr = ~is_test
        if not tr.any():
            return
        df_tr, cl_tr = df[tr], client[tr]
        for c in range(NUM_CLIENTS):
            m = cl_tr == c
            if not m.any():
                continue
            part = df_tr[m]
            self.w_client[c].write_table(pa.Table.from_pandas(part, schema=self.schema, preserve_index=False), row_group_size=ROW_GROUP)
            self.n_client[c] += len(part)
            np.add.at(self.cat_client[c], part["category_id"].to_numpy(), 1)

    def close(self, stats: dict, scaler: StandardScaler, seed: int) -> None:
        for w in self.w_client:
            w.close()
        self.w_test.close()
        self.w_eval.close()
        joblib.dump(scaler, self.arts / "scaler.joblib")
        (self.arts / "label_map.json").write_text(json.dumps({
            "label_to_id": LABEL_TO_ID,
            "id_to_label": {i: l for l, i in LABEL_TO_ID.items()},
            "category_to_id": CATEGORY_TO_ID,
        }, indent=2))
        kept = {ATTACK_LABELS[i]: int(self.used[i])
                for i in range(len(ATTACK_LABELS)) if self.used[i]}
        (self.arts / "preprocess_stats.json").write_text(json.dumps({
            "n_rows": int(self.n_client.sum() + self.n_test),
            "n_features": len(ALL_FEATURES),
            "scaled_feature_dtype": self.feature_dtype,
            "n_raw_rows_scanned": stats["n_raw_rows"],
            "subsampled": False,
            "label_counts": kept,
            "seed": stats['seed'],
            "augmentation_seed": stats['seed'],
            "partition_seed": seed,
            "source": "src/data/build_scaled.py (streaming full-corpus builder)",
        }, indent=2))
        manifest = {
            "preprocessing_version": PREPROCESSING_VERSION,
            "feature_dtype": self.feature_dtype,
            "representation_version": 'scaled-' + self.feature_dtype + '-v1',
            "scaler_fit_rows": int(scaler.n_samples_seen_),
            "evaluation_subset_method": "global stride over held-out rows",
            "num_clients": NUM_CLIENTS,
            "dirichlet_alpha": DIRICHLET_ALPHA,
            "test_fraction": TEST_FRACTION,
            "seed": seed,
            "augmentation_seed": stats['seed'],
            "partition_seed": seed,
            "holdout_method": "every kth row per class; independent of seeds",
            "n_train": int(self.n_client.sum()),
            "n_test": int(self.n_test),
            "n_eval_subset": int(self.n_eval),
            "row_group_size": ROW_GROUP,
            "clients": {
                f"client_{c:02d}": {
                    "n_rows": int(self.n_client[c]),
                    "category_counts": {
                        CATEGORIES[j]: int(self.cat_client[c][j])
                        for j in range(len(CATEGORIES)) if self.cat_client[c][j]},
                } for c in range(NUM_CLIENTS)},
        }
        (self.parts / "partition_manifest.json").write_text(
            json.dumps(manifest, indent=2))


def pass2(interim: Path, stats: dict, root: _Root, seed: int,
          quiet: bool = False) -> None:
    if stats.get("preprocessing_version") != PREPROCESSING_VERSION:
        raise ValueError("Incompatible preprocessing version; rebuild pass 1 with train-only float64 scaling")
    scaler = _scaler_from_stats(stats)
    rng = np.random.default_rng(seed)

    # Per-class Dirichlet client proportions, drawn once per class exactly as
    # partition.dirichlet_partition does, so the skew is comparable.
    props = np.stack([rng.dirichlet(np.full(NUM_CLIENTS, DIRICHLET_ALPHA))
                      for _ in ATTACK_LABELS])

    # Stratified test hold-out: every k-th row of each class, so the split is
    # exact per class even for the classes with only a handful of rows.
    seen = np.zeros(len(ATTACK_LABELS), dtype=np.int64)

    # Evaluation subset: sample test rows at a rate that lands near the target.
    total_test_est = max(1, int(stats["n_rows"] * TEST_FRACTION))
    eval_stride = max(1, total_test_est // EVAL_SUBSET_TARGET)
    test_seen = 0

    pf = pq.ParquetFile(interim)
    t0 = time.time()
    done = 0
    for b, batch in enumerate(pf.iter_batches(batch_size=CHUNK_SIZE)):
        df = batch.to_pandas()
        lids = df["label_id"].to_numpy()

        # Explicit storage precision; downstream loaders must honor the manifest.
        feats = scaler.transform(df[ALL_FEATURES].to_numpy(dtype=np.float64))
        if not np.isfinite(feats).all():
            raise FloatingPointError('Nonfinite scaled features; no zero substitution is permitted')
        with np.errstate(over='ignore', invalid='ignore'):
            scaled32 = feats.astype(root.feature_dtype)
        if not np.isfinite(scaled32).all():
            raise FloatingPointError('Scaled features overflow requested storage precision')
        df[ALL_FEATURES] = scaled32

        # deterministic per-class test hold-out
        is_test = _test_mask(lids, seen)

        # evaluation subset: a strided sample of the test rows
        eval_mask = np.zeros(len(df), dtype=bool)
        ti = np.flatnonzero(is_test)
        if len(ti):
            pick = ti[((np.arange(len(ti)) + test_seen) % eval_stride) == 0]
            eval_mask[pick] = True
            test_seen += len(ti)

        # client assignment for training rows, by that class's proportions
        client = np.empty(len(df), dtype=np.int16)
        for lid in np.unique(lids):
            m = lids == lid
            client[m] = rng.choice(NUM_CLIENTS, size=int(m.sum()), p=props[lid])

        root.count(lids)
        root.write(df, is_test, client, eval_mask)
        done += len(df)
        if not quiet and b % 25 == 0:
            print(f"  pass2 batch {b:>5}  {done:>12,} rows  "
                  f"({time.time() - t0:.0f}s)", flush=True)

    if int(root.n_client.sum()) != stats["scaler_n"]:
        raise ValueError("Training row count differs from scaler fitting population")
    root.close(stats, scaler, seed)
    if not quiet:
        print(f"  pass2 done: {done:,} rows ({time.time() - t0:.0f}s)")


# ══════════════════════════════════════════════════════════════════════════
def build(out: Path, max_files: int | None = None, chunk_rows: int = CHUNK_SIZE,
          seed: int = SEED, interim: Path | None = None,
          reuse_interim: bool = False, keep_interim: bool = False,
          quiet: bool = False, partition_seed: int | None = None,
          feature_dtype: str = 'float32') -> dict:
    if feature_dtype not in ('float32', 'float64'):
        raise ValueError('Expected float32 or float64 feature dtype')
    # Preserve old behavior unless the caller explicitly separates seed roles.
    partition_seed = seed if partition_seed is None else partition_seed
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing dataset: {out}")
    interim = interim or (out.parent / (out.name + "_interim") / "corpus.parquet")
    stats_path = interim.with_suffix(".stats.json")

    if reuse_interim and interim.exists() and stats_path.exists():
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        if stats.get("preprocessing_version") != PREPROCESSING_VERSION:
            raise ValueError("Cannot reuse legacy scaler statistics; use a new interim path")
        if any(stats.get(k) != v for k, v in
               (("seed", seed), ("max_files", max_files), ("chunk_rows", chunk_rows))):
            raise ValueError("Interim configuration differs from requested build")
        if not quiet:
            print(f"[1/2] reusing interim {interim} ({stats['n_rows']:,} rows)")
    else:
        if interim.exists() or stats_path.exists():
            raise FileExistsError("Interim artifacts already exist; reuse compatible artifacts or choose a new path")
        if not quiet:
            print(f"[1/2] streaming corpus -> {interim}")
        stats = pass1(interim, max_files, chunk_rows, seed, quiet)
        stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    if not quiet:
        print(f"[2/2] scaling + partitioning -> {out}")
    pass2(interim, stats, _Root(out, feature_dtype), partition_seed, quiet)

    # The interim file is a scratch artifact the size of the output — worth
    # reclaiming by default, since this is expected to run on a laptop.
    if not keep_interim:
        try:
            size = interim.stat().st_size
            interim.unlink()
            stats_path.unlink(missing_ok=True)
            try:
                interim.parent.rmdir()
            except OSError:
                pass
            if not quiet:
                print(f"  removed interim scratch file ({size / 1e9:.1f} GB freed; "
                      "pass --keep-interim to retain it for a faster rebuild)")
        except FileNotFoundError:
            pass
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Stream the full CICIoT2023 corpus into scaled non-IID gNB shards.")
    ap.add_argument("--out", default="data_full_trainonly_v3",
                    help="new output dataset root; existing artifacts are never overwritten")
    ap.add_argument("--max-files", type=int, default=None,
                    help="only read the first N raw CSVs (smoke tests)")
    ap.add_argument("--chunk-rows", type=int, default=CHUNK_SIZE)
    ap.add_argument("--seed", type=int, default=SEED,
                    help="synthetic augmentation seed; also partition seed unless overridden")
    ap.add_argument("--partition-seed", type=int, default=None,
                    help="client assignment seed only; does not change interim features or holdout")
    ap.add_argument('--feature-dtype', choices=('float32', 'float64'), default='float32',
                    help='saved and model-input precision; float64 prevents final float32 rounding')
    ap.add_argument("--interim", default=None,
                    help="path for the interim parquet (default data_interim/corpus.parquet)")
    ap.add_argument("--reuse-interim", action="store_true",
                    help="skip pass 1 if the interim file and its stats exist")
    ap.add_argument("--keep-interim", action="store_true",
                    help="keep the interim scratch file (same size as the output) "
                         "so a rebuild can skip pass 1")
    args = ap.parse_args()

    def _path(v):
        if v is None:
            return None
        p = Path(v)
        return p if p.is_absolute() else PROJECT_ROOT / p

    stats = build(out=_path(args.out), max_files=args.max_files,
                  chunk_rows=args.chunk_rows, seed=args.seed,
                  interim=_path(args.interim), reuse_interim=args.reuse_interim,
                  keep_interim=args.keep_interim, partition_seed=args.partition_seed,
                  feature_dtype=args.feature_dtype)
    print(f"\nCorpus rows processed: {stats['n_rows']:,}")


if __name__ == "__main__":
    main()
