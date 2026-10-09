"""
Torch data loaders for the DP-FL-IDS framework.

Reads the preprocessed, already-scaled parquet artifacts produced by the data
pipeline (``src/data/build_scaled.py``), under whichever dataset root a run
names — ``data_full/`` for the complete corpus:

  * ``<root>/partitions/client_XX.parquet`` — one non-IID shard per gNB
  * ``<root>/partitions/global_test.parquet`` — held-out global test set
  * ``<root>/artifacts/label_map.json`` — label <-> id mapping (34 classes)

Every shard has 53 numeric feature columns (``config.ALL_FEATURES``) plus the
meta columns ``label, category, label_id, category_id``. Features are already
StandardScaler-transformed (fit on train only), so loaders here do **no**
further scaling — they only slice out the feature matrix and the integer
``label_id`` target and wrap them as torch tensors.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, TensorDataset

from .config import ALL_FEATURES, RAN_FEATURES, ARTIFACT_DIR, PARTITION_DIR, PROJECT_ROOT

LABEL_ID_COL = "label_id"
N_FEATURES = len(ALL_FEATURES)  # 53


# ── label map ─────────────────────────────────────────────────────────────
# Roots to look in, in order, when nothing names one explicitly. The label map
# is identical across prepared roots — the 34 labels come from config.py — but
# the metrics helpers resolve it without knowing which root a run used, so the
# lookup has to succeed whichever roots happen to be on disk.
_LABEL_MAP_DIRS = (ARTIFACT_DIR, PROJECT_ROOT / "data_full" / "artifacts")


def load_label_map() -> dict[str, int]:
    """Return the ``label -> id`` dict (34 fine-grained classes)."""
    for d in _LABEL_MAP_DIRS:
        path = d / "label_map.json"
        if path.exists():
            obj = json.loads(path.read_text(encoding="utf-8"))
            return {str(k): int(v) for k, v in obj["label_to_id"].items()}
    raise FileNotFoundError(
        "label_map.json not found in " + " or ".join(str(d) for d in _LABEL_MAP_DIRS)
        + " — build the dataset first (`python main.py --build-data`).")


def num_classes() -> int:
    """Number of output classes = number of fine-grained labels (34)."""
    return len(load_label_map())


def id_to_label() -> dict[int, str]:
    return {v: k for k, v in load_label_map().items()}


# ── frame -> arrays ───────────────────────────────────────────────────────
def _frame_to_xy(df: pd.DataFrame, *, mask_ran_features: bool = False,
                 features=None, radio_features=None, dtype='float32') -> tuple[np.ndarray, np.ndarray]:
    features = ALL_FEATURES if features is None else features
    radio_features = RAN_FEATURES if radio_features is None else radio_features
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise KeyError(f"Parquet is missing {len(missing)} feature columns: {missing[:5]}…")
    with np.errstate(over='ignore', invalid='ignore'):
        x = df[features].to_numpy(dtype=dtype, copy=True)
    # Dataset manifests select float32 (legacy) or float64 (precision-preserving).
    # Invalid values, including overflow during conversion, must never become
    # plausible zeros or be hidden by the radio-feature ablation mask.
    if not np.isfinite(x).all():
        raise FloatingPointError('Nonfinite prepared features; rebuild or inspect the dataset')
    if mask_ran_features:
        # StandardScaler maps the training-set mean to zero.  Masking these
        # columns therefore removes their information without changing model
        # dimensionality or requiring a second prepared corpus.
        x[:, [features.index(c) for c in radio_features]] = 0.0
    y = df[LABEL_ID_COL].to_numpy(dtype=np.int64, copy=True)
    return x, y


def load_partition(client_id: int, partition_dir: Path = PARTITION_DIR
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Load one gNB shard as ``(X[float32, N×53], y[int64, N])``."""
    path = partition_dir / f"client_{client_id:02d}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run `python -m src.data.partition`.")
    return _frame_to_xy(pd.read_parquet(path))


def load_global_test(partition_dir: Path = PARTITION_DIR
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Load the held-out global test set as ``(X, y)``."""
    path = partition_dir / "global_test.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run `python -m src.data.partition`.")
    return _frame_to_xy(pd.read_parquet(path))


def count_clients(partition_dir: Path = PARTITION_DIR) -> int:
    return len(sorted(partition_dir.glob("client_*.parquet")))


class _ParquetBatches:
    """Iterable of ``(x, y)`` tensor batches read straight from a parquet file.

    Deliberately not a ``DataLoader``: everything that consumes an evaluation
    loader in this project only iterates it, so a generator over row groups is
    enough and keeps peak memory at one batch instead of the whole file. Exposes
    ``.dataset`` with ``__len__`` because callers report the row count.
    """

    class _Sized:
        def __init__(self, n: int):
            self.n = n

        def __len__(self) -> int:
            return self.n

    def __init__(self, path: Path, batch_size: int = 4096, *, mask_ran_features: bool = False,
                 features=None, radio_features=None, dtype='float32'):
        self.dtype = dtype
        self.path, self.batch_size, self.mask_ran_features = path, batch_size, mask_ran_features
        self.features = list(ALL_FEATURES if features is None else features)
        self.radio_features = list(RAN_FEATURES if radio_features is None else radio_features)
        self._n = pq.ParquetFile(path).metadata.num_rows
        self.dataset = self._Sized(self._n)

    def __len__(self) -> int:
        return max(1, (self._n + self.batch_size - 1) // self.batch_size)

    def __iter__(self):
        pf = pq.ParquetFile(self.path)
        for batch in pf.iter_batches(batch_size=self.batch_size,
                                     columns=self.features + [LABEL_ID_COL]):
            x, y = _frame_to_xy(batch.to_pandas(), mask_ran_features=self.mask_ran_features,
                                features=self.features, radio_features=self.radio_features, dtype=self.dtype)
            yield torch.from_numpy(x), torch.from_numpy(y)


# ── alternate dataset roots (full-corpus shards live outside data/) ────────
class DataRoot:
    """Where one dataset's shards and artifacts live.

    ``data_full/`` is the complete 46.7M-flow corpus and is what every entry
    point runs on. The builder can write other roots in the same layout, so
    anything that reads data can be pointed at a root by name instead of at the
    module-level default paths.
    """

    def __init__(self, root: str | Path | None = None):
        if root is None:
            self.partitions, self.artifacts = PARTITION_DIR, ARTIFACT_DIR
            self.name = "data"
        else:
            base = Path(root)
            # Datasets live beside code/, not inside it, so a bare name like
            # "data_full" must resolve against the project root rather than the
            # current directory — otherwise it depends on where you ran from.
            if not base.is_absolute():
                base = PROJECT_ROOT / base
            self.partitions = base / "partitions"
            self.artifacts = base / "artifacts"
            self.name = base.name
        if not self.partitions.exists():
            raise FileNotFoundError(
                f"{self.partitions} not found — build it with "
                "`python main.py --build-data`.")
        self.features, self.radio_features = list(ALL_FEATURES), list(RAN_FEATURES)
        self.feature_dtype = self.manifest().get('feature_dtype', 'float32')
        if self.feature_dtype not in ('float32', 'float64'):
            raise ValueError('Unsupported model-input precision')
        schema_path = self.artifacts / 'feature_schema.json'
        if schema_path.exists():
            from .schema39 import schema_manifest
            declared = json.loads(schema_path.read_text(encoding='utf-8'))
            supported = schema_manifest()
            for key in ('schema_id', 'network_features', 'radio_features', 'feature_order', 'n_features'):
                if declared.get(key) != supported[key]:
                    raise ValueError(f'Unsupported or inconsistent feature schema: {key}')
            self.features = list(declared['feature_order'])
            self.radio_features = list(declared['radio_features'])

    def num_features(self) -> int:
        return len(self.features)

    def _xy(self, frame, mask_ran_features=False):
        return _frame_to_xy(frame, mask_ran_features=mask_ran_features,
                            features=self.features, radio_features=self.radio_features, dtype=self.feature_dtype)

    # -- metadata only, no row data read --
    def client_path(self, c: int) -> Path:
        return self.partitions / f"client_{c:02d}.parquet"

    def n_clients(self) -> int:
        return len(sorted(self.partitions.glob("client_*.parquet")))

    def client_rows(self, c: int) -> int:
        return pq.ParquetFile(self.client_path(c)).metadata.num_rows

    def label_map(self) -> dict[str, int]:
        obj = json.loads((self.artifacts / "label_map.json").read_text(encoding="utf-8"))
        return {str(k): int(v) for k, v in obj["label_to_id"].items()}

    def num_classes(self) -> int:
        return len(self.label_map())

    def manifest(self) -> dict:
        p = self.partitions / "partition_manifest.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}

    # -- data --
    def load_client(self, c: int, *, mask_ran_features: bool = False) -> tuple[np.ndarray, np.ndarray]:
        return self._xy(pd.read_parquet(self.client_path(c)), mask_ran_features)

    def load_client_chunk(self, c: int, chunk: int, n_chunks: int, *, mask_ran_features: bool = False
                          ) -> tuple[np.ndarray, np.ndarray]:
        """Load only chunk ``chunk`` of ``n_chunks`` equal slices of a shard.

        Round *r* of a federated run trains on chunk *r*, so ``n_chunks``
        rounds together consume the client's shard exactly once with no row
        seen twice. Only the parquet row groups overlapping the slice are read,
        which is what keeps a 3.7M-row shard off the heap.
        """
        pf = pq.ParquetFile(self.client_path(c))
        total = pf.metadata.num_rows
        if n_chunks <= 1 or total == 0:
            return self.load_client(c, mask_ran_features=mask_ran_features)
        start = (total * chunk) // n_chunks
        stop = (total * (chunk + 1)) // n_chunks
        if stop <= start:
            return (np.empty((0, self.num_features()), dtype=self.feature_dtype),
                    np.empty(0, dtype=np.int64))

        groups, lo, offset = [], 0, None
        for g in range(pf.metadata.num_row_groups):
            hi = lo + pf.metadata.row_group(g).num_rows
            if hi > start and lo < stop:
                groups.append(g)
                if offset is None:
                    offset = start - lo
            lo = hi
        table = pf.read_row_groups(groups, columns=self.features + [LABEL_ID_COL])
        df = table.slice(offset or 0, stop - start).to_pandas()
        return self._xy(df, mask_ran_features)

    def stream_test(self, batch_size: int = 4096, subset: bool = False,
                    mask_ran_features: bool = False
                    ) -> "_ParquetBatches":
        """Iterate the held-out set in parquet row-group batches.

        The complete held-out set is 9.3M rows — ~2 GB as float32 — and the
        final evaluation of every run reads it. Materialising that is a needless
        spike on a laptop, and nothing downstream needs random access: both
        ``metrics.predict`` and ``training.evaluate`` only iterate.
        """
        path = self.partitions / ("eval_subset.parquet" if subset else "global_test.parquet")
        if not path.exists():
            path = self.partitions / "global_test.parquet"
        return _ParquetBatches(path, batch_size, mask_ran_features=mask_ran_features,
                               features=self.features, radio_features=self.radio_features, dtype=self.feature_dtype)

    def test_loader(self, batch_size: int = 256, subset: bool = True,
                    mask_ran_features: bool = False) -> DataLoader:
        """Per-round evaluation loader.

        ``subset=True`` prefers ``eval_subset.parquet`` when the builder wrote
        one: scoring a 9M-row held-out set after each of 30 rounds costs far
        more than it tells us. Final reported metrics use ``subset=False``.
        """
        path = self.partitions / "eval_subset.parquet"
        if not (subset and path.exists()):
            path = self.partitions / "global_test.parquet"
        x, y = self._xy(pd.read_parquet(path), mask_ran_features)
        return make_loader(x, y, batch_size=batch_size, shuffle=False)


# ── torch wrappers ────────────────────────────────────────────────────────
def make_dataset(x: np.ndarray, y: np.ndarray) -> TensorDataset:
    return TensorDataset(torch.from_numpy(x), torch.from_numpy(y))


def make_loader(x: np.ndarray, y: np.ndarray, batch_size: int = 64,
                shuffle: bool = True, seed: int | None = None) -> DataLoader:
    """Wrap ``(X, y)`` in a DataLoader.

    A fixed ``seed`` makes shuffling reproducible (used so Opacus/FL runs are
    deterministic across the experiment matrix).
    """
    ds = make_dataset(x, y)
    gen = None
    if seed is not None:
        gen = torch.Generator()
        gen.manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=gen,
                      drop_last=False)


def client_loader(client_id: int, batch_size: int = 64, shuffle: bool = True,
                  seed: int | None = None) -> DataLoader:
    x, y = load_partition(client_id)
    return make_loader(x, y, batch_size=batch_size, shuffle=shuffle, seed=seed)


def global_test_loader(batch_size: int = 256) -> DataLoader:
    x, y = load_global_test()
    return make_loader(x, y, batch_size=batch_size, shuffle=False)
