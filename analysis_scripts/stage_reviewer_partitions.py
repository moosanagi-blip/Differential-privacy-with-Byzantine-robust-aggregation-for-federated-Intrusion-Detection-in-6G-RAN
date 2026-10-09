"""Build and fully scan paired float64 partitions without training or overwriting."""
from pathlib import Path
from itertools import zip_longest
import json
import shutil
import sys
import hashlib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import joblib

BASE = Path(__file__).resolve().parents[1]
PROJECT = BASE / 'paper gpt/dp-fl-ids code'
sys.path.insert(0, str(PROJECT / 'code'))
from src.data.build_scaled import build
from src.data.config import ALL_FEATURES, ATTACK_LABELS, CATEGORIES, label_to_category

REFERENCE = PROJECT / 'data_original53_v3_f64_p42'
INTERIM = PROJECT / 'data_original53_v3_shared_interim/corpus.parquet'

def scan(root):
    manifest = json.loads((root / 'partitions/partition_manifest.json').read_text())
    scaler = joblib.load(root / 'artifacts/scaler.joblib')
    counts = np.zeros(len(ATTACK_LABELS), dtype=np.int64)
    category_lut = np.array([CATEGORIES.index(label_to_category(x)) for x in ATTACK_LABELS])
    fingerprint = [0, 0]
    files = {}
    train = 0
    for path in sorted((root / 'partitions').glob('*.parquet')):
        pf = pq.ParquetFile(path)
        assert pf.schema_arrow.names == ALL_FEATURES + ['label_id', 'category_id']
        assert all(pf.schema_arrow.field(c).type == pa.float64() for c in ALL_FEATURES)
        n = 0
        for batch in pf.iter_batches(batch_size=65536):
            frame = batch.to_pandas()
            assert np.isfinite(frame[ALL_FEATURES].to_numpy()).all()
            ids = frame.label_id.to_numpy()
            assert np.issubdtype(ids.dtype, np.integer)
            assert (ids >= 0).all() and (ids < len(ATTACK_LABELS)).all()
            np.testing.assert_array_equal(frame.category_id.to_numpy(), category_lut[ids])
            n += len(frame)
            if path.stem.startswith('client_'):
                counts += np.bincount(ids, minlength=len(ATTACK_LABELS))
                # Order-independent row fingerprint is probabilistic, not an exact multiset proof.
                hashes = pd.util.hash_pandas_object(frame, index=False).to_numpy(dtype=np.uint64)
                fingerprint[0] = (fingerprint[0] + int(hashes.sum(dtype=np.uint64))) % 2**64
                fingerprint[1] ^= int(np.bitwise_xor.reduce(hashes, initial=np.uint64(0)))
        expected = (manifest['clients'][path.stem]['n_rows'] if path.stem.startswith('client_')
                    else manifest['n_test'] if path.stem == 'global_test' else manifest['n_eval_subset'])
        assert n == expected == pf.metadata.num_rows
        files[path.name] = n
        if path.stem.startswith('client_'): train += n
        print(f'Validated {root.name}/{path.name}: {n}', flush=True)
    assert train == manifest['n_train'] == manifest['scaler_fit_rows'] == int(scaler.n_samples_seen_)
    assert len([p for p in files if p.startswith('client_')]) == 10
    return {'files':files, 'train_label_counts':counts.tolist(),
            'training_row_fingerprint':fingerprint, 'n_train':train}, manifest, scaler

def exact_parquet_equal(left, right):
    for a,b in zip_longest(pq.ParquetFile(left).iter_batches(batch_size=65536),
                          pq.ParquetFile(right).iter_batches(batch_size=65536)):
        assert a is not None and b is not None and a.equals(b)

def main():
    if shutil.disk_usage(BASE).free < 20 * 2**30:
        raise RuntimeError('At least 20 GiB free required before staging')
    for seed in (43,44):
        if (PROJECT / f'data_original53_v3_f64_p{seed}').exists():
            raise FileExistsError('Refusing to overwrite any existing partition')
    reference, rm, rs = scan(REFERENCE)
    for seed in (43,44):
        root = PROJECT / f'data_original53_v3_f64_p{seed}'
        build(root, seed=42, partition_seed=seed, interim=INTERIM,
              keep_interim=True, reuse_interim=True, feature_dtype='float64')
        result, manifest, scaler = scan(root)
        assert manifest['augmentation_seed'] == 42 and manifest['partition_seed'] == seed
        assert result['train_label_counts'] == reference['train_label_counts']
        assert result['training_row_fingerprint'] == reference['training_row_fingerprint']
        assert result['n_train'] == reference['n_train']
        assert manifest['clients'] != rm['clients']
        for attr in ('mean_', 'scale_', 'var_', 'n_samples_seen_'):
            np.testing.assert_array_equal(getattr(scaler,attr), getattr(rs,attr))
        for filename in ('global_test.parquet','eval_subset.parquet'):
            exact_parquet_equal(REFERENCE/'partitions'/filename, root/'partitions'/filename)
        result.update(status='passed', partition_seed=seed, augmentation_seed=42,
            holdout_and_eval='exact row-wise equality with verified seed-42 root',
            scaler='exact equality with seed-42 scaler',
            training_content='full finite/schema/count scan plus probabilistic 64-bit sum/XOR row fingerprint',
            caveat='Not an event-independence proof; two original exact train/test matches remain applicable',
            validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        with (root/'reviewer_validation.json').open('x') as handle:
            json.dump(result, handle, indent=2, allow_nan=False)
        print(f'PASS: partition {seed}', flush=True)

if __name__ == '__main__':
    main()
