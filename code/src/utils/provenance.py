"""Local, read-only fingerprints for experimental provenance.

Metadata fingerprints do not certify the full contents of large parquet files.
No raw examples, model updates, or network requests are involved.
"""
import hashlib
import importlib.metadata
import json
import platform
from datetime import datetime, timezone
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def capture(config, root, code_root=None):
    code_root = Path(code_root) if code_root else Path(__file__).resolve().parents[2]
    files = list((code_root/'src').rglob('*.py')) + list((code_root/'scripts').glob('*.py'))
    if (code_root/'main.py').exists():
        files.append(code_root/'main.py')
    sources = {p.relative_to(code_root).as_posix():digest(p) for p in sorted(files)}
    config_dict = config.to_dict()
    versions = {}
    for package in ('torch','opacus','flwr','numpy','scikit-learn','pandas','pyarrow'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    metadata = {}
    inventory = []
    for attr in ('artifacts','partitions'):
        folder = getattr(root,attr,None)
        if folder is None:
            continue
        folder = Path(folder)
        for p in sorted(folder.glob('*')):
            if p.is_file() and p.suffix in ('.json','.joblib'):
                metadata[f'{attr}/{p.name}'] = digest(p)
            elif p.is_file() and p.suffix == '.parquet':
                stat = p.stat()
                inventory.append({'path':f'{attr}/{p.name}', 'bytes':stat.st_size,
                                  'modified_ns':stat.st_mtime_ns})
    canonical = lambda obj: json.dumps(obj,sort_keys=True,separators=(',',':')).encode()
    return {
        'schema':'audit-provenance-v1',
        'started_utc':datetime.now(timezone.utc).isoformat(),
        'python':platform.python_version(), 'platform':platform.platform(),
        'packages':versions, 'source_sha256':sources,
        'source_fingerprint':hashlib.sha256(canonical(sources)).hexdigest(),
        'config_sha256':hashlib.sha256(canonical(config_dict)).hexdigest(),
        'dataset_name':root.name, 'dataset_metadata_sha256':metadata,
        'dataset_metadata_fingerprint':hashlib.sha256(canonical(metadata)).hexdigest(),
        'parquet_inventory':inventory,
        'full_parquet_contents_hashed':False,
        'limitations':'Metadata hashes and file stats are not full dataset-content verification.',
    }
