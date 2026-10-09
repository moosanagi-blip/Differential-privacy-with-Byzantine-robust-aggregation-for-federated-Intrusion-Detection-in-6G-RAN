import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from src.experiment_config import RunConfig
from src.utils.provenance import capture


class Provenance(unittest.TestCase):
    def test_changes_are_detected_and_hash_scope_is_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp)
            (base/'src').mkdir()
            (base/'artifacts').mkdir()
            (base/'partitions').mkdir()
            source=base/'src'/'example.py'
            source.write_text('x=1',encoding='utf-8')
            manifest=base/'partitions'/'partition_manifest.json'
            manifest.write_text('{"n_train":10}',encoding='utf-8')
            root=SimpleNamespace(name='synthetic',artifacts=base/'artifacts',partitions=base/'partitions')
            a=capture(RunConfig(),root,base)
            b=capture(RunConfig(),root,base)
            self.assertEqual(a['source_fingerprint'],b['source_fingerprint'])
            self.assertEqual(a['dataset_metadata_fingerprint'],b['dataset_metadata_fingerprint'])
            source.write_text('x=2',encoding='utf-8')
            manifest.write_text('{"n_train":11}',encoding='utf-8')
            c=capture(RunConfig(seed=43),root,base)
            self.assertNotEqual(a['source_fingerprint'],c['source_fingerprint'])
            self.assertNotEqual(a['dataset_metadata_fingerprint'],c['dataset_metadata_fingerprint'])
            self.assertNotEqual(a['config_sha256'],c['config_sha256'])
            self.assertFalse(c['full_parquet_contents_hashed'])


if __name__=='__main__': unittest.main()
