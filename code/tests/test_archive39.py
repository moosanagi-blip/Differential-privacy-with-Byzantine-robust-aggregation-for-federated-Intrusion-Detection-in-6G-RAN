from io import BytesIO
import unittest
import zipfile

from src.data.archive39 import iter_validated
from src.data.schema39 import NETWORK_FEATURES


class Archive39(unittest.TestCase):
    def archive(self, lines, header=None):
        stream = BytesIO()
        with zipfile.ZipFile(stream, 'w') as z:
            z.writestr('MERGED_CSV/Merged01.csv',
                       ','.join(header or NETWORK_FEATURES + ['Label']) + '\n' + '\n'.join(lines))
        stream.seek(0)
        return stream

    def test_rejections_have_exact_locations_and_counts(self):
        good = ','.join(['1'] * 39 + ['BENIGN'])
        bad_numeric = ','.join(['inf'] + ['1'] * 38 + ['BENIGN'])
        bad_label = ','.join(['1'] * 39 + ['UNKNOWN'])
        rejected = []
        batches = list(iter_validated(self.archive([good, '1,2', bad_numeric, bad_label, '', good]),
                                     layout='merged', reject=rejected.append, chunk_rows=2))
        self.assertEqual([i for _, frame in batches for i in frame.index], [2, 7])
        self.assertEqual({r['line']: r['reason'] for r in rejected},
                         {3: 'field_count', 4: 'non_numeric_or_nonfinite',
                          5: 'unknown_or_missing_label', 6: 'blank'})

    def test_header_mismatch_fails_and_quotes_are_explicit(self):
        with self.assertRaisesRegex(ValueError, 'Unexpected header'):
            list(iter_validated(self.archive([], ['wrong']), layout='merged', reject=lambda x: None))
        rejected = []
        list(iter_validated(self.archive(['"1",2']), layout='merged', reject=rejected.append))
        self.assertEqual(rejected[0]['reason'], 'quoted_record_unsupported')


if __name__ == '__main__':
    unittest.main()
