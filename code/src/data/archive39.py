"""Bounded-memory, read-only reader for the unquoted downloaded CSV release."""
from __future__ import annotations

from io import BytesIO
from pathlib import PurePosixPath
import zipfile

import numpy as np
import pandas as pd

from .schema39 import NETWORK_FEATURES, validate_chunk


def iter_validated(archive, *, layout, reject, chunk_rows=50000):
    """Yield (member name, validated frame), indexed by physical source line.

    ``reject`` must consume each rejection record, e.g. persist JSONL. This
    specialized reader rejects quoted records instead of guessing multiline
    CSV semantics. The inspected release contains no quoted data fields.
    Header, ZIP, decoding and I/O errors abort; row defects are logged.
    """
    if layout not in ('per_attack', 'merged') or chunk_rows < 1:
        raise ValueError('Choose per_attack/merged and positive chunk_rows')
    expected = NETWORK_FEATURES + (['Label'] if layout == 'merged' else [])
    with zipfile.ZipFile(archive) as z:
        names = [i.filename for i in z.infolist() if i.filename.lower().endswith('.csv')]
        if not names or len(set(names)) != len(names):
            raise ValueError('Empty archive or duplicate CSV member names')
        for member in names:
            directory = PurePosixPath(member).parent.name if layout == 'per_attack' else None
            with z.open(member) as stream:
                header = stream.readline()
                columns = header.decode('utf-8-sig').rstrip('\r\n').split(',')
                if columns != expected:
                    raise ValueError(f'Unexpected header: {member}')
                validate_chunk(pd.DataFrame(columns=columns), directory_label=directory)
                rows, locations = [], []

                def flush():
                    frame = pd.read_csv(BytesIO(b''.join(rows)), header=None,
                                        names=columns, float_precision='round_trip')
                    if len(frame) != len(locations):
                        raise ValueError(f'Parser changed record count: {member}')
                    frame.index = locations
                    accepted, _ = validate_chunk(frame, directory_label=directory)
                    for line in frame.index.difference(accepted.index):
                        numeric = pd.to_numeric(frame.loc[line, NETWORK_FEATURES], errors='coerce')
                        reason = ('non_numeric_or_nonfinite' if not np.isfinite(
                            numeric.to_numpy(dtype=np.float64)).all() else 'unknown_or_missing_label')
                        reject({'member': member, 'line': int(line), 'reason': reason})
                    return accepted

                for line_number, raw in enumerate(stream, start=2):
                    body = raw.rstrip(b'\r\n')
                    reason = ('blank' if not body.strip() else
                              'quoted_record_unsupported' if b'"' in body else
                              'field_count' if body.count(b',') + 1 != len(columns) else None)
                    if reason:
                        reject({'member': member, 'line': line_number, 'reason': reason,
                                'fields': body.count(b',') + 1, 'expected_fields': len(columns)})
                        continue
                    rows.append(body + b'\n')
                    locations.append(line_number)
                    if len(rows) == chunk_rows:
                        yield member, flush()
                        rows, locations = [], []
                if rows:
                    yield member, flush()
