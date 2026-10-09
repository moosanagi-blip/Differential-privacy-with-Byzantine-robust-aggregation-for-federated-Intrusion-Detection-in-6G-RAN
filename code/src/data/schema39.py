"""Explicit adapter for the downloaded 39-feature release.

Separate from the legacy 46-network-feature pipeline. This module validates
input chunks; it does not select a release, merge archives, split, or scale.
Radio augmentation remains synthetic and label-conditioned.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ATTACK_LABELS, RAN_FEATURES
from .ran_augment import add_ran_features

SCHEMA_ID = 'ciciot2023-network39-ran7-v1'
NETWORK_FEATURES = [
    'Header_Length', 'Protocol Type', 'Time_To_Live', 'Rate',
    'fin_flag_number', 'syn_flag_number', 'rst_flag_number',
    'psh_flag_number', 'ack_flag_number', 'ece_flag_number',
    'cwr_flag_number', 'ack_count', 'syn_count', 'fin_count', 'rst_count',
    'HTTP', 'HTTPS', 'DNS', 'Telnet', 'SMTP', 'SSH', 'IRC', 'TCP', 'UDP',
    'DHCP', 'ARP', 'ICMP', 'IGMP', 'IPv', 'LLC', 'Tot sum', 'Min', 'Max',
    'AVG', 'Std', 'Tot size', 'IAT', 'Number', 'Variance',
]
ALL_FEATURES = NETWORK_FEATURES + list(RAN_FEATURES)
_LABELS = {label.upper(): label for label in ATTACK_LABELS}
_LABELS.update({'BENIGN': 'BenignTraffic', 'BENIGN_FINAL': 'BenignTraffic'})


def validate_chunk(frame: pd.DataFrame, *, directory_label: str | None = None):
    """Return accepted float64 rows and explicit, mutually exclusive counts.

    For merged files require ``Label``; for per-attack files require an
    explicit directory label. No feature imputation or duplicate removal.
    The caller must preserve and report rejection counts before using output.
    CSV field-width checking must happen before pandas parsing: a malformed
    record can otherwise be padded or realigned by the CSV parser.
    """
    expected = NETWORK_FEATURES + ([] if directory_label is not None else ['Label'])
    if list(frame.columns) != expected:
        raise ValueError('Unexpected 39-feature release header or column order')
    if directory_label is not None:
        canonical = _LABELS.get(directory_label.strip().upper())
        if canonical is None:
            raise ValueError(f'Unknown source directory label: {directory_label}')
        labels = pd.Series(canonical, index=frame.index)
    else:
        labels = frame['Label'].astype('string').str.strip().str.upper().map(_LABELS)
    features = frame[NETWORK_FEATURES].apply(pd.to_numeric, errors='coerce').astype('float64')
    invalid_numeric = ~np.isfinite(features.to_numpy()).all(axis=1)
    invalid_label = ~invalid_numeric & labels.isna().to_numpy()
    accepted = ~(invalid_numeric | invalid_label)
    out = features.loc[accepted].copy()
    out['label'] = labels.loc[accepted]
    report = {'schema_id': SCHEMA_ID, 'input_rows': len(frame),
              'accepted_rows': int(accepted.sum()),
              'rejected_non_numeric_or_nonfinite': int(invalid_numeric.sum()),
              'rejected_unknown_or_missing_label': int(invalid_label.sum()),
              'duplicates_removed': 0}
    return out, report


def augment_validated(frame: pd.DataFrame, *, seed: int):
    """Append seven simulated columns without claiming measured radio data."""
    if list(frame.columns) != NETWORK_FEATURES + ['label']:
        raise ValueError('Expected validated 39-feature frame and canonical label')
    if not frame['label'].isin(ATTACK_LABELS).all():
        raise ValueError('Unknown canonical label')
    if not np.isfinite(frame[NETWORK_FEATURES].to_numpy(dtype=np.float64)).all():
        raise ValueError('Nonfinite network features')
    augmented = add_ran_features(frame, np.random.default_rng(seed))
    return augmented[ALL_FEATURES + ['label']]


def schema_manifest():
    return {'schema_id': SCHEMA_ID, 'network_features': list(NETWORK_FEATURES),
            'radio_features': list(RAN_FEATURES), 'feature_order': list(ALL_FEATURES),
            'n_features': len(ALL_FEATURES), 'raw_numeric_precision': 'float64',
            'radio_source': 'synthetic, label-conditioned; not measured telemetry',
            'label_mapping': dict(_LABELS)}
