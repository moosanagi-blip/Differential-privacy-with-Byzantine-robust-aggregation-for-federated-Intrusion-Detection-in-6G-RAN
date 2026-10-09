"""Offline simulator diagnostics of submitted payloads, never training inputs.

Ground-truth attacker IDs are simulation annotations, not an operational detector.
These summaries do not establish an additional privacy guarantee for telemetry.
"""
import numpy as np

from .params import flatten


def round_diagnostics(updates, participants, malicious, selected_positions, scores=None):
    n = len(updates)
    if not n or len(participants) != n or len(set(participants)) != n:
        raise ValueError('Expected distinct participant IDs matching payloads')
    chosen = list(selected_positions)
    if len(set(chosen)) != len(chosen) or any(not isinstance(i, (int, np.integer)) or not 0 <= i < n for i in chosen):
        raise ValueError('Invalid selected positions')
    bad = np.array([c in malicious for c in participants], dtype=bool)
    vectors = np.stack([flatten(u) for u in updates])
    if not np.isfinite(vectors).all():
        raise FloatingPointError('Non-finite diagnostic payload')
    honest = vectors[~bad]
    spread = displacement = ratio = None
    if len(honest):
        centre = honest.mean(axis=0)
        # Euclidean RMS radius around the honest submitted-payload centroid.
        spread = float(np.sqrt(np.mean(np.sum((honest-centre)**2, axis=1))))
        if bad.any():
            displacement = float(np.sqrt(np.mean(np.sum((vectors[bad]-centre)**2, axis=1))))
            ratio = displacement/spread if spread > 0 else None
    result = {
        'version': 'submitted-payload-v1',
        'scope': 'offline simulator analysis; not an additional telemetry privacy guarantee',
        'participants': [int(c) for c in participants],
        'malicious_participant_count': int(bad.sum()),
        'malicious_selected_count': int(bad[chosen].sum()) if chosen else 0,
        'honest_rms_radius': spread,
        'malicious_rms_distance_to_honest_centroid': displacement,
        'malicious_distance_over_honest_radius': ratio,
        'geometry_definition': 'Euclidean distances of submitted model payloads, not counterfactual attack perturbations; null ratio when honest radius is zero',
        'score_selection': None,
    }
    if scores is not None:
        s = np.asarray(scores, dtype=np.float64)
        if s.shape != (n,) or not np.isfinite(s).all():
            raise ValueError('Expected one finite score per participant')
        rejected = [i for i in range(n) if i not in chosen]
        threshold = float(max(s[chosen])) if chosen else None
        def quantiles(values):
            return dict(zip(('min', 'q25', 'median', 'q75', 'max'),
                            map(float, np.quantile(values, [0, .25, .5, .75, 1])))) if len(values) else None
        result['score_selection'] = {
            'max_selected_score': threshold,
            'min_rejected_score': float(min(s[rejected])) if rejected else None,
            'boundary_gap': float(min(s[rejected])-threshold) if rejected and chosen else None,
            'honest_score_quantiles': quantiles(s[~bad]),
            'malicious_score_quantiles': quantiles(s[bad]),
            'clients': [{'client': int(c), 'score': float(s[i]),
                         'selected': i in chosen, 'malicious': bool(bad[i]),
                         'signed_margin': float(threshold-s[i]) if threshold is not None else None}
                        for i, c in enumerate(participants)],
            'margin_definition': 'max selected score minus client score; positive is below cutoff; zero can be selected or rejected in a tie; actual selected flag is authoritative',
        }
    # Catch overflow in derived geometry as well as non-finite inputs.
    for value in (spread, displacement, ratio):
        if value is not None and not np.isfinite(value):
            raise FloatingPointError('Non-finite derived geometry')
    return result
