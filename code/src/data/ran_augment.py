"""
Synthetic 6G RAN telemetry augmentation.

No public 6G RAN intrusion dataset exists, so for each CICIoT2023 flow we
generate 7 attack-conditioned radio-layer metrics. The perturbations are
attack-conditional (not random noise): e.g. DDoS drives PRB utilisation
toward saturation, jamming-like attacks degrade SINR/RSRQ, and MITM/spoofing
inflates the handover rate. Ranges are anchored to 3GPP specifications:

  RSRP (dBm)  TS 38.215 reportable ~[-156,-31];  usable cell ~[-110,-70]
  SINR (dB)   TS 38.215 ~[-23,+40];              usable cell ~[-5,+30]
  RSRQ (dB)   TS 38.215 ~[-43,+20];              usable cell ~[-20,-3]
  PRB_Util(%) 0-100
  Handover_Rate (per hr)   normal 0-10
  Beam_Failure_Count (per hr) normal 0-2
  Throughput (Mbps)        per-UE typical 50-2000

This module supersedes an earlier single-row prototype: 7 RAN features rather
than 6 (RSRQ added) and vectorised so it scales to hundreds of thousands of
rows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import RAN_FEATURES, label_to_category

# Per-category (low, high) uniform ranges for each RAN feature.
# Gaussian jitter (sd) is added afterwards to avoid hard boundaries.
DIST: dict[str, dict[str, tuple[float, float]]] = {
    #              rsrp           sinr        rsrq         prb         ho         bf        thr
    "Benign":     dict(rsrp=(-95, -75),  sinr=(10, 25), rsrq=(-12, -5),  prb=(10, 45),  ho=(0, 3),  bf=(0, 1), thr=(300, 1500)),
    "DDoS":       dict(rsrp=(-95, -75),  sinr=(8, 22),  rsrq=(-15, -7),  prb=(90, 100), ho=(0, 4),  bf=(0, 2), thr=(20, 200)),
    "DoS":        dict(rsrp=(-95, -75),  sinr=(8, 22),  rsrq=(-15, -7),  prb=(75, 95),  ho=(0, 4),  bf=(0, 2), thr=(40, 250)),
    "Mirai":      dict(rsrp=(-100, -80), sinr=(5, 20),  rsrq=(-17, -8),  prb=(85, 100), ho=(0, 5),  bf=(0, 2), thr=(30, 200)),
    "Recon":      dict(rsrp=(-95, -75),  sinr=(10, 25), rsrq=(-12, -5),  prb=(15, 50),  ho=(0, 3),  bf=(0, 1), thr=(200, 1200)),
    "MITM_Spoof": dict(rsrp=(-100, -75), sinr=(5, 20),  rsrq=(-18, -9),  prb=(20, 60),  ho=(8, 30), bf=(1, 4), thr=(100, 900)),
    "Injection":  dict(rsrp=(-95, -75),  sinr=(10, 25), rsrq=(-12, -5),  prb=(15, 50),  ho=(0, 3),  bf=(0, 1), thr=(200, 1200)),
    "Malware_BF": dict(rsrp=(-100, -78), sinr=(6, 22),  rsrq=(-16, -7),  prb=(20, 65),  ho=(0, 6),  bf=(0, 2), thr=(80, 800)),
}

# (uniform-range key, gaussian sd, clip-low, clip-high, integer?)
_SPEC: dict[str, tuple[str, float, float | None, float | None, bool]] = {
    "RSRP_dBm":           ("rsrp", 1.5, -156, -31,  False),
    "SINR_dB":            ("sinr", 1.0, -23,  40,    False),
    "RSRQ_dB":            ("rsrq", 1.0, -43,  20,    False),
    "PRB_Util_percent":   ("prb",  2.0, 0,    100,   False),
    "Handover_Rate_hr":   ("ho",   0.5, 0,    None,  True),
    "Beam_Failure_Count": ("bf",   0.3, 0,    None,  True),
    "Throughput_Mbps":    ("thr",  20.0, 0,   None,  False),
}


def add_ran_features(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Append the 7 attack-conditioned RAN telemetry columns to ``df``.

    ``df`` must contain a ``label`` column. Returns a copy with RAN columns
    inserted just before ``label`` (matching the prototype layout).
    """
    n = len(df)
    cats = df["label"].astype(str).map(label_to_category).to_numpy()

    out_cols: dict[str, np.ndarray] = {}
    for feat in RAN_FEATURES:
        key, sd, lo, hi, is_int = _SPEC[feat]
        low = np.empty(n)
        high = np.empty(n)
        for cat, d in DIST.items():
            mask = cats == cat
            if mask.any():
                low[mask], high[mask] = d[key]
        vals = rng.uniform(low, high) + rng.normal(0.0, sd, size=n)
        if lo is not None or hi is not None:
            vals = np.clip(vals, lo if lo is not None else -np.inf,
                           hi if hi is not None else np.inf)
        if is_int:
            vals = np.maximum(0, np.round(vals)).astype(int)
        else:
            vals = np.round(vals, 2)
        out_cols[feat] = vals

    ran_df = pd.DataFrame(out_cols, index=df.index)
    label = df["label"]
    base = df.drop(columns=["label"])
    return pd.concat([base, ran_df, label], axis=1)
