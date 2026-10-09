"""The reported thesis figures, for printing beside live demo numbers.

The demo runs a handful of rounds so it finishes while someone is watching.
Those numbers show the right *trends* but sit below the reported results, which
use 30 rounds over the same complete corpus. Rather than let that gap go
unexplained, every experimental step prints the corresponding reported figure
next to its live one, read from the campaign's own result files under
``results/Research Objective */`` so the demo never hard-codes a number it
cannot point at.

Configurations the reported campaign did not run return ``n/a`` rather than a
stand-in taken from some other run.
"""

from __future__ import annotations

import json
from functools import lru_cache

from ..data.config import PROJECT_ROOT

CAMPAIGN = PROJECT_ROOT / "results"

# How many rounds the reported campaign used, for labelling the column.
ROUNDS = 30

# ``block/name`` as the steps ask for it -> the campaign's file stem.
_PREFIX = {
    "ablation": "ablation_",
    "privacy_utility": "pu_",
    "byzantine": "byz_",
    "adaptive_eps": "eps_",
}

_METRIC_FIELD = {"acc": "final_accuracy", "f1": "final_macro_f1",
                 "fpr": "final_fpr", "eps": "final_epsilon"}


def _stem(config: str) -> str | None:
    """``"privacy_utility/eps5"`` -> ``"pu_eps5"``. None if not in this campaign."""
    block, _, name = config.partition("/")
    prefix = _PREFIX.get(block)
    return f"{prefix}{name}" if prefix and name else None


@lru_cache(maxsize=1)
def _load() -> dict[str, dict]:
    """Map file stem -> record, read once from the campaign directory."""
    if not CAMPAIGN.exists():
        return {}
    out: dict[str, dict] = {}
    for path in CAMPAIGN.glob("**/*.json"):
        try:
            rec = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if "final_accuracy" in rec:
            out[path.stem] = rec
    return out


def get(config: str) -> dict | None:
    """One reported record, e.g. ``get("privacy_utility/eps5")``."""
    stem = _stem(config)
    return _load().get(stem) if stem else None


def fmt(config: str, metric: str = "acc") -> str:
    """``"0.839"``, or ``"n/a"`` when the campaign has no such run."""
    rec = get(config)
    if not rec:
        return "n/a"
    value = rec.get(_METRIC_FIELD.get(metric, f"final_{metric}"))
    return f"{value:.3f}" if isinstance(value, (int, float)) else "n/a"


def available() -> bool:
    return bool(_load())


def column(metric: str = "") -> str:
    """Header for the reported-figure column, so every table labels it the same
    way and nobody has to guess which campaign the number came from."""
    what = f"reported {metric}" if metric else "reported"
    return f"{what} ({ROUNDS} rounds, full corpus)"


def compare_rows(pairs: list[tuple[str, str, float]]) -> list[list]:
    """Build ``[label, live, reported]`` rows.

    ``pairs`` is ``(label, campaign_config_key, live_value)``.
    """
    rows = []
    for label, key, live in pairs:
        rows.append([label, f"{live:.3f}" if live is not None else "-", fmt(key)])
    return rows
