"""
Parameter <-> model helpers.

Models are exchanged as an ordered ``list[np.ndarray]`` taken from
``model.parameters()`` (NOT ``state_dict``). This keeps the representation
identical whether or not the model is wrapped by Opacus' ``GradSampleModule``
(which renames state-dict keys with a ``_module.`` prefix but preserves the
parameter *order*), so DP and non-DP clients stay interoperable.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

Params = list[np.ndarray]


def get_params(model: nn.Module) -> Params:
    return [p.detach().cpu().numpy().copy() for p in model.parameters()]


def set_params(model: nn.Module, params: Params) -> None:
    with torch.no_grad():
        for p, new in zip(model.parameters(), params):
            p.copy_(torch.as_tensor(new, dtype=p.dtype, device=p.device))


def flatten(params: Params) -> np.ndarray:
    """Concatenate all layers into one 1-D vector (for distance-based rules)."""
    return np.concatenate([p.ravel() for p in params]).astype(np.float64)


def zeros_like(params: Params) -> Params:
    return [np.zeros_like(p) for p in params]
