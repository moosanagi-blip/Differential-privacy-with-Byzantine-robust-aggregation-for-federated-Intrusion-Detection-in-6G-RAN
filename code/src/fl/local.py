"""
One gNB's local update for a single FL round — the shared "fit" logic used by
both the sequential engine and the Flower ClientApp.

Pipeline: load global weights → (if Byzantine data-poisoning) flip labels →
train locally (optionally with Opacus DP-SGD at the round's noise multiplier σ)
→ (if Byzantine update-poisoning) transform the update → return new parameters.
"""

from __future__ import annotations

import numpy as np
import torch

from ..attacks import byzantine
from ..models.lstm_ids import build_model
from ..privacy.dp_engine import attach_dp
from ..training import train_epochs
from .params import Params, get_params, set_params


def local_train(*, global_params: Params, x: np.ndarray, y: np.ndarray,
                n_features: int, n_classes: int, model_name: str, hidden: int,
                epochs: int, batch_size: int, lr: float, device: torch.device,
                is_byzantine: bool = False, attack: str = "none",
                attack_scale: float = 10.0, attack_noise_std: float = 1.0,
                dp: bool = False, noise_multiplier: float = 1.0,
                max_grad_norm: float = 1.0, seed: int = 0,
                optimizer_name: str = "adam", momentum: float = 0.9
                ) -> tuple[Params, int, dict]:
    """Return ``(new_params, num_examples, info)`` for one client-round."""
    if dp and (not np.isfinite(noise_multiplier) or noise_multiplier <= 0):
        raise ValueError("DP training requires a positive finite noise multiplier")
    torch.manual_seed(seed)
    model = build_model(model_name, n_features, n_classes, hidden=hidden)
    dtype = np.asarray(global_params[0]).dtype
    if dtype not in (np.dtype('float32'), np.dtype('float64')) or np.asarray(x).dtype != dtype:
        raise ValueError('Model and input precision must agree; refusing implicit precision loss')
    if dtype == np.dtype('float64'):
        model = model.double()
    set_params(model, global_params)
    model.to(device)

    # Data poisoning (label flip) happens before training.
    y_train = byzantine.maybe_flip_labels(
        y, n_classes, active=(is_byzantine and attack == "label_flip"), seed=seed)

    # Preserve the explicitly matched input/model precision, including float64.
    # np.asarray avoids duplicating an already correctly typed client slice.
    ds = torch.utils.data.TensorDataset(
        torch.from_numpy(np.asarray(x, dtype=dtype)),
        torch.from_numpy(np.asarray(y_train, dtype=np.int64)))
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)

    info: dict = {"dp": bool(dp), "num_examples": int(len(y))}
    # SGD is the canonical DP-SGD optimizer; Adam's per-coordinate normalisation
    # fights the clip+noise and badly hurts DP utility (see RO2/RO3 analysis).
    if optimizer_name.lower() == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if dp and noise_multiplier and noise_multiplier > 0:
        engine, model, optimizer, loader = attach_dp(
            model, optimizer, loader, noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm)
        train_epochs(model, loader, epochs=epochs, device=device, optimizer=optimizer)
        history = [(float(s), float(q), int(n)) for s, q, n in engine.accountant.history]
        q = float(loader.sample_rate)
        steps = sum(n for _, _, n in history)
        info.update(noise_multiplier=float(noise_multiplier), sample_rate=float(q),
                    steps=int(steps), accountant_history=history,
                    expected_batch_size=int(optimizer.expected_batch_size))
    else:
        train_epochs(model, loader, epochs=epochs, device=device, optimizer=optimizer)
        info.update(noise_multiplier=0.0, sample_rate=0.0, steps=0)

    new_params = get_params(model)

    # Update poisoning (applied to the honest delta).
    if is_byzantine and attack in byzantine.UPDATE_ATTACKS:
        new_params = byzantine.apply_update_attack(
            attack, global_params, new_params,
            scale=attack_scale, noise_std=attack_noise_std, seed=seed)
        info["poisoned"] = True

    return new_params, int(len(y)), info
