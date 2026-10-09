"""
Single source of truth for a DP-FL-IDS experiment run.

A ``RunConfig`` captures everything needed to reproduce one federated run —
model, FL rounds, DP settings, aggregator, and Byzantine attack — so the
orchestrator, the sequential engine, and the Flower engine all share identical
semantics. Every experiment in the RO3/RO4 matrix is one ``RunConfig``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class RunConfig:
    name: str = "run"

    # ── model / local training ──
    model: str = "lstm"              # lstm | mlp | transformer
    hidden: int = 64
    local_epochs: int = 1
    batch_size: int = 64
    lr: float = 1e-3
    optimizer: str = "adam"          # adam | sgd  (SGD is the canonical DP-SGD optimizer)
    momentum: float = 0.9            # used only by SGD

    # ── federated learning ──
    rounds: int = 15
    num_clients: int = 10
    clients_per_round: int | None = None   # None = all
    aggregator: str = "fedavg"        # fedavg | uniformavg | krum | multikrum | trimmedmean

    # ── differential privacy (RO1) ──
    dp: bool = False
    noise_multiplier: float = 1.0     # σ (static baseline)
    max_grad_norm: float = 1.0        # C
    delta: float = 1e-5
    target_epsilon: float | None = None   # if set, calibrate σ to this total ε

    # ── adaptive ε (RO2) ──
    adaptive_eps: bool = False
    eps_schedule: str = "decreasing_noise"   # see privacy/adaptive_eps.py

    # ── Byzantine attack (RO4) ──
    attack: str = "none"              # none | label_flip | sign_flip | scale | gaussian | random | adaptive_krum | boundary_krum
    byzantine_fraction: float = 0.0
    attack_scale: float = 10.0
    attack_noise_std: float = 1.0
    # Deployment rarely reveals the true attacker count.  ``None`` preserves
    # the original oracle setting; an integer is the operator's estimate.
    n_byzantine_estimate: int | None = None
    # When enabled, a same-sized compromised coalition is redrawn each round.
    rotate_byzantine: bool = False

    # ── dataset selection / scale ──
    # Which prepared dataset root to read: "data_full" is the complete corpus
    # built by src.data.build_scaled. None falls back to the module-level
    # default paths in src/data/config.py.
    data_root: str | None = None
    # Split each client's shard into disjoint chunks and train round r on chunk
    # r, so the run consumes every row exactly once. Required at full corpus
    # scale, where one local epoch over a 3.7M-row shard is 57,500 steps.
    round_chunks: bool = False
    # How many chunks to cut each shard into. Defaults to `rounds`, which is
    # what makes a run consume the corpus exactly once. Set it explicitly for a
    # short run that should still see a normal round's worth of data — a
    # single-round run would otherwise take the whole 3.7M-row shard, which is
    # ~8 GB across ten clients.
    n_chunks: int | None = None
    # Score each round on the smaller stratified eval subset; final metrics are
    # always computed on the complete held-out set.
    eval_on_subset: bool = True
    # Which client the privacy budget is calibrated for. Under Dirichlet skew the
    # smallest shard samples at the highest rate and so spends the most epsilon:
    # Corrected execution calibrates over all planned client/round profiles and
    # reports the maximum executed accountant bound. Legacy mean calibration is
    # rejected; historical reproduction requires the preserved historical code.
    dp_calibrate: str = "worst"       # corrected execution rejects legacy mean calibration
    # Feature ablation: retain the 53-column model interface but replace the
    # seven standardized synthetic RAN columns with their training-set mean.
    mask_ran_features: bool = False

    # ── misc ──
    seed: int = 42
    device: str = "auto"             # auto | cpu | cuda
    engine: str = "sequential"       # sequential | flower
    conv_threshold: float = 0.90      # accuracy for "rounds-to-threshold" metric

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "RunConfig":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})
