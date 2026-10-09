#!/usr/bin/env python3
"""
Confusion matrix + per-category recall figure for the headline model (RO3).

Trains one federated global model (FL + Multi-Krum, no attack by default) and
plots (a) a row-normalised 34×34 confusion matrix and (b) per-category (8-family)
recall, on the held-out global test set. Writes PNGs + a per-class CSV into the
chosen results dir. Reuses the engine (`run_federated(..., return_model=True)`)
and the metric helpers so numbers match the main campaign.

Run:  python scripts/confusion_figure.py                       # MLP, 30 rounds
      python scripts/confusion_figure.py --model lstm --rounds 30 --local-epochs 3
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib                                    # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402
import numpy as np                                    # noqa: E402

from src.data import loaders                          # noqa: E402
from src.data.config import PROJECT_ROOT             # noqa: E402
from src.evaluation import metrics as M               # noqa: E402
from src.experiment_config import RunConfig           # noqa: E402
from src.fl.simulation import run_federated, _resolve_device  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlp", choices=["mlp", "lstm", "transformer"])
    ap.add_argument("--rounds", type=int, default=30)
    ap.add_argument("--local-epochs", type=int, default=1, dest="local_epochs")
    ap.add_argument("--device", default="cpu", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--optimizer", default="sgd", choices=["adam", "sgd"])
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--data-root", default="data_full", dest="data_root",
                    help="prepared dataset root (default: data_full)")
    ap.add_argument("--out", default=str(PROJECT_ROOT / "results" / "Research Objective 3"))
    args = ap.parse_args()
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    lr = args.lr if args.lr is not None else (0.1 if args.optimizer == "sgd" else 1e-3)

    cfg = RunConfig(name=f"confusion_{args.model}", model=args.model,
                    aggregator="multikrum", dp=False, attack="none",
                    rounds=args.rounds, local_epochs=args.local_epochs,
                    optimizer=args.optimizer, lr=lr, device=args.device,
                    data_root=args.data_root, round_chunks=True)
    print(f"training {args.model} (rounds={args.rounds}, epochs={args.local_epochs}) …")
    result, model = run_federated(cfg, verbose=True, return_model=True)

    device = _resolve_device(args.device)
    # Stream the complete held-out set rather than materialising it: at full
    # corpus scale it is 9.34M rows, about 2 GB as float32.
    root = loaders.DataRoot(args.data_root)
    test_loader = root.stream_test(batch_size=4096, subset=False)
    y_true, y_pred = M.predict(model, test_loader, device)

    id2label = {v: k for k, v in root.label_map().items()}
    n = root.num_classes()
    labels = [id2label[i] for i in range(n)]
    cm = M.confusion(y_true, y_pred, n).astype(np.float64)
    row = cm.sum(axis=1, keepdims=True)
    cmn = np.divide(cm, row, out=np.zeros_like(cm), where=row > 0)

    # (a) confusion matrix heatmap
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(cmn, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Row-normalised confusion matrix — {args.model.upper()} "
                 f"(acc={result['final_accuracy']:.3f})")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="recall")
    fig.tight_layout(); fig.savefig(outdir / f"fig_confusion_{args.model}.png", dpi=150)
    plt.close(fig)

    # (b) per-category recall
    pc = M.per_category_metrics(y_true, y_pred)
    cats = sorted(pc)
    rec = [pc[c]["recall"] for c in cats]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(cats, rec, color="#2a7fb8")
    ax.set_ylim(0, 1); ax.set_ylabel("Recall")
    ax.set_title(f"Per-category recall — {args.model.upper()}")
    ax.tick_params(axis="x", rotation=30)
    for i, v in enumerate(rec):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
    fig.tight_layout(); fig.savefig(outdir / f"fig_per_category_{args.model}.png", dpi=150)
    plt.close(fig)

    # per-class recall CSV
    diag = np.diag(cm); support = cm.sum(axis=1)
    lines = ["label,support,recall"]
    for i in range(n):
        r = diag[i] / support[i] if support[i] > 0 else 0.0
        lines.append(f"{labels[i]},{int(support[i])},{r:.4f}")
    (outdir / f"per_class_recall_{args.model}.csv").write_text("\n".join(lines), encoding="utf-8")

    print(f"wrote fig_confusion_{args.model}.png, fig_per_category_{args.model}.png, "
          f"per_class_recall_{args.model}.csv → {outdir}")


if __name__ == "__main__":
    main()
