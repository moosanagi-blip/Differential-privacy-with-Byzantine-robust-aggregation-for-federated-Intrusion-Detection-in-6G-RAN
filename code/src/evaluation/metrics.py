"""
Evaluation metrics for the DP-FL-IDS (RO3).

Reports the detection-quality metrics used throughout the results chapter:
accuracy, macro-averaged F1, and the **false-positive rate** (benign flows
predicted as an attack — the operational false-alarm rate). Convergence speed,
cumulative privacy budget, and per-round overhead are time-series metrics
tracked by the FL engine, not here. Also provides the confusion matrix and a
per-category (8-family) breakdown for the analysis chapter.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)

from ..data import loaders
from ..data.config import label_to_category


@torch.no_grad()
def predict(model: nn.Module, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.to(device).eval()
    ys, ps = [], []
    for xb, yb in loader:
        logits = model(xb.to(device))
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Non-finite evaluation logits; refusing to report classification metrics")
        ps.append(logits.argmax(1).cpu().numpy())
        ys.append(yb.numpy())
    return np.concatenate(ys), np.concatenate(ps)


def _benign_id() -> int | None:
    return loaders.load_label_map().get("BenignTraffic")


def false_positive_rate(y_true: np.ndarray, y_pred: np.ndarray, label_map=None) -> float:
    """Benign samples wrongly flagged as an attack / all benign samples."""
    bid = _benign_id() if label_map is None else label_map.get('BenignTraffic')
    if bid is None:
        return float("nan")
    benign = y_true == bid
    n = int(benign.sum())
    if n == 0:
        return float("nan")
    false_alarms = int(((y_pred != bid) & benign).sum())
    return false_alarms / n


def core_metrics(y_true: np.ndarray, y_pred: np.ndarray, label_map=None) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "fpr": false_positive_rate(y_true, y_pred, label_map),
    }


def class_metrics(y_true, y_pred, label_map):
    """Auditable class counts; undefined class rates are JSON null, not NaN.

    Matrix rows are true classes and columns predicted classes, ordered by ID.
    Macro metrics elsewhere retain their existing observed-label convention.
    """
    ordered = sorted(label_map.items(), key=lambda item: item[1])
    if not ordered or [i for _, i in ordered] != list(range(len(ordered))):
        raise ValueError('Class IDs must be unique contiguous integers from zero')
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if y_true.ndim != 1 or y_true.shape != y_pred.shape or not len(y_true):
        raise ValueError('Expected nonempty equal-length one-dimensional labels')
    for values in (y_true, y_pred):
        if not np.issubdtype(values.dtype, np.integer) or (values < 0).any() or (values >= len(ordered)).any():
            raise ValueError('Observed class ID outside the declared label map')
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(ordered))))
    support, predicted = matrix.sum(axis=1), matrix.sum(axis=0)
    total = int(matrix.sum())
    classes = {}
    for name, i in ordered:
        tp = int(matrix[i, i]); fn = int(support[i]) - tp; fp = int(predicted[i]) - tp
        classes[name] = {'class_id': i, 'support': int(support[i]),
            'predicted_count': int(predicted[i]), 'true_positive': tp,
            'false_negative': fn, 'false_positive': fp, 'true_negative': total-tp-fn-fp,
            'precision': tp/int(predicted[i]) if predicted[i] else None,
            'recall': tp/int(support[i]) if support[i] else None,
            'f1': 2*tp/(int(support[i])+int(predicted[i])) if support[i]+predicted[i] else None}
    return {'class_order': [name for name,_ in ordered], 'confusion_matrix': matrix.tolist(),
            'confusion_matrix_axes': 'rows=true; columns=predicted', 'per_class': classes,
            'evaluated_rows': total, 'undefined_class_rates': 'null when denominator is zero',
            'macro_label_scope': 'existing core metrics use observed true/predicted label union'}


def evaluate_full(model: nn.Module, loader, device, label_map=None) -> dict:
    y_true, y_pred = predict(model, loader, device)
    label_map = loaders.load_label_map() if label_map is None else label_map
    details = class_metrics(y_true, y_pred, label_map)
    m = core_metrics(y_true, y_pred, label_map)
    m["per_category"] = per_category_metrics(y_true, y_pred, label_map)
    m.update(details)
    return m


def _category_lut(label_map=None) -> tuple[np.ndarray, list[str]]:
    """``(label_id -> category_index)`` lookup table, plus the category names.

    Built once per process. The previous implementation resolved the category of
    *every test row* with a dict lookup plus a chain of string comparisons, twice
    — on a 9M-row held-out set that is ~18M Python-level string classifications
    per evaluation. There are only 34 labels, so a 34-element table indexed by
    label id gives the same answer as one vectorised gather.
    """
    global _LUT_CACHE
    if label_map is not None:
        id2label = {i: label for label,i in label_map.items()}
        categories = [label_to_category(id2label[i]) for i in range(len(id2label))]
        names = sorted(set(categories))
        return np.array([names.index(c) for c in categories], dtype=np.int16), names
    if _LUT_CACHE is None:
        id2label = loaders.id_to_label()
        n = (max(id2label) + 1) if id2label else 0
        cats: list[str] = []
        lut = np.zeros(n, dtype=np.int16)
        for i in range(n):
            cat = label_to_category(id2label[i]) if i in id2label else "Unknown"
            if cat not in cats:
                cats.append(cat)
            lut[i] = cats.index(cat)
        _LUT_CACHE = (lut, cats)
    return _LUT_CACHE


_LUT_CACHE: tuple[np.ndarray, list[str]] | None = None


def per_category_metrics(y_true: np.ndarray, y_pred: np.ndarray, label_map=None) -> dict:
    """Recall + support within each of the 8 attack families."""
    lut, cats = _category_lut(label_map)
    ct = lut[np.clip(y_true, 0, len(lut) - 1)]
    cp = lut[np.clip(y_pred, 0, len(lut) - 1)]
    support = np.bincount(ct, minlength=len(cats))
    hits = np.bincount(ct[ct == cp], minlength=len(cats))
    return {cats[i]: {"support": int(support[i]),
                      "recall": float(hits[i] / support[i]) if support[i] else 0.0}
            for i in np.argsort(cats).tolist() if support[i]}


def confusion(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
