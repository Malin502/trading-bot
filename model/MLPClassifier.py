from __future__ import annotations

from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn


class MLPClassifier(nn.Module):
    """
    Binary classifier returning logits.
    Input:  (B, in_dim)
    Output: (B,) logits
    """
    def __init__(self, in_dim: int = 32, hidden: List[int] | None = None, dropout: float = 0.2):
        super().__init__()
        if hidden is None:
            hidden = [128, 64]

        layers: List[nn.Module] = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.GELU(), nn.Dropout(dropout)]
            d = h
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def compute_pos_weight(y: np.ndarray) -> float:
    """
    pos_weight for BCEWithLogitsLoss.
    """
    pos = float((y == 1).sum())
    neg = float((y == 0).sum())
    if pos <= 0:
        return 1.0
    return neg / pos


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, thr: float = 0.5) -> Dict[str, float | int]:
    y_true = y_true.astype(np.int64)
    y_pred = (y_prob >= thr).astype(np.int64)

    acc = float((y_pred == y_true).mean())

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

    eps = 1e-12
    p = np.clip(y_prob, eps, 1 - eps)
    logloss = float(-(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)).mean())

    return {
        "acc": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "logloss": logloss,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }

def find_best_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    metric: str = "f1",
    grid: np.ndarray | None = None,
) -> Tuple[float, Dict[str, float | int]]:
    """
    Find best threshold on validation set.
    Default: maximize F1. Tie-break: higher precision, then higher recall.

    Returns:
      best_thr, metrics_at_best_thr
    """
    if grid is None:
        grid = np.linspace(0.05, 0.95, 19)  # 0.05刻み

    best_thr = 0.5
    best_m = None
    best_score = -1e18

    for thr in grid:
        m = compute_metrics(y_true, y_prob, thr=float(thr))

        if metric == "f1":
            score = float(m["f1"])
        elif metric == "precision":
            score = float(m["precision"])
        elif metric == "recall":
            score = float(m["recall"])
        elif metric == "acc":
            score = float(m["acc"])
        else:
            raise ValueError(f"Unknown metric: {metric}")

        # tie-break: precision > recall
        if (score > best_score + 1e-12) or (
            abs(score - best_score) <= 1e-12 and best_m is not None and (
                (m["precision"] > best_m["precision"] + 1e-12) or
                (abs(m["precision"] - best_m["precision"]) <= 1e-12 and m["recall"] > best_m["recall"] + 1e-12)
            )
        ):
            best_score = score
            best_thr = float(thr)
            best_m = m

    assert best_m is not None
    return best_thr, best_m