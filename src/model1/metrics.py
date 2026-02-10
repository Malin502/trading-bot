from __future__ import annotations
import numpy as np

def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))

def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))

def corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])

def rank_ic(pred: np.ndarray, target: np.ndarray) -> float:
    """Rank Information Coefficient (Spearman correlation)"""
    from scipy.stats import spearmanr
    if len(pred) < 2:
        return 0.0
    rho, _ = spearmanr(pred, target, nan_policy='omit')
    if np.isnan(rho):
        return 0.0
    return float(rho)
