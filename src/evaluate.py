"""RMSE, Pearson, and score clipping."""

from __future__ import annotations

import numpy as np

from .paths import SCORE_HIGH, SCORE_LOW


def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def pearson(y_true, y_pred) -> float:
    """Pearson correlation. A constant vector has undefined correlation (NaN)."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    # min == max, not std == 0: a constant float can have a tiny computed std.
    if (
        y_true.size < 2
        or y_true.min() == y_true.max()
        or y_pred.min() == y_pred.max()
    ):
        return float("nan")
    corr = float(np.corrcoef(y_true, y_pred)[0, 1])
    if corr != corr:
        return float("nan")
    return corr


def clip_scores(y_pred, lo: float = SCORE_LOW, hi: float = SCORE_HIGH) -> np.ndarray:
    return np.clip(np.asarray(y_pred, dtype=np.float64), lo, hi)
