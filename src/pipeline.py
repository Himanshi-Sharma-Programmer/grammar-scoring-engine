"""Cross-validation and the final fit.

Scalers live inside the Ridge pipeline and are fit only on each training
partition. The 20% holdout is for plots; it is not used to choose the model.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from .evaluate import clip_scores, pearson, rmse
from .models import make_estimator, make_hgbr
from .paths import HOLDOUT_FRACTION, N_SPLITS, SEED


@dataclass
class FoldResult:
    fold: int
    n_train: int
    n_val: int
    train_rmse: float
    val_rmse: float
    val_pearson: float


def feature_matrix(feat_df: pd.DataFrame, columns: list[str]) -> np.ndarray:
    missing = [name for name in columns if name not in feat_df.columns]
    if missing:
        raise KeyError(f"Missing feature columns: {missing}")
    return feat_df[columns].to_numpy(dtype=np.float64)


def _nanmean(values: list[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or np.all(np.isnan(array)):
        return float("nan")
    return float(np.nanmean(array))


def _nanstd(values: list[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or np.all(np.isnan(array)):
        return float("nan")
    return float(np.nanstd(array))


def cross_validate(
    X: np.ndarray,
    y: np.ndarray,
    strata: np.ndarray,
    factory: Callable[[], Any] | None,
    n_splits: int = N_SPLITS,
    seed: int = SEED,
) -> dict[str, Any]:
    """Stratified K-fold.

    `factory` builds a fresh estimator for each fold. None predicts the
    training-fold mean. Metrics use predictions clipped to [0, 5].
    """
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds: list[FoldResult] = []
    oof_pred = np.full(len(y), np.nan, dtype=np.float64)
    for fold, (tr, va) in enumerate(splitter.split(np.zeros(len(y)), strata), start=1):
        y_tr, y_va = y[tr], y[va]
        if factory is None:
            mean_value = float(np.mean(y_tr))
            pred_tr = np.full(y_tr.shape, mean_value, dtype=np.float64)
            pred_va = np.full(y_va.shape, mean_value, dtype=np.float64)
        else:
            model = factory()
            model.fit(X[tr], y_tr)
            pred_tr = clip_scores(model.predict(X[tr]))
            pred_va = clip_scores(model.predict(X[va]))
        folds.append(
            FoldResult(
                fold=fold,
                n_train=int(len(tr)),
                n_val=int(len(va)),
                train_rmse=rmse(y_tr, pred_tr),
                val_rmse=rmse(y_va, pred_va),
                val_pearson=pearson(y_va, pred_va),
            )
        )
        oof_pred[va] = pred_va

    train_rmses = [fold.train_rmse for fold in folds]
    val_rmses = [fold.val_rmse for fold in folds]
    val_pearsons = [fold.val_pearson for fold in folds]
    return {
        "folds": [asdict(fold) for fold in folds],
        "mean_train_rmse": float(np.mean(train_rmses)),
        "mean_val_rmse": float(np.mean(val_rmses)),
        "mean_val_pearson": _nanmean(val_pearsons),
        "std_val_rmse": float(np.std(val_rmses)),
        "std_val_pearson": _nanstd(val_pearsons),
        "oof_pred": oof_pred,
    }


def stratified_cv(
    X: np.ndarray,
    y: np.ndarray,
    strata: np.ndarray,
    model_name: str | None,
    n_splits: int = N_SPLITS,
    seed: int = SEED,
) -> dict[str, Any]:
    """Cross-validate a named model: None (mean), 'ridge', or 'hgbr'."""
    factory = None if model_name is None else (lambda: make_estimator(model_name))
    result = cross_validate(X, y, strata, factory, n_splits=n_splits, seed=seed)
    result["model"] = model_name or "mean"
    return result


def holdout_split(
    X: np.ndarray,
    y: np.ndarray,
    strata: np.ndarray,
    model_name: str,
    test_size: float = HOLDOUT_FRACTION,
    seed: int = SEED,
) -> dict[str, Any]:
    """One stratified holdout for residual plots. Not used for model choice."""
    index = np.arange(len(y))
    tr, va = train_test_split(index, test_size=test_size, random_state=seed, stratify=strata)
    model = make_estimator(model_name)
    model.fit(X[tr], y[tr])
    pred_tr = clip_scores(model.predict(X[tr]))
    pred_va = clip_scores(model.predict(X[va]))
    return {
        "model": model_name,
        "n_train": int(len(tr)),
        "n_val": int(len(va)),
        "train_idx": tr,
        "val_idx": va,
        "train_rmse": rmse(y[tr], pred_tr),
        "val_rmse": rmse(y[va], pred_va),
        "val_pearson": pearson(y[va], pred_va),
        "y_val": y[va],
        "pred_val": pred_va,
    }


def slice_metrics(y: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    return {
        "n": int(np.sum(mask)),
        "rmse": rmse(y[mask], pred[mask]),
        "pearson": pearson(y[mask], pred[mask]),
    }


def fit_final(X: np.ndarray, y: np.ndarray, model_name: str):
    """Fit on every labeled row. The returned RMSE is in-sample."""
    model = make_estimator(model_name)
    model.fit(X, y)
    prediction = clip_scores(model.predict(X))
    return model, prediction, rmse(y, prediction)


def boosting_factory(**overrides):
    """Fresh HistGradientBoostingRegressor, with optional parameter overrides."""
    return lambda: make_hgbr(**overrides)
