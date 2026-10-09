"""Leakage-safe estimators. Scalers are fit only when the estimator is fit."""

from __future__ import annotations

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .paths import SEED

# Alphas are chosen inside each training fold by RidgeCV's own cross-validation.
RIDGE_ALPHAS = (0.01, 0.1, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0)

# Compared with stronger L2, larger leaves, and shallower trees on the same
# folds. Those settings did not beat this one by more than fold noise.
HGBR_PARAMS = {
    "max_depth": 3,
    "learning_rate": 0.05,
    "max_iter": 200,
    "min_samples_leaf": 20,
    "l2_regularization": 1.0,
    "random_state": SEED,
}


def make_ridge() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", RidgeCV(alphas=RIDGE_ALPHAS, scoring="neg_mean_squared_error")),
        ]
    )


def make_hgbr(**overrides) -> HistGradientBoostingRegressor:
    params = {**HGBR_PARAMS, **overrides}
    return HistGradientBoostingRegressor(**params)


def make_estimator(name: str):
    if name == "ridge":
        return make_ridge()
    if name == "hgbr":
        return make_hgbr()
    raise ValueError(f"Unknown model {name!r}")
