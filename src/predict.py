"""Write the assessment submission from test.csv order."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .evaluate import clip_scores
from .paths import N_TEST_EXPECTED, ROOT_SUBMISSION_CSV, SUBMISSION_CSV, TEST_CSV, ensure_output_dirs


def write_submission(filenames: list[str], predictions: np.ndarray, path: Path | None = None) -> pd.DataFrame:
    ensure_output_dirs()
    path = path or SUBMISSION_CSV
    test_order = pd.read_csv(TEST_CSV)["filename"].astype(str).tolist()
    if len(test_order) != N_TEST_EXPECTED:
        raise ValueError(f"test.csv has {len(test_order)} rows; expected {N_TEST_EXPECTED}")
    if list(filenames) != test_order:
        raise ValueError("Submission filenames must match test.csv order exactly")
    if len(predictions) != len(test_order):
        raise ValueError(f"Expected {len(test_order)} predictions, got {len(predictions)}")
    if not np.isfinite(np.asarray(predictions, dtype=np.float64)).all():
        raise ValueError("Predictions contain non-finite values")
    labels = clip_scores(predictions)
    if not np.isfinite(labels).all() or np.any(labels < 0) or np.any(labels > 5):
        raise ValueError("Predictions escaped the [0, 5] clip")
    out = pd.DataFrame({"filename": filenames, "label": labels})
    out.to_csv(path, index=False)
    if Path(path).resolve() == SUBMISSION_CSV.resolve():
        out.to_csv(ROOT_SUBMISSION_CSV, index=False)
    return out


def verify_submission(path: Path | None = None) -> dict:
    path = path or SUBMISSION_CSV
    pred = pd.read_csv(path)
    test = pd.read_csv(TEST_CSV)
    issues = []
    if list(pred.columns) != ["filename", "label"]:
        issues.append(f"columns={list(pred.columns)}")
    if len(pred) != N_TEST_EXPECTED or len(pred) != len(test):
        issues.append(f"n_rows={len(pred)}")
    if list(pred["filename"].astype(str)) != list(test["filename"].astype(str)):
        issues.append("filename order does not match test.csv")
    labels = pred["label"].to_numpy(dtype=np.float64)
    if not np.isfinite(labels).all():
        issues.append("non-finite labels")
    elif labels.min() < 0 or labels.max() > 5:
        issues.append(f"label range [{labels.min()}, {labels.max()}]")
    if np.any(labels == -1):
        issues.append("placeholder -1 still present")
    return {
        "path": str(path),
        "n_rows": int(len(pred)),
        "columns": list(pred.columns),
        "label_min": float(np.nanmin(labels)) if len(labels) else float("nan"),
        "label_max": float(np.nanmax(labels)) if len(labels) else float("nan"),
        "label_mean": float(np.nanmean(labels)) if len(labels) else float("nan"),
        "ok": not issues,
        "issues": issues,
    }
