"""Load CSVs with split-aware audio paths. Test labels are never used."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .paths import TEST_CSV, TRAIN_CSV, audio_path

SCORE_BINS = ("0", "1-2", "2.5-3.5", "4-5")


def score_bin(score: float) -> str:
    """Stratification bins: 0 | 1–2 | 2.5–3.5 | 4–5."""
    if score == 0:
        return "0"
    if score <= 2:
        return "1-2"
    if score <= 3.5:
        return "2.5-3.5"
    return "4-5"


def load_train() -> pd.DataFrame:
    df = pd.read_csv(TRAIN_CSV)
    if list(df.columns) != ["filename", "label"]:
        raise ValueError(f"Unexpected train.csv columns: {list(df.columns)}")
    df = df.copy()
    df["label"] = df["label"].astype(float)
    df["split"] = "train"
    df["path"] = [str(audio_path(name, "train")) for name in df["filename"]]
    missing = [p for p in df["path"] if not Path(p).is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} train audio files missing, e.g. {missing[:3]}")
    df["stratum"] = df["label"].map(score_bin)
    df["is_zero"] = df["label"] == 0.0
    return df.reset_index(drop=True)


def load_test_ids() -> pd.DataFrame:
    """Return test identifiers in CSV order. The placeholder label is discarded."""
    df = pd.read_csv(TEST_CSV)
    if "filename" not in df.columns:
        raise ValueError(f"test.csv missing filename column: {list(df.columns)}")
    out = pd.DataFrame({"filename": df["filename"].astype(str)})
    out["split"] = "test"
    out["path"] = [str(audio_path(name, "test")) for name in out["filename"]]
    missing = [p for p in out["path"] if not Path(p).is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} test audio files missing, e.g. {missing[:3]}")
    return out.reset_index(drop=True)
