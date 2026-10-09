"""Split-aware filesystem paths and fixed experiment settings."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "Dataset_Final"
TRAIN_CSV = DATA_DIR / "train.csv"
TEST_CSV = DATA_DIR / "test.csv"
SAMPLE_SUBMISSION_CSV = DATA_DIR / "sample_submission.csv"
TRAIN_AUDIO_DIR = DATA_DIR / "train"
TEST_AUDIO_DIR = DATA_DIR / "test"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
FEATURE_CACHE = OUTPUT_DIR / "feature_cache.csv"
METRICS_JSON = OUTPUT_DIR / "metrics.json"
SUBMISSION_CSV = OUTPUT_DIR / "submission.csv"
ROOT_SUBMISSION_CSV = PROJECT_ROOT / "submission.csv"

SEED = 42
N_SPLITS = 5
HOLDOUT_FRACTION = 0.2
SCORE_LOW = 0.0
SCORE_HIGH = 5.0
N_TEST_EXPECTED = 216


def audio_path(filename: str, split: str) -> Path:
    """Resolve a WAV path inside exactly one split directory."""
    if split == "train":
        directory = TRAIN_AUDIO_DIR
    elif split == "test":
        directory = TEST_AUDIO_DIR
    else:
        raise ValueError(f"Unknown split {split!r}; expected 'train' or 'test'")
    path = directory / filename
    if path.parent.resolve() != directory.resolve():
        raise ValueError(f"Refusing path that escapes {directory}: {filename}")
    return path


def ensure_output_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
