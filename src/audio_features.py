"""Handcrafted acoustic features computed with numpy and scipy.

Every feature is computed per file. Labels are not used. Duration columns are
kept in the cache so the duration ablation can reuse one extraction pass.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.fft import dct
from scipy.signal import stft

FEATURE_VERSION = "v1"
N_MFCC = 13
N_MELS = 20
FRAME_MS = 20
SILENCE_ABS = 0.01
VOICED_RMS = 0.02
PAUSE_MIN_SEC = 0.20
N_FFT = 512

DURATION_FEATURES = (
    "duration_sec",
    "log_duration",
    "voiced_sec",
    "n_stft_frames",
)

# Collinear with zero-crossing rate (|r| > 0.93) and with each other.
# band_very_high_frac is identically zero at 16 kHz because the top edge
# equals the Nyquist frequency, so that bin is empty.
SPECTRAL_FEATURES = (
    "centroid_mean",
    "centroid_std",
    "bandwidth_mean",
    "bandwidth_std",
    "rolloff_mean",
    "flatness_mean",
    "band_low_frac",
    "band_mid_frac",
    "band_high_frac",
    "band_very_high_frac",
)

# clip_frac barely correlates with the score. Absolute pause lengths are not
# duration-normalized and add nothing beyond pause_rate.
WEAK_FEATURES = (
    "clip_frac",
    "mean_pause_sec",
    "max_pause_sec",
)

EXCLUDED_FEATURES = SPECTRAL_FEATURES + WEAK_FEATURES

ENERGY_FEATURES = (
    "rms",
    "peak_abs",
    "crest_factor",
    "silence_frac",
    "clip_frac",
    "voiced_frac",
    "energy_std",
    "energy_p10",
    "energy_p90",
    "energy_iqr",
)
PAUSE_FEATURES = ("pause_rate", "mean_pause_sec", "max_pause_sec")
ZCR_FEATURES = ("zcr_mean", "zcr_std")
MFCC_MEAN_FEATURES = tuple(f"mfcc{i}_mean" for i in range(1, N_MFCC + 1))
MFCC_STD_FEATURES = tuple(f"mfcc{i}_std" for i in range(1, N_MFCC + 1))
MFCC_FEATURES = tuple(
    name for i in range(1, N_MFCC + 1) for name in (f"mfcc{i}_mean", f"mfcc{i}_std")
)

FEATURE_GROUPS = {
    "energy": ENERGY_FEATURES,
    "pause": PAUSE_FEATURES,
    "zcr": ZCR_FEATURES,
    "spectral": SPECTRAL_FEATURES,
    "mfcc_mean": MFCC_MEAN_FEATURES,
    "mfcc_std": MFCC_STD_FEATURES,
}

ALL_FEATURES = (
    "duration_sec",
    "log_duration",
    "rms",
    "peak_abs",
    "crest_factor",
    "silence_frac",
    "clip_frac",
    "voiced_frac",
    "voiced_sec",
    "energy_std",
    "energy_p10",
    "energy_p90",
    "energy_iqr",
    "pause_rate",
    "mean_pause_sec",
    "max_pause_sec",
    "zcr_mean",
    "zcr_std",
    "centroid_mean",
    "centroid_std",
    "bandwidth_mean",
    "bandwidth_std",
    "rolloff_mean",
    "flatness_mean",
    "band_low_frac",
    "band_mid_frac",
    "band_high_frac",
    "band_very_high_frac",
    "n_stft_frames",
    *MFCC_FEATURES,
)

_FILTERBANKS: dict[tuple[int, int, int], np.ndarray] = {}


def feature_columns(include_duration: bool) -> list[str]:
    if include_duration:
        return list(ALL_FEATURES)
    return [name for name in ALL_FEATURES if name not in DURATION_FEATURES]


def model_feature_columns(include_duration: bool = False) -> list[str]:
    """Features used by the submitted model.

    Duration is omitted because train clips are mostly 60 s and test clips
    mostly 45 s. Spectral summaries that duplicate zero-crossing rate, a band
    that is always zero at 16 kHz, clip fraction, and absolute pause lengths
    are omitted because they did not help cross-validation.
    """
    return [name for name in feature_columns(include_duration) if name not in EXCLUDED_FEATURES]


def load_wav_mono(path: str | Path) -> tuple[np.ndarray, int]:
    path = Path(path)
    with wave.open(str(path), "rb") as handle:
        n_channels = handle.getnchannels()
        sampwidth = handle.getsampwidth()
        sample_rate = handle.getframerate()
        n_frames = handle.getnframes()
        raw = handle.readframes(n_frames)
    if sampwidth == 2:
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 1:
        audio = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sampwidth == 4:
        audio = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported sample width {sampwidth} for {path}")
    if n_channels > 1:
        audio = audio.reshape(-1, n_channels).mean(axis=1)
    return np.ascontiguousarray(audio), int(sample_rate)


def _hz_to_mel(freq: np.ndarray | float) -> np.ndarray | float:
    return 2595.0 * np.log10(1.0 + np.asarray(freq) / 700.0)


def _mel_to_hz(mel: np.ndarray | float) -> np.ndarray | float:
    return 700.0 * (10.0 ** (np.asarray(mel) / 2595.0) - 1.0)


def _build_mel_filterbank(sample_rate: int, n_fft: int, n_mels: int) -> np.ndarray:
    f_max = sample_rate / 2.0
    mel_pts = np.linspace(_hz_to_mel(20.0), _hz_to_mel(f_max), n_mels + 2)
    hz_pts = _mel_to_hz(mel_pts)
    bins = np.floor((n_fft + 1) * hz_pts / sample_rate).astype(int)
    n_bins = n_fft // 2 + 1
    fb = np.zeros((n_mels, n_bins), dtype=np.float64)
    for i in range(n_mels):
        left, center, right = int(bins[i]), int(bins[i + 1]), int(bins[i + 2])
        if center <= left:
            center = left + 1
        if right <= center:
            right = center + 1
        right = min(right, n_bins)
        center = min(center, n_bins - 1)
        left_span = max(center - left, 1)
        right_span = max(right - center, 1)
        for j in range(left, center):
            if 0 <= j < n_bins:
                fb[i, j] = (j - left) / left_span
        for j in range(center, right):
            if 0 <= j < n_bins:
                fb[i, j] = (right - j) / right_span
    return fb


def _mel_filterbank(sample_rate: int, n_fft: int, n_mels: int = N_MELS) -> np.ndarray:
    key = (sample_rate, n_fft, n_mels)
    cached = _FILTERBANKS.get(key)
    if cached is None:
        cached = _build_mel_filterbank(sample_rate, n_fft, n_mels)
        _FILTERBANKS[key] = cached
    return cached


def _frame_rms(audio: np.ndarray, frame_len: int) -> np.ndarray:
    n_frames = audio.size // frame_len
    if n_frames == 0:
        return np.array([np.sqrt(np.mean(audio ** 2))], dtype=np.float64)
    frames = audio[: n_frames * frame_len].reshape(n_frames, frame_len)
    return np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)


def _zero_crossing_rate(audio: np.ndarray, frame_len: int, hop: int) -> np.ndarray:
    """Fraction of adjacent samples that change sign, per hop-sized frame."""
    if audio.size < frame_len:
        signs = np.sign(audio)
        signs[signs == 0] = 1
        return np.array([np.mean(np.abs(np.diff(signs)) > 0)], dtype=np.float64)

    signs = np.sign(audio)
    signs[signs == 0] = 1
    pair_change = signs[1:] * signs[:-1] < 0
    cumulative = np.empty(pair_change.size + 1, dtype=np.int32)
    cumulative[0] = 0
    cumulative[1:] = np.cumsum(pair_change, dtype=np.int32)
    n_frames = 1 + (audio.size - frame_len) // hop
    starts = np.arange(n_frames, dtype=np.int32) * hop
    counts = cumulative[starts + (frame_len - 1)] - cumulative[starts]
    return counts / np.float64(frame_len - 1)


def _pause_lengths(voiced: np.ndarray, seconds_per_frame: float) -> np.ndarray:
    """Unvoiced runs at least as long as the original frame-count threshold."""
    min_frames = max(int(PAUSE_MIN_SEC / seconds_per_frame), 1)
    unvoiced = ~voiced
    if unvoiced.size == 0 or not np.any(unvoiced):
        return np.array([], dtype=np.float64)
    padded = np.concatenate(([False], unvoiced, [False]))
    changes = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    run_frames = ends - starts
    kept = run_frames[run_frames >= min_frames]
    return kept.astype(np.float64) * seconds_per_frame


def extract_file_features(path: str | Path) -> dict[str, float]:
    audio, sample_rate = load_wav_mono(path)
    duration = float(audio.size / sample_rate) if sample_rate else 0.0
    if audio.size == 0:
        raise ValueError(f"Empty audio file: {path}")

    rms = float(np.sqrt(np.mean(audio ** 2) + 1e-12))
    peak = float(np.max(np.abs(audio)))
    silence_frac = float(np.mean(np.abs(audio) < SILENCE_ABS))
    clip_frac = float(np.mean(np.abs(audio) >= 0.99))

    frame_len = max(int(sample_rate * FRAME_MS / 1000.0), 1)
    hop = frame_len // 2
    frame_e = _frame_rms(audio, frame_len)
    voiced = frame_e > VOICED_RMS
    voiced_frac = float(np.mean(voiced))
    voiced_sec = float(voiced.sum() * (frame_len / sample_rate))
    energy_std = float(np.std(frame_e))
    energy_p10 = float(np.percentile(frame_e, 10))
    energy_p90 = float(np.percentile(frame_e, 90))
    energy_iqr = float(np.percentile(frame_e, 75) - np.percentile(frame_e, 25))

    seconds_per_frame = frame_len / sample_rate
    pause_lens = _pause_lengths(voiced, seconds_per_frame)
    pause_rate = float(pause_lens.size / max(duration, 1e-6))
    mean_pause = float(pause_lens.mean()) if pause_lens.size else 0.0
    max_pause = float(pause_lens.max()) if pause_lens.size else 0.0

    zcr = _zero_crossing_rate(audio, frame_len=frame_len, hop=hop)

    _freqs, times, spec = stft(
        audio,
        fs=sample_rate,
        nperseg=N_FFT,
        noverlap=N_FFT // 2,
        boundary=None,
        padded=False,
    )
    freqs = _freqs
    mag = np.abs(spec) + 1e-12
    power = mag ** 2
    freq_col = freqs[:, None]
    spec_sum = power.sum(axis=0) + 1e-12
    centroid = (freq_col * power).sum(axis=0) / spec_sum
    bandwidth = np.sqrt(((freq_col - centroid) ** 2 * power).sum(axis=0) / spec_sum)
    cumsum = np.cumsum(power, axis=0)
    rolloff_idx = np.argmax(cumsum >= 0.85 * spec_sum, axis=0)
    rolloff = freqs[np.clip(rolloff_idx, 0, len(freqs) - 1)]
    flatness = np.exp(np.mean(np.log(mag), axis=0)) / (np.mean(mag, axis=0) + 1e-12)

    nyquist = sample_rate / 2.0
    edges = (0.0, 500.0, 2000.0, min(8000.0, nyquist), nyquist)
    total_band = float(power.sum()) + 1e-12
    band_fracs = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (freqs >= lo) & (freqs < hi)
        band_fracs.append(float(power[mask].sum() / total_band) if np.any(mask) else 0.0)

    features: dict[str, float] = {
        "duration_sec": duration,
        "log_duration": float(np.log(duration + 1e-6)),
        "rms": rms,
        "peak_abs": peak,
        "crest_factor": float(peak / (rms + 1e-12)),
        "silence_frac": silence_frac,
        "clip_frac": clip_frac,
        "voiced_frac": voiced_frac,
        "voiced_sec": voiced_sec,
        "energy_std": energy_std,
        "energy_p10": energy_p10,
        "energy_p90": energy_p90,
        "energy_iqr": energy_iqr,
        "pause_rate": pause_rate,
        "mean_pause_sec": mean_pause,
        "max_pause_sec": max_pause,
        "zcr_mean": float(np.mean(zcr)),
        "zcr_std": float(np.std(zcr)),
        "centroid_mean": float(np.mean(centroid)),
        "centroid_std": float(np.std(centroid)),
        "bandwidth_mean": float(np.mean(bandwidth)),
        "bandwidth_std": float(np.std(bandwidth)),
        "rolloff_mean": float(np.mean(rolloff)),
        "flatness_mean": float(np.mean(flatness)),
        "band_low_frac": band_fracs[0],
        "band_mid_frac": band_fracs[1],
        "band_high_frac": band_fracs[2],
        "band_very_high_frac": band_fracs[3],
        "n_stft_frames": float(times.size),
    }

    mel = np.maximum(_mel_filterbank(sample_rate, N_FFT) @ power, 1e-12)
    mfcc = dct(np.log(mel), type=2, axis=0, norm="ortho")[:N_MFCC]
    mfcc_mean = mfcc.mean(axis=1)
    mfcc_std = mfcc.std(axis=1)
    for i in range(N_MFCC):
        features[f"mfcc{i + 1}_mean"] = float(mfcc_mean[i])
        features[f"mfcc{i + 1}_std"] = float(mfcc_std[i])
    return features


def extract_table(paths: list[str], filenames: list[str], progress_every: int = 100) -> pd.DataFrame:
    rows = []
    for i, (path, name) in enumerate(zip(paths, filenames), start=1):
        feats = extract_file_features(path)
        feats["filename"] = name
        rows.append(feats)
        if progress_every and i % progress_every == 0:
            print(f"  extracted {i}/{len(paths)}", flush=True)
    return pd.DataFrame(rows)


def align_feature_frame(feat: pd.DataFrame, filenames: list[str]) -> pd.DataFrame:
    """Return one row per filename, in the requested order."""
    indexed = feat.drop_duplicates("filename").set_index("filename")
    missing = [name for name in filenames if name not in indexed.index]
    if missing:
        raise ValueError(f"Feature table is missing {len(missing)} files, e.g. {missing[:3]}")
    ordered = indexed.loc[list(filenames)].reset_index()
    if ordered.drop(columns=["filename"]).isna().any().any():
        raise ValueError("Feature table has missing values after alignment")
    return ordered


def load_or_build_feature_cache(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cache_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load cached features, or extract train and test audio once and store them.

    Returned frames follow train_df and test_df order, not the cache's row order.
    """
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    needed = set(ALL_FEATURES)
    if cache_path.is_file():
        cached = pd.read_csv(cache_path)
        version_ok = (
            "feature_version" in cached.columns
            and (cached["feature_version"].astype(str) == FEATURE_VERSION).all()
        )
        if version_ok and {"split", "filename"}.issubset(cached.columns) and needed.issubset(cached.columns):
            train_cached = cached[cached["split"] == "train"]
            test_cached = cached[cached["split"] == "test"]
            if set(train_cached["filename"]) == set(train_df["filename"]) and set(test_cached["filename"]) == set(
                test_df["filename"]
            ):
                print(f"Loaded feature cache {cache_path} ({FEATURE_VERSION})", flush=True)
                return (
                    align_feature_frame(train_cached, train_df["filename"].tolist()),
                    align_feature_frame(test_cached, test_df["filename"].tolist()),
                )
        print("Feature cache stale; rebuilding.", flush=True)

    print(f"Extracting train features ({len(train_df)} files)...", flush=True)
    train_feat = extract_table(train_df["path"].tolist(), train_df["filename"].tolist())
    print(f"Extracting test features ({len(test_df)} files)...", flush=True)
    test_feat = extract_table(test_df["path"].tolist(), test_df["filename"].tolist())
    stored = pd.concat(
        [
            train_feat.assign(split="train", feature_version=FEATURE_VERSION),
            test_feat.assign(split="test", feature_version=FEATURE_VERSION),
        ],
        ignore_index=True,
    )
    stored.to_csv(cache_path, index=False)
    print(f"Wrote feature cache {cache_path}", flush=True)
    return train_feat, test_feat
