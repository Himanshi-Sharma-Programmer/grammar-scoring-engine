"""Tests for leakage controls, feature definitions, and the submission schema."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from src.audio_features import (
    ALL_FEATURES,
    DURATION_FEATURES,
    EXCLUDED_FEATURES,
    FEATURE_GROUPS,
    align_feature_frame,
    extract_file_features,
    feature_columns,
    model_feature_columns,
)
from src.data import SCORE_BINS, load_test_ids, load_train, score_bin
from src.evaluate import clip_scores, pearson, rmse
from src.paths import (
    N_TEST_EXPECTED,
    SAMPLE_SUBMISSION_CSV,
    TEST_AUDIO_DIR,
    TEST_CSV,
    TRAIN_AUDIO_DIR,
    audio_path,
)
from src.pipeline import feature_matrix, stratified_cv
from src.predict import verify_submission, write_submission


ROOT = Path(__file__).resolve().parents[1]
DATA_PRESENT = (ROOT / "Dataset_Final" / "train.csv").is_file()
BASELINE_PRESENT = (ROOT / "baseline_snapshot" / "src" / "audio_features.py").is_file()
requires_data = unittest.skipUnless(DATA_PRESENT, "challenge audio and CSVs are local only")
requires_baseline = unittest.skipUnless(
    DATA_PRESENT and BASELINE_PRESENT,
    "local baseline snapshot is not part of the published copy",
)


def _baseline_audio():
    path = ROOT / "baseline_snapshot" / "src" / "audio_features.py"
    spec = importlib.util.spec_from_file_location("baseline_audio_features", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ScoreAndPathTests(unittest.TestCase):
    def test_score_bins_match_the_assessment_cuts(self):
        self.assertEqual(score_bin(0), "0")
        self.assertEqual(score_bin(1), "1-2")
        self.assertEqual(score_bin(1.5), "1-2")
        self.assertEqual(score_bin(2), "1-2")
        self.assertEqual(score_bin(2.5), "2.5-3.5")
        self.assertEqual(score_bin(3.5), "2.5-3.5")
        self.assertEqual(score_bin(4), "4-5")
        self.assertEqual(score_bin(5), "4-5")
        self.assertEqual(SCORE_BINS, ("0", "1-2", "2.5-3.5", "4-5"))

    def test_audio_path_stays_inside_the_split_directory(self):
        train_file = audio_path("audio_192.wav", "train")
        test_file = audio_path("audio_192.wav", "test")
        self.assertEqual(train_file.parent.resolve(), TRAIN_AUDIO_DIR.resolve())
        self.assertEqual(test_file.parent.resolve(), TEST_AUDIO_DIR.resolve())
        self.assertNotEqual(train_file.resolve(), test_file.resolve())
        with self.assertRaises(ValueError):
            audio_path("../test/audio_192.wav", "train")
        with self.assertRaises(ValueError):
            audio_path("audio_192.wav", "both")

    @requires_data
    def test_train_keeps_zeros_and_test_labels_are_discarded(self):
        train = load_train()
        test = load_test_ids()
        raw_test = pd.read_csv(TEST_CSV)
        self.assertEqual(len(train), 769)
        self.assertEqual(int(train["is_zero"].sum()), 37)
        self.assertEqual(set(train["stratum"]), set(SCORE_BINS))
        self.assertEqual(len(test), N_TEST_EXPECTED)
        self.assertNotIn("label", test.columns)
        self.assertEqual(set(raw_test["label"].unique()), {-1})
        self.assertEqual(test["filename"].tolist(), raw_test["filename"].astype(str).tolist())
        self.assertTrue(all(Path(path).parent.name == "train" for path in train["path"]))
        self.assertTrue(all(Path(path).parent.name == "test" for path in test["path"]))
        overlap = set(train["filename"]) & set(test["filename"])
        self.assertGreater(len(overlap), 0)
        self.assertTrue(all(Path(train.loc[train["filename"] == name, "path"].iloc[0]).resolve()
                            != Path(test.loc[test["filename"] == name, "path"].iloc[0]).resolve()
                            for name in list(overlap)[:5]))
        sample = pd.read_csv(SAMPLE_SUBMISSION_CSV)
        self.assertEqual(list(sample.columns), ["filename", "label"])
        self.assertNotEqual(len(sample), len(test))


class MetricTests(unittest.TestCase):
    def test_rmse_pearson_and_clip(self):
        y = np.array([0.0, 2.0, 4.0])
        pred = np.array([0.0, 2.0, 4.0])
        self.assertAlmostEqual(rmse(y, pred), 0.0)
        self.assertAlmostEqual(pearson(y, pred), 1.0)
        self.assertAlmostEqual(rmse(y, np.array([1.0, 1.0, 1.0])), np.sqrt(np.mean((y - 1.0) ** 2)))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.assertTrue(np.isnan(pearson(y, np.array([3.0, 3.0, 3.0]))))
            self.assertTrue(np.isnan(pearson(y, np.full(3, 1.0 / 3.0))))
        self.assertEqual(caught, [])
        clipped = clip_scores(np.array([-1.0, 2.5, 9.0, np.nan]))
        self.assertEqual(clipped[:3].tolist(), [0.0, 2.5, 5.0])


class FeatureTests(unittest.TestCase):
    def test_groups_partition_the_no_duration_columns(self):
        grouped = [name for names in FEATURE_GROUPS.values() for name in names]
        self.assertEqual(sorted(grouped), sorted(feature_columns(False)))
        self.assertEqual(len(grouped), len(set(grouped)))
        self.assertTrue(set(EXCLUDED_FEATURES).issubset(ALL_FEATURES))
        selected = model_feature_columns(False)
        self.assertEqual(len(selected), 38)
        self.assertTrue(set(selected).isdisjoint(EXCLUDED_FEATURES))
        self.assertTrue(set(selected).isdisjoint(DURATION_FEATURES))
        self.assertEqual(feature_columns(True), list(ALL_FEATURES))

    def test_alignment_follows_requested_filename_order(self):
        frame = pd.DataFrame(
            {
                "filename": ["b", "a", "c", "a"],
                "rms": [2.0, 1.0, 3.0, 9.0],
                "split": ["train", "train", "train", "train"],
            }
        )
        ordered = align_feature_frame(frame, ["c", "a", "b"])
        self.assertEqual(ordered["filename"].tolist(), ["c", "a", "b"])
        self.assertEqual(ordered["rms"].tolist(), [3.0, 1.0, 2.0])

    @requires_baseline
    def test_new_extractor_matches_the_baseline_extractor(self):
        old = _baseline_audio()
        self.assertEqual(old.feature_columns(True), feature_columns(True))
        self.assertEqual(old.feature_columns(False), feature_columns(False))
        train = load_train()
        for path in train["path"].head(3):
            before = old.extract_file_features(path)
            after = extract_file_features(path)
            self.assertEqual(list(before), list(after))
            for key in before:
                self.assertTrue(
                    np.isclose(before[key], after[key], rtol=1e-8, atol=1e-8),
                    f"{key} changed for {path}: {before[key]} vs {after[key]}",
                )


class SubmissionTests(unittest.TestCase):
    @requires_data
    def test_submission_matches_test_csv_order_and_range(self):
        filenames = pd.read_csv(TEST_CSV)["filename"].astype(str).tolist()
        predictions = np.linspace(0, 5, len(filenames))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "submission.csv"
            written = write_submission(filenames, predictions, path)
            self.assertEqual(list(written.columns), ["filename", "label"])
            report = verify_submission(path)
            self.assertTrue(report["ok"], report["issues"])
            self.assertEqual(report["n_rows"], N_TEST_EXPECTED)
        with self.assertRaises(ValueError):
            write_submission(list(reversed(filenames)), predictions)
        with self.assertRaises(ValueError):
            write_submission(filenames, np.array([np.nan] * len(filenames)))

    @requires_data
    def test_full_feature_hgbr_cv_matches_the_recorded_baseline(self):
        from src.audio_features import load_or_build_feature_cache
        from src.paths import FEATURE_CACHE, SEED

        train = load_train()
        test = load_test_ids()
        train_feat, _ = load_or_build_feature_cache(train, test, FEATURE_CACHE)
        merged = train[["filename", "label", "stratum"]].merge(train_feat, on="filename", how="left")
        y = merged["label"].to_numpy(dtype=np.float64)
        strata = merged["stratum"].to_numpy()
        matrix = feature_matrix(merged, feature_columns(False))
        result = stratified_cv(matrix, y, strata, "hgbr", n_splits=5, seed=SEED)
        self.assertAlmostEqual(result["mean_val_rmse"], 0.7764477276093398, places=6)
        self.assertAlmostEqual(result["mean_val_pearson"], 0.7766257014748094, places=6)
        self.assertAlmostEqual(result["mean_train_rmse"], 0.4582023257368203, places=6)


if __name__ == "__main__":
    unittest.main()
