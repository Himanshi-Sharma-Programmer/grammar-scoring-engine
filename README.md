# Grammar Scoring Engine

This project predicts a grammar score between 0 and 5 from a spoken WAV recording. It measures the recording directly and fits a scikit-learn model. It does not call an external service, a speech-recognition system, or a pretrained model.

The recordings and label files are not included. Place them on your machine as follows:

- `Dataset_Final/train.csv` with audio in `Dataset_Final/train/`
- `Dataset_Final/test.csv` with audio in `Dataset_Final/test/`

Find each recording from its directory. The same filename in the two files refers to two different recordings. Ignore the placeholder labels in `test.csv`.

## Approach

Each recording is converted to a fixed set of acoustic measurements. Those measurements describe loudness, pauses, and the short-term spectrum. They are proxies for fluency and recording conditions, not a grammatical parse of the words.

A constant-mean predictor, a ridge regression, and a shallow gradient-boosting model were compared on the same five folds. The boosting model was kept because its validation error was lower by more than the variation across folds. The final model is then fit on every training recording, including the 37 zero scores.

## Features

The submitted model uses 38 measurements:

- Loudness and silence: root-mean-square level, peak level, crest factor, the share of silence, the share of voiced frames, and the spread of frame energy.
- Pause rate, measured as pauses of at least 0.2 seconds.
- Mean and standard deviation of the zero-crossing rate.
- Mean and standard deviation of 13 mel-frequency cepstral coefficients.

Duration is left out because the training recordings are mostly about 60 seconds and the test recordings are mostly about 45 seconds. Spectral summaries that repeat the zero-crossing rate, and a few weak pause and clipping measures, are also left out. They did not improve validation.

## Model

The model is `HistGradientBoostingRegressor` with depth 3, learning rate 0.05, 200 iterations, at least 20 samples per leaf, and L2 regularization of 1.0. The random seed is 42. Stronger regularization, larger leaves, and shallower trees were tried on the same folds and were not better.

## Validation

Scores are clipped to the range 0 to 5 before each metric is computed. The five folds are stratified by score band (`0`, `1–2`, `2.5–3.5`, and `4–5`) and use seed 42. A separate 20 percent holdout is used only as a check. It was not used to choose the model. These figures are from the training labels only. They are not a test-set score.

| Check | RMSE | Pearson correlation |
| --- | ---: | ---: |
| Constant mean, all 769 recordings | 1.238 | — |
| Training fit, all 769 recordings | 0.502 | — |
| Five-fold validation | 0.775 ± 0.050 | 0.778 ± 0.044 |
| 20 percent holdout | 0.793 | 0.764 |
| Ridge regression, five-fold validation | 0.863 ± 0.043 | 0.717 ± 0.045 |

## Setup

The reported results used NumPy 1.24.3, SciPy 1.15.3, and scikit-learn 1.9.0. The requirements file pins those packages together with pandas, matplotlib, and the tools needed to run the notebook.

```bash
python -m pip install -r requirements.txt
```

## Run

```bash
python -m unittest discover -s tests -v
python -m nbconvert --to notebook --execute grammar_scoring_engine.ipynb --output grammar_scoring_engine.ipynb --ExecutePreprocessor.timeout=3600
```

The notebook loads the data, builds the features, runs the validation, fits the final model, and writes `outputs/submission.csv`.

## Submission

`outputs/submission.csv` is included. It has 216 rows, in the same order as `test.csv`, with the columns `filename` and `label`. Every score is finite and lies between 0 and 5. The file contains predictions only. It does not contain the recordings or the training labels.
