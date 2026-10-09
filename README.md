# Grammar Scoring Engine

This project predicts a grammar score between 0 and 5 from a spoken WAV recording. It uses acoustic measurements of the audio and a scikit-learn model. It does not call an external service or a pretrained speech model.

The recordings and label files are not included. Place them on your machine in the following layout:

- `Dataset_Final/train.csv` with audio in `Dataset_Final/train/`
- `Dataset_Final/test.csv` with audio in `Dataset_Final/test/`

Use the directory to find each recording. A filename that appears in both files refers to two different recordings. The test labels are placeholders and should be ignored.

## Setup

NumPy, SciPy, and scikit-learn should already be installed. The requirements file adds compatible versions of the remaining packages.

```bash
python -m pip install -r requirements.txt --upgrade-strategy only-if-needed
```

## Run

```bash
python -m unittest discover -s tests -v
python -m nbconvert --to notebook --execute grammar_scoring_engine.ipynb --output grammar_scoring_engine.ipynb --ExecutePreprocessor.timeout=3600
```

The notebook writes `outputs/submission.csv`. That file has 216 rows, in the same order as `test.csv`, with the columns `filename` and `label`. Every score is kept between 0 and 5.

## Method

The model is a shallow gradient-boosting regressor trained on 38 measurements: loudness, pause rate, zero-crossing rate, and 13 mel-frequency cepstral coefficients. Evaluation uses five stratified folds with seed 42, plus a separate 20 percent holdout reserved for plots. The final model is fit on all 769 training recordings, including the zero scores.
