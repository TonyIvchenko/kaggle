# Contradictory, My Dear Watson

Workspace for Kaggle competition:

- competition slug: `contradictory-my-dear-watson`
- task: natural language inference (text classification)
- expected submission: `id` + one target column from `sample_submission.csv`

## Layout

- `scripts/download_data.py`: downloads competition files with Kaggle CLI
- `scripts/train_model.py`: trains a TF-IDF + logistic regression baseline and writes submission
- `models/baseline.py`: schema inference, text preprocessing, holdout evaluation utilities
- `tests/`: unit tests for downloader and baseline pipeline

## Quick Start

```bash
python competitions/contradictory_my_dear_watson/scripts/download_data.py
python competitions/contradictory_my_dear_watson/scripts/train_model.py
```

## Artifacts

- model bundle: `models/contradictory_my_dear_watson.joblib`
- metrics: `models/contradictory_my_dear_watson_metrics.json`
- holdout predictions: `data/processed/holdout_predictions.csv`
- submission: `submissions/submission.csv`

