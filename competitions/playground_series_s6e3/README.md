# Playground Series S6E3

Workspace for Kaggle competition:

- competition slug: `playground-series-s6e3`
- task: infer target from tabular train/test files
- expected submission: `id` + one target column from `sample_submission.csv`

## Layout

- `scripts/download_data.py`: downloads competition files via Kaggle CLI
- `scripts/train_model.py`: runs holdout evaluation, trains final model, writes submission
- `models/baseline.py`: schema inference, tabular model candidates, evaluation utilities
- `notebooks/playground_series_s6e3.ipynb`: quick schema + metrics notebook
- `tests/`: unit tests for downloader and baseline pipeline

## Data Flow

```bash
python competitions/playground_series_s6e3/scripts/download_data.py
python competitions/playground_series_s6e3/scripts/train_model.py
```

Optional flags:

- `--all-files` in downloader to fetch every competition file (default already fetches train/test/sample).
- `--target-column <name>` in trainer if schema inference is ambiguous.
- `--holdout-fraction 0.2` in trainer to tune offline validation split size.

## Notes

- Baseline automatically infers:
  - id column
  - target column
  - feature columns shared between train and test
  - task type (`classification` or `regression`)
- Two candidates are evaluated on holdout:
  - linear pipeline (impute + one-hot + logistic/ridge)
  - tree pipeline (impute + ordinal encoding + histogram gradient boosting)
- Best strategy is selected by holdout score and retrained on full train set.

Artifacts written by trainer:

- model bundle: `models/playground_series_s6e3.joblib`
- metrics: `models/playground_series_s6e3_metrics.json`
- holdout predictions: `data/processed/holdout_predictions.csv`
- submission: `submissions/submission.csv`

