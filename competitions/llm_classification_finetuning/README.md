# LLM Classification Finetuning

Workspace for Kaggle competition:

- competition slug: `llm-classification-finetuning`
- task: text classification
- expected submission: `id` + one target column from `sample_submission.csv`

## Layout

- `scripts/download_data.py`: downloads competition files with Kaggle CLI
- `scripts/train_model.py`: trains a baseline text model and writes submission
- `models/baseline.py`: schema inference, text preprocessing, holdout scoring, submission generation (single-label or multi-probability)
- `tests/`: unit tests for downloader and baseline behavior

## Quick Start

```bash
python competitions/llm_classification_finetuning/scripts/download_data.py
python competitions/llm_classification_finetuning/scripts/train_model.py
```

## Artifacts

- model bundle: `models/llm_classification_finetuning.joblib`
- metrics: `models/llm_classification_finetuning_metrics.json`
- holdout predictions: `data/processed/holdout_predictions.csv`
- submission: `submissions/submission.csv`
