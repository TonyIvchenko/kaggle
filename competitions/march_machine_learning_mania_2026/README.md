# March Machine Learning Mania 2026

Baseline workspace for the Kaggle competition:

- competition slug: `march-machine-learning-mania-2026`
- task: forecast NCAA tournament matchups
- metric: mean squared error on predicted win probabilities

## Layout

- `scripts/download_data.py`: downloads and extracts competition files with the Kaggle CLI
- `scripts/train_model.py`: builds a PyTorch baseline, evaluates on the latest completed season, and writes a submission file
- `models/baseline.py`: shared feature engineering and PyTorch model code (linear + MLP + residual candidates, with optional gender-specific ensembles)
- `notebooks/march_machine_learning_mania_2026.ipynb`: EDA + evaluation notebook
- `tests/`: unit tests for file discovery and modeling logic

## Expected Data Flow

```bash
python competitions/march_machine_learning_mania_2026/scripts/download_data.py
python competitions/march_machine_learning_mania_2026/scripts/train_model.py --device-preference mps
```

The download step requires a working Kaggle API configuration at `~/.kaggle/kaggle.json`.
The training step defaults to MPS so Mac Apple Silicon can use GPU acceleration.
By default, the trainer fits separate men/women ensembles; use `--no-split-by-gender` to disable.
