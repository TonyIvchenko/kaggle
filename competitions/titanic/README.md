# Titanic

Workspace for Kaggle competition:

- competition slug: `titanic`
- task: binary classification (`Survived`)
- expected submission: `PassengerId` + `Survived`

## Layout

- `scripts/download_data.py`: downloads competition files via Kaggle CLI
- `scripts/train_model.py`: trains baseline models and writes submission
- `models/baseline.py`: Titanic feature engineering + model utilities
- `tests/`: unit tests for downloader and baseline behavior

## Quick Start

```bash
python competitions/titanic/scripts/download_data.py
python competitions/titanic/scripts/train_model.py
```

