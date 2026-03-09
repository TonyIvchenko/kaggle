# Deep Past Initiative: Machine Translation

Workspace for the Kaggle competition:

- competition slug: `deep-past-initiative-machine-translation`
- task: predict translated text for each source segment
- expected submission: `id` + predicted translation text

## Layout

- `scripts/download_data.py`: downloads competition files via Kaggle CLI
- `scripts/train_model.py`: trains/evaluates baseline models and writes a submission
- `models/baseline.py`: feature prep, retrieval models, training, and inference
- `notebooks/deep_past_initiative_machine_translation.ipynb`: EDA + evaluation notebook
- `tests/`: unit tests for downloader and model pipeline

## Expected Data Flow

```bash
python competitions/deep_past_initiative_machine_translation/scripts/download_data.py
python competitions/deep_past_initiative_machine_translation/scripts/train_model.py --device-preference mps
```

By default, the downloader only pulls `train.csv`, `test.csv`, and `sample_submission.csv`.
Use `--all-files` to fetch every competition file (includes large supplemental resources).

If download returns `403 Forbidden`, open the competition page in your browser and complete:

1. Join / Accept Rules: https://www.kaggle.com/competitions/deep-past-initiative-machine-translation
2. Account verification (if prompted): https://www.kaggle.com/settings

## Training Notes

- Holdout metrics reported by the trainer are:
  - `char_f1`
  - `sequence_ratio`
  - `exact_match`
- Candidate strategies:
  - character TF-IDF retriever
  - PyTorch bi-encoder retriever (MPS/CPU)
  - confidence-threshold hybrid of both
