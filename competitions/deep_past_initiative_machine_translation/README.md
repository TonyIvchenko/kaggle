# Deep Past Initiative: Machine Translation

Workspace for the Kaggle competition:

- competition slug: `deep-past-initiative-machine-translation`
- task: predict translated text for each source segment
- expected submission: `id` + predicted translation text

## Layout

- `scripts/download_data.py`: downloads competition files via Kaggle CLI
- `scripts/train_model.py`: trains/evaluates baseline models and writes a submission
- `scripts/build_doc_memory_submission.py`: document-level translation memory submission builder
- `scripts/train_transformer_editor.py`: pretrained transformer editor (`source + draft -> final`) with accelerator support
- `models/baseline.py`: feature prep, retrieval models, training, and inference
- `notebooks/deep_past_initiative_machine_translation.ipynb`: EDA + evaluation notebook
- `tests/`: unit tests for downloader and model pipeline

## Expected Data Flow

```bash
python competitions/deep_past_initiative_machine_translation/scripts/download_data.py --all-files
python competitions/deep_past_initiative_machine_translation/scripts/train_model.py --device-preference mps
python competitions/deep_past_initiative_machine_translation/scripts/build_doc_memory_submission.py
python competitions/deep_past_initiative_machine_translation/scripts/train_transformer_editor.py --device-preference auto
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

## Advanced Submission Strategy

`build_doc_memory_submission.py` is designed for test sets that contain grouped segments
(for example, `text_id` + `line_start`).

Flow:

1. Load and use all competition files:
   - `train.csv`, `test.csv`, `sample_submission.csv`
   - `Sentences_Oare_FirstWord_LinNum.csv`, `published_texts.csv`
   - `OA_Lexicon_eBL.csv`, `eBL_Dictionary.csv`
   - `publications.csv`, `bibliography.csv`, `resources.csv`
2. Normalize transliteration tokens with lexicon mappings.
3. Retrieve top candidate documents from sentence metadata by character TF-IDF.
4. Use `line_start`/`line_end` and sentence `line_number` to select candidate sentence translations per row.
5. Convert sentence-style translations into train-style output where possible via learned sentence-to-train mapping.
6. Rerank candidates using retrieval confidence, line coverage, and an auxiliary English-vocabulary penalty built from supplemental corpora.
7. Fall back to row-level nearest-neighbor retrieval when candidate coverage is weak.

It writes:

- submission CSV: `submissions/submission_doc_memory.csv`
- diagnostics report: `models/submission_doc_memory_report.json`

## Transformer Editor Strategy

`train_transformer_editor.py` adds a pretrained seq2seq stage:

1. Builds nearest-neighbor retrieval drafts from train data (plus optional supplemental sentence memory).
2. Fine-tunes `google/byt5-small` (or another HF seq2seq model) on prompts:
   - source transliteration
   - retrieval draft
   - target translation
3. Evaluates holdout with:
   - `char_f1`, `sequence_ratio`, `exact_match`
   - `BLEU`, `chrF++`
4. Trains on full train set and exports:
   - submission CSV: `submissions/submission_transformer_editor.csv`
   - metrics: `models/transformer_editor_metrics.json`
   - model artifacts: `models/transformer_editor/`

Device options:

- local Mac: `--device-preference mps`
- Kaggle GPU notebook: `--device-preference cuda`
- automatic: `--device-preference auto`
