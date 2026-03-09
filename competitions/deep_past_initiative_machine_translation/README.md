# Deep Past Initiative: Machine Translation

Workspace for the Kaggle competition:

- competition slug: `deep-past-initiative-machine-translation`
- task: predict translated text for each source segment
- expected submission: `id` + predicted translation text

## Layout

- `scripts/download_data.py`: downloads competition files via Kaggle CLI
- `scripts/train_model.py`: trains/evaluates baseline models and writes a submission
- `scripts/build_doc_memory_submission.py`: document-level translation memory submission builder
- `models/baseline.py`: feature prep, retrieval models, training, and inference
- `notebooks/deep_past_initiative_machine_translation.ipynb`: EDA + evaluation notebook
- `tests/`: unit tests for downloader and model pipeline

## Expected Data Flow

```bash
python competitions/deep_past_initiative_machine_translation/scripts/download_data.py --all-files
python competitions/deep_past_initiative_machine_translation/scripts/train_model.py --device-preference mps
python competitions/deep_past_initiative_machine_translation/scripts/build_doc_memory_submission.py
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

1. Concatenate source rows inside each test document group.
2. Build sentence-level translation memory from supplemental files:
   - `Sentences_Oare_FirstWord_LinNum.csv`
   - `published_texts.csv`
3. Retrieve top candidate documents from the sentence memory by character TF-IDF.
4. Align grouped test rows to contiguous sentence spans inside each candidate document using dynamic programming.
5. Blend document-level alignment confidence with document retrieval confidence.
6. Fall back to row-level nearest-neighbor retrieval when document alignment confidence is low.

It writes:

- submission CSV: `submissions/submission_doc_memory.csv`
- diagnostics report: `models/submission_doc_memory_report.json`
