from __future__ import annotations

from pathlib import Path

import pandas as pd

from competitions.llm_classification_finetuning.models.baseline import (
    build_dataset,
    discover_competition_files,
    fit_and_score_holdout,
    fit_final_model,
    generate_submission,
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def _prepare_dir(tmp_path: Path) -> Path:
    train_rows = [
        {"id": 1, "prompt": "classify sentiment", "text": "I love this movie", "label": "positive"},
        {"id": 2, "prompt": "classify sentiment", "text": "This is fantastic", "label": "positive"},
        {"id": 3, "prompt": "classify sentiment", "text": "I hate this movie", "label": "negative"},
        {"id": 4, "prompt": "classify sentiment", "text": "This is terrible", "label": "negative"},
        {"id": 5, "prompt": "topic label", "text": "The match ended 2-1", "label": "sports"},
        {"id": 6, "prompt": "topic label", "text": "Team won the cup", "label": "sports"},
        {"id": 7, "prompt": "topic label", "text": "The stock dropped today", "label": "finance"},
        {"id": 8, "prompt": "topic label", "text": "Markets closed lower", "label": "finance"},
        {"id": 9, "prompt": "classify sentiment", "text": "Amazing and wonderful", "label": "positive"},
        {"id": 10, "prompt": "classify sentiment", "text": "Awful and boring", "label": "negative"},
        {"id": 11, "prompt": "topic label", "text": "Coach announced lineup", "label": "sports"},
        {"id": 12, "prompt": "topic label", "text": "Investors bought bonds", "label": "finance"},
    ]
    test_rows = [
        {"id": 101, "prompt": "classify sentiment", "text": "Wonderful direction"},
        {"id": 102, "prompt": "topic label", "text": "Quarterly earnings report"},
    ]
    sample_rows = [
        {"id": 101, "label": "positive"},
        {"id": 102, "label": "positive"},
    ]

    _write_csv(tmp_path / "train.csv", train_rows)
    _write_csv(tmp_path / "test.csv", test_rows)
    _write_csv(tmp_path / "sample_submission.csv", sample_rows)
    return tmp_path


def test_holdout_and_submission(tmp_path: Path):
    raw_dir = _prepare_dir(tmp_path)
    files = discover_competition_files(raw_dir)
    dataset = build_dataset(files)

    selection, metrics, holdout_predictions = fit_and_score_holdout(
        dataset=dataset,
        holdout_fraction=0.25,
        seed=7,
    )
    final_model = fit_final_model(dataset=dataset, selection=selection, seed=7)
    submission = generate_submission(final_model, dataset=dataset)

    assert selection["selected_strategy"] in {"word_logreg", "word_svm", "char_svm"}
    assert "accuracy" in metrics
    assert list(submission.columns) == ["id", "label"]
    assert list(holdout_predictions.columns) == ["id", "label", "prediction"]
    assert len(submission) == 2
