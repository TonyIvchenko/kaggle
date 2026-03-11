from __future__ import annotations

from pathlib import Path

import pandas as pd

from competitions.contradictory_my_dear_watson.models.baseline import (
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
        {"id": 1, "premise": "Dogs are animals.", "hypothesis": "Dogs are living beings.", "label": "entailment"},
        {"id": 2, "premise": "Cats can fly.", "hypothesis": "Cats can fly in the sky.", "label": "entailment"},
        {"id": 3, "premise": "Cats can fly.", "hypothesis": "Cats cannot fly.", "label": "contradiction"},
        {"id": 4, "premise": "It is raining now.", "hypothesis": "The ground is dry.", "label": "contradiction"},
        {"id": 5, "premise": "A person is reading.", "hypothesis": "Someone is reading a book.", "label": "entailment"},
        {"id": 6, "premise": "Birds are singing.", "hypothesis": "There may be birds nearby.", "label": "neutral"},
        {"id": 7, "premise": "He is sleeping.", "hypothesis": "He is awake.", "label": "contradiction"},
        {"id": 8, "premise": "People are walking.", "hypothesis": "The street is busy.", "label": "neutral"},
        {"id": 9, "premise": "Sun is bright.", "hypothesis": "It might be daytime.", "label": "neutral"},
    ]
    test_rows = [
        {"id": 101, "premise": "Kids are playing.", "hypothesis": "Children are outdoors."},
        {"id": 102, "premise": "Water is dry.", "hypothesis": "Water is not wet."},
    ]
    sample_rows = [
        {"id": 101, "label": "entailment"},
        {"id": 102, "label": "entailment"},
    ]

    _write_csv(tmp_path / "train.csv", train_rows)
    _write_csv(tmp_path / "test.csv", test_rows)
    _write_csv(tmp_path / "sample_submission.csv", sample_rows)
    return tmp_path


def test_holdout_and_submission(tmp_path: Path):
    raw_dir = _prepare_dir(tmp_path)
    files = discover_competition_files(raw_dir)
    dataset = build_dataset(files)

    _, metrics, holdout_predictions = fit_and_score_holdout(
        dataset=dataset,
        holdout_fraction=0.33,
        seed=7,
    )
    final_model = fit_final_model(dataset=dataset, seed=7)
    submission = generate_submission(final_model, dataset=dataset)

    assert "accuracy" in metrics
    assert list(submission.columns) == ["id", "label"]
    assert list(holdout_predictions.columns) == ["id", "label", "prediction"]
    assert len(submission) == 2

