from __future__ import annotations

from pathlib import Path

import pandas as pd

from competitions.playground_series_s6e3.models.baseline import (
    build_datasets,
    discover_competition_files,
    fit_and_score_holdout,
    fit_final_model,
    generate_submission,
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def _prepare_classification_dir(tmp_path: Path) -> Path:
    train_rows = [
        {"id": 1, "num_a": 0.1, "num_b": 10.0, "segment": "A", "target": 0},
        {"id": 2, "num_a": 0.2, "num_b": 12.0, "segment": "A", "target": 0},
        {"id": 3, "num_a": 1.2, "num_b": 20.0, "segment": "B", "target": 1},
        {"id": 4, "num_a": 1.5, "num_b": 18.0, "segment": "B", "target": 1},
        {"id": 5, "num_a": 0.3, "num_b": 11.0, "segment": "A", "target": 0},
        {"id": 6, "num_a": 1.8, "num_b": 21.0, "segment": "B", "target": 1},
        {"id": 7, "num_a": 0.4, "num_b": 10.5, "segment": "A", "target": 0},
        {"id": 8, "num_a": 2.0, "num_b": 22.0, "segment": "B", "target": 1},
        {"id": 9, "num_a": 0.5, "num_b": 9.5, "segment": "A", "target": 0},
        {"id": 10, "num_a": 2.1, "num_b": 23.0, "segment": "B", "target": 1},
    ]
    test_rows = [
        {"id": 101, "num_a": 0.15, "num_b": 10.2, "segment": "A"},
        {"id": 102, "num_a": 1.9, "num_b": 21.2, "segment": "B"},
        {"id": 103, "num_a": 0.45, "num_b": 10.1, "segment": "A"},
    ]
    sample_rows = [
        {"id": 101, "target": 0.5},
        {"id": 102, "target": 0.5},
        {"id": 103, "target": 0.5},
    ]

    _write_csv(tmp_path / "train.csv", train_rows)
    _write_csv(tmp_path / "test.csv", test_rows)
    _write_csv(tmp_path / "sample_submission.csv", sample_rows)
    return tmp_path


def _prepare_regression_dir(tmp_path: Path) -> Path:
    train_rows = [
        {"id": 1, "x": 1.0, "group": "g1", "target": 3.0},
        {"id": 2, "x": 2.0, "group": "g1", "target": 5.0},
        {"id": 3, "x": 3.0, "group": "g2", "target": 7.1},
        {"id": 4, "x": 4.0, "group": "g2", "target": 8.9},
        {"id": 5, "x": 5.0, "group": "g2", "target": 11.2},
        {"id": 6, "x": 6.0, "group": "g3", "target": 12.8},
        {"id": 7, "x": 7.0, "group": "g3", "target": 15.0},
        {"id": 8, "x": 8.0, "group": "g3", "target": 17.1},
        {"id": 9, "x": 9.0, "group": "g4", "target": 18.7},
        {"id": 10, "x": 10.0, "group": "g4", "target": 21.0},
    ]
    test_rows = [
        {"id": 201, "x": 1.5, "group": "g1"},
        {"id": 202, "x": 7.5, "group": "g3"},
    ]
    sample_rows = [
        {"id": 201, "target": 0.0},
        {"id": 202, "target": 0.0},
    ]

    _write_csv(tmp_path / "train.csv", train_rows)
    _write_csv(tmp_path / "test.csv", test_rows)
    _write_csv(tmp_path / "sample_submission.csv", sample_rows)
    return tmp_path


def test_holdout_and_submission_classification(tmp_path: Path):
    raw_dir = _prepare_classification_dir(tmp_path)
    files = discover_competition_files(raw_dir)
    dataset = build_datasets(files)

    selection, metrics, holdout_predictions = fit_and_score_holdout(
        dataset=dataset,
        holdout_fraction=0.3,
        seed=7,
    )
    final_model = fit_final_model(dataset=dataset, selection=selection, seed=7)
    submission = generate_submission(final_model, dataset=dataset)

    assert dataset.task_type == "classification"
    assert dataset.probability_output is True
    assert selection["selected_strategy"] in {"linear", "tree"}
    assert "accuracy" in metrics and "log_loss" in metrics
    assert list(holdout_predictions.columns)[0] == dataset.id_column
    assert list(submission.columns) == ["id", "target"]
    assert submission["target"].between(0.0, 1.0).all()


def test_holdout_and_submission_regression(tmp_path: Path):
    raw_dir = _prepare_regression_dir(tmp_path)
    files = discover_competition_files(raw_dir)
    dataset = build_datasets(files)

    selection, metrics, _ = fit_and_score_holdout(
        dataset=dataset,
        holdout_fraction=0.3,
        seed=11,
    )
    final_model = fit_final_model(dataset=dataset, selection=selection, seed=11)
    submission = generate_submission(final_model, dataset=dataset)

    assert dataset.task_type == "regression"
    assert selection["selected_strategy"] in {"linear", "tree"}
    assert "rmse" in metrics and "mae" in metrics
    assert list(submission.columns) == ["id", "target"]
    assert pd.to_numeric(submission["target"], errors="coerce").notna().all()

