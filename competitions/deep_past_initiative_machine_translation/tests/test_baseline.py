from __future__ import annotations

from pathlib import Path

import pandas as pd

from competitions.deep_past_initiative_machine_translation.models.baseline import (
    TorchRetrieverConfig,
    build_datasets,
    discover_competition_files,
    fit_and_score_holdout,
    fit_final_model,
    generate_submission,
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def _prepare_competition_dir(tmp_path: Path) -> Path:
    rows = [
        (1, "lugal gal", "royal title", "great king"),
        (2, "lugal tur", "royal title", "young king"),
        (3, "e2 gal", "temple", "great house"),
        (4, "e2 tur", "temple", "small house"),
        (5, "dumu nin", "family", "daughter of lady"),
        (6, "dumu lugal", "family", "son of king"),
        (7, "ki kalam", "toponym", "land place"),
        (8, "ki uri", "toponym", "city place"),
        (9, "nin gal", "title", "great lady"),
        (10, "nin tur", "title", "young lady"),
        (11, "udu gu4", "animals", "sheep and ox"),
        (12, "sze gu4", "animals", "barley for ox"),
    ]
    train_rows = [
        {"id": item[0], "source": item[1], "domain": item[2], "translation": item[3]}
        for item in rows
    ]
    test_rows = [
        {"id": 101, "source": "lugal gal", "domain": "royal title"},
        {"id": 102, "source": "nin gal", "domain": "title"},
        {"id": 103, "source": "e2 tur", "domain": "temple"},
        {"id": 104, "source": "udu gu4", "domain": "animals"},
    ]

    _write_csv(tmp_path / "train.csv", train_rows)
    _write_csv(tmp_path / "test.csv", test_rows)
    _write_csv(
        tmp_path / "sample_submission.csv",
        [
            {"id": 101, "translation": ""},
            {"id": 102, "translation": ""},
            {"id": 103, "translation": ""},
            {"id": 104, "translation": ""},
        ],
    )
    return tmp_path


def test_discover_files_and_build_datasets(tmp_path: Path):
    raw_dir = _prepare_competition_dir(tmp_path)
    files = discover_competition_files(raw_dir)
    dataset = build_datasets(files)

    assert dataset.id_column == "id"
    assert dataset.target_column == "translation"
    assert set(dataset.source_columns) == {"source", "domain"}
    assert len(dataset.train_source_texts) == 12
    assert len(dataset.test_source_texts) == 4


def test_holdout_and_submission_tfidf_only(tmp_path: Path):
    raw_dir = _prepare_competition_dir(tmp_path)
    files = discover_competition_files(raw_dir)
    dataset = build_datasets(files)

    selection, metrics, holdout_predictions = fit_and_score_holdout(
        dataset=dataset,
        holdout_fraction=0.25,
        seed=123,
        device_preference="cpu",
        use_torch=False,
    )
    final_model = fit_final_model(dataset=dataset, selection=selection, seed=123, device_preference="cpu")
    submission = generate_submission(final_model, dataset=dataset, device_preference="cpu")

    assert selection["selected_strategy"] == "tfidf"
    assert 0.0 <= metrics["char_f1"] <= 1.0
    assert list(holdout_predictions.columns) == ["id", "translation", "prediction"]
    assert list(submission.columns) == ["id", "translation"]
    assert submission["translation"].map(len).ge(0).all()


def test_holdout_with_torch_candidates(tmp_path: Path):
    raw_dir = _prepare_competition_dir(tmp_path)
    files = discover_competition_files(raw_dir)
    dataset = build_datasets(files)
    configs = [
        TorchRetrieverConfig(
            name="tiny_torch",
            embedding_dim=24,
            hidden_dim=48,
            dropout=0.1,
            learning_rate=2e-3,
            weight_decay=1e-4,
            batch_size=4,
            epochs=4,
            patience=2,
            max_length=40,
            temperature=0.08,
        )
    ]

    selection, metrics, _ = fit_and_score_holdout(
        dataset=dataset,
        holdout_fraction=0.25,
        seed=7,
        device_preference="cpu",
        use_torch=True,
        torch_configs=configs,
    )

    strategies = [row["strategy"] for row in metrics["strategy_metrics"]]
    assert "tfidf" in strategies
    assert "torch" in strategies
    assert selection["selected_strategy"] in {"tfidf", "torch", "hybrid"}

