from __future__ import annotations

from pathlib import Path

import pandas as pd

from competitions.titanic.models.baseline import (
    build_datasets,
    discover_competition_files,
    fit_and_score_holdout,
    fit_final_model,
    generate_submission,
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def _prepare_dir(tmp_path: Path) -> Path:
    train_rows = [
        {
            "PassengerId": 1,
            "Survived": 0,
            "Pclass": 3,
            "Name": "Allen, Mr. William",
            "Sex": "male",
            "Age": 35,
            "SibSp": 0,
            "Parch": 0,
            "Ticket": "A/5 21171",
            "Fare": 7.25,
            "Cabin": None,
            "Embarked": "S",
        },
        {
            "PassengerId": 2,
            "Survived": 1,
            "Pclass": 1,
            "Name": "Cumings, Mrs. John",
            "Sex": "female",
            "Age": 38,
            "SibSp": 1,
            "Parch": 0,
            "Ticket": "PC 17599",
            "Fare": 71.2833,
            "Cabin": "C85",
            "Embarked": "C",
        },
        {
            "PassengerId": 3,
            "Survived": 1,
            "Pclass": 3,
            "Name": "Heikkinen, Miss. Laina",
            "Sex": "female",
            "Age": 26,
            "SibSp": 0,
            "Parch": 0,
            "Ticket": "STON/O2. 3101282",
            "Fare": 7.925,
            "Cabin": None,
            "Embarked": "S",
        },
        {
            "PassengerId": 4,
            "Survived": 1,
            "Pclass": 1,
            "Name": "Futrelle, Mrs. Jacques",
            "Sex": "female",
            "Age": 35,
            "SibSp": 1,
            "Parch": 0,
            "Ticket": "113803",
            "Fare": 53.1,
            "Cabin": "C123",
            "Embarked": "S",
        },
        {
            "PassengerId": 5,
            "Survived": 0,
            "Pclass": 3,
            "Name": "Allen, Mr. James",
            "Sex": "male",
            "Age": 35,
            "SibSp": 0,
            "Parch": 0,
            "Ticket": "373450",
            "Fare": 8.05,
            "Cabin": None,
            "Embarked": "S",
        },
        {
            "PassengerId": 6,
            "Survived": 0,
            "Pclass": 3,
            "Name": "Moran, Mr. James",
            "Sex": "male",
            "Age": None,
            "SibSp": 0,
            "Parch": 0,
            "Ticket": "330877",
            "Fare": 8.4583,
            "Cabin": None,
            "Embarked": "Q",
        },
        {
            "PassengerId": 7,
            "Survived": 0,
            "Pclass": 1,
            "Name": "McCarthy, Mr. Timothy",
            "Sex": "male",
            "Age": 54,
            "SibSp": 0,
            "Parch": 0,
            "Ticket": "17463",
            "Fare": 51.8625,
            "Cabin": "E46",
            "Embarked": "S",
        },
        {
            "PassengerId": 8,
            "Survived": 0,
            "Pclass": 3,
            "Name": "Palsson, Master. Gosta",
            "Sex": "male",
            "Age": 2,
            "SibSp": 3,
            "Parch": 1,
            "Ticket": "349909",
            "Fare": 21.075,
            "Cabin": None,
            "Embarked": "S",
        },
        {
            "PassengerId": 9,
            "Survived": 1,
            "Pclass": 3,
            "Name": "Johnson, Mrs. Oscar",
            "Sex": "female",
            "Age": 27,
            "SibSp": 0,
            "Parch": 2,
            "Ticket": "347742",
            "Fare": 11.1333,
            "Cabin": None,
            "Embarked": "S",
        },
        {
            "PassengerId": 10,
            "Survived": 1,
            "Pclass": 2,
            "Name": "Nasser, Mrs. Nicholas",
            "Sex": "female",
            "Age": 14,
            "SibSp": 1,
            "Parch": 0,
            "Ticket": "237736",
            "Fare": 30.0708,
            "Cabin": None,
            "Embarked": "C",
        },
    ]
    test_rows = [
        {
            "PassengerId": 101,
            "Pclass": 3,
            "Name": "Allen, Mr. Bob",
            "Sex": "male",
            "Age": 34,
            "SibSp": 0,
            "Parch": 0,
            "Ticket": "330911",
            "Fare": 7.8292,
            "Cabin": None,
            "Embarked": "Q",
        },
        {
            "PassengerId": 102,
            "Pclass": 1,
            "Name": "Cumings, Mrs. Jane",
            "Sex": "female",
            "Age": 47,
            "SibSp": 1,
            "Parch": 0,
            "Ticket": "PC 17558",
            "Fare": 79.2,
            "Cabin": "B28",
            "Embarked": "C",
        },
    ]
    sample_rows = [
        {"PassengerId": 101, "Survived": 0},
        {"PassengerId": 102, "Survived": 1},
    ]

    _write_csv(tmp_path / "train.csv", train_rows)
    _write_csv(tmp_path / "test.csv", test_rows)
    _write_csv(tmp_path / "gender_submission.csv", sample_rows)
    return tmp_path


def test_holdout_and_submission(tmp_path: Path):
    raw_dir = _prepare_dir(tmp_path)
    files = discover_competition_files(raw_dir)
    dataset = build_datasets(files)

    selection, metrics, holdout_predictions = fit_and_score_holdout(
        dataset=dataset,
        holdout_fraction=0.3,
        seed=7,
    )
    final_model = fit_final_model(dataset=dataset, selection=selection, seed=7)
    submission = generate_submission(final_model, dataset=dataset)

    assert selection["selected_strategy"] in {"linear", "forest", "hist", "catboost"}
    assert "accuracy" in metrics and "log_loss" in metrics
    assert list(submission.columns) == ["PassengerId", "Survived"]
    assert submission["Survived"].isin([0, 1]).all()
    assert list(holdout_predictions.columns)[0] == "PassengerId"

