from __future__ import annotations

from pathlib import Path

import pandas as pd

from competitions.store_sales_time_series_forecasting.models.baseline import (
    build_dataset,
    discover_competition_files,
    fit_and_score_holdout,
    fit_final_model,
    generate_submission,
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def _prepare_dir(tmp_path: Path) -> Path:
    train_rows: list[dict[str, object]] = []
    test_rows: list[dict[str, object]] = []
    sample_rows: list[dict[str, object]] = []

    families = ["AUTOMOTIVE", "BEAUTY"]
    train_dates = pd.date_range("2017-01-01", periods=28, freq="D")
    test_dates = pd.date_range("2017-01-29", periods=4, freq="D")
    next_id = 1

    for date in train_dates:
        dow = int(date.dayofweek)
        for idx, family in enumerate(families):
            promo = int((date.day + idx) % 3 == 0)
            sales = 20.0 + 4.0 * idx + 1.5 * dow + 2.0 * promo + float(date.day % 5)
            train_rows.append(
                {
                    "id": next_id,
                    "date": date.strftime("%Y-%m-%d"),
                    "store_nbr": 1,
                    "family": family,
                    "sales": sales,
                    "onpromotion": promo,
                }
            )
            next_id += 1

    for date in test_dates:
        for idx, family in enumerate(families):
            promo = int((date.day + idx) % 2 == 0)
            test_rows.append(
                {
                    "id": next_id,
                    "date": date.strftime("%Y-%m-%d"),
                    "store_nbr": 1,
                    "family": family,
                    "onpromotion": promo,
                }
            )
            sample_rows.append({"id": next_id, "sales": 0.0})
            next_id += 1

    stores_rows = [{"store_nbr": 1, "city": "Quito", "state": "Pichincha", "type": "D", "cluster": 13}]
    oil_rows = [
        {"date": date.strftime("%Y-%m-%d"), "dcoilwtico": 50.0 + idx * 0.1}
        for idx, date in enumerate(pd.date_range("2017-01-01", periods=40, freq="D"))
    ]
    holiday_rows = [
        {
            "date": "2017-01-10",
            "type": "Holiday",
            "locale": "National",
            "locale_name": "Ecuador",
            "description": "Holiday",
            "transferred": False,
        },
        {
            "date": "2017-01-20",
            "type": "Event",
            "locale": "Local",
            "locale_name": "Quito",
            "description": "Concert",
            "transferred": False,
        },
    ]
    transactions_rows = [
        {"date": date.strftime("%Y-%m-%d"), "store_nbr": 1, "transactions": 1000 + idx}
        for idx, date in enumerate(train_dates)
    ]

    _write_csv(tmp_path / "train.csv", train_rows)
    _write_csv(tmp_path / "test.csv", test_rows)
    _write_csv(tmp_path / "stores.csv", stores_rows)
    _write_csv(tmp_path / "oil.csv", oil_rows)
    _write_csv(tmp_path / "holidays_events.csv", holiday_rows)
    _write_csv(tmp_path / "transactions.csv", transactions_rows)
    _write_csv(tmp_path / "sample_submission.csv", sample_rows)
    return tmp_path


def test_holdout_and_submission(tmp_path: Path):
    raw_dir = _prepare_dir(tmp_path)
    files = discover_competition_files(raw_dir)
    dataset = build_dataset(files)

    selection, metrics, holdout_predictions = fit_and_score_holdout(
        dataset=dataset,
        holdout_days=4,
        recent_train_start="2017-01-10",
        max_train_rows=5000,
        candidate_strategies=("seasonal_naive",),
        seed=7,
    )
    final_model = fit_final_model(
        dataset=dataset,
        selection=selection,
        recent_train_start="2017-01-10",
        max_train_rows=5000,
        seed=7,
    )
    submission = generate_submission(final_model, dataset=dataset)

    assert selection["selected_strategy"] == "seasonal_naive"
    assert "rmsle" in metrics and "mae" in metrics
    assert list(holdout_predictions.columns) == ["id", "date", "store_nbr", "family", "actual", "prediction"]
    assert list(submission.columns) == ["id", "sales"]
    assert len(submission) == 8
    assert submission["sales"].ge(0).all()
