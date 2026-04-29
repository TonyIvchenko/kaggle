"""Baseline models for Store Sales - Time Series Forecasting."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

try:
    import lightgbm as lgb
except ImportError:  # pragma: no cover - optional dependency
    lgb = None

try:
    import xgboost as xgb
except ImportError:  # pragma: no cover - optional dependency
    xgb = None


COMPETITION_SLUG = "store-sales-time-series-forecasting"
COMPETITION_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = COMPETITION_ROOT / "data" / "raw"
DEFAULT_PROCESSED_DIR = COMPETITION_ROOT / "data" / "processed"
DEFAULT_MODEL_PATH = COMPETITION_ROOT / "models" / "store_sales_time_series_forecasting.joblib"
DEFAULT_METRICS_PATH = DEFAULT_PROCESSED_DIR / "metrics.json"
DEFAULT_SUBMISSION_PATH = COMPETITION_ROOT / "submissions" / "submission.csv"

GROUP_COLUMNS = ("store_nbr", "family")
HOLIDAY_COUNT_SUFFIXES = (
    "holiday_count",
    "event_count",
    "additional_count",
    "bridge_count",
    "workday_count",
    "transferred_count",
)
SALES_LAGS = (1, 7, 14, 28, 56, 364)
ROLL_WINDOWS = (7, 14, 28)
PROMO_LAGS = (1, 7, 14, 28)
PROMO_ROLL_WINDOWS = (7, 14)
HISTORY_LENGTH = max(max(SALES_LAGS), max(ROLL_WINDOWS), max(PROMO_LAGS), max(PROMO_ROLL_WINDOWS)) + 8

# Reserve the most recent calendar days of the supervised frame as an early-stopping
# validation set for the gradient-boosted models.
EARLY_STOPPING_VALID_DAYS = 28
EARLY_STOPPING_ROUNDS = 50

BASE_FEATURE_COLUMNS = [
    "store_nbr",
    "family_code",
    "onpromotion",
    "city_code",
    "state_code",
    "type_code",
    "cluster",
    "oil",
    "transactions_dow_mean",
    "dayofweek",
    "day",
    "month",
    "year",
    "weekofyear",
    "is_weekend",
    "is_month_start",
    "is_month_end",
    "is_payday",
    "days_to_next_national_holiday",
    "days_since_prev_national_holiday",
    "national_holiday_count",
    "national_event_count",
    "national_additional_count",
    "national_bridge_count",
    "national_workday_count",
    "national_transferred_count",
    "regional_holiday_count",
    "regional_event_count",
    "regional_additional_count",
    "regional_bridge_count",
    "regional_workday_count",
    "regional_transferred_count",
    "local_holiday_count",
    "local_event_count",
    "local_additional_count",
    "local_bridge_count",
    "local_workday_count",
    "local_transferred_count",
]
LAG_FEATURE_COLUMNS = [f"sales_lag_{lag}" for lag in SALES_LAGS]
ROLL_FEATURE_COLUMNS = [f"sales_roll_mean_{window}" for window in ROLL_WINDOWS]
ROLL_STD_FEATURE_COLUMNS = [f"sales_roll_std_{window}" for window in ROLL_WINDOWS]
PROMO_FEATURE_COLUMNS = [f"promo_lag_{lag}" for lag in PROMO_LAGS] + [
    f"promo_roll_mean_{window}" for window in PROMO_ROLL_WINDOWS
]
FEATURE_COLUMNS = [
    *BASE_FEATURE_COLUMNS,
    *LAG_FEATURE_COLUMNS,
    *ROLL_FEATURE_COLUMNS,
    *ROLL_STD_FEATURE_COLUMNS,
    *PROMO_FEATURE_COLUMNS,
]


@dataclass(frozen=True)
class CompetitionFiles:
    train_path: Path
    test_path: Path
    stores_path: Path
    oil_path: Path
    holidays_path: Path
    transactions_path: Path
    sample_submission_path: Path


@dataclass(frozen=True)
class DatasetBundle:
    train_frame: pd.DataFrame
    test_frame: pd.DataFrame
    stores_frame: pd.DataFrame
    oil_frame: pd.DataFrame
    holidays_frame: pd.DataFrame
    transactions_frame: pd.DataFrame
    sample_submission: pd.DataFrame
    id_column: str
    target_column: str
    competition: str = COMPETITION_SLUG


def discover_competition_files(raw_dir: Path = DEFAULT_RAW_DIR) -> CompetitionFiles:
    raw_dir = Path(raw_dir)
    files = CompetitionFiles(
        train_path=raw_dir / "train.csv",
        test_path=raw_dir / "test.csv",
        stores_path=raw_dir / "stores.csv",
        oil_path=raw_dir / "oil.csv",
        holidays_path=raw_dir / "holidays_events.csv",
        transactions_path=raw_dir / "transactions.csv",
        sample_submission_path=raw_dir / "sample_submission.csv",
    )
    missing = [str(path.name) for path in files.__dict__.values() if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required competition files in "
            f"{raw_dir}: {', '.join(sorted(missing))}. "
            "Run the download script first."
        )
    return files


def build_dataset(files: CompetitionFiles) -> DatasetBundle:
    # Explicit dtypes keep the 3M-row train file from defaulting to int64/float64
    # and a duplicated object-dtype ``family`` column, roughly halving its memory.
    train = pd.read_csv(
        files.train_path,
        parse_dates=["date"],
        dtype={"id": "int32", "store_nbr": "int16", "family": "category", "sales": "float32", "onpromotion": "int32"},
    )
    test = pd.read_csv(
        files.test_path,
        parse_dates=["date"],
        dtype={"id": "int32", "store_nbr": "int16", "family": "category", "onpromotion": "int32"},
    )
    stores = pd.read_csv(files.stores_path, dtype={"store_nbr": "int16", "cluster": "int16"})
    oil = pd.read_csv(files.oil_path, parse_dates=["date"], dtype={"dcoilwtico": "float32"})
    holidays = pd.read_csv(files.holidays_path, parse_dates=["date"])
    transactions = pd.read_csv(
        files.transactions_path,
        parse_dates=["date"],
        dtype={"store_nbr": "int16", "transactions": "int32"},
    )
    sample_submission = pd.read_csv(files.sample_submission_path, dtype={"id": "int32", "sales": "float32"})

    return DatasetBundle(
        train_frame=train,
        test_frame=test,
        stores_frame=stores,
        oil_frame=oil,
        holidays_frame=holidays,
        transactions_frame=transactions,
        sample_submission=sample_submission,
        id_column="id",
        target_column="sales",
    )


def _build_category_mapping(values: pd.Series) -> dict[str, int]:
    clean = sorted({str(value) for value in values.dropna().astype(str).tolist()})
    return {value: idx for idx, value in enumerate(clean)}


def _build_encoders(dataset: DatasetBundle) -> dict[str, dict[str, int]]:
    family_values = pd.concat(
        [
            dataset.train_frame["family"].astype("string"),
            dataset.test_frame["family"].astype("string"),
        ],
        ignore_index=True,
    )
    return {
        "family": _build_category_mapping(family_values),
        "city": _build_category_mapping(dataset.stores_frame["city"].astype("string")),
        "state": _build_category_mapping(dataset.stores_frame["state"].astype("string")),
        "type": _build_category_mapping(dataset.stores_frame["type"].astype("string")),
    }


def _encode_with_mapping(values: pd.Series, mapping: dict[str, int]) -> pd.Series:
    encoded = values.astype("string").map(lambda value: mapping.get(str(value), -1))
    return encoded.fillna(-1).astype("int16")


def _prepare_oil_frame(dataset: DatasetBundle) -> pd.DataFrame:
    full_dates = pd.date_range(
        start=min(dataset.train_frame["date"].min(), dataset.test_frame["date"].min()),
        end=max(dataset.train_frame["date"].max(), dataset.test_frame["date"].max()),
        freq="D",
    )
    oil = dataset.oil_frame.loc[:, ["date", "dcoilwtico"]].copy()
    oil = oil.sort_values("date").drop_duplicates("date", keep="last").set_index("date")
    oil = oil.reindex(full_dates).rename_axis("date")
    oil["dcoilwtico"] = oil["dcoilwtico"].ffill().bfill().fillna(0.0)
    oil = oil.rename(columns={"dcoilwtico": "oil"}).reset_index()
    oil["oil"] = oil["oil"].astype("float32")
    return oil


def _prepare_transaction_profile(dataset: DatasetBundle) -> pd.DataFrame:
    """Average transactions per (store, day-of-week).

    The transactions file only covers the training period (the test horizon has
    none), so instead of forecasting it we summarise it into a stable per-store
    weekly seasonality profile that can be joined onto both train and test rows.
    """
    transactions = dataset.transactions_frame.loc[:, ["date", "store_nbr", "transactions"]].copy()
    transactions["date"] = pd.to_datetime(transactions["date"])
    transactions["store_nbr"] = transactions["store_nbr"].astype("int16")
    transactions["dayofweek"] = transactions["date"].dt.dayofweek.astype("int8")
    profile = (
        transactions.groupby(["store_nbr", "dayofweek"], observed=True)["transactions"]
        .mean()
        .reset_index()
        .rename(columns={"transactions": "transactions_dow_mean"})
    )
    profile["transactions_dow_mean"] = profile["transactions_dow_mean"].astype("float32")
    return profile


HOLIDAY_PROXIMITY_CAP = 60


def _active_national_holiday_dates(holidays: pd.DataFrame) -> np.ndarray:
    """Dates on which a national holiday actually lands (same rule as the counts)."""
    national = holidays.loc[holidays["locale"].eq("National")]
    if national.empty:
        return np.array([], dtype="datetime64[D]")
    transferred = national["transferred"].fillna(False).astype(bool)
    type_col = national["type"].astype("string")
    active = (type_col.eq("Holiday") & ~transferred) | type_col.eq("Transfer")
    dates = pd.to_datetime(national.loc[active, "date"]).dropna().drop_duplicates().sort_values()
    return dates.to_numpy().astype("datetime64[D]")


def _holiday_proximity(
    dates: pd.Series, holiday_dates: np.ndarray, cap: int = HOLIDAY_PROXIMITY_CAP
) -> tuple[np.ndarray, np.ndarray]:
    """Signed-free day distance to the next and previous national holiday, capped."""
    day_index = pd.to_datetime(dates).to_numpy().astype("datetime64[D]").astype("int64")
    days_to_next = np.full(len(day_index), float(cap), dtype="float32")
    days_since_prev = np.full(len(day_index), float(cap), dtype="float32")
    if holiday_dates.size == 0:
        return days_to_next, days_since_prev

    holiday_index = holiday_dates.astype("int64")
    # Next holiday at-or-after the date (0 on the holiday itself).
    next_pos = np.searchsorted(holiday_index, day_index, side="left")
    has_next = next_pos < holiday_index.size
    safe_next = np.clip(next_pos, 0, holiday_index.size - 1)
    days_to_next[has_next] = (holiday_index[safe_next] - day_index).astype("float32")[has_next]

    # Previous holiday at-or-before the date (0 on the holiday itself).
    prev_pos = np.searchsorted(holiday_index, day_index, side="right") - 1
    has_prev = prev_pos >= 0
    safe_prev = np.clip(prev_pos, 0, holiday_index.size - 1)
    days_since_prev[has_prev] = (day_index - holiday_index[safe_prev]).astype("float32")[has_prev]

    np.clip(days_to_next, 0.0, float(cap), out=days_to_next)
    np.clip(days_since_prev, 0.0, float(cap), out=days_since_prev)
    return days_to_next, days_since_prev


def _aggregate_holiday_features(
    holidays: pd.DataFrame,
    locale: str,
    group_key: str | None,
    prefix: str,
) -> pd.DataFrame:
    subset = holidays.loc[holidays["locale"].eq(locale)].copy()
    if subset.empty:
        merge_columns = ["date"] if group_key is None else ["date", group_key]
        return pd.DataFrame(
            columns=[*merge_columns, *[f"{prefix}_{name}" for name in HOLIDAY_COUNT_SUFFIXES]]
        )

    if group_key is not None:
        subset = subset.rename(columns={"locale_name": group_key})
        group_columns = ["date", group_key]
    else:
        group_columns = ["date"]

    transferred = subset["transferred"].fillna(False).astype(bool)
    type_col = subset["type"].astype("string")
    # A "Holiday" with transferred=True did NOT happen on this date (it was moved
    # elsewhere), while a "Transfer" row marks the date the day off actually lands.
    # Treat the latter as an active holiday and drop the former from the count.
    is_active_holiday = (type_col.eq("Holiday") & ~transferred) | type_col.eq("Transfer")
    subset["is_holiday"] = is_active_holiday.astype("int8")
    subset["is_event"] = type_col.eq("Event").astype("int8")
    subset["is_additional"] = type_col.eq("Additional").astype("int8")
    subset["is_bridge"] = type_col.eq("Bridge").astype("int8")
    subset["is_workday"] = type_col.eq("Work Day").astype("int8")
    subset["is_transferred"] = transferred.astype("int8")

    aggregated = (
        subset.groupby(group_columns, observed=True)
        .agg(
            holiday_count=("is_holiday", "sum"),
            event_count=("is_event", "sum"),
            additional_count=("is_additional", "sum"),
            bridge_count=("is_bridge", "sum"),
            workday_count=("is_workday", "sum"),
            transferred_count=("is_transferred", "sum"),
        )
        .reset_index()
    )
    return aggregated.rename(
        columns={column: f"{prefix}_{column}" for column in aggregated.columns if column not in group_columns}
    )


def _merge_holiday_features(frame: pd.DataFrame, holidays: pd.DataFrame) -> pd.DataFrame:
    national = _aggregate_holiday_features(holidays, locale="National", group_key=None, prefix="national")
    regional = _aggregate_holiday_features(holidays, locale="Regional", group_key="state", prefix="regional")
    local = _aggregate_holiday_features(holidays, locale="Local", group_key="city", prefix="local")

    work = frame.merge(national, on="date", how="left")
    work = work.merge(regional, on=["date", "state"], how="left")
    work = work.merge(local, on=["date", "city"], how="left")

    holiday_columns = [
        column
        for column in work.columns
        if any(column.endswith(f"_{suffix}") for suffix in HOLIDAY_COUNT_SUFFIXES)
    ]
    for column in holiday_columns:
        work[column] = pd.to_numeric(work[column], errors="coerce").fillna(0.0).astype("float32")
    return work


def _build_base_frame(frame: pd.DataFrame, dataset: DatasetBundle, encoders: dict[str, dict[str, int]]) -> pd.DataFrame:
    work = frame.copy()
    work["date"] = pd.to_datetime(work["date"])
    stores = dataset.stores_frame.copy()
    oil = _prepare_oil_frame(dataset)

    work = work.merge(stores, on="store_nbr", how="left", validate="many_to_one")
    work = work.merge(oil, on="date", how="left", validate="many_to_one")
    work = _merge_holiday_features(work, dataset.holidays_frame)

    work["family_code"] = _encode_with_mapping(work["family"], encoders["family"])
    work["city_code"] = _encode_with_mapping(work["city"], encoders["city"])
    work["state_code"] = _encode_with_mapping(work["state"], encoders["state"])
    work["type_code"] = _encode_with_mapping(work["type"], encoders["type"])

    work["cluster"] = work["cluster"].fillna(-1).astype("int16")
    work["store_nbr"] = work["store_nbr"].astype("int16")
    work["onpromotion"] = work["onpromotion"].fillna(0).astype("float32")
    work["oil"] = work["oil"].fillna(0).astype("float32")

    date_parts = work["date"].dt
    work["dayofweek"] = date_parts.dayofweek.astype("int8")
    work["day"] = date_parts.day.astype("int8")
    work["month"] = date_parts.month.astype("int8")
    work["year"] = date_parts.year.astype("int16")
    work["weekofyear"] = date_parts.isocalendar().week.astype("int16")
    work["is_weekend"] = work["dayofweek"].isin([5, 6]).astype("int8")
    work["is_month_start"] = date_parts.is_month_start.astype("int8")
    work["is_month_end"] = date_parts.is_month_end.astype("int8")
    work["is_payday"] = date_parts.day.isin([1, 15]).astype("int8")

    national_holiday_dates = _active_national_holiday_dates(dataset.holidays_frame)
    days_to_next, days_since_prev = _holiday_proximity(work["date"], national_holiday_dates)
    work["days_to_next_national_holiday"] = days_to_next
    work["days_since_prev_national_holiday"] = days_since_prev

    transaction_profile = _prepare_transaction_profile(dataset)
    work = work.merge(transaction_profile, on=["store_nbr", "dayofweek"], how="left", validate="many_to_one")
    overall_mean = (
        float(transaction_profile["transactions_dow_mean"].mean())
        if not transaction_profile.empty
        else 0.0
    )
    if np.isnan(overall_mean):
        overall_mean = 0.0
    work["transactions_dow_mean"] = work["transactions_dow_mean"].fillna(overall_mean).astype("float32")

    sort_columns = ["store_nbr", "family", "date"]
    return work.sort_values(sort_columns).reset_index(drop=True)


def _add_history_features(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.sort_values(["store_nbr", "family", "date"]).copy()
    work["group_id"] = work.groupby(list(GROUP_COLUMNS), observed=True).ngroup().astype("int32")

    sales_group = work.groupby("group_id", sort=False, observed=True)["sales"]
    promo_group = work.groupby("group_id", sort=False, observed=True)["onpromotion"]

    for lag in SALES_LAGS:
        work[f"sales_lag_{lag}"] = sales_group.shift(lag).astype("float32")
    shifted_sales = sales_group.shift(1)
    shifted_sales_group = shifted_sales.groupby(work["group_id"], sort=False, observed=True)
    for window in ROLL_WINDOWS:
        work[f"sales_roll_mean_{window}"] = (
            shifted_sales_group.rolling(window).mean().reset_index(level=0, drop=True).astype("float32")
        )
        work[f"sales_roll_std_{window}"] = (
            shifted_sales_group.rolling(window).std().reset_index(level=0, drop=True).astype("float32")
        )

    for lag in PROMO_LAGS:
        work[f"promo_lag_{lag}"] = promo_group.shift(lag).astype("float32")
    shifted_promo = promo_group.shift(1)
    for window in PROMO_ROLL_WINDOWS:
        work[f"promo_roll_mean_{window}"] = (
            shifted_promo.groupby(work["group_id"], sort=False, observed=True)
            .rolling(window)
            .mean()
            .reset_index(level=0, drop=True)
            .astype("float32")
        )

    return work.drop(columns=["group_id"])


def _prepare_supervised_training_frame(
    dataset: DatasetBundle,
    encoders: dict[str, dict[str, int]],
    cutoff_date: pd.Timestamp | None,
    recent_train_start: str | None,
    max_train_rows: int | None,
) -> pd.DataFrame:
    train_source = dataset.train_frame.copy()
    if cutoff_date is not None:
        train_source = train_source.loc[train_source["date"] < cutoff_date].copy()

    feature_source = _build_base_frame(train_source, dataset=dataset, encoders=encoders)
    feature_source = _add_history_features(feature_source)
    if recent_train_start:
        feature_source = feature_source.loc[feature_source["date"] >= pd.Timestamp(recent_train_start)].copy()
    if max_train_rows is not None and len(feature_source) > max_train_rows:
        # Keep the most *recent* rows, not the tail of the (store, family, date)
        # sort order. A plain ``iloc[-N:]`` biases the cap toward the highest
        # store/family groups; sorting by date first keeps the latest history.
        feature_source = (
            feature_source.sort_values("date", kind="stable").iloc[-int(max_train_rows) :].copy()
        )

    feature_source = feature_source.reset_index(drop=True)
    return feature_source


def _build_histories(
    history_frame: pd.DataFrame,
) -> tuple[dict[tuple[int, str], deque[float]], dict[tuple[int, str], deque[float]]]:
    sales_histories: dict[tuple[int, str], deque[float]] = {}
    promo_histories: dict[tuple[int, str], deque[float]] = {}

    ordered = history_frame.sort_values(["store_nbr", "family", "date"])
    for row in ordered.itertuples(index=False):
        key = (int(row.store_nbr), str(row.family))
        sales_histories.setdefault(key, deque(maxlen=HISTORY_LENGTH)).append(float(getattr(row, "sales", 0.0)))
        promo_histories.setdefault(key, deque(maxlen=HISTORY_LENGTH)).append(float(getattr(row, "onpromotion", 0.0)))
    return sales_histories, promo_histories


def _safe_lag(history: deque[float], lag: int) -> float:
    if len(history) < lag:
        return float("nan")
    return float(history[-lag])


def _safe_mean(history: deque[float], window: int) -> float:
    if not history:
        return float("nan")
    values = list(history)[-window:]
    if not values:
        return float("nan")
    return float(np.mean(values))


def _safe_std(history: deque[float], window: int) -> float:
    # Mirror the training-time rolling std (full window, sample ddof=1) so the
    # recursive forecast sees features on the same scale it was trained on.
    if len(history) < window:
        return float("nan")
    values = list(history)[-window:]
    return float(np.std(values, ddof=1))


def _attach_future_history_features(
    batch: pd.DataFrame,
    sales_histories: dict[tuple[int, str], deque[float]],
    promo_histories: dict[tuple[int, str], deque[float]],
) -> pd.DataFrame:
    work = batch.copy()
    sales_feature_values: dict[str, list[float]] = {
        column: [] for column in [*LAG_FEATURE_COLUMNS, *ROLL_FEATURE_COLUMNS, *ROLL_STD_FEATURE_COLUMNS]
    }
    promo_feature_values: dict[str, list[float]] = {column: [] for column in PROMO_FEATURE_COLUMNS}

    for row in work.itertuples(index=False):
        key = (int(row.store_nbr), str(row.family))
        sales_history = sales_histories.get(key, deque(maxlen=HISTORY_LENGTH))
        promo_history = promo_histories.get(key, deque(maxlen=HISTORY_LENGTH))

        for lag in SALES_LAGS:
            sales_feature_values[f"sales_lag_{lag}"].append(_safe_lag(sales_history, lag))
        for window in ROLL_WINDOWS:
            sales_feature_values[f"sales_roll_mean_{window}"].append(_safe_mean(sales_history, window))
            sales_feature_values[f"sales_roll_std_{window}"].append(_safe_std(sales_history, window))
        for lag in PROMO_LAGS:
            promo_feature_values[f"promo_lag_{lag}"].append(_safe_lag(promo_history, lag))
        for window in PROMO_ROLL_WINDOWS:
            promo_feature_values[f"promo_roll_mean_{window}"].append(_safe_mean(promo_history, window))

    for column, values in sales_feature_values.items():
        work[column] = pd.Series(values, index=work.index, dtype="float32")
    for column, values in promo_feature_values.items():
        work[column] = pd.Series(values, index=work.index, dtype="float32")
    return work


def _training_fill_values(frame: pd.DataFrame) -> dict[str, float]:
    fill_values: dict[str, float] = {}
    for column in FEATURE_COLUMNS:
        median = float(frame[column].median()) if column in frame.columns else 0.0
        if np.isnan(median):
            median = 0.0
        fill_values[column] = median
    return fill_values


def _feature_matrix(frame: pd.DataFrame, fill_values: dict[str, float]) -> pd.DataFrame:
    matrix = frame.loc[:, FEATURE_COLUMNS].copy()
    for column, value in fill_values.items():
        matrix[column] = matrix[column].fillna(value)
    return matrix.astype("float32")


def _log_target(sales: Any) -> np.ndarray:
    """Log1p-transform the regression target, guarding against negative sales.

    Sales are non-negative by definition, but returns/refunds or corrupted rows
    can produce small negatives that would make ``log1p`` emit NaN and poison
    training. Clip to zero first so the transform stays finite.
    """
    values = np.asarray(sales, dtype=np.float32)
    return np.log1p(np.clip(values, 0.0, None))


def _recent_sample_weight(frame: pd.DataFrame) -> np.ndarray:
    day_index = (frame["date"] - frame["date"].min()).dt.days.to_numpy(dtype=np.float32)
    if len(day_index) == 0:
        return np.asarray([], dtype=np.float32)
    if float(day_index.max()) <= 0:
        return np.ones_like(day_index, dtype=np.float32)
    return 1.0 + day_index / float(day_index.max())


def _lightgbm_available() -> bool:
    return lgb is not None


def _xgboost_available() -> bool:
    return xgb is not None


def _time_validation_split(
    frame: pd.DataFrame, valid_days: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split off the most recent ``valid_days`` calendar days for early stopping.

    Returns ``(train_part, valid_part)``. When there is not enough history to
    carve out a non-empty split on both sides, the validation part is empty and
    callers fall back to training on the full frame without early stopping.
    """
    unique_dates = np.sort(frame["date"].drop_duplicates().to_numpy())
    if valid_days <= 0 or len(unique_dates) <= valid_days + 1:
        return frame, frame.iloc[0:0]
    cutoff = pd.Timestamp(unique_dates[-valid_days])
    train_part = frame.loc[frame["date"] < cutoff]
    valid_part = frame.loc[frame["date"] >= cutoff]
    if train_part.empty or valid_part.empty:
        return frame, frame.iloc[0:0]
    return train_part, valid_part


def _fit_trained_strategy(
    train_frame: pd.DataFrame,
    strategy: str,
    seed: int,
) -> dict[str, Any]:
    # Compute fill values on the full frame so train/validation share the same
    # imputation, then carve off the most recent days for early stopping.
    fill_values = _training_fill_values(train_frame)
    fit_frame, valid_frame = _time_validation_split(train_frame, EARLY_STOPPING_VALID_DAYS)
    use_early_stopping = not valid_frame.empty

    x_train = _feature_matrix(fit_frame, fill_values=fill_values)
    y_train = _log_target(fit_frame["sales"].to_numpy())
    sample_weight = _recent_sample_weight(fit_frame)
    eval_set = None
    if use_early_stopping:
        x_valid = _feature_matrix(valid_frame, fill_values=fill_values)
        y_valid = _log_target(valid_frame["sales"].to_numpy())
        eval_set = [(x_valid, y_valid)]

    if strategy == "lightgbm":
        if not _lightgbm_available():
            raise RuntimeError("lightgbm strategy selected but lightgbm is not installed.")
        model = lgb.LGBMRegressor(
            objective="rmse",
            n_estimators=700,
            learning_rate=0.05,
            num_leaves=127,
            min_child_samples=50,
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=0.85,
            reg_alpha=0.05,
            reg_lambda=1.0,
            random_state=seed,
            n_jobs=-1,
            # Silence the per-tree training chatter and make fits reproducible
            # regardless of thread count / dataset size.
            verbose=-1,
            deterministic=True,
            force_row_wise=True,
        )
        fit_kwargs: dict[str, Any] = {"sample_weight": sample_weight}
        if use_early_stopping:
            fit_kwargs["eval_set"] = eval_set
            fit_kwargs["callbacks"] = [
                lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                lgb.log_evaluation(0),
            ]
        model.fit(x_train, y_train, **fit_kwargs)
    elif strategy == "xgboost":
        if not _xgboost_available():
            raise RuntimeError("xgboost strategy selected but xgboost is not installed.")
        xgb_params: dict[str, Any] = dict(
            objective="reg:squarederror",
            n_estimators=550,
            learning_rate=0.05,
            max_depth=10,
            min_child_weight=6.0,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.0,
            reg_lambda=1.0,
            tree_method="hist",
            random_state=seed,
            n_jobs=-1,
        )
        if use_early_stopping:
            xgb_params["early_stopping_rounds"] = EARLY_STOPPING_ROUNDS
        model = xgb.XGBRegressor(**xgb_params)
        fit_kwargs = {"sample_weight": sample_weight}
        if use_early_stopping:
            fit_kwargs["eval_set"] = eval_set
            fit_kwargs["verbose"] = False
        model.fit(x_train, y_train, **fit_kwargs)
    else:
        raise ValueError(f"Unsupported trained strategy: {strategy}")

    return {
        "selected_strategy": strategy,
        "feature_columns": list(FEATURE_COLUMNS),
        "fill_values": fill_values,
        "model": model,
    }


def _seasonal_naive_prediction(batch: pd.DataFrame) -> np.ndarray:
    lag_7 = batch["sales_lag_7"].fillna(batch["sales_lag_14"])
    lag_14 = batch["sales_lag_14"].fillna(batch["sales_roll_mean_14"])
    lag_28 = batch["sales_lag_28"].fillna(batch["sales_roll_mean_28"])
    base = 0.55 * lag_7 + 0.30 * lag_14 + 0.15 * lag_28
    base = base.fillna(batch["sales_roll_mean_28"]).fillna(batch["sales_lag_1"]).fillna(0.0)

    promo_reference = batch["promo_lag_7"].fillna(batch["promo_roll_mean_7"]).fillna(0.0)
    promo_delta = batch["onpromotion"] - promo_reference
    adjustment = 1.0 + 0.015 * np.clip(promo_delta.to_numpy(dtype=np.float32), -20.0, 80.0)
    prediction = np.clip(base.to_numpy(dtype=np.float32) * adjustment, 0.0, None)
    return prediction


def _predict_future(
    strategy_bundle: dict[str, Any],
    history_frame: pd.DataFrame,
    future_frame: pd.DataFrame,
) -> pd.DataFrame:
    sales_histories, promo_histories = _build_histories(history_frame)
    ordered_future = future_frame.sort_values(["date", "store_nbr", "family"]).copy()
    prediction_batches: list[pd.DataFrame] = []

    strategy = str(strategy_bundle["selected_strategy"])
    fill_values = dict(strategy_bundle.get("fill_values", {}))
    model = strategy_bundle.get("model")

    for current_date, date_batch in ordered_future.groupby("date", sort=True):
        batch = _attach_future_history_features(date_batch, sales_histories=sales_histories, promo_histories=promo_histories)
        if strategy == "seasonal_naive":
            prediction = _seasonal_naive_prediction(batch)
        else:
            matrix = _feature_matrix(batch, fill_values=fill_values)
            prediction = np.expm1(np.asarray(model.predict(matrix), dtype=np.float32))
            prediction = np.clip(prediction, 0.0, None)

        batch["prediction"] = prediction.astype("float32")
        prediction_batches.append(batch)

        for row in batch.itertuples(index=False):
            key = (int(row.store_nbr), str(row.family))
            sales_histories.setdefault(key, deque(maxlen=HISTORY_LENGTH)).append(float(row.prediction))
            promo_histories.setdefault(key, deque(maxlen=HISTORY_LENGTH)).append(float(row.onpromotion))

    combined = pd.concat(prediction_batches, ignore_index=True) if prediction_batches else ordered_future
    return combined.sort_values(["date", "store_nbr", "family"]).reset_index(drop=True)


def _rmsle(actual: np.ndarray, prediction: np.ndarray) -> float:
    actual = np.clip(np.asarray(actual, dtype=np.float64), 0.0, None)
    prediction = np.clip(np.asarray(prediction, dtype=np.float64), 0.0, None)
    return float(np.sqrt(mean_squared_error(np.log1p(actual), np.log1p(prediction))))


def fit_and_score_holdout(
    dataset: DatasetBundle,
    holdout_days: int = 16,
    recent_train_start: str = "2016-01-01",
    max_train_rows: int | None = 1_200_000,
    candidate_strategies: tuple[str, ...] = ("seasonal_naive", "lightgbm", "xgboost"),
    seed: int = 42,
    force_strategy: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame]:
    encoders = _build_encoders(dataset)
    if force_strategy is not None and force_strategy not in candidate_strategies:
        raise ValueError(
            f"force_strategy={force_strategy!r} must be one of the candidate strategies "
            f"{list(candidate_strategies)} so its holdout metrics can be reported."
        )
    train_dates = np.sort(dataset.train_frame["date"].drop_duplicates().to_numpy())
    if holdout_days <= 0 or holdout_days >= len(train_dates):
        raise ValueError("holdout_days must be between 1 and the number of unique train dates - 1.")
    holdout_start = pd.Timestamp(train_dates[-holdout_days])

    train_history = _build_base_frame(
        dataset.train_frame.loc[dataset.train_frame["date"] < holdout_start].copy(),
        dataset=dataset,
        encoders=encoders,
    )
    holdout_future = _build_base_frame(
        dataset.train_frame.loc[dataset.train_frame["date"] >= holdout_start].copy(),
        dataset=dataset,
        encoders=encoders,
    )
    supervised_train = _prepare_supervised_training_frame(
        dataset=dataset,
        encoders=encoders,
        cutoff_date=holdout_start,
        recent_train_start=recent_train_start,
        max_train_rows=max_train_rows,
    )

    if supervised_train.empty:
        raise ValueError("Supervised training frame is empty. Lower recent_train_start or increase available history.")

    print(
        "Prepared holdout setup: "
        f"holdout_start={holdout_start.date()}, "
        f"history_rows={len(train_history)}, "
        f"supervised_rows={len(supervised_train)}"
    )

    strategy_metrics: list[dict[str, Any]] = []
    holdout_predictions_by_strategy: dict[str, pd.DataFrame] = {}

    for strategy in candidate_strategies:
        print(f"Evaluating strategy: {strategy}")
        if strategy == "seasonal_naive":
            strategy_bundle = {"selected_strategy": "seasonal_naive"}
        else:
            strategy_bundle = _fit_trained_strategy(supervised_train, strategy=strategy, seed=seed)

        predicted = _predict_future(
            strategy_bundle=strategy_bundle,
            history_frame=train_history,
            future_frame=holdout_future,
        )
        actual = predicted[dataset.target_column].to_numpy(dtype=np.float32)
        forecast = predicted["prediction"].to_numpy(dtype=np.float32)
        metrics_row = {
            "name": strategy,
            "rmsle": _rmsle(actual, forecast),
            "mae": float(mean_absolute_error(actual, forecast)),
        }
        print(
            f"Completed strategy: {strategy} "
            f"(rmsle={metrics_row['rmsle']:.6f}, mae={metrics_row['mae']:.6f})"
        )
        strategy_metrics.append(metrics_row)
        holdout_predictions_by_strategy[strategy] = predicted

    strategy_metrics.sort(key=lambda row: (row["rmsle"], row["mae"], row["name"]))
    metrics_by_name = {str(row["name"]): row for row in strategy_metrics}
    if force_strategy is not None:
        selected_strategy = str(force_strategy)
    else:
        selected_strategy = str(strategy_metrics[0]["name"])
    selected_metrics = metrics_by_name[selected_strategy]
    selected_predictions = holdout_predictions_by_strategy[selected_strategy]
    selection = {
        "selected_strategy": selected_strategy,
        "holdout_days": int(holdout_days),
        "recent_train_start": recent_train_start,
        "max_train_rows": None if max_train_rows is None else int(max_train_rows),
        "candidate_strategies": list(candidate_strategies),
        "feature_columns": list(FEATURE_COLUMNS),
    }
    holdout_end = pd.Timestamp(train_dates[-1])
    holdout_metrics = {
        "selected_strategy": selected_strategy,
        "rmsle": float(selected_metrics["rmsle"]),
        "mae": float(selected_metrics["mae"]),
        "holdout_start": str(holdout_start.date()),
        "holdout_end": str(holdout_end.date()),
        "history_rows": int(len(train_history)),
        "holdout_rows": int(len(holdout_future)),
        "supervised_rows": int(len(supervised_train)),
        "strategy_metrics": strategy_metrics,
    }
    holdout_predictions = selected_predictions.loc[
        :,
        [dataset.id_column, "date", "store_nbr", "family", dataset.target_column, "prediction"],
    ].rename(columns={dataset.target_column: "actual"})
    return selection, holdout_metrics, holdout_predictions


def fit_final_model(
    dataset: DatasetBundle,
    selection: dict[str, Any],
    recent_train_start: str = "2016-01-01",
    max_train_rows: int | None = 1_200_000,
    seed: int = 42,
) -> dict[str, Any]:
    encoders = _build_encoders(dataset)
    strategy = str(selection["selected_strategy"])

    if strategy == "seasonal_naive":
        return {
            "competition": COMPETITION_SLUG,
            "selected_strategy": strategy,
            "feature_columns": list(FEATURE_COLUMNS),
            "encoders": encoders,
            "seed": int(seed),
        }

    supervised_train = _prepare_supervised_training_frame(
        dataset=dataset,
        encoders=encoders,
        cutoff_date=None,
        recent_train_start=recent_train_start,
        max_train_rows=max_train_rows,
    )
    trained = _fit_trained_strategy(supervised_train, strategy=strategy, seed=seed)
    return {
        "competition": COMPETITION_SLUG,
        "selected_strategy": strategy,
        "feature_columns": list(FEATURE_COLUMNS),
        "encoders": encoders,
        "fill_values": trained["fill_values"],
        "model": trained["model"],
        "seed": int(seed),
    }


def generate_submission(model_bundle: dict[str, Any], dataset: DatasetBundle) -> pd.DataFrame:
    encoders = dict(model_bundle.get("encoders", _build_encoders(dataset)))
    train_history = _build_base_frame(dataset.train_frame, dataset=dataset, encoders=encoders)
    test_future = _build_base_frame(dataset.test_frame, dataset=dataset, encoders=encoders)
    predicted = _predict_future(
        strategy_bundle=model_bundle,
        history_frame=train_history,
        future_frame=test_future,
    )
    submission = predicted.loc[:, [dataset.id_column, "prediction"]].rename(columns={"prediction": dataset.target_column})
    # validate="one_to_one" turns a duplicated/forecast id into a hard error
    # instead of silently fanning the submission out to the wrong length.
    ordered = dataset.sample_submission[[dataset.id_column]].merge(
        submission, on=dataset.id_column, how="left", validate="one_to_one"
    )
    if ordered[dataset.target_column].isna().any():
        raise ValueError("Submission contains missing predictions after merge. Check id alignment.")
    if len(ordered) != len(dataset.sample_submission):
        raise ValueError("Submission row count does not match the sample submission.")
    target = ordered[dataset.target_column].to_numpy(dtype=np.float64)
    if not np.isfinite(target).all():
        raise ValueError("Submission contains non-finite predictions.")
    if (target < 0).any():
        raise ValueError("Submission contains negative predictions.")
    ordered[dataset.target_column] = ordered[dataset.target_column].astype("float32")
    return ordered


def save_model_bundle(model_bundle: dict[str, Any], path: Path = DEFAULT_MODEL_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_bundle, path)


def load_model_bundle(path: Path = DEFAULT_MODEL_PATH) -> dict[str, Any]:
    return joblib.load(path)


def save_metrics(payload: dict[str, Any], path: Path = DEFAULT_METRICS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)
        handle.write("\n")
