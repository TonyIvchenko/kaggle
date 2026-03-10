"""Generic tabular baselines for Playground Series S6E3."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_float_dtype,
    is_integer_dtype,
    is_numeric_dtype,
    is_object_dtype,
)
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, log_loss, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


COMPETITION_SLUG = "playground-series-s6e3"
COMPETITION_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = COMPETITION_ROOT / "data" / "raw"
DEFAULT_PROCESSED_DIR = COMPETITION_ROOT / "data" / "processed"
DEFAULT_MODEL_PATH = COMPETITION_ROOT / "models" / "playground_series_s6e3.joblib"
DEFAULT_METRICS_PATH = COMPETITION_ROOT / "models" / "playground_series_s6e3_metrics.json"
DEFAULT_SUBMISSION_PATH = COMPETITION_ROOT / "submissions" / "submission.csv"


@dataclass(frozen=True)
class CompetitionFiles:
    raw_dir: Path
    train: Path
    test: Path
    sample_submission: Path


@dataclass(frozen=True)
class DatasetBundle:
    train_frame: pd.DataFrame
    test_frame: pd.DataFrame
    sample_submission: pd.DataFrame
    id_column: str
    target_column: str
    feature_columns: tuple[str, ...]
    task_type: str
    probability_output: bool


@dataclass(frozen=True)
class CandidateMetrics:
    name: str
    task_type: str
    score: float
    accuracy: float | None = None
    log_loss: float | None = None
    rmse: float | None = None
    mae: float | None = None


def available_csv_files(raw_dir: Path) -> list[Path]:
    return sorted(path for path in raw_dir.rglob("*.csv") if path.is_file())


def _pick_file(csv_files: list[Path], candidates: tuple[str, ...]) -> Path:
    by_name = {path.name.lower(): path for path in csv_files}
    for name in candidates:
        match = by_name.get(name.lower())
        if match is not None:
            return match
    available = ", ".join(sorted(path.name for path in csv_files))
    raise FileNotFoundError(f"Could not find any of {candidates}. Available CSV files: {available}")


def discover_competition_files(raw_dir: Path = DEFAULT_RAW_DIR) -> CompetitionFiles:
    csv_files = available_csv_files(raw_dir)
    if not csv_files:
        raise FileNotFoundError(f"No CSV files were found under {raw_dir}. Run download_data.py first.")
    return CompetitionFiles(
        raw_dir=raw_dir,
        train=_pick_file(csv_files, ("train.csv",)),
        test=_pick_file(csv_files, ("test.csv",)),
        sample_submission=_pick_file(csv_files, ("sample_submission.csv", "sampleSubmission.csv")),
    )


def _choose_id_column(train: pd.DataFrame, test: pd.DataFrame, sample_submission: pd.DataFrame) -> str:
    preferred = ("id", "ID", "Id")
    for column in preferred:
        if column in test.columns and column in sample_submission.columns:
            return column

    shared = [column for column in sample_submission.columns if column in test.columns]
    if not shared:
        raise ValueError("Could not infer id column: sample_submission and test share no columns.")
    for column in shared:
        if column in train.columns:
            return str(column)
    return str(shared[0])


def _choose_target_column(
    train: pd.DataFrame,
    test: pd.DataFrame,
    sample_submission: pd.DataFrame,
    id_column: str,
    target_override: str | None = None,
) -> str:
    if target_override:
        if target_override not in train.columns:
            raise ValueError(f"Requested target column '{target_override}' is not in train.csv.")
        if target_override not in sample_submission.columns:
            raise ValueError(f"Requested target column '{target_override}' is not in sample_submission.csv.")
        return target_override

    candidates = [column for column in sample_submission.columns if column != id_column]
    if len(candidates) == 1:
        return str(candidates[0])
    for column in candidates:
        if column in train.columns and column not in test.columns:
            return str(column)
    for column in candidates:
        if column in train.columns:
            return str(column)
    raise ValueError(
        "Could not infer target column from schema. Pass --target-column explicitly."
    )


def _choose_feature_columns(
    train: pd.DataFrame,
    test: pd.DataFrame,
    id_column: str,
    target_column: str,
) -> tuple[str, ...]:
    columns = [column for column in test.columns if column != id_column and column in train.columns]
    columns = [column for column in columns if column != target_column]
    if not columns:
        fallback = [column for column in train.columns if column not in {id_column, target_column}]
        if not fallback:
            raise ValueError("Could not infer feature columns.")
        columns = fallback
    return tuple(str(column) for column in columns)


def _infer_task_type(target: pd.Series) -> str:
    values = target.dropna()
    if values.empty:
        return "regression"
    if is_object_dtype(values) or isinstance(values.dtype, pd.CategoricalDtype) or is_bool_dtype(values):
        return "classification"

    unique_count = int(values.nunique())
    if is_integer_dtype(values):
        threshold = max(20, int(0.02 * len(values)))
        return "classification" if unique_count <= threshold else "regression"

    if is_float_dtype(values):
        threshold = max(20, int(0.02 * len(values)))
        rounded = bool(np.allclose(values.to_numpy(dtype=float), np.round(values.to_numpy(dtype=float)), atol=1e-8))
        if rounded and unique_count <= threshold:
            return "classification"

    return "regression"


def _infer_probability_output(sample_submission: pd.DataFrame, target_column: str, task_type: str) -> bool:
    if task_type != "classification":
        return False
    candidate = pd.to_numeric(sample_submission[target_column], errors="coerce")
    return bool(candidate.notna().all())


def build_datasets(files: CompetitionFiles, target_column: str | None = None) -> DatasetBundle:
    train = pd.read_csv(files.train)
    test = pd.read_csv(files.test)
    sample_submission = pd.read_csv(files.sample_submission)

    id_column = _choose_id_column(train, test, sample_submission)
    target = _choose_target_column(
        train=train,
        test=test,
        sample_submission=sample_submission,
        id_column=id_column,
        target_override=target_column,
    )
    features = _choose_feature_columns(train, test, id_column=id_column, target_column=target)
    task_type = _infer_task_type(train[target])
    probability_output = _infer_probability_output(sample_submission, target_column=target, task_type=task_type)

    return DatasetBundle(
        train_frame=train.copy(),
        test_frame=test.copy(),
        sample_submission=sample_submission.copy(),
        id_column=id_column,
        target_column=target,
        feature_columns=features,
        task_type=task_type,
        probability_output=probability_output,
    )


def _split_feature_types(frame: pd.DataFrame, feature_columns: tuple[str, ...]) -> tuple[list[str], list[str]]:
    numeric_cols = [column for column in feature_columns if is_numeric_dtype(frame[column])]
    categorical_cols = [column for column in feature_columns if column not in numeric_cols]
    return numeric_cols, categorical_cols


def _build_linear_pipeline(
    frame: pd.DataFrame,
    feature_columns: tuple[str, ...],
    task_type: str,
    seed: int,
) -> Pipeline:
    numeric_cols, categorical_cols = _split_feature_types(frame, feature_columns)
    transformers: list[tuple[str, Any, list[str]]] = []

    if numeric_cols:
        transformers.append(
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler(with_mean=False)),
                    ]
                ),
                numeric_cols,
            )
        )

    if categorical_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                min_frequency=5,
                            ),
                        ),
                    ]
                ),
                categorical_cols,
            )
        )

    if not transformers:
        raise ValueError("No feature columns available for linear pipeline.")

    preprocessor = ColumnTransformer(transformers=transformers, sparse_threshold=1.0)
    if task_type == "classification":
        model = LogisticRegression(
            solver="saga",
            max_iter=1500,
            random_state=seed,
        )
    else:
        model = Ridge(alpha=1.0, random_state=seed)
    return Pipeline(steps=[("preprocess", preprocessor), ("model", model)])


def _build_tree_pipeline(
    frame: pd.DataFrame,
    feature_columns: tuple[str, ...],
    task_type: str,
    seed: int,
) -> Pipeline:
    numeric_cols, categorical_cols = _split_feature_types(frame, feature_columns)
    transformers: list[tuple[str, Any, list[str]]] = []

    if numeric_cols:
        transformers.append(
            (
                "num",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
                numeric_cols,
            )
        )

    if categorical_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "ordinal",
                            OrdinalEncoder(
                                handle_unknown="use_encoded_value",
                                unknown_value=-1,
                                encoded_missing_value=-1,
                            ),
                        ),
                    ]
                ),
                categorical_cols,
            )
        )

    if not transformers:
        raise ValueError("No feature columns available for tree pipeline.")

    preprocessor = ColumnTransformer(transformers=transformers, sparse_threshold=0.0)
    if task_type == "classification":
        model = HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_depth=8,
            max_iter=350,
            min_samples_leaf=40,
            random_state=seed,
        )
    else:
        model = HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_depth=8,
            max_iter=350,
            min_samples_leaf=40,
            random_state=seed,
        )
    return Pipeline(steps=[("preprocess", preprocessor), ("model", model)])


def _evaluate_classification(
    y_true: pd.Series,
    pred_labels: np.ndarray,
    pred_proba: np.ndarray | None,
    classes: np.ndarray | None,
) -> tuple[float, float]:
    accuracy = float(accuracy_score(y_true, pred_labels))
    if pred_proba is None:
        return accuracy, float("nan")
    try:
        if classes is None:
            loss = float(log_loss(y_true, pred_proba))
        else:
            loss = float(log_loss(y_true, pred_proba, labels=list(classes)))
    except ValueError:
        loss = float("nan")
    return accuracy, loss


def _evaluate_regression(y_true: pd.Series, pred_values: np.ndarray) -> tuple[float, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, pred_values)))
    mae = float(mean_absolute_error(y_true, pred_values))
    return rmse, mae


def _stratify_target(target: pd.Series, task_type: str) -> pd.Series | None:
    if task_type != "classification":
        return None
    counts = target.value_counts(dropna=False)
    if len(counts) < 2 or int(counts.min()) < 2:
        return None
    return target


def _candidate_score(metrics: CandidateMetrics) -> float:
    if metrics.task_type == "classification":
        if metrics.log_loss is not None and np.isfinite(metrics.log_loss):
            return float(-metrics.log_loss)
        return float(metrics.accuracy or 0.0)
    return float(-(metrics.rmse or np.inf))


def fit_and_score_holdout(
    dataset: DatasetBundle,
    holdout_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame]:
    if not 0.05 <= holdout_fraction <= 0.5:
        raise ValueError("holdout_fraction should be between 0.05 and 0.5")

    work = dataset.train_frame.copy()
    x_all = work.loc[:, dataset.feature_columns]
    y_all = work.loc[:, dataset.target_column]

    train_idx, holdout_idx = train_test_split(
        np.arange(len(work), dtype=np.int64),
        test_size=float(holdout_fraction),
        random_state=seed,
        shuffle=True,
        stratify=_stratify_target(y_all, dataset.task_type),
    )
    x_train = x_all.iloc[train_idx]
    y_train = y_all.iloc[train_idx]
    x_holdout = x_all.iloc[holdout_idx]
    y_holdout = y_all.iloc[holdout_idx]

    candidates = {
        "linear": _build_linear_pipeline(work, dataset.feature_columns, dataset.task_type, seed=seed),
        "tree": _build_tree_pipeline(work, dataset.feature_columns, dataset.task_type, seed=seed),
    }

    candidate_rows: list[CandidateMetrics] = []
    prediction_cache: dict[str, tuple[np.ndarray, np.ndarray | None, np.ndarray | None]] = {}

    for name, model in candidates.items():
        model.fit(x_train, y_train)
        pred_labels = model.predict(x_holdout)
        pred_proba: np.ndarray | None = None
        classes: np.ndarray | None = None
        if dataset.task_type == "classification" and hasattr(model, "predict_proba"):
            pred_proba = model.predict_proba(x_holdout)
            classes = getattr(model.named_steps["model"], "classes_", None)

        if dataset.task_type == "classification":
            acc, ll = _evaluate_classification(
                y_true=y_holdout,
                pred_labels=np.asarray(pred_labels),
                pred_proba=pred_proba,
                classes=classes,
            )
            row = CandidateMetrics(
                name=name,
                task_type=dataset.task_type,
                accuracy=acc,
                log_loss=ll,
                score=0.0,
            )
        else:
            rmse, mae = _evaluate_regression(y_true=y_holdout, pred_values=np.asarray(pred_labels, dtype=float))
            row = CandidateMetrics(
                name=name,
                task_type=dataset.task_type,
                rmse=rmse,
                mae=mae,
                score=0.0,
            )
        row = CandidateMetrics(**{**asdict(row), "score": _candidate_score(row)})
        candidate_rows.append(row)
        prediction_cache[name] = (np.asarray(pred_labels), pred_proba, classes)

    best = max(candidate_rows, key=lambda row: row.score)
    selected_labels, selected_proba, selected_classes = prediction_cache[best.name]

    selection = {
        "selected_strategy": best.name,
        "task_type": dataset.task_type,
        "id_column": dataset.id_column,
        "target_column": dataset.target_column,
        "feature_columns": list(dataset.feature_columns),
        "probability_output": dataset.probability_output,
        "seed": int(seed),
    }

    if dataset.task_type == "classification":
        holdout_metrics = {
            "rows": int(len(y_holdout)),
            "task_type": dataset.task_type,
            "selected_strategy": best.name,
            "accuracy": float(best.accuracy or 0.0),
            "log_loss": float(best.log_loss) if best.log_loss is not None else float("nan"),
            "strategy_metrics": [asdict(row) for row in candidate_rows],
        }
    else:
        holdout_metrics = {
            "rows": int(len(y_holdout)),
            "task_type": dataset.task_type,
            "selected_strategy": best.name,
            "rmse": float(best.rmse or 0.0),
            "mae": float(best.mae or 0.0),
            "strategy_metrics": [asdict(row) for row in candidate_rows],
        }

    output_prediction: np.ndarray | pd.Series
    if dataset.task_type == "classification" and dataset.probability_output and selected_proba is not None:
        if selected_proba.shape[1] == 2:
            output_prediction = selected_proba[:, 1]
        else:
            output_prediction = selected_labels
    else:
        output_prediction = selected_labels

    holdout_predictions = pd.DataFrame(
        {
            dataset.id_column: work.iloc[holdout_idx][dataset.id_column].tolist(),
            dataset.target_column: y_holdout.tolist(),
            "prediction": pd.Series(output_prediction).tolist(),
        }
    )
    if dataset.task_type == "classification":
        holdout_predictions["prediction_label"] = pd.Series(selected_labels).tolist()
        if selected_proba is not None and selected_proba.shape[1] == 2:
            holdout_predictions["prediction_proba"] = selected_proba[:, 1].tolist()
        if selected_classes is not None:
            holdout_predictions["classes"] = [json.dumps([str(v) for v in selected_classes])] * len(
                holdout_predictions
            )
    return selection, holdout_metrics, holdout_predictions


def fit_final_model(dataset: DatasetBundle, selection: dict[str, Any], seed: int = 42) -> dict[str, Any]:
    strategy = str(selection["selected_strategy"])
    if strategy == "tree":
        model = _build_tree_pipeline(dataset.train_frame, dataset.feature_columns, dataset.task_type, seed=seed)
    else:
        model = _build_linear_pipeline(dataset.train_frame, dataset.feature_columns, dataset.task_type, seed=seed)
        strategy = "linear"

    x_train = dataset.train_frame.loc[:, dataset.feature_columns]
    y_train = dataset.train_frame.loc[:, dataset.target_column]
    model.fit(x_train, y_train)

    return {
        "competition": COMPETITION_SLUG,
        "selected_strategy": strategy,
        "task_type": dataset.task_type,
        "id_column": dataset.id_column,
        "target_column": dataset.target_column,
        "feature_columns": list(dataset.feature_columns),
        "probability_output": dataset.probability_output,
        "model": model,
        "seed": int(seed),
    }


def generate_submission(model_bundle: dict[str, Any], dataset: DatasetBundle) -> pd.DataFrame:
    model = model_bundle["model"]
    x_test = dataset.test_frame.loc[:, dataset.feature_columns]

    if dataset.task_type == "classification":
        labels = np.asarray(model.predict(x_test))
        prediction: np.ndarray | pd.Series
        if dataset.probability_output and hasattr(model, "predict_proba"):
            proba = np.asarray(model.predict_proba(x_test))
            prediction = proba[:, 1] if proba.shape[1] == 2 else labels
        else:
            prediction = labels
    else:
        prediction = np.asarray(model.predict(x_test), dtype=float)

    prediction_frame = pd.DataFrame(
        {
            dataset.id_column: dataset.test_frame[dataset.id_column].tolist(),
            dataset.target_column: pd.Series(prediction).tolist(),
        }
    )
    ordered = dataset.sample_submission[[dataset.id_column]].merge(
        prediction_frame,
        on=dataset.id_column,
        how="left",
    )
    if ordered[dataset.target_column].isna().any():
        raise ValueError("Submission contains missing predictions after merge. Check id alignment.")
    return ordered


def save_model_bundle(model_bundle: dict[str, Any], path: Path = DEFAULT_MODEL_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_bundle, path)


def load_model_bundle(path: Path = DEFAULT_MODEL_PATH) -> dict[str, Any]:
    return joblib.load(path)


def save_metrics(payload: dict[str, Any], path: Path = DEFAULT_METRICS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
