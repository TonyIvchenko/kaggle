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
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
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
    auc: float | None = None
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
    raise ValueError("Could not infer target column from schema. Pass --target-column explicitly.")


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
        rounded = bool(
            np.allclose(values.to_numpy(dtype=float), np.round(values.to_numpy(dtype=float)), atol=1e-8)
        )
        if rounded and unique_count <= threshold:
            return "classification"

    return "regression"


def _infer_probability_output(sample_submission: pd.DataFrame, target_column: str, task_type: str) -> bool:
    if task_type != "classification":
        return False
    candidate = pd.to_numeric(sample_submission[target_column], errors="coerce")
    return bool(candidate.notna().all())


def _normalize_label(value: Any) -> str:
    return str(value).strip().lower()


def _infer_positive_label(target: pd.Series) -> Any | None:
    values = [value for value in target.dropna().unique().tolist()]
    if len(values) != 2:
        return None

    priority = {"yes", "true", "1", "y", "churn", "positive"}
    for value in values:
        if _normalize_label(value) in priority:
            return value

    for value in values:
        try:
            if float(value) == 1.0:
                return value
        except Exception:
            continue

    try:
        return sorted(values, key=lambda x: float(x))[-1]
    except Exception:
        return sorted(values, key=lambda x: str(x))[-1]


def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _feature_engineering(
    train: pd.DataFrame,
    test: pd.DataFrame,
    id_column: str,
    target_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_out = train.copy()
    test_out = test.copy()

    frames = (train_out, test_out)
    for frame in frames:
        if "TotalCharges" in frame.columns:
            frame["TotalCharges"] = _safe_numeric(frame["TotalCharges"])
        if "MonthlyCharges" in frame.columns:
            frame["MonthlyCharges"] = _safe_numeric(frame["MonthlyCharges"])
        if "tenure" in frame.columns:
            frame["tenure"] = _safe_numeric(frame["tenure"])

    has_charge_triplet = all(
        column in train_out.columns and column in test_out.columns
        for column in ("TotalCharges", "MonthlyCharges", "tenure")
    )
    if has_charge_triplet:
        for frame in frames:
            tenure_non_zero = frame["tenure"].replace(0, np.nan)
            frame["charge_per_tenure"] = frame["TotalCharges"] / tenure_non_zero
            frame["monthly_share_of_total"] = frame["MonthlyCharges"] / (frame["TotalCharges"] + 1.0)
            frame["charge_delta"] = frame["TotalCharges"] - (frame["MonthlyCharges"] * frame["tenure"])
            frame["is_new_customer"] = (frame["tenure"] <= 1).astype("int8")

    service_columns = [
        "PhoneService",
        "MultipleLines",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]
    if all(column in train_out.columns and column in test_out.columns for column in service_columns):
        inactive_values = {"no", "no internet service", "no phone service", "__nan__"}
        for frame in frames:
            active = pd.DataFrame(index=frame.index)
            for column in service_columns:
                normalized = frame[column].astype("string").fillna("__nan__").str.strip().str.lower()
                active[column] = (~normalized.isin(inactive_values)).astype("int8")
            frame["active_service_count"] = active.sum(axis=1).astype("int16")

    for frame in frames:
        if "Contract" in frame.columns:
            contract_norm = frame["Contract"].astype("string").str.strip().str.lower()
            frame["contract_is_month_to_month"] = contract_norm.eq("month-to-month").astype("int8")
            frame["contract_is_annual"] = contract_norm.isin({"one year", "two year"}).astype("int8")
        if "PaperlessBilling" in frame.columns:
            paperless_norm = frame["PaperlessBilling"].astype("string").str.strip().str.lower()
            frame["paperless_flag"] = paperless_norm.eq("yes").astype("int8")

    drop_candidates = [column for column in (id_column, target_column) if column in train_out.columns]
    keep = [column for column in train_out.columns if column not in drop_candidates]
    train_out = train_out[[id_column, *keep, target_column]]
    return train_out, test_out


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
    train, test = _feature_engineering(train=train, test=test, id_column=id_column, target_column=target)
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
            max_iter=450,
            min_samples_leaf=50,
            random_state=seed,
        )
    else:
        model = HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_depth=8,
            max_iter=450,
            min_samples_leaf=50,
            random_state=seed,
        )
    return Pipeline(steps=[("preprocess", preprocessor), ("model", model)])


def _extract_classes(model: Any) -> np.ndarray | None:
    if hasattr(model, "classes_"):
        return np.asarray(model.classes_)
    inner = getattr(model, "named_steps", {}).get("model")
    if inner is not None and hasattr(inner, "classes_"):
        return np.asarray(inner.classes_)
    return None


def _positive_class_index(classes: np.ndarray | None, positive_label: Any | None, proba: np.ndarray) -> int | None:
    if proba.ndim != 2 or proba.shape[1] < 2:
        return None
    if classes is not None:
        if positive_label is not None:
            for idx, value in enumerate(classes.tolist()):
                if value == positive_label or str(value) == str(positive_label):
                    return int(idx)
        if len(classes) == 2:
            return 1
        return None
    return 1 if proba.shape[1] == 2 else None


def _evaluate_classification(
    y_true: pd.Series,
    pred_labels: np.ndarray,
    pred_proba: np.ndarray | None,
    classes: np.ndarray | None,
    positive_label: Any | None,
) -> tuple[float, float, float]:
    accuracy = float(accuracy_score(y_true, pred_labels))
    loss = float("nan")
    auc = float("nan")

    if pred_proba is None:
        return accuracy, loss, auc

    try:
        if classes is None:
            loss = float(log_loss(y_true, pred_proba))
        else:
            loss = float(log_loss(y_true, pred_proba, labels=list(classes)))
    except ValueError:
        loss = float("nan")

    pos_idx = _positive_class_index(classes=classes, positive_label=positive_label, proba=pred_proba)
    if pos_idx is None:
        return accuracy, loss, auc
    y_binary = pd.Series(y_true).map(lambda value: 1 if value == positive_label else 0).to_numpy(dtype=np.int8)
    if np.unique(y_binary).size < 2:
        return accuracy, loss, auc
    try:
        auc = float(roc_auc_score(y_binary, pred_proba[:, pos_idx]))
    except ValueError:
        auc = float("nan")
    return accuracy, loss, auc


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
        if metrics.auc is not None and np.isfinite(metrics.auc):
            return float(metrics.auc)
        if metrics.log_loss is not None and np.isfinite(metrics.log_loss):
            return float(-metrics.log_loss)
        return float(metrics.accuracy or 0.0)
    return float(-(metrics.rmse or np.inf))


def _prepare_catboost_frame(frame: pd.DataFrame, categorical_cols: list[str]) -> pd.DataFrame:
    prepared = frame.copy()
    for column in categorical_cols:
        prepared[column] = prepared[column].astype("string").fillna("__nan__")
    return prepared


def _fit_catboost_holdout(
    *,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_holdout: pd.DataFrame,
    y_holdout: pd.Series,
    feature_columns: tuple[str, ...],
    seed: int,
) -> tuple[Any, np.ndarray, np.ndarray, np.ndarray]:
    try:
        from catboost import CatBoostClassifier
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("catboost not available") from exc

    numeric_cols, categorical_cols = _split_feature_types(x_train, feature_columns)
    x_train_cb = _prepare_catboost_frame(x_train.loc[:, feature_columns], categorical_cols)
    x_holdout_cb = _prepare_catboost_frame(x_holdout.loc[:, feature_columns], categorical_cols)

    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        iterations=1800,
        learning_rate=0.04,
        depth=8,
        l2_leaf_reg=6.0,
        bagging_temperature=0.2,
        random_strength=1.0,
        random_seed=seed,
        allow_writing_files=False,
        verbose=False,
        task_type="CPU",
        thread_count=-1,
    )
    _ = numeric_cols  # keep explicit split for readability/debugging
    model.fit(
        x_train_cb,
        y_train,
        cat_features=categorical_cols,
        eval_set=(x_holdout_cb, y_holdout),
        use_best_model=True,
        early_stopping_rounds=180,
        verbose=False,
    )
    pred_proba = np.asarray(model.predict_proba(x_holdout_cb))
    pred_labels = np.asarray(model.predict(x_holdout_cb)).reshape(-1)
    classes = np.asarray(model.classes_)
    return model, pred_labels, pred_proba, classes


def _catboost_available() -> bool:
    try:
        import catboost  # noqa: F401
    except Exception:
        return False
    return True


def _prediction_for_submission(
    *,
    labels: np.ndarray,
    proba: np.ndarray | None,
    classes: np.ndarray | None,
    positive_label: Any | None,
    probability_output: bool,
) -> np.ndarray:
    if not probability_output or proba is None:
        return labels
    pos_idx = _positive_class_index(classes=classes, positive_label=positive_label, proba=proba)
    if pos_idx is None:
        return labels
    return proba[:, pos_idx]


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
    positive_label = _infer_positive_label(y_all) if dataset.task_type == "classification" else None

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
    selection_extras: dict[str, Any] = {}

    for name, model in candidates.items():
        model.fit(x_train, y_train)
        pred_labels = np.asarray(model.predict(x_holdout))
        pred_proba: np.ndarray | None = None
        classes: np.ndarray | None = None
        if dataset.task_type == "classification" and hasattr(model, "predict_proba"):
            pred_proba = np.asarray(model.predict_proba(x_holdout))
            classes = _extract_classes(model)

        if dataset.task_type == "classification":
            acc, ll, auc = _evaluate_classification(
                y_true=y_holdout,
                pred_labels=pred_labels,
                pred_proba=pred_proba,
                classes=classes,
                positive_label=positive_label,
            )
            row = CandidateMetrics(
                name=name,
                task_type=dataset.task_type,
                accuracy=acc,
                auc=auc,
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
        prediction_cache[name] = (pred_labels, pred_proba, classes)

    if dataset.task_type == "classification" and _catboost_available():
        try:
            cat_model, pred_labels, pred_proba, classes = _fit_catboost_holdout(
                x_train=x_train,
                y_train=y_train,
                x_holdout=x_holdout,
                y_holdout=y_holdout,
                feature_columns=dataset.feature_columns,
                seed=seed,
            )
            acc, ll, auc = _evaluate_classification(
                y_true=y_holdout,
                pred_labels=pred_labels,
                pred_proba=pred_proba,
                classes=classes,
                positive_label=positive_label,
            )
            row = CandidateMetrics(
                name="catboost",
                task_type=dataset.task_type,
                accuracy=acc,
                auc=auc,
                log_loss=ll,
                score=0.0,
            )
            row = CandidateMetrics(**{**asdict(row), "score": _candidate_score(row)})
            candidate_rows.append(row)
            prediction_cache["catboost"] = (pred_labels, pred_proba, classes)
            selection_extras["catboost_best_iteration"] = int(max(0, cat_model.get_best_iteration()))
        except RuntimeError:
            pass

    best = max(candidate_rows, key=lambda row: row.score)
    selected_labels, selected_proba, selected_classes = prediction_cache[best.name]

    selection = {
        "selected_strategy": best.name,
        "task_type": dataset.task_type,
        "id_column": dataset.id_column,
        "target_column": dataset.target_column,
        "feature_columns": list(dataset.feature_columns),
        "probability_output": dataset.probability_output,
        "positive_label": positive_label,
        "seed": int(seed),
        **selection_extras,
    }

    if dataset.task_type == "classification":
        holdout_metrics = {
            "rows": int(len(y_holdout)),
            "task_type": dataset.task_type,
            "selected_strategy": best.name,
            "positive_label": positive_label,
            "accuracy": float(best.accuracy or 0.0),
            "auc": float(best.auc) if best.auc is not None else float("nan"),
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
    if dataset.task_type == "classification":
        output_prediction = _prediction_for_submission(
            labels=selected_labels,
            proba=selected_proba,
            classes=selected_classes,
            positive_label=positive_label,
            probability_output=dataset.probability_output,
        )
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
        if selected_proba is not None:
            pos_idx = _positive_class_index(
                classes=selected_classes,
                positive_label=positive_label,
                proba=selected_proba,
            )
            if pos_idx is not None:
                holdout_predictions["prediction_proba"] = selected_proba[:, pos_idx].tolist()
        if selected_classes is not None:
            holdout_predictions["classes"] = [json.dumps([str(v) for v in selected_classes])] * len(
                holdout_predictions
            )
    return selection, holdout_metrics, holdout_predictions


def fit_final_model(dataset: DatasetBundle, selection: dict[str, Any], seed: int = 42) -> dict[str, Any]:
    strategy = str(selection["selected_strategy"])
    positive_label = selection.get("positive_label")
    x_train = dataset.train_frame.loc[:, dataset.feature_columns]
    y_train = dataset.train_frame.loc[:, dataset.target_column]

    if strategy == "catboost":
        if not _catboost_available():
            raise RuntimeError("catboost strategy selected but catboost is not installed.")
        from catboost import CatBoostClassifier

        _, categorical_cols = _split_feature_types(dataset.train_frame, dataset.feature_columns)
        x_train_cb = _prepare_catboost_frame(x_train, categorical_cols)
        iterations = 1800
        best_iteration = selection.get("catboost_best_iteration")
        if isinstance(best_iteration, int) and best_iteration > 0:
            iterations = max(600, best_iteration + 120)

        model = CatBoostClassifier(
            loss_function="Logloss",
            eval_metric="AUC",
            iterations=iterations,
            learning_rate=0.04,
            depth=8,
            l2_leaf_reg=6.0,
            bagging_temperature=0.2,
            random_strength=1.0,
            random_seed=seed,
            allow_writing_files=False,
            verbose=False,
            task_type="CPU",
            thread_count=-1,
        )
        model.fit(x_train_cb, y_train, cat_features=categorical_cols, verbose=False)
        return {
            "competition": COMPETITION_SLUG,
            "selected_strategy": "catboost",
            "task_type": dataset.task_type,
            "id_column": dataset.id_column,
            "target_column": dataset.target_column,
            "feature_columns": list(dataset.feature_columns),
            "catboost_categorical_columns": categorical_cols,
            "probability_output": dataset.probability_output,
            "positive_label": positive_label,
            "model": model,
            "seed": int(seed),
        }

    if strategy == "tree":
        model = _build_tree_pipeline(dataset.train_frame, dataset.feature_columns, dataset.task_type, seed=seed)
    else:
        model = _build_linear_pipeline(dataset.train_frame, dataset.feature_columns, dataset.task_type, seed=seed)
        strategy = "linear"
    model.fit(x_train, y_train)

    return {
        "competition": COMPETITION_SLUG,
        "selected_strategy": strategy,
        "task_type": dataset.task_type,
        "id_column": dataset.id_column,
        "target_column": dataset.target_column,
        "feature_columns": list(dataset.feature_columns),
        "probability_output": dataset.probability_output,
        "positive_label": positive_label,
        "model": model,
        "seed": int(seed),
    }


def generate_submission(model_bundle: dict[str, Any], dataset: DatasetBundle) -> pd.DataFrame:
    strategy = str(model_bundle["selected_strategy"])
    model = model_bundle["model"]
    positive_label = model_bundle.get("positive_label")
    x_test = dataset.test_frame.loc[:, dataset.feature_columns]

    if dataset.task_type == "classification":
        if strategy == "catboost":
            categorical_cols = list(model_bundle.get("catboost_categorical_columns", []))
            x_test_cb = _prepare_catboost_frame(x_test, categorical_cols)
            labels = np.asarray(model.predict(x_test_cb)).reshape(-1)
            proba = np.asarray(model.predict_proba(x_test_cb))
            classes = np.asarray(model.classes_)
        else:
            labels = np.asarray(model.predict(x_test))
            proba = np.asarray(model.predict_proba(x_test)) if hasattr(model, "predict_proba") else None
            classes = _extract_classes(model)

        prediction = _prediction_for_submission(
            labels=labels,
            proba=proba,
            classes=classes,
            positive_label=positive_label,
            probability_output=dataset.probability_output,
        )
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
