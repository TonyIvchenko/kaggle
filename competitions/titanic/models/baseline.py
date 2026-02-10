"""Tabular baselines for Titanic."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
from typing import Any

import joblib
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


COMPETITION_SLUG = "titanic"
COMPETITION_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = COMPETITION_ROOT / "data" / "raw"
DEFAULT_PROCESSED_DIR = COMPETITION_ROOT / "data" / "processed"
DEFAULT_MODEL_PATH = COMPETITION_ROOT / "models" / "titanic.joblib"
DEFAULT_METRICS_PATH = COMPETITION_ROOT / "models" / "titanic_metrics.json"
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


@dataclass(frozen=True)
class CandidateMetrics:
    name: str
    accuracy: float
    log_loss: float | None


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
        sample_submission=_pick_file(csv_files, ("gender_submission.csv", "sample_submission.csv")),
    )


def _choose_id_column(train: pd.DataFrame, test: pd.DataFrame, sample_submission: pd.DataFrame) -> str:
    for column in ("PassengerId", "id", "ID", "Id"):
        if column in test.columns and column in sample_submission.columns:
            return column
    shared = [column for column in sample_submission.columns if column in test.columns]
    if not shared:
        raise ValueError("Could not infer id column from schema.")
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
    raise ValueError("Could not infer target column from schema.")


def _normalize_title(raw: str) -> str:
    normalized = raw.strip().lower()
    mapping = {
        "mlle": "Miss",
        "ms": "Miss",
        "mme": "Mrs",
        "lady": "Rare",
        "countess": "Rare",
        "capt": "Rare",
        "col": "Rare",
        "don": "Rare",
        "dr": "Rare",
        "major": "Rare",
        "rev": "Rare",
        "sir": "Rare",
        "jonkheer": "Rare",
        "dona": "Rare",
    }
    if normalized in {"mr", "mrs", "miss", "master"}:
        return normalized.title()
    return mapping.get(normalized, "Rare")


def _feature_engineering(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_out = train.copy()
    test_out = test.copy()

    for frame in (train_out, test_out):
        if "Name" in frame.columns:
            titles = frame["Name"].astype(str).str.extract(r",\s*([^\.]+)\.", expand=False).fillna("Rare")
            frame["Title"] = titles.map(_normalize_title)
        if "Cabin" in frame.columns:
            frame["CabinDeck"] = frame["Cabin"].astype(str).str[0].replace({"n": "U"}).fillna("U")
            frame.loc[frame["Cabin"].isna(), "CabinDeck"] = "U"
        if "Ticket" in frame.columns:
            prefixes = (
                frame["Ticket"]
                .astype(str)
                .str.replace(r"[\.\/]", " ", regex=True)
                .str.replace(r"\s+", " ", regex=True)
                .str.strip()
                .str.extract(r"^([A-Za-z]+)", expand=False)
                .fillna("NONE")
                .str.upper()
            )
            frame["TicketPrefix"] = prefixes

        for column in ("Age", "Fare", "SibSp", "Parch", "Pclass"):
            if column in frame.columns:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")

        if "SibSp" in frame.columns and "Parch" in frame.columns:
            frame["FamilySize"] = frame["SibSp"].fillna(0) + frame["Parch"].fillna(0) + 1
            frame["IsAlone"] = (frame["FamilySize"] == 1).astype("int8")
        if "Fare" in frame.columns and "FamilySize" in frame.columns:
            frame["FarePerPerson"] = frame["Fare"] / frame["FamilySize"].replace(0, np.nan)

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

    train, test = _feature_engineering(train, test)
    preferred = [
        "Pclass",
        "Sex",
        "Age",
        "SibSp",
        "Parch",
        "Fare",
        "Embarked",
        "Title",
        "CabinDeck",
        "TicketPrefix",
        "FamilySize",
        "IsAlone",
        "FarePerPerson",
    ]
    feature_columns = [column for column in preferred if column in train.columns and column in test.columns]
    if not feature_columns:
        fallback = [column for column in test.columns if column not in {id_column, target} and column in train.columns]
        feature_columns = [str(column) for column in fallback]
    if not feature_columns:
        raise ValueError("Could not infer feature columns.")

    return DatasetBundle(
        train_frame=train.copy(),
        test_frame=test.copy(),
        sample_submission=sample_submission.copy(),
        id_column=id_column,
        target_column=target,
        feature_columns=tuple(feature_columns),
    )


def _split_feature_types(frame: pd.DataFrame, feature_columns: tuple[str, ...]) -> tuple[list[str], list[str]]:
    numeric_cols = [column for column in feature_columns if is_numeric_dtype(frame[column])]
    categorical_cols = [column for column in feature_columns if column not in numeric_cols]
    return numeric_cols, categorical_cols


def _build_linear_pipeline(frame: pd.DataFrame, feature_columns: tuple[str, ...], seed: int) -> Pipeline:
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
                        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=2)),
                    ]
                ),
                categorical_cols,
            )
        )
    if not transformers:
        raise ValueError("No features available for linear model.")

    preprocessor = ColumnTransformer(transformers=transformers, sparse_threshold=1.0)
    model = LogisticRegression(C=1.5, max_iter=4000, random_state=seed)
    return Pipeline(steps=[("preprocess", preprocessor), ("model", model)])


def _build_forest_pipeline(frame: pd.DataFrame, feature_columns: tuple[str, ...], seed: int) -> Pipeline:
    numeric_cols, categorical_cols = _split_feature_types(frame, feature_columns)
    transformers: list[tuple[str, Any, list[str]]] = []

    if numeric_cols:
        transformers.append(("num", Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]), numeric_cols))
    if categorical_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_cols,
            )
        )
    if not transformers:
        raise ValueError("No features available for forest model.")

    preprocessor = ColumnTransformer(transformers=transformers, sparse_threshold=1.0)
    model = RandomForestClassifier(
        n_estimators=900,
        min_samples_leaf=2,
        random_state=seed,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    return Pipeline(steps=[("preprocess", preprocessor), ("model", model)])


def _build_hist_pipeline(frame: pd.DataFrame, feature_columns: tuple[str, ...], seed: int) -> Pipeline:
    numeric_cols, categorical_cols = _split_feature_types(frame, feature_columns)
    transformers: list[tuple[str, Any, list[str]]] = []

    if numeric_cols:
        transformers.append(("num", Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]), numeric_cols))
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
        raise ValueError("No features available for hist model.")

    preprocessor = ColumnTransformer(transformers=transformers, sparse_threshold=0.0)
    model = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_depth=5,
        max_iter=500,
        min_samples_leaf=5,
        random_state=seed,
    )
    return Pipeline(steps=[("preprocess", preprocessor), ("model", model)])


def _prepare_catboost_frame(frame: pd.DataFrame, categorical_cols: list[str]) -> pd.DataFrame:
    out = frame.copy()
    for column in categorical_cols:
        out[column] = out[column].astype("string").fillna("__nan__")
    return out


def _catboost_available() -> bool:
    try:
        import catboost  # noqa: F401
    except Exception:
        return False
    return True


def _fit_catboost_holdout(
    *,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_holdout: pd.DataFrame,
    y_holdout: pd.Series,
    feature_columns: tuple[str, ...],
    seed: int,
) -> tuple[Any, np.ndarray, np.ndarray]:
    from catboost import CatBoostClassifier

    _, categorical_cols = _split_feature_types(x_train.assign(_tmp=0), feature_columns)
    x_train_cb = _prepare_catboost_frame(x_train.loc[:, feature_columns], categorical_cols)
    x_holdout_cb = _prepare_catboost_frame(x_holdout.loc[:, feature_columns], categorical_cols)
    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="Accuracy",
        iterations=1200,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=4.0,
        random_seed=seed,
        allow_writing_files=False,
        verbose=False,
        thread_count=-1,
    )
    model.fit(
        x_train_cb,
        y_train,
        cat_features=categorical_cols,
        eval_set=(x_holdout_cb, y_holdout),
        use_best_model=True,
        early_stopping_rounds=100,
        verbose=False,
    )
    labels = np.asarray(model.predict(x_holdout_cb)).reshape(-1)
    proba = np.asarray(model.predict_proba(x_holdout_cb))
    return model, labels, proba


def _metrics_for_classification(y_true: pd.Series, labels: np.ndarray, proba: np.ndarray | None) -> tuple[float, float]:
    acc = float(accuracy_score(y_true, labels))
    ll = float("nan")
    if proba is not None:
        try:
            ll = float(log_loss(y_true, proba))
        except ValueError:
            ll = float("nan")
    return acc, ll


def fit_and_score_holdout(
    dataset: DatasetBundle,
    holdout_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame]:
    if not 0.05 <= holdout_fraction <= 0.5:
        raise ValueError("holdout_fraction should be between 0.05 and 0.5")

    work = dataset.train_frame.copy()
    x_all = work.loc[:, dataset.feature_columns]
    y_all = pd.to_numeric(work[dataset.target_column], errors="coerce").fillna(0).astype(int)

    train_idx, holdout_idx = train_test_split(
        np.arange(len(work), dtype=np.int64),
        test_size=float(holdout_fraction),
        random_state=seed,
        shuffle=True,
        stratify=y_all,
    )
    x_train = x_all.iloc[train_idx]
    y_train = y_all.iloc[train_idx]
    x_holdout = x_all.iloc[holdout_idx]
    y_holdout = y_all.iloc[holdout_idx]

    candidates: dict[str, Any] = {
        "linear": _build_linear_pipeline(work, dataset.feature_columns, seed=seed),
        "forest": _build_forest_pipeline(work, dataset.feature_columns, seed=seed),
        "hist": _build_hist_pipeline(work, dataset.feature_columns, seed=seed),
    }

    metrics_rows: list[CandidateMetrics] = []
    prediction_cache: dict[str, tuple[np.ndarray, np.ndarray | None]] = {}
    selection_extras: dict[str, Any] = {}

    for name, model in candidates.items():
        model.fit(x_train, y_train)
        labels = np.asarray(model.predict(x_holdout))
        proba = np.asarray(model.predict_proba(x_holdout)) if hasattr(model, "predict_proba") else None
        acc, ll = _metrics_for_classification(y_holdout, labels, proba)
        metrics_rows.append(CandidateMetrics(name=name, accuracy=acc, log_loss=ll))
        prediction_cache[name] = (labels, proba)

    if _catboost_available():
        try:
            cat_model, labels, proba = _fit_catboost_holdout(
                x_train=x_train,
                y_train=y_train,
                x_holdout=x_holdout,
                y_holdout=y_holdout,
                feature_columns=dataset.feature_columns,
                seed=seed,
            )
            acc, ll = _metrics_for_classification(y_holdout, labels, proba)
            metrics_rows.append(CandidateMetrics(name="catboost", accuracy=acc, log_loss=ll))
            prediction_cache["catboost"] = (labels, proba)
            selection_extras["catboost_best_iteration"] = int(max(0, cat_model.get_best_iteration()))
        except Exception:
            pass

    def rank_key(row: CandidateMetrics) -> tuple[float, float]:
        loss_score = -1e9 if row.log_loss is None or not np.isfinite(row.log_loss) else -float(row.log_loss)
        return float(row.accuracy), loss_score

    best = max(metrics_rows, key=rank_key)
    labels, proba = prediction_cache[best.name]

    holdout_metrics = {
        "rows": int(len(y_holdout)),
        "accuracy": float(best.accuracy),
        "log_loss": float(best.log_loss) if best.log_loss is not None else float("nan"),
        "selected_strategy": best.name,
        "strategy_metrics": [asdict(row) for row in metrics_rows],
    }
    selection = {
        "selected_strategy": best.name,
        "id_column": dataset.id_column,
        "target_column": dataset.target_column,
        "feature_columns": list(dataset.feature_columns),
        "seed": int(seed),
        **selection_extras,
    }

    holdout_predictions = pd.DataFrame(
        {
            dataset.id_column: work.iloc[holdout_idx][dataset.id_column].tolist(),
            dataset.target_column: y_holdout.tolist(),
            "prediction": labels.tolist(),
        }
    )
    if proba is not None and proba.ndim == 2 and proba.shape[1] == 2:
        holdout_predictions["prediction_proba"] = proba[:, 1].tolist()

    return selection, holdout_metrics, holdout_predictions


def fit_final_model(dataset: DatasetBundle, selection: dict[str, Any], seed: int = 42) -> dict[str, Any]:
    strategy = str(selection["selected_strategy"])
    x_train = dataset.train_frame.loc[:, dataset.feature_columns]
    y_train = pd.to_numeric(dataset.train_frame[dataset.target_column], errors="coerce").fillna(0).astype(int)

    if strategy == "catboost":
        if not _catboost_available():
            raise RuntimeError("catboost strategy selected but catboost is not installed.")
        from catboost import CatBoostClassifier

        _, categorical_cols = _split_feature_types(dataset.train_frame, dataset.feature_columns)
        x_train_cb = _prepare_catboost_frame(x_train, categorical_cols)
        iterations = 1200
        best_iteration = selection.get("catboost_best_iteration")
        if isinstance(best_iteration, int) and best_iteration > 0:
            iterations = max(300, best_iteration + 80)
        model = CatBoostClassifier(
            loss_function="Logloss",
            eval_metric="Accuracy",
            iterations=iterations,
            learning_rate=0.03,
            depth=6,
            l2_leaf_reg=4.0,
            random_seed=seed,
            allow_writing_files=False,
            verbose=False,
            thread_count=-1,
        )
        model.fit(x_train_cb, y_train, cat_features=categorical_cols, verbose=False)
        return {
            "competition": COMPETITION_SLUG,
            "selected_strategy": "catboost",
            "id_column": dataset.id_column,
            "target_column": dataset.target_column,
            "feature_columns": list(dataset.feature_columns),
            "catboost_categorical_columns": categorical_cols,
            "model": model,
            "seed": int(seed),
        }

    if strategy == "forest":
        model = _build_forest_pipeline(dataset.train_frame, dataset.feature_columns, seed=seed)
    elif strategy == "hist":
        model = _build_hist_pipeline(dataset.train_frame, dataset.feature_columns, seed=seed)
    else:
        model = _build_linear_pipeline(dataset.train_frame, dataset.feature_columns, seed=seed)
        strategy = "linear"
    model.fit(x_train, y_train)
    return {
        "competition": COMPETITION_SLUG,
        "selected_strategy": strategy,
        "id_column": dataset.id_column,
        "target_column": dataset.target_column,
        "feature_columns": list(dataset.feature_columns),
        "model": model,
        "seed": int(seed),
    }


def generate_submission(model_bundle: dict[str, Any], dataset: DatasetBundle) -> pd.DataFrame:
    strategy = str(model_bundle["selected_strategy"])
    model = model_bundle["model"]
    x_test = dataset.test_frame.loc[:, dataset.feature_columns]

    if strategy == "catboost":
        categorical_cols = list(model_bundle.get("catboost_categorical_columns", []))
        x_test = _prepare_catboost_frame(x_test, categorical_cols)
        predictions = np.asarray(model.predict(x_test)).reshape(-1)
    else:
        predictions = np.asarray(model.predict(x_test)).reshape(-1)

    prediction_frame = pd.DataFrame(
        {
            dataset.id_column: dataset.test_frame[dataset.id_column].tolist(),
            dataset.target_column: pd.Series(predictions).astype(int).tolist(),
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


def save_metrics(payload: dict[str, Any], path: Path = DEFAULT_METRICS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

