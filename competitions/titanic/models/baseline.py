"""Tabular baselines for Titanic."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any
import warnings

import joblib
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import StratifiedKFold
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


def _xgboost_available() -> bool:
    try:
        import xgboost  # noqa: F401
    except Exception:
        return False
    return True


def _lightgbm_available() -> bool:
    try:
        import lightgbm  # noqa: F401
    except Exception:
        return False
    return True


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


def _build_xgboost_pipeline(frame: pd.DataFrame, feature_columns: tuple[str, ...], seed: int) -> Pipeline:
    from xgboost import XGBClassifier

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
        raise ValueError("No features available for xgboost model.")

    preprocessor = ColumnTransformer(transformers=transformers, sparse_threshold=1.0)
    model = XGBClassifier(
        n_estimators=1200,
        learning_rate=0.02,
        max_depth=4,
        min_child_weight=2,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        reg_lambda=1.2,
        random_state=seed,
        n_jobs=-1,
    )
    return Pipeline(steps=[("preprocess", preprocessor), ("model", model)])


def _build_lightgbm_pipeline(frame: pd.DataFrame, feature_columns: tuple[str, ...], seed: int) -> Pipeline:
    from lightgbm import LGBMClassifier

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
        raise ValueError("No features available for lightgbm model.")

    preprocessor = ColumnTransformer(transformers=transformers, sparse_threshold=0.0)
    model = LGBMClassifier(
        n_estimators=1400,
        learning_rate=0.02,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=0.6,
        verbosity=-1,
        random_state=seed,
        n_jobs=-1,
    )
    return Pipeline(steps=[("preprocess", preprocessor), ("model", model)])


def _build_strategy_model(
    *,
    strategy: str,
    frame: pd.DataFrame,
    feature_columns: tuple[str, ...],
    seed: int,
) -> Any:
    if strategy == "linear":
        return _build_linear_pipeline(frame, feature_columns, seed=seed)
    if strategy == "forest":
        return _build_forest_pipeline(frame, feature_columns, seed=seed)
    if strategy == "hist":
        return _build_hist_pipeline(frame, feature_columns, seed=seed)
    if strategy == "xgboost":
        if not _xgboost_available():
            raise RuntimeError("xgboost is not installed.")
        return _build_xgboost_pipeline(frame, feature_columns, seed=seed)
    if strategy == "lightgbm":
        if not _lightgbm_available():
            raise RuntimeError("lightgbm is not installed.")
        return _build_lightgbm_pipeline(frame, feature_columns, seed=seed)
    raise ValueError(f"Unknown strategy: {strategy}")


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


def _prepare_overlap_keys(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "Name" in out.columns:
        out["__surname"] = (
            out["Name"]
            .astype(str)
            .str.split(",", n=1)
            .str[0]
            .str.strip()
            .str.lower()
        )
    else:
        out["__surname"] = ""
    if "Ticket" in out.columns:
        out["__ticket_norm"] = (
            out["Ticket"]
            .astype(str)
            .str.replace(r"\s+", "", regex=True)
            .str.upper()
        )
    else:
        out["__ticket_norm"] = ""
    fare = pd.to_numeric(out.get("Fare", pd.Series(index=out.index, dtype=float)), errors="coerce").fillna(-1.0)
    out["__fare_round"] = fare.round(0)
    out["__family_key"] = (
        out["__surname"]
        + "_"
        + out.get("Pclass", pd.Series(index=out.index, dtype=object)).astype(str)
        + "_"
        + out["__fare_round"].astype(str)
    )
    return out


def _build_overlap_rules(train_frame: pd.DataFrame, target_column: str) -> dict[str, dict[str, int]]:
    frame = _prepare_overlap_keys(train_frame)
    y = pd.to_numeric(frame[target_column], errors="coerce").fillna(0).astype(int)
    frame = frame.assign(__target=y.values)

    def summarize(key: str) -> tuple[dict[str, int], dict[str, int]]:
        grouped = frame.groupby(key)["__target"].agg(["mean", "count"])
        pure = grouped[(grouped["count"] >= 2) & ((grouped["mean"] == 0.0) | (grouped["mean"] == 1.0))]
        strong = grouped[(grouped["count"] >= 3) & ((grouped["mean"] <= 0.2) | (grouped["mean"] >= 0.8))]
        pure_map = pure["mean"].round().astype(int).to_dict()
        strong_map = strong["mean"].round().astype(int).to_dict()
        return pure_map, strong_map

    pure_ticket, strong_ticket = summarize("__ticket_norm")
    pure_family, strong_family = summarize("__family_key")
    return {
        "pure_ticket": pure_ticket,
        "pure_family": pure_family,
        "strong_ticket": strong_ticket,
        "strong_family": strong_family,
    }


def _predict_with_overlap_rules(test_frame: pd.DataFrame, rules: dict[str, dict[str, int]]) -> np.ndarray:
    frame = _prepare_overlap_keys(test_frame)
    out = np.full(len(frame), -1, dtype=np.int64)

    def fill(mask_values: pd.Series) -> None:
        nonlocal out
        values = pd.to_numeric(mask_values, errors="coerce")
        mask = values.notna().to_numpy() & (out < 0)
        if mask.any():
            out[mask] = values.to_numpy(dtype=np.float64)[mask].astype(np.int64)

    fill(frame["__ticket_norm"].map(rules.get("pure_ticket", {})))
    fill(frame["__family_key"].map(rules.get("pure_family", {})))
    fill(frame["__ticket_norm"].map(rules.get("strong_ticket", {})))
    fill(frame["__family_key"].map(rules.get("strong_family", {})))
    return out


def _apply_overlap_overrides(
    *,
    train_frame: pd.DataFrame,
    valid_frame: pd.DataFrame,
    target_column: str,
    labels: np.ndarray,
    proba: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray | None, int]:
    rules = _build_overlap_rules(train_frame=train_frame, target_column=target_column)
    forced = _predict_with_overlap_rules(valid_frame, rules)
    mask = forced >= 0
    if not mask.any():
        return labels, proba, 0

    updated_labels = np.asarray(labels).copy()
    updated_labels[mask] = forced[mask]
    updated_proba = None if proba is None else np.asarray(proba).copy()
    if updated_proba is not None and updated_proba.ndim == 2 and updated_proba.shape[1] == 2:
        forced_float = forced[mask].astype(np.float64)
        updated_proba[mask, 1] = forced_float
        updated_proba[mask, 0] = 1.0 - forced_float
    return updated_labels, updated_proba, int(mask.sum())


def _find_best_pairwise_blend(
    *,
    y_true: pd.Series,
    metrics_rows: list[CandidateMetrics],
    prediction_cache: dict[str, tuple[np.ndarray, np.ndarray | None]],
) -> tuple[str, np.ndarray, np.ndarray, dict[str, Any]] | None:
    ranked = sorted(
        metrics_rows,
        key=lambda row: (
            float(row.accuracy),
            -float(row.log_loss) if row.log_loss is not None and np.isfinite(row.log_loss) else -1e9,
        ),
        reverse=True,
    )
    probabilistic = [
        row.name
        for row in ranked
        if row.name in prediction_cache
        and prediction_cache[row.name][1] is not None
        and np.asarray(prediction_cache[row.name][1]).ndim == 2
        and np.asarray(prediction_cache[row.name][1]).shape[1] == 2
    ]
    if len(probabilistic) < 2:
        return None

    first, second = probabilistic[0], probabilistic[1]
    first_proba = np.asarray(prediction_cache[first][1], dtype=np.float64)[:, 1]
    second_proba = np.asarray(prediction_cache[second][1], dtype=np.float64)[:, 1]

    best: tuple[float, float, float, np.ndarray] | None = None
    for weight in np.arange(0.05, 1.0, 0.05):
        mixed = weight * first_proba + (1.0 - weight) * second_proba
        labels = (mixed >= 0.5).astype(int)
        proba = np.column_stack([1.0 - mixed, mixed])
        acc, ll = _metrics_for_classification(y_true, labels, proba)
        loss_score = -1e9 if not np.isfinite(ll) else -ll
        score = (acc, loss_score)
        if best is None or score > (best[0], best[1]):
            best = (float(acc), float(loss_score), float(weight), mixed)

    if best is None:
        return None

    _, _, best_weight, mixed = best
    labels = (mixed >= 0.5).astype(int)
    proba = np.column_stack([1.0 - mixed, mixed])
    blend_name = f"blend:{first}+{second}"
    extras = {
        "blend_primary_strategy": first,
        "blend_secondary_strategy": second,
        "blend_primary_weight": float(best_weight),
    }
    return blend_name, labels, proba, extras


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

    class_counts = y_all.value_counts()
    if class_counts.empty:
        raise ValueError("No target classes found.")
    max_splits = int(class_counts.min())
    n_splits = max(2, min(5, max_splits))
    folds = list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(x_all, y_all))

    candidate_names = ["linear", "forest", "hist"]
    if _xgboost_available():
        candidate_names.append("xgboost")
    if _lightgbm_available():
        candidate_names.append("lightgbm")

    metrics_rows: list[CandidateMetrics] = []
    prediction_cache: dict[str, tuple[np.ndarray, np.ndarray | None]] = {}
    selection_extras: dict[str, Any] = {}

    for name in candidate_names:
        try:
            oof_labels = np.zeros(len(work), dtype=np.int64)
            oof_proba = np.full((len(work), 2), np.nan, dtype=np.float64)
            oof_labels_overlap = np.zeros(len(work), dtype=np.int64)
            oof_proba_overlap = np.full((len(work), 2), np.nan, dtype=np.float64)
            has_binary_proba = True
            has_binary_proba_overlap = True
            overlap_total = 0
            for fold_id, (train_idx, valid_idx) in enumerate(folds):
                model = _build_strategy_model(
                    strategy=name,
                    frame=work,
                    feature_columns=dataset.feature_columns,
                    seed=seed + fold_id,
                )
                x_train = x_all.iloc[train_idx]
                y_train = y_all.iloc[train_idx]
                x_valid = x_all.iloc[valid_idx]

                if name == "lightgbm":
                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                            "ignore",
                            message="X does not have valid feature names, but LGBMClassifier was fitted with feature names",
                        )
                        model.fit(x_train, y_train)
                        labels = np.asarray(model.predict(x_valid)).reshape(-1).astype(np.int64)
                        proba = np.asarray(model.predict_proba(x_valid)) if hasattr(model, "predict_proba") else None
                else:
                    model.fit(x_train, y_train)
                    labels = np.asarray(model.predict(x_valid)).reshape(-1).astype(np.int64)
                    proba = np.asarray(model.predict_proba(x_valid)) if hasattr(model, "predict_proba") else None

                oof_labels[valid_idx] = labels
                if proba is None or proba.ndim != 2 or proba.shape[1] != 2:
                    has_binary_proba = False
                else:
                    oof_proba[valid_idx] = proba

                labels_overlap, proba_overlap, forced = _apply_overlap_overrides(
                    train_frame=work.iloc[train_idx],
                    valid_frame=work.iloc[valid_idx],
                    target_column=dataset.target_column,
                    labels=labels,
                    proba=proba,
                )
                overlap_total += int(forced)
                oof_labels_overlap[valid_idx] = labels_overlap
                if proba_overlap is None or proba_overlap.ndim != 2 or proba_overlap.shape[1] != 2:
                    has_binary_proba_overlap = False
                else:
                    oof_proba_overlap[valid_idx] = proba_overlap

            proba_eval = oof_proba if has_binary_proba and np.isfinite(oof_proba).all() else None
            acc, ll = _metrics_for_classification(y_all, oof_labels, proba_eval)
            metrics_rows.append(CandidateMetrics(name=name, accuracy=acc, log_loss=ll))
            prediction_cache[name] = (oof_labels, proba_eval)

            if overlap_total > 0:
                proba_overlap_eval = (
                    oof_proba_overlap
                    if has_binary_proba_overlap and np.isfinite(oof_proba_overlap).all()
                    else None
                )
                acc_overlap, ll_overlap = _metrics_for_classification(y_all, oof_labels_overlap, proba_overlap_eval)
                overlap_name = f"{name}+overlap"
                metrics_rows.append(CandidateMetrics(name=overlap_name, accuracy=acc_overlap, log_loss=ll_overlap))
                prediction_cache[overlap_name] = (oof_labels_overlap, proba_overlap_eval)
        except Exception:
            continue

    if _catboost_available():
        try:
            oof_labels = np.zeros(len(work), dtype=np.int64)
            oof_proba = np.full((len(work), 2), np.nan, dtype=np.float64)
            oof_labels_overlap = np.zeros(len(work), dtype=np.int64)
            oof_proba_overlap = np.full((len(work), 2), np.nan, dtype=np.float64)
            best_iterations: list[int] = []
            overlap_total = 0
            for fold_id, (train_idx, valid_idx) in enumerate(folds):
                x_train = x_all.iloc[train_idx]
                y_train = y_all.iloc[train_idx]
                x_valid = x_all.iloc[valid_idx]
                y_valid = y_all.iloc[valid_idx]
                cat_model, labels, proba = _fit_catboost_holdout(
                    x_train=x_train,
                    y_train=y_train,
                    x_holdout=x_valid,
                    y_holdout=y_valid,
                    feature_columns=dataset.feature_columns,
                    seed=seed + fold_id,
                )
                oof_labels[valid_idx] = labels.reshape(-1).astype(np.int64)
                oof_proba[valid_idx] = proba

                labels_overlap, proba_overlap, forced = _apply_overlap_overrides(
                    train_frame=work.iloc[train_idx],
                    valid_frame=work.iloc[valid_idx],
                    target_column=dataset.target_column,
                    labels=labels.reshape(-1).astype(np.int64),
                    proba=proba,
                )
                overlap_total += int(forced)
                oof_labels_overlap[valid_idx] = labels_overlap
                if proba_overlap is not None and proba_overlap.ndim == 2 and proba_overlap.shape[1] == 2:
                    oof_proba_overlap[valid_idx] = proba_overlap
                try:
                    best_iteration = int(cat_model.get_best_iteration())
                    if best_iteration > 0:
                        best_iterations.append(best_iteration)
                except Exception:
                    pass

            acc, ll = _metrics_for_classification(y_all, oof_labels, oof_proba)
            metrics_rows.append(CandidateMetrics(name="catboost", accuracy=acc, log_loss=ll))
            prediction_cache["catboost"] = (oof_labels, oof_proba)
            if overlap_total > 0 and np.isfinite(oof_proba_overlap).all():
                acc_overlap, ll_overlap = _metrics_for_classification(y_all, oof_labels_overlap, oof_proba_overlap)
                metrics_rows.append(CandidateMetrics(name="catboost+overlap", accuracy=acc_overlap, log_loss=ll_overlap))
                prediction_cache["catboost+overlap"] = (oof_labels_overlap, oof_proba_overlap)
            if best_iterations:
                selection_extras["catboost_best_iteration"] = int(np.median(np.asarray(best_iterations)))
        except Exception:
            pass

    if not metrics_rows:
        raise RuntimeError("No candidate models were evaluated successfully.")

    def rank_key(row: CandidateMetrics) -> tuple[float, float]:
        loss_score = -1e9 if row.log_loss is None or not np.isfinite(row.log_loss) else -float(row.log_loss)
        return float(row.accuracy), loss_score

    best = max(metrics_rows, key=rank_key)
    labels, proba = prediction_cache[best.name]

    holdout_metrics = {
        "rows": int(len(y_all)),
        "accuracy": float(best.accuracy),
        "log_loss": float(best.log_loss) if best.log_loss is not None else float("nan"),
        "selected_strategy": best.name,
        "selection_method": "stratified_kfold_oof",
        "cv_folds": int(n_splits),
        "strategy_metrics": [asdict(row) for row in metrics_rows],
    }
    selection = {
        "selected_strategy": best.name,
        "id_column": dataset.id_column,
        "target_column": dataset.target_column,
        "feature_columns": list(dataset.feature_columns),
        "seed": int(seed),
        "selection_method": "stratified_kfold_oof",
        "cv_folds": int(n_splits),
        **selection_extras,
    }

    holdout_predictions = pd.DataFrame(
        {
            dataset.id_column: work[dataset.id_column].tolist(),
            dataset.target_column: y_all.tolist(),
            "prediction": labels.tolist(),
        }
    )
    if proba is not None and proba.ndim == 2 and proba.shape[1] == 2:
        holdout_predictions["prediction_proba"] = proba[:, 1].tolist()

    return selection, holdout_metrics, holdout_predictions


def _fit_strategy_model(
    *,
    dataset: DatasetBundle,
    strategy: str,
    selection: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
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
            "selected_strategy": "catboost",
            "feature_columns": list(dataset.feature_columns),
            "catboost_categorical_columns": categorical_cols,
            "model": model,
        }

    if strategy == "forest":
        model = _build_forest_pipeline(dataset.train_frame, dataset.feature_columns, seed=seed)
    elif strategy == "hist":
        model = _build_hist_pipeline(dataset.train_frame, dataset.feature_columns, seed=seed)
    elif strategy == "xgboost":
        if not _xgboost_available():
            raise RuntimeError("xgboost strategy selected but xgboost is not installed.")
        model = _build_xgboost_pipeline(dataset.train_frame, dataset.feature_columns, seed=seed)
    elif strategy == "lightgbm":
        if not _lightgbm_available():
            raise RuntimeError("lightgbm strategy selected but lightgbm is not installed.")
        model = _build_lightgbm_pipeline(dataset.train_frame, dataset.feature_columns, seed=seed)
    else:
        model = _build_linear_pipeline(dataset.train_frame, dataset.feature_columns, seed=seed)
        strategy = "linear"
    model.fit(x_train, y_train)
    return {
        "selected_strategy": strategy,
        "feature_columns": list(dataset.feature_columns),
        "model": model,
    }


def fit_final_model(dataset: DatasetBundle, selection: dict[str, Any], seed: int = 42) -> dict[str, Any]:
    strategy = str(selection["selected_strategy"])
    use_overlap_postprocess = strategy.endswith("+overlap") and not strategy.startswith("blend:")
    base_strategy = strategy[:-8] if use_overlap_postprocess else strategy

    if base_strategy.startswith("blend:"):
        primary = str(selection.get("blend_primary_strategy", "catboost"))
        secondary = str(selection.get("blend_secondary_strategy", "linear"))
        weight = float(selection.get("blend_primary_weight", 0.5))
        primary_bundle = _fit_strategy_model(dataset=dataset, strategy=primary, selection=selection, seed=seed)
        secondary_bundle = _fit_strategy_model(dataset=dataset, strategy=secondary, selection=selection, seed=seed)
        return {
            "competition": COMPETITION_SLUG,
            "selected_strategy": strategy,
            "id_column": dataset.id_column,
            "target_column": dataset.target_column,
            "feature_columns": list(dataset.feature_columns),
            "blend_primary_strategy": primary,
            "blend_secondary_strategy": secondary,
            "blend_primary_weight": weight,
            "primary_model_bundle": primary_bundle,
            "secondary_model_bundle": secondary_bundle,
            "seed": int(seed),
        }

    trained = _fit_strategy_model(dataset=dataset, strategy=base_strategy, selection=selection, seed=seed)
    overlap_rules: dict[str, dict[str, int]] | None = None
    if use_overlap_postprocess:
        overlap_rules = _build_overlap_rules(
            train_frame=dataset.train_frame,
            target_column=dataset.target_column,
        )
    decision_threshold = float(selection.get("decision_threshold", 0.5))
    return {
        "competition": COMPETITION_SLUG,
        "selected_strategy": strategy,
        "id_column": dataset.id_column,
        "target_column": dataset.target_column,
        "feature_columns": list(dataset.feature_columns),
        "model": trained["model"],
        "seed": int(seed),
        "decision_threshold": decision_threshold,
        **({"overlap_rules": overlap_rules} if overlap_rules is not None else {}),
        **({"catboost_categorical_columns": trained["catboost_categorical_columns"]} if "catboost_categorical_columns" in trained else {}),
    }


def _predict_probabilities(
    model_bundle: dict[str, Any],
    dataset: DatasetBundle,
) -> np.ndarray:
    strategy = str(model_bundle["selected_strategy"])
    model = model_bundle["model"]
    x_test = dataset.test_frame.loc[:, dataset.feature_columns]

    if strategy == "catboost":
        categorical_cols = list(model_bundle.get("catboost_categorical_columns", []))
        x_test = _prepare_catboost_frame(x_test, categorical_cols)
        proba = np.asarray(model.predict_proba(x_test))
    elif hasattr(model, "predict_proba"):
        if strategy == "lightgbm":
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="X does not have valid feature names, but LGBMClassifier was fitted with feature names",
                )
                proba = np.asarray(model.predict_proba(x_test))
        else:
            proba = np.asarray(model.predict_proba(x_test))
    else:
        labels = np.asarray(model.predict(x_test)).astype(int).reshape(-1)
        proba = np.column_stack([1 - labels, labels])

    if proba.ndim != 2 or proba.shape[1] != 2:
        raise ValueError("Expected binary-class probabilities with 2 columns.")
    return proba


def generate_submission(model_bundle: dict[str, Any], dataset: DatasetBundle) -> pd.DataFrame:
    strategy = str(model_bundle["selected_strategy"])
    threshold = float(model_bundle.get("decision_threshold", 0.5))
    if strategy.startswith("blend:"):
        primary_bundle = dict(model_bundle["primary_model_bundle"])
        secondary_bundle = dict(model_bundle["secondary_model_bundle"])
        weight = float(model_bundle.get("blend_primary_weight", 0.5))
        primary_proba = _predict_probabilities(primary_bundle, dataset)[:, 1]
        secondary_proba = _predict_probabilities(secondary_bundle, dataset)[:, 1]
        combined = weight * primary_proba + (1.0 - weight) * secondary_proba
        predictions = (combined >= threshold).astype(int)
    else:
        proba = _predict_probabilities(model_bundle, dataset)
        predictions = (proba[:, 1] >= threshold).astype(int)

    overlap_rules = model_bundle.get("overlap_rules")
    if isinstance(overlap_rules, dict):
        forced = _predict_with_overlap_rules(dataset.test_frame, overlap_rules)
        mask = forced >= 0
        if mask.any():
            predictions = np.asarray(predictions).copy()
            predictions[mask] = forced[mask]

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
