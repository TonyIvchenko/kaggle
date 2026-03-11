"""Text baselines for LLM Classification Finetuning."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


COMPETITION_SLUG = "llm-classification-finetuning"
COMPETITION_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = COMPETITION_ROOT / "data" / "raw"
DEFAULT_PROCESSED_DIR = COMPETITION_ROOT / "data" / "processed"
DEFAULT_MODEL_PATH = COMPETITION_ROOT / "models" / "llm_classification_finetuning.joblib"
DEFAULT_METRICS_PATH = COMPETITION_ROOT / "models" / "llm_classification_finetuning_metrics.json"
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
    x_train: pd.Series
    x_test: pd.Series
    y_train: pd.Series


@dataclass(frozen=True)
class CandidateMetrics:
    name: str
    accuracy: float


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
    for column in ("id", "ID", "Id"):
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
    raise ValueError("Could not infer target column. Pass --target-column explicitly.")


def _choose_feature_columns(
    train: pd.DataFrame,
    test: pd.DataFrame,
    id_column: str,
    target_column: str,
) -> list[str]:
    candidates = [column for column in test.columns if column != id_column and column in train.columns]
    if target_column in candidates:
        candidates.remove(target_column)
    if not candidates:
        raise ValueError("Could not infer feature columns from test/train schema.")
    return [str(column) for column in candidates]


def _build_text_corpus(frame: pd.DataFrame, feature_columns: tuple[str, ...]) -> pd.Series:
    text_frame = frame.loc[:, feature_columns].copy()
    text_frame = text_frame.fillna("")
    for column in feature_columns:
        text_frame[column] = text_frame[column].astype(str).str.strip()
    return text_frame.agg(" [SEP] ".join, axis=1)


def build_dataset(files: CompetitionFiles, target_column: str | None = None) -> DatasetBundle:
    train_frame = pd.read_csv(files.train)
    test_frame = pd.read_csv(files.test)
    sample_submission = pd.read_csv(files.sample_submission)

    id_column = _choose_id_column(train_frame, test_frame, sample_submission)
    target = _choose_target_column(
        train=train_frame,
        test=test_frame,
        sample_submission=sample_submission,
        id_column=id_column,
        target_override=target_column,
    )
    feature_columns = tuple(
        _choose_feature_columns(
            train=train_frame,
            test=test_frame,
            id_column=id_column,
            target_column=target,
        )
    )
    x_train = _build_text_corpus(train_frame, feature_columns)
    x_test = _build_text_corpus(test_frame, feature_columns)
    y_train = train_frame[target].astype(str)

    return DatasetBundle(
        train_frame=train_frame,
        test_frame=test_frame,
        sample_submission=sample_submission,
        id_column=id_column,
        target_column=target,
        feature_columns=feature_columns,
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
    )


def _build_word_logreg(seed: int) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.995,
                    max_features=220_000,
                    sublinear_tf=True,
                ),
            ),
            (
                "model",
                LogisticRegression(
                    C=4.0,
                    max_iter=6000,
                    random_state=seed,
                ),
            ),
        ]
    )


def _build_word_svm(seed: int) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.995,
                    max_features=220_000,
                    sublinear_tf=True,
                ),
            ),
            ("model", LinearSVC(C=1.0, random_state=seed)),
        ]
    )


def _build_char_svm(seed: int) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=2,
                    lowercase=True,
                    strip_accents="unicode",
                    max_features=180_000,
                    sublinear_tf=True,
                ),
            ),
            ("model", LinearSVC(C=1.2, random_state=seed)),
        ]
    )


def build_candidate_models(seed: int = 42) -> dict[str, BaseEstimator]:
    return {
        "word_logreg": _build_word_logreg(seed=seed),
        "word_svm": _build_word_svm(seed=seed),
        "char_svm": _build_char_svm(seed=seed),
    }


def fit_and_score_holdout(
    dataset: DatasetBundle,
    holdout_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame]:
    if not 0.05 <= holdout_fraction <= 0.5:
        raise ValueError("holdout_fraction should be between 0.05 and 0.5")

    indices = dataset.train_frame.index.to_numpy()
    classes = dataset.y_train.astype(str).unique().tolist()
    class_count = len(classes)
    n_rows = len(indices)

    stratify_target: pd.Series | None = dataset.y_train
    holdout_size = max(int(round(n_rows * float(holdout_fraction))), 1)
    if class_count > 1 and n_rows >= class_count * 2:
        holdout_size = max(holdout_size, class_count)
        holdout_size = min(holdout_size, n_rows - class_count)
        if holdout_size < class_count:
            stratify_target = None
    else:
        stratify_target = None

    train_idx, holdout_idx = train_test_split(
        indices,
        test_size=int(holdout_size),
        random_state=seed,
        shuffle=True,
        stratify=stratify_target,
    )

    x_fit = dataset.x_train.loc[train_idx]
    y_fit = dataset.y_train.loc[train_idx]
    x_holdout = dataset.x_train.loc[holdout_idx]
    y_holdout = dataset.y_train.loc[holdout_idx]

    candidates = build_candidate_models(seed=seed)
    metrics_rows: list[CandidateMetrics] = []
    predictions: dict[str, np.ndarray] = {}

    for name, model in candidates.items():
        model.fit(x_fit, y_fit)
        pred = np.asarray(model.predict(x_holdout))
        acc = float(accuracy_score(y_holdout, pred))
        metrics_rows.append(CandidateMetrics(name=name, accuracy=acc))
        predictions[name] = pred

    best = max(metrics_rows, key=lambda row: row.accuracy)
    best_predictions = predictions[best.name]

    holdout_metrics = {
        "rows": int(len(holdout_idx)),
        "accuracy": float(best.accuracy),
        "selected_strategy": best.name,
        "classes": sorted(classes),
        "strategy_metrics": [asdict(row) for row in metrics_rows],
    }
    selection = {
        "selected_strategy": best.name,
        "id_column": dataset.id_column,
        "target_column": dataset.target_column,
        "feature_columns": list(dataset.feature_columns),
        "seed": int(seed),
    }
    holdout_predictions = pd.DataFrame(
        {
            dataset.id_column: dataset.train_frame.loc[holdout_idx, dataset.id_column].tolist(),
            dataset.target_column: y_holdout.tolist(),
            "prediction": best_predictions.tolist(),
        }
    )

    return selection, holdout_metrics, holdout_predictions


def fit_final_model(dataset: DatasetBundle, selection: dict[str, Any], seed: int = 42) -> dict[str, Any]:
    strategy = str(selection.get("selected_strategy", "word_logreg"))
    candidates = build_candidate_models(seed=seed)
    model = candidates.get(strategy, candidates["word_logreg"])
    if strategy not in candidates:
        strategy = "word_logreg"
    model.fit(dataset.x_train, dataset.y_train)
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
    predictions = pd.Series(model_bundle["model"].predict(dataset.x_test))
    prediction_frame = pd.DataFrame(
        {
            dataset.id_column: dataset.test_frame[dataset.id_column].tolist(),
            dataset.target_column: predictions.tolist(),
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
