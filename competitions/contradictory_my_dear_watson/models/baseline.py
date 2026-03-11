"""Text baseline for Contradictory, My Dear Watson."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


COMPETITION_SLUG = "contradictory-my-dear-watson"
COMPETITION_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = COMPETITION_ROOT / "data" / "raw"
DEFAULT_PROCESSED_DIR = COMPETITION_ROOT / "data" / "processed"
DEFAULT_MODEL_PATH = COMPETITION_ROOT / "models" / "contradictory_my_dear_watson.joblib"
DEFAULT_METRICS_PATH = COMPETITION_ROOT / "models" / "contradictory_my_dear_watson_metrics.json"
DEFAULT_SUBMISSION_PATH = COMPETITION_ROOT / "submissions" / "submission.csv"


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


def discover_competition_files(raw_dir: Path = DEFAULT_RAW_DIR) -> dict[str, Path]:
    csv_files = available_csv_files(raw_dir)
    if not csv_files:
        raise FileNotFoundError(f"No CSV files were found under {raw_dir}. Run download_data.py first.")
    return {
        "raw_dir": raw_dir,
        "train": _pick_file(csv_files, ("train.csv",)),
        "test": _pick_file(csv_files, ("test.csv",)),
        "sample_submission": _pick_file(csv_files, ("sample_submission.csv", "sampleSubmission.csv")),
    }


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


def _choose_text_feature_columns(train: pd.DataFrame, test: pd.DataFrame, id_column: str, target_column: str) -> list[str]:
    candidates = [column for column in test.columns if column != id_column and column in train.columns]
    if target_column in candidates:
        candidates.remove(target_column)
    if not candidates:
        raise ValueError("Could not infer text feature columns from test/train schema.")
    return [str(column) for column in candidates]


def _build_text_corpus(frame: pd.DataFrame, feature_columns: list[str]) -> pd.Series:
    text_frame = frame.loc[:, feature_columns].copy()
    text_frame = text_frame.fillna("")
    for column in feature_columns:
        text_frame[column] = text_frame[column].astype(str).str.strip()
    return text_frame.agg(" [SEP] ".join, axis=1)


def build_dataset(files: dict[str, Path], target_column: str | None = None) -> dict[str, Any]:
    train_frame = pd.read_csv(files["train"])
    test_frame = pd.read_csv(files["test"])
    sample_submission = pd.read_csv(files["sample_submission"])

    id_column = _choose_id_column(train_frame, test_frame, sample_submission)
    target = _choose_target_column(
        train=train_frame,
        test=test_frame,
        sample_submission=sample_submission,
        id_column=id_column,
        target_override=target_column,
    )
    feature_columns = _choose_text_feature_columns(
        train=train_frame,
        test=test_frame,
        id_column=id_column,
        target_column=target,
    )
    x_train = _build_text_corpus(train_frame, feature_columns)
    x_test = _build_text_corpus(test_frame, feature_columns)

    return {
        "train_frame": train_frame,
        "test_frame": test_frame,
        "sample_submission": sample_submission,
        "id_column": id_column,
        "target_column": target,
        "feature_columns": feature_columns,
        "x_train": x_train,
        "x_test": x_test,
        "y_train": train_frame[target].astype(str),
    }


def build_pipeline(seed: int = 42) -> Pipeline:
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
                    sublinear_tf=True,
                ),
            ),
            (
                "model",
                LogisticRegression(
                    C=4.0,
                    max_iter=5000,
                    random_state=seed,
                ),
            ),
        ]
    )


def fit_and_score_holdout(
    dataset: dict[str, Any],
    holdout_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[Pipeline, dict[str, Any], pd.DataFrame]:
    x_train = dataset["x_train"]
    y_train = dataset["y_train"]
    source = dataset["train_frame"]

    train_idx, holdout_idx = train_test_split(
        source.index.to_numpy(),
        test_size=float(holdout_fraction),
        random_state=seed,
        shuffle=True,
        stratify=y_train,
    )

    x_fit = x_train.loc[train_idx]
    y_fit = y_train.loc[train_idx]
    x_holdout = x_train.loc[holdout_idx]
    y_holdout = y_train.loc[holdout_idx]

    model = build_pipeline(seed=seed)
    model.fit(x_fit, y_fit)
    holdout_pred = pd.Series(model.predict(x_holdout), index=holdout_idx)
    acc = float(accuracy_score(y_holdout, holdout_pred))

    holdout_predictions = pd.DataFrame(
        {
            dataset["id_column"]: source.loc[holdout_idx, dataset["id_column"]].tolist(),
            dataset["target_column"]: y_holdout.tolist(),
            "prediction": holdout_pred.tolist(),
        }
    )
    metrics = {
        "rows": int(len(holdout_idx)),
        "accuracy": acc,
        "classes": sorted(y_train.unique().tolist()),
    }
    return model, metrics, holdout_predictions


def fit_final_model(dataset: dict[str, Any], seed: int = 42) -> dict[str, Any]:
    model = build_pipeline(seed=seed)
    model.fit(dataset["x_train"], dataset["y_train"])
    return {
        "competition": COMPETITION_SLUG,
        "id_column": dataset["id_column"],
        "target_column": dataset["target_column"],
        "feature_columns": list(dataset["feature_columns"]),
        "model": model,
        "seed": int(seed),
    }


def generate_submission(model_bundle: dict[str, Any], dataset: dict[str, Any]) -> pd.DataFrame:
    predictions = pd.Series(model_bundle["model"].predict(dataset["x_test"]))
    prediction_frame = pd.DataFrame(
        {
            dataset["id_column"]: dataset["test_frame"][dataset["id_column"]].tolist(),
            dataset["target_column"]: predictions.tolist(),
        }
    )
    ordered = dataset["sample_submission"][[dataset["id_column"]]].merge(
        prediction_frame,
        on=dataset["id_column"],
        how="left",
    )
    if ordered[dataset["target_column"]].isna().any():
        raise ValueError("Submission contains missing predictions after merge. Check id alignment.")
    return ordered


def save_model_bundle(model_bundle: dict[str, Any], path: Path = DEFAULT_MODEL_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_bundle, path)


def save_metrics(payload: dict[str, Any], path: Path = DEFAULT_METRICS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
