"""Train a blended ensemble for Playground Series S6E3."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from xgboost import XGBClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from competitions.playground_series_s6e3.models.baseline import (  # noqa: E402
    COMPETITION_SLUG,
    DEFAULT_PROCESSED_DIR,
    DEFAULT_RAW_DIR,
    DEFAULT_SUBMISSION_PATH,
    build_datasets,
    discover_competition_files,
)


DEFAULT_METRICS_PATH = PROJECT_ROOT / "competitions" / "playground_series_s6e3" / "models" / "ensemble_metrics.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and blend CatBoost + LightGBM models for S6E3.")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR, help="Directory with competition CSV files.")
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=DEFAULT_PROCESSED_DIR,
        help="Directory used for holdout artifacts.",
    )
    parser.add_argument(
        "--submission-path",
        type=Path,
        default=DEFAULT_SUBMISSION_PATH,
        help="Path to save submission CSV.",
    )
    parser.add_argument(
        "--metrics-path",
        type=Path,
        default=DEFAULT_METRICS_PATH,
        help="Path to save ensemble metrics JSON.",
    )
    parser.add_argument(
        "--holdout-fraction",
        type=float,
        default=0.2,
        help="Fraction of train rows used for holdout evaluation.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--blend-search-trials",
        type=int,
        default=2500,
        help="Random blend samples for weight search.",
    )
    return parser.parse_args()


def _normalize_label(value: Any) -> str:
    return str(value).strip().lower()


def _infer_positive_label(target: pd.Series) -> Any:
    values = [value for value in target.dropna().unique().tolist()]
    if len(values) != 2:
        raise ValueError(f"Expected binary target for this ensemble script, got {len(values)} classes.")

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


def _split_feature_types(frame: pd.DataFrame, feature_columns: list[str]) -> tuple[list[str], list[str]]:
    numeric = [column for column in feature_columns if pd.api.types.is_numeric_dtype(frame[column])]
    categorical = [column for column in feature_columns if column not in numeric]
    return numeric, categorical


def _prepare_catboost_frame(frame: pd.DataFrame, categorical_columns: list[str]) -> pd.DataFrame:
    out = frame.copy()
    for column in categorical_columns:
        out[column] = out[column].astype("string").fillna("__nan__")
    return out


def _rank_normalized(values: np.ndarray) -> np.ndarray:
    if values.size <= 1:
        return np.zeros_like(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(values.size, dtype=float)
    return ranks / float(values.size - 1)


def _fit_catboost(
    *,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_valid: pd.DataFrame | None,
    y_valid: pd.Series | None,
    categorical_columns: list[str],
    positive_label: Any,
    params: dict[str, Any],
) -> tuple[CatBoostClassifier, np.ndarray | None]:
    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        allow_writing_files=False,
        thread_count=-1,
        verbose=False,
        **params,
    )
    fit_kwargs: dict[str, Any] = {"cat_features": categorical_columns, "verbose": False}
    if x_valid is not None and y_valid is not None:
        fit_kwargs["eval_set"] = (x_valid, y_valid)
        fit_kwargs["use_best_model"] = True
        fit_kwargs["early_stopping_rounds"] = 160
    model.fit(x_train, y_train, **fit_kwargs)

    pred: np.ndarray | None = None
    if x_valid is not None:
        classes = list(model.classes_)
        pred = np.asarray(model.predict_proba(x_valid))[:, classes.index(positive_label)]
    return model, pred


def _fit_lightgbm(
    *,
    x_train_matrix: Any,
    y_train_binary: np.ndarray,
    x_valid_matrix: Any | None,
    y_valid_binary: np.ndarray | None,
    params: dict[str, Any],
    n_estimators: int,
) -> tuple[LGBMClassifier, np.ndarray | None]:
    model = LGBMClassifier(
        objective="binary",
        n_estimators=int(n_estimators),
        random_state=42,
        n_jobs=-1,
        **params,
    )
    fit_kwargs: dict[str, Any] = {}
    if x_valid_matrix is not None and y_valid_binary is not None:
        fit_kwargs["eval_set"] = [(x_valid_matrix, y_valid_binary)]
        fit_kwargs["eval_metric"] = "auc"
        fit_kwargs["callbacks"] = [early_stopping(140, verbose=False), log_evaluation(0)]
    model.fit(x_train_matrix, y_train_binary, **fit_kwargs)
    pred: np.ndarray | None = None
    if x_valid_matrix is not None:
        pred = np.asarray(model.predict_proba(x_valid_matrix))[:, 1]
    return model, pred


def _fit_xgboost(
    *,
    x_train_matrix: Any,
    y_train_binary: np.ndarray,
    x_valid_matrix: Any | None,
    y_valid_binary: np.ndarray | None,
    params: dict[str, Any],
    n_estimators: int,
) -> tuple[XGBClassifier, np.ndarray | None]:
    has_valid = x_valid_matrix is not None and y_valid_binary is not None
    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        max_bin=256,
        random_state=42,
        n_jobs=-1,
        n_estimators=int(n_estimators),
        early_stopping_rounds=160 if has_valid else None,
        **params,
    )
    fit_kwargs: dict[str, Any] = {"verbose": False}
    if has_valid:
        fit_kwargs["eval_set"] = [(x_valid_matrix, y_valid_binary)]
    model.fit(x_train_matrix, y_train_binary, **fit_kwargs)
    pred: np.ndarray | None = None
    if x_valid_matrix is not None:
        pred = np.asarray(model.predict_proba(x_valid_matrix))[:, 1]
    return model, pred


def _blend_search(
    prediction_map: dict[str, np.ndarray],
    y_binary: np.ndarray,
    trials: int,
    seed: int,
) -> dict[str, Any]:
    names = list(prediction_map.keys())
    matrix = np.column_stack([prediction_map[name] for name in names])
    rng = np.random.default_rng(seed)

    baseline_rows: list[dict[str, Any]] = []
    for idx, name in enumerate(names):
        pred = matrix[:, idx]
        baseline_rows.append(
            {
                "name": name,
                "auc": float(roc_auc_score(y_binary, pred)),
                "log_loss": float(log_loss(y_binary, pred)),
            }
        )

    best: dict[str, Any] | None = None
    for mode in ("probability", "rank"):
        if mode == "rank":
            source = np.column_stack([_rank_normalized(matrix[:, idx]) for idx in range(matrix.shape[1])])
        else:
            source = matrix

        for _ in range(max(1, trials)):
            weights = rng.dirichlet(np.ones(len(names), dtype=float))
            blend = source @ weights
            auc = float(roc_auc_score(y_binary, blend))
            if best is None or auc > best["auc"]:
                best = {
                    "mode": mode,
                    "auc": auc,
                    "log_loss": float(log_loss(y_binary, blend)),
                    "weights": {name: float(weights[idx]) for idx, name in enumerate(names)},
                }

    assert best is not None
    return {"single_model_metrics": baseline_rows, "best_blend": best}


def _apply_blend(
    *,
    prediction_map: dict[str, np.ndarray],
    mode: str,
    weights: dict[str, float],
) -> np.ndarray:
    names = list(prediction_map.keys())
    matrix = np.column_stack([prediction_map[name] for name in names])
    weight_vector = np.asarray([weights.get(name, 0.0) for name in names], dtype=float)
    if mode == "rank":
        matrix = np.column_stack([_rank_normalized(matrix[:, idx]) for idx in range(matrix.shape[1])])
    return matrix @ weight_vector


def main() -> None:
    args = parse_args()
    files = discover_competition_files(args.raw_dir)
    dataset = build_datasets(files)
    if dataset.task_type != "classification":
        raise ValueError("train_ensemble.py supports classification tasks only.")
    if not dataset.probability_output:
        raise ValueError("train_ensemble.py expects probability output competition.")

    train_frame = dataset.train_frame.copy()
    test_frame = dataset.test_frame.copy()
    feature_columns = list(dataset.feature_columns)
    target_column = dataset.target_column
    id_column = dataset.id_column
    positive_label = _infer_positive_label(train_frame[target_column])

    x_all = train_frame.loc[:, feature_columns]
    y_all = train_frame.loc[:, target_column]
    y_all_binary = (y_all == positive_label).astype(np.int8).to_numpy()
    x_test = test_frame.loc[:, feature_columns]

    train_idx, holdout_idx = train_test_split(
        np.arange(len(train_frame), dtype=np.int64),
        test_size=float(args.holdout_fraction),
        random_state=args.seed,
        shuffle=True,
        stratify=y_all,
    )
    x_train = x_all.iloc[train_idx].copy()
    y_train = y_all.iloc[train_idx].copy()
    y_train_binary = (y_train == positive_label).astype(np.int8).to_numpy()
    x_holdout = x_all.iloc[holdout_idx].copy()
    y_holdout = y_all.iloc[holdout_idx].copy()
    y_holdout_binary = (y_holdout == positive_label).astype(np.int8).to_numpy()

    numeric_columns, categorical_columns = _split_feature_types(train_frame, feature_columns)
    x_train_cb = _prepare_catboost_frame(x_train, categorical_columns)
    x_holdout_cb = _prepare_catboost_frame(x_holdout, categorical_columns)

    cat_a_params = {
        "iterations": 1800,
        "learning_rate": 0.04,
        "depth": 8,
        "l2_leaf_reg": 6.0,
        "bagging_temperature": 0.2,
        "random_strength": 1.0,
        "random_seed": args.seed,
    }
    print("Fitting holdout catboost_a...")
    cat_a_model, cat_a_holdout = _fit_catboost(
        x_train=x_train_cb,
        y_train=y_train,
        x_valid=x_holdout_cb,
        y_valid=y_holdout,
        categorical_columns=categorical_columns,
        positive_label=positive_label,
        params=cat_a_params,
    )
    cat_a_best_iter = int(max(0, cat_a_model.get_best_iteration()))
    print(f"  catboost_a best_iteration={cat_a_best_iter}")

    lgb_preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]), numeric_columns),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_columns,
            ),
        ],
        sparse_threshold=1.0,
    )
    x_train_lgb = lgb_preprocessor.fit_transform(x_train)
    x_holdout_lgb = lgb_preprocessor.transform(x_holdout)
    x_test_lgb = lgb_preprocessor.transform(x_test)

    lgb_params = {
        "learning_rate": 0.03,
        "num_leaves": 63,
        "max_depth": -1,
        "min_child_samples": 20,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "min_split_gain": 0.0,
    }
    xgb_params = {
        "learning_rate": 0.02,
        "max_depth": 6,
        "min_child_weight": 1,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
    }
    print("Fitting holdout lightgbm...")
    lgb_model, lgb_holdout = _fit_lightgbm(
        x_train_matrix=x_train_lgb,
        y_train_binary=y_train_binary,
        x_valid_matrix=x_holdout_lgb,
        y_valid_binary=y_holdout_binary,
        params=lgb_params,
        n_estimators=3000,
    )
    lgb_best_iter = int(max(1, int(getattr(lgb_model, "best_iteration_", 0) or 0)))
    print(f"  lightgbm best_iteration={lgb_best_iter}")

    print("Fitting holdout xgboost...")
    xgb_model, xgb_holdout = _fit_xgboost(
        x_train_matrix=x_train_lgb,
        y_train_binary=y_train_binary,
        x_valid_matrix=x_holdout_lgb,
        y_valid_binary=y_holdout_binary,
        params=xgb_params,
        n_estimators=3000,
    )
    xgb_best_iter = int(max(1, int(getattr(xgb_model, "best_iteration", 0) or 0)))
    print(f"  xgboost best_iteration={xgb_best_iter}")

    assert cat_a_holdout is not None and lgb_holdout is not None and xgb_holdout is not None
    holdout_prediction_map = {
        "catboost_a": cat_a_holdout,
        "lightgbm": lgb_holdout,
        "xgboost": xgb_holdout,
    }
    blend = _blend_search(
        prediction_map=holdout_prediction_map,
        y_binary=y_holdout_binary,
        trials=args.blend_search_trials,
        seed=args.seed,
    )
    best_blend = blend["best_blend"]
    print(
        "Selected holdout blend: "
        f"mode={best_blend['mode']}, auc={best_blend['auc']:.6f}, log_loss={best_blend['log_loss']:.6f}"
    )

    x_all_cb = _prepare_catboost_frame(x_all, categorical_columns)
    x_test_cb = _prepare_catboost_frame(x_test, categorical_columns)
    cat_a_full_params = dict(cat_a_params)
    if cat_a_best_iter > 0:
        cat_a_full_params["iterations"] = int(max(450, cat_a_best_iter + 80))

    print("Fitting full catboost_a...")
    cat_a_full, _ = _fit_catboost(
        x_train=x_all_cb,
        y_train=y_all,
        x_valid=None,
        y_valid=None,
        categorical_columns=categorical_columns,
        positive_label=positive_label,
        params=cat_a_full_params,
    )
    x_all_lgb = lgb_preprocessor.fit_transform(x_all)
    lgb_full_estimators = int(max(350, lgb_best_iter + 100))
    print(f"Fitting full lightgbm (n_estimators={lgb_full_estimators})...")
    lgb_full, _ = _fit_lightgbm(
        x_train_matrix=x_all_lgb,
        y_train_binary=y_all_binary,
        x_valid_matrix=None,
        y_valid_binary=None,
        params=lgb_params,
        n_estimators=lgb_full_estimators,
    )
    xgb_full_estimators = int(max(400, xgb_best_iter + 120))
    print(f"Fitting full xgboost (n_estimators={xgb_full_estimators})...")
    xgb_full, _ = _fit_xgboost(
        x_train_matrix=x_all_lgb,
        y_train_binary=y_all_binary,
        x_valid_matrix=None,
        y_valid_binary=None,
        params=xgb_params,
        n_estimators=xgb_full_estimators,
    )

    cat_a_test = np.asarray(cat_a_full.predict_proba(x_test_cb))[:, list(cat_a_full.classes_).index(positive_label)]
    lgb_test = np.asarray(lgb_full.predict_proba(x_test_lgb))[:, 1]
    xgb_test = np.asarray(xgb_full.predict_proba(x_test_lgb))[:, 1]
    test_prediction_map = {
        "catboost_a": cat_a_test,
        "lightgbm": lgb_test,
        "xgboost": xgb_test,
    }
    blended_test = _apply_blend(
        prediction_map=test_prediction_map,
        mode=str(best_blend["mode"]),
        weights={str(key): float(value) for key, value in best_blend["weights"].items()},
    )

    args.submission_path.parent.mkdir(parents=True, exist_ok=True)
    submission = pd.DataFrame(
        {
            id_column: test_frame[id_column].tolist(),
            target_column: blended_test.tolist(),
        }
    )
    ordered = dataset.sample_submission[[id_column]].merge(submission, on=id_column, how="left")
    if ordered[target_column].isna().any():
        raise ValueError("Submission contains missing predictions after merge.")
    ordered.to_csv(args.submission_path, index=False)

    args.processed_dir.mkdir(parents=True, exist_ok=True)
    holdout_prediction = _apply_blend(
        prediction_map=holdout_prediction_map,
        mode=str(best_blend["mode"]),
        weights={str(key): float(value) for key, value in best_blend["weights"].items()},
    )
    holdout_df = pd.DataFrame(
        {
            id_column: train_frame.iloc[holdout_idx][id_column].tolist(),
            target_column: y_holdout.tolist(),
            "prediction": holdout_prediction.tolist(),
        }
    )
    holdout_path = args.processed_dir / "holdout_ensemble_predictions.csv"
    holdout_df.to_csv(holdout_path, index=False)

    metrics = {
        "competition": COMPETITION_SLUG,
        "task_type": dataset.task_type,
        "id_column": id_column,
        "target_column": target_column,
        "training_rows": int(len(train_frame)),
        "test_rows": int(len(test_frame)),
        "holdout_rows": int(len(holdout_idx)),
        "positive_label": str(positive_label),
        "single_model_metrics": blend["single_model_metrics"],
        "best_blend": best_blend,
        "blended_holdout_auc": float(roc_auc_score(y_holdout_binary, holdout_prediction)),
        "blended_holdout_log_loss": float(log_loss(y_holdout_binary, holdout_prediction)),
        "models": {
            "catboost_a": cat_a_params,
            "lightgbm": lgb_params,
            "xgboost": xgb_params,
        },
        "full_train_iterations": {
            "catboost_a": int(cat_a_full_params["iterations"]),
            "lightgbm": int(lgb_full_estimators),
            "xgboost": int(xgb_full_estimators),
        },
    }
    args.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Saved holdout ensemble predictions to: {holdout_path}")
    print(f"Saved submission to: {args.submission_path}")
    print(f"Saved metrics to: {args.metrics_path}")
    print(
        "Best blend: "
        f"mode={best_blend['mode']}, "
        f"auc={best_blend['auc']:.6f}, "
        f"log_loss={best_blend['log_loss']:.6f}"
    )
    for row in blend["single_model_metrics"]:
        print(f"  - {row['name']}: auc={row['auc']:.6f}, log_loss={row['log_loss']:.6f}")


if __name__ == "__main__":
    main()
