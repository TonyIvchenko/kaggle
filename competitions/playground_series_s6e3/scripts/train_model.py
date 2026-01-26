"""Train and evaluate Playground Series S6E3 baselines."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from competitions.playground_series_s6e3.models.baseline import (
    COMPETITION_SLUG,
    DEFAULT_METRICS_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_PROCESSED_DIR,
    DEFAULT_RAW_DIR,
    DEFAULT_SUBMISSION_PATH,
    build_datasets,
    discover_competition_files,
    fit_and_score_holdout,
    fit_final_model,
    generate_submission,
    save_metrics,
    save_model_bundle,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Playground Series S6E3 baseline models.")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR, help="Directory with competition CSV files.")
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=DEFAULT_PROCESSED_DIR,
        help="Directory used for processed holdout artifacts.",
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH, help="Path to save the trained model.")
    parser.add_argument("--metrics-path", type=Path, default=DEFAULT_METRICS_PATH, help="Path to save metrics JSON.")
    parser.add_argument(
        "--submission-path",
        type=Path,
        default=DEFAULT_SUBMISSION_PATH,
        help="Path to save the submission CSV.",
    )
    parser.add_argument(
        "--holdout-fraction",
        type=float,
        default=0.2,
        help="Fraction of train rows used for holdout evaluation.",
    )
    parser.add_argument(
        "--target-column",
        type=str,
        default=None,
        help="Optional target column override when schema inference is ambiguous.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = discover_competition_files(args.raw_dir)
    dataset = build_datasets(files, target_column=args.target_column)

    args.processed_dir.mkdir(parents=True, exist_ok=True)
    holdout_predictions_path = args.processed_dir / "holdout_predictions.csv"
    train_preview_path = args.processed_dir / "train_preview.csv"
    test_preview_path = args.processed_dir / "test_preview.csv"
    train_preview_cols = [dataset.id_column, *dataset.feature_columns, dataset.target_column]
    test_preview_cols = [dataset.id_column, *dataset.feature_columns]
    dataset.train_frame[train_preview_cols].head(1000).to_csv(train_preview_path, index=False)
    dataset.test_frame[test_preview_cols].head(1000).to_csv(test_preview_path, index=False)

    selection, holdout_metrics, holdout_predictions = fit_and_score_holdout(
        dataset=dataset,
        holdout_fraction=args.holdout_fraction,
        seed=args.seed,
    )
    holdout_predictions.to_csv(holdout_predictions_path, index=False)

    final_model = fit_final_model(dataset=dataset, selection=selection, seed=args.seed)
    final_model["holdout_metrics"] = holdout_metrics
    final_model["selection"] = selection
    save_model_bundle(final_model, args.model_path)

    submission = generate_submission(final_model, dataset=dataset)
    args.submission_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.submission_path, index=False)

    metrics_payload = {
        "competition": COMPETITION_SLUG,
        "training_rows": int(len(dataset.train_frame)),
        "test_rows": int(len(dataset.test_frame)),
        "id_column": dataset.id_column,
        "target_column": dataset.target_column,
        "feature_columns": list(dataset.feature_columns),
        "task_type": dataset.task_type,
        "probability_output": dataset.probability_output,
        "selection": selection,
        "holdout_metrics": holdout_metrics,
    }
    save_metrics(metrics_payload, args.metrics_path)

    print(f"Saved train preview to: {train_preview_path}")
    print(f"Saved test preview to: {test_preview_path}")
    print(f"Saved holdout predictions to: {holdout_predictions_path}")
    print(f"Saved model bundle to: {args.model_path}")
    print(f"Saved metrics to: {args.metrics_path}")
    print(f"Saved submission to: {args.submission_path}")
    print(f"Selected strategy: {selection['selected_strategy']}")
    if dataset.task_type == "classification":
        print(
            "Holdout metrics: "
            f"auc={holdout_metrics.get('auc', float('nan')):.6f}, "
            f"accuracy={holdout_metrics['accuracy']:.6f}, "
            f"log_loss={holdout_metrics['log_loss']:.6f}"
        )
    else:
        print(
            "Holdout metrics: "
            f"rmse={holdout_metrics['rmse']:.6f}, "
            f"mae={holdout_metrics['mae']:.6f}"
        )
    print("Candidate strategy metrics:")
    for row in holdout_metrics["strategy_metrics"]:
        if dataset.task_type == "classification":
            print(
                f"  - {row['name']}: auc={row.get('auc', float('nan')):.6f}, "
                f"accuracy={row['accuracy']:.6f}, "
                f"log_loss={row['log_loss']:.6f}, score={row['score']:.6f}"
            )
        else:
            print(
                f"  - {row['name']}: rmse={row['rmse']:.6f}, "
                f"mae={row['mae']:.6f}, score={row['score']:.6f}"
            )


if __name__ == "__main__":
    main()
