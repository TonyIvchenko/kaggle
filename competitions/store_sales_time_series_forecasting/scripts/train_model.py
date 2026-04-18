"""Train and evaluate Store Sales baseline models."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from competitions.store_sales_time_series_forecasting.models.baseline import (  # noqa: E402
    COMPETITION_SLUG,
    DEFAULT_METRICS_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_PROCESSED_DIR,
    DEFAULT_RAW_DIR,
    DEFAULT_SUBMISSION_PATH,
    build_dataset,
    discover_competition_files,
    fit_and_score_holdout,
    fit_final_model,
    generate_submission,
    save_metrics,
    save_model_bundle,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Store Sales baseline models.")
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
    parser.add_argument("--holdout-days", type=int, default=16, help="Number of final train dates to reserve for holdout.")
    parser.add_argument(
        "--recent-train-start",
        type=str,
        default="2016-01-01",
        help="Earliest date kept in the supervised training frame after lag generation.",
    )
    parser.add_argument(
        "--max-train-rows",
        type=int,
        default=1_200_000,
        help="Cap the supervised training rows used for model fitting.",
    )
    parser.add_argument(
        "--candidate-strategies",
        type=str,
        default="seasonal_naive,lightgbm,xgboost",
        help="Comma-separated candidate strategies to evaluate on holdout.",
    )
    parser.add_argument(
        "--force-strategy",
        type=str,
        default=None,
        help="Optional strategy override (e.g. seasonal_naive, lightgbm, xgboost).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = discover_competition_files(args.raw_dir)
    dataset = build_dataset(files)
    print(
        f"Competition: {COMPETITION_SLUG}\n"
        f"Rows: train={len(dataset.train_frame)}, test={len(dataset.test_frame)}\n"
        f"Candidate strategies: {args.candidate_strategies}"
    )

    args.processed_dir.mkdir(parents=True, exist_ok=True)
    train_preview_path = args.processed_dir / "train_preview.csv"
    test_preview_path = args.processed_dir / "test_preview.csv"
    holdout_predictions_path = args.processed_dir / "holdout_predictions.csv"

    dataset.train_frame.head(2000).to_csv(train_preview_path, index=False)
    dataset.test_frame.head(2000).to_csv(test_preview_path, index=False)

    candidate_strategies = [part.strip() for part in args.candidate_strategies.split(",") if part.strip()]
    force_strategy = str(args.force_strategy).strip() if args.force_strategy else None
    if force_strategy and force_strategy not in candidate_strategies:
        candidate_strategies.append(force_strategy)
    selection, holdout_metrics, holdout_predictions = fit_and_score_holdout(
        dataset=dataset,
        holdout_days=args.holdout_days,
        recent_train_start=args.recent_train_start,
        max_train_rows=args.max_train_rows,
        candidate_strategies=tuple(candidate_strategies),
        seed=args.seed,
        force_strategy=force_strategy,
    )
    holdout_predictions.to_csv(holdout_predictions_path, index=False)

    final_model = fit_final_model(
        dataset=dataset,
        selection=selection,
        recent_train_start=args.recent_train_start,
        max_train_rows=args.max_train_rows,
        seed=args.seed,
    )
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
    print(
        "Holdout metrics: "
        f"rmsle={holdout_metrics['rmsle']:.6f}, "
        f"mae={holdout_metrics['mae']:.6f}"
    )
    print("Candidate strategy metrics:")
    for row in holdout_metrics["strategy_metrics"]:
        print(f"  - {row['name']}: rmsle={row['rmsle']:.6f}, mae={row['mae']:.6f}")


if __name__ == "__main__":
    main()
