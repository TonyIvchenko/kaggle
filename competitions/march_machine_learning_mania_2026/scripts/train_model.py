"""Train and evaluate a March Machine Learning Mania 2026 baseline."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from competitions.march_machine_learning_mania_2026.models.baseline import (
    DEFAULT_METRICS_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_PROCESSED_DIR,
    DEFAULT_RAW_DIR,
    DEFAULT_SUBMISSION_PATH,
    build_datasets,
    default_holdout_season,
    discover_competition_files,
    fit_and_score_holdout,
    fit_final_model,
    generate_submission,
    save_metrics,
    save_model_bundle,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a March Mania 2026 baseline model.")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR, help="Directory with extracted CSV files.")
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=DEFAULT_PROCESSED_DIR,
        help="Directory used for processed feature tables and holdout predictions.",
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH, help="Path to the saved model bundle.")
    parser.add_argument(
        "--metrics-path",
        type=Path,
        default=DEFAULT_METRICS_PATH,
        help="Path to the saved holdout metrics JSON.",
    )
    parser.add_argument(
        "--submission-path",
        type=Path,
        default=DEFAULT_SUBMISSION_PATH,
        help="Path to the generated submission CSV.",
    )
    parser.add_argument("--target-season", type=int, default=2026, help="Competition season to score.")
    parser.add_argument(
        "--holdout-season",
        type=int,
        default=None,
        help="Optional historical season to reserve for offline evaluation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = discover_competition_files(args.raw_dir)
    training_frame, submission_frame, team_stats = build_datasets(files)

    args.processed_dir.mkdir(parents=True, exist_ok=True)
    training_output_path = args.processed_dir / "training_features.csv"
    submission_features_path = args.processed_dir / "submission_features.csv"
    holdout_predictions_path = args.processed_dir / "holdout_predictions.csv"
    team_stats_path = args.processed_dir / "team_stats.csv"

    training_frame.to_csv(training_output_path, index=False)
    submission_frame.to_csv(submission_features_path, index=False)
    team_stats.to_csv(team_stats_path, index=False)

    holdout_season = args.holdout_season or default_holdout_season(training_frame, args.target_season)
    _, holdout_metrics, holdout_predictions = fit_and_score_holdout(
        training_frame=training_frame,
        target_season=args.target_season,
        holdout_season=holdout_season,
    )
    holdout_predictions.to_csv(holdout_predictions_path, index=False)

    final_model = fit_final_model(training_frame, target_season=args.target_season)
    final_model["holdout_metrics"] = holdout_metrics
    final_model["holdout_season"] = holdout_season
    save_model_bundle(final_model, args.model_path)

    submission = generate_submission(final_model, submission_frame)
    args.submission_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.submission_path, index=False)

    metrics_payload = {
        "competition": "march-machine-learning-mania-2026",
        "target_season": int(args.target_season),
        "holdout_season": int(holdout_season),
        "training_rows": int(len(training_frame)),
        "submission_rows": int(len(submission_frame)),
        "holdout_metrics": holdout_metrics,
        "train_seasons": final_model["train_seasons"],
    }
    save_metrics(metrics_payload, args.metrics_path)

    print(f"Saved team stats to: {team_stats_path}")
    print(f"Saved training features to: {training_output_path}")
    print(f"Saved holdout predictions to: {holdout_predictions_path}")
    print(f"Saved model bundle to: {args.model_path}")
    print(f"Saved metrics to: {args.metrics_path}")
    print(f"Saved submission to: {args.submission_path}")
    print(
        "Holdout metrics: "
        f"MSE={holdout_metrics['mse']:.6f}, "
        f"log_loss={holdout_metrics['log_loss']:.6f}, "
        f"accuracy={holdout_metrics['accuracy']:.4f}"
    )


if __name__ == "__main__":
    main()
