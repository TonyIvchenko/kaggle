"""Train and evaluate baseline models for Deep Past Initiative translation."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from competitions.deep_past_initiative_machine_translation.models.baseline import (
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
    parser = argparse.ArgumentParser(description="Train baseline models for Deep Past Initiative translation.")
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
        default=0.15,
        help="Fraction of training rows used for holdout evaluation.",
    )
    parser.add_argument(
        "--device-preference",
        type=str,
        default="mps",
        choices=["auto", "mps", "cpu"],
        help="Torch device preference.",
    )
    parser.add_argument(
        "--target-column",
        type=str,
        default=None,
        help="Optional target column override when schema inference is ambiguous.",
    )
    parser.add_argument(
        "--use-torch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable or disable torch bi-encoder retriever candidates.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = discover_competition_files(args.raw_dir)
    dataset = build_datasets(files, target_column=args.target_column)

    args.processed_dir.mkdir(parents=True, exist_ok=True)
    train_pairs_path = args.processed_dir / "train_text_pairs.csv"
    test_pairs_path = args.processed_dir / "test_texts.csv"
    holdout_predictions_path = args.processed_dir / "holdout_predictions.csv"

    train_pairs = dataset.train_frame[[dataset.train_id_column, *dataset.source_columns, dataset.target_column]].copy()
    train_pairs["source_text"] = dataset.train_source_texts
    test_pairs = dataset.test_frame[[dataset.id_column, *dataset.source_columns]].copy()
    test_pairs["source_text"] = dataset.test_source_texts
    train_pairs.to_csv(train_pairs_path, index=False)
    test_pairs.to_csv(test_pairs_path, index=False)

    selection, holdout_metrics, holdout_predictions = fit_and_score_holdout(
        dataset=dataset,
        holdout_fraction=args.holdout_fraction,
        seed=args.seed,
        device_preference=args.device_preference,
        use_torch=bool(args.use_torch),
    )
    holdout_predictions.to_csv(holdout_predictions_path, index=False)

    final_model = fit_final_model(
        dataset=dataset,
        selection=selection,
        seed=args.seed,
        device_preference=args.device_preference,
    )
    final_model["holdout_metrics"] = holdout_metrics
    final_model["selection"] = selection
    save_model_bundle(final_model, args.model_path)

    submission = generate_submission(final_model, dataset=dataset, device_preference=args.device_preference)
    args.submission_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.submission_path, index=False)

    metrics_payload = {
        "competition": COMPETITION_SLUG,
        "training_rows": int(len(dataset.train_frame)),
        "test_rows": int(len(dataset.test_frame)),
        "train_id_column": dataset.train_id_column,
        "id_column": dataset.id_column,
        "target_column": dataset.target_column,
        "source_columns": list(dataset.source_columns),
        "selection": selection,
        "holdout_metrics": holdout_metrics,
        "device_used": final_model["device_used"],
    }
    save_metrics(metrics_payload, args.metrics_path)

    print(f"Saved train text pairs to: {train_pairs_path}")
    print(f"Saved test text rows to: {test_pairs_path}")
    print(f"Saved holdout predictions to: {holdout_predictions_path}")
    print(f"Saved model bundle to: {args.model_path}")
    print(f"Saved metrics to: {args.metrics_path}")
    print(f"Saved submission to: {args.submission_path}")
    print(f"Selected strategy: {holdout_metrics['selected_strategy']}")
    print(
        "Holdout metrics: "
        f"char_f1={holdout_metrics['char_f1']:.6f}, "
        f"sequence_ratio={holdout_metrics['sequence_ratio']:.6f}, "
        f"exact_match={holdout_metrics['exact_match']:.6f}"
    )
    print("Strategy leaderboard:")
    for row in holdout_metrics["strategy_metrics"]:
        print(
            f"  - {row['strategy']}: char_f1={row['char_f1']:.6f}, "
            f"sequence_ratio={row['sequence_ratio']:.6f}, exact_match={row['exact_match']:.6f}"
        )


if __name__ == "__main__":
    main()
