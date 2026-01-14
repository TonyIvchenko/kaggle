from __future__ import annotations

from pathlib import Path

import pandas as pd

from competitions.march_machine_learning_mania_2026.models.baseline import (
    TEAM_STAT_COLUMNS,
    TorchCandidateConfig,
    build_datasets,
    build_matchup_features,
    build_team_stats,
    default_holdout_season,
    discover_competition_files,
    evaluate_predictions,
    feature_columns,
    fit_and_score_holdout,
    fit_and_score_holdout_by_gender,
    fit_final_model,
    fit_final_model_by_gender,
    generate_submission,
    load_compact_results,
    load_seeds,
    parse_seed,
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def _prepare_competition_dir(tmp_path: Path) -> Path:
    _write_csv(
        tmp_path / "MRegularSeasonCompactResults.csv",
        [
            {"Season": 2023, "DayNum": 10, "WTeamID": 1101, "WScore": 80, "LTeamID": 1102, "LScore": 70, "WLoc": "H", "NumOT": 0},
            {"Season": 2023, "DayNum": 20, "WTeamID": 1103, "WScore": 77, "LTeamID": 1104, "LScore": 60, "WLoc": "N", "NumOT": 0},
            {"Season": 2024, "DayNum": 10, "WTeamID": 1102, "WScore": 79, "LTeamID": 1101, "LScore": 72, "WLoc": "A", "NumOT": 0},
            {"Season": 2024, "DayNum": 20, "WTeamID": 1104, "WScore": 83, "LTeamID": 1103, "LScore": 65, "WLoc": "N", "NumOT": 0},
            {"Season": 2025, "DayNum": 10, "WTeamID": 1101, "WScore": 81, "LTeamID": 1102, "LScore": 68, "WLoc": "H", "NumOT": 0},
            {"Season": 2025, "DayNum": 20, "WTeamID": 1104, "WScore": 88, "LTeamID": 1103, "LScore": 71, "WLoc": "N", "NumOT": 0},
            {"Season": 2026, "DayNum": 10, "WTeamID": 1101, "WScore": 84, "LTeamID": 1104, "LScore": 73, "WLoc": "N", "NumOT": 0},
            {"Season": 2026, "DayNum": 20, "WTeamID": 1102, "WScore": 76, "LTeamID": 1103, "LScore": 70, "WLoc": "H", "NumOT": 0},
        ],
    )
    _write_csv(
        tmp_path / "WRegularSeasonCompactResults.csv",
        [
            {"Season": 2023, "DayNum": 10, "WTeamID": 2101, "WScore": 75, "LTeamID": 2102, "LScore": 60, "WLoc": "H", "NumOT": 0},
            {"Season": 2023, "DayNum": 20, "WTeamID": 2103, "WScore": 79, "LTeamID": 2104, "LScore": 61, "WLoc": "N", "NumOT": 0},
            {"Season": 2024, "DayNum": 10, "WTeamID": 2102, "WScore": 73, "LTeamID": 2101, "LScore": 70, "WLoc": "A", "NumOT": 0},
            {"Season": 2024, "DayNum": 20, "WTeamID": 2104, "WScore": 82, "LTeamID": 2103, "LScore": 74, "WLoc": "N", "NumOT": 0},
            {"Season": 2025, "DayNum": 10, "WTeamID": 2101, "WScore": 77, "LTeamID": 2102, "LScore": 69, "WLoc": "H", "NumOT": 0},
            {"Season": 2025, "DayNum": 20, "WTeamID": 2104, "WScore": 85, "LTeamID": 2103, "LScore": 76, "WLoc": "N", "NumOT": 0},
            {"Season": 2026, "DayNum": 10, "WTeamID": 2101, "WScore": 80, "LTeamID": 2104, "LScore": 68, "WLoc": "N", "NumOT": 0},
            {"Season": 2026, "DayNum": 20, "WTeamID": 2102, "WScore": 75, "LTeamID": 2103, "LScore": 71, "WLoc": "H", "NumOT": 0},
        ],
    )
    _write_csv(
        tmp_path / "MNCAATourneyCompactResults.csv",
        [
            {"Season": 2023, "DayNum": 140, "WTeamID": 1101, "WScore": 76, "LTeamID": 1104, "LScore": 71, "WLoc": "N", "NumOT": 0},
            {"Season": 2023, "DayNum": 141, "WTeamID": 1103, "WScore": 70, "LTeamID": 1102, "LScore": 68, "WLoc": "N", "NumOT": 0},
            {"Season": 2024, "DayNum": 140, "WTeamID": 1102, "WScore": 77, "LTeamID": 1103, "LScore": 69, "WLoc": "N", "NumOT": 0},
            {"Season": 2024, "DayNum": 141, "WTeamID": 1104, "WScore": 75, "LTeamID": 1101, "LScore": 70, "WLoc": "N", "NumOT": 0},
            {"Season": 2025, "DayNum": 140, "WTeamID": 1101, "WScore": 82, "LTeamID": 1103, "LScore": 74, "WLoc": "N", "NumOT": 0},
            {"Season": 2025, "DayNum": 141, "WTeamID": 1104, "WScore": 79, "LTeamID": 1102, "LScore": 73, "WLoc": "N", "NumOT": 0},
        ],
    )
    _write_csv(
        tmp_path / "WNCAATourneyCompactResults.csv",
        [
            {"Season": 2023, "DayNum": 140, "WTeamID": 2101, "WScore": 71, "LTeamID": 2104, "LScore": 66, "WLoc": "N", "NumOT": 0},
            {"Season": 2023, "DayNum": 141, "WTeamID": 2103, "WScore": 75, "LTeamID": 2102, "LScore": 67, "WLoc": "N", "NumOT": 0},
            {"Season": 2024, "DayNum": 140, "WTeamID": 2102, "WScore": 74, "LTeamID": 2103, "LScore": 72, "WLoc": "N", "NumOT": 0},
            {"Season": 2024, "DayNum": 141, "WTeamID": 2104, "WScore": 78, "LTeamID": 2101, "LScore": 69, "WLoc": "N", "NumOT": 0},
            {"Season": 2025, "DayNum": 140, "WTeamID": 2101, "WScore": 79, "LTeamID": 2103, "LScore": 72, "WLoc": "N", "NumOT": 0},
            {"Season": 2025, "DayNum": 141, "WTeamID": 2104, "WScore": 81, "LTeamID": 2102, "LScore": 70, "WLoc": "N", "NumOT": 0},
        ],
    )
    _write_csv(
        tmp_path / "MNCAATourneySeeds.csv",
        [
            {"Season": 2023, "Seed": "W01", "TeamID": 1101},
            {"Season": 2023, "Seed": "W02", "TeamID": 1102},
            {"Season": 2023, "Seed": "X03", "TeamID": 1103},
            {"Season": 2023, "Seed": "X04", "TeamID": 1104},
            {"Season": 2024, "Seed": "W01", "TeamID": 1102},
            {"Season": 2024, "Seed": "W02", "TeamID": 1101},
            {"Season": 2024, "Seed": "X03", "TeamID": 1103},
            {"Season": 2024, "Seed": "X04", "TeamID": 1104},
            {"Season": 2025, "Seed": "W01", "TeamID": 1101},
            {"Season": 2025, "Seed": "W02", "TeamID": 1102},
            {"Season": 2025, "Seed": "X03", "TeamID": 1103},
            {"Season": 2025, "Seed": "X04", "TeamID": 1104},
            {"Season": 2026, "Seed": "W01", "TeamID": 1101},
            {"Season": 2026, "Seed": "W02", "TeamID": 1102},
            {"Season": 2026, "Seed": "X03", "TeamID": 1103},
            {"Season": 2026, "Seed": "X04", "TeamID": 1104},
        ],
    )
    _write_csv(
        tmp_path / "WNCAATourneySeeds.csv",
        [
            {"Season": 2023, "Seed": "W01", "TeamID": 2101},
            {"Season": 2023, "Seed": "W02", "TeamID": 2102},
            {"Season": 2023, "Seed": "X03", "TeamID": 2103},
            {"Season": 2023, "Seed": "X04", "TeamID": 2104},
            {"Season": 2024, "Seed": "W01", "TeamID": 2102},
            {"Season": 2024, "Seed": "W02", "TeamID": 2101},
            {"Season": 2024, "Seed": "X03", "TeamID": 2103},
            {"Season": 2024, "Seed": "X04", "TeamID": 2104},
            {"Season": 2025, "Seed": "W01", "TeamID": 2101},
            {"Season": 2025, "Seed": "W02", "TeamID": 2102},
            {"Season": 2025, "Seed": "X03", "TeamID": 2103},
            {"Season": 2025, "Seed": "X04", "TeamID": 2104},
            {"Season": 2026, "Seed": "W01", "TeamID": 2101},
            {"Season": 2026, "Seed": "W02", "TeamID": 2102},
            {"Season": 2026, "Seed": "X03", "TeamID": 2103},
            {"Season": 2026, "Seed": "X04", "TeamID": 2104},
        ],
    )
    _write_csv(
        tmp_path / "MTeams.csv",
        [
            {"TeamID": 1101, "TeamName": "M Team 1"},
            {"TeamID": 1102, "TeamName": "M Team 2"},
            {"TeamID": 1103, "TeamName": "M Team 3"},
            {"TeamID": 1104, "TeamName": "M Team 4"},
        ],
    )
    _write_csv(
        tmp_path / "WTeams.csv",
        [
            {"TeamID": 2101, "TeamName": "W Team 1"},
            {"TeamID": 2102, "TeamName": "W Team 2"},
            {"TeamID": 2103, "TeamName": "W Team 3"},
            {"TeamID": 2104, "TeamName": "W Team 4"},
        ],
    )
    pd.DataFrame(
        {
            "ID": [
                "2026_1101_1104",
                "2026_1102_1103",
                "2026_2101_2104",
                "2026_2102_2103",
            ],
            "Pred": [0.5, 0.5, 0.5, 0.5],
        }
    ).to_csv(tmp_path / "SampleSubmissionStage1.csv", index=False)
    return tmp_path


def test_parse_seed_reads_two_digit_seed():
    assert parse_seed("W01") == 1.0
    assert parse_seed("X16b") == 16.0
    assert parse_seed(None) == 17.0


def test_discover_competition_files_and_build_datasets(tmp_path: Path):
    raw_dir = _prepare_competition_dir(tmp_path)
    files = discover_competition_files(raw_dir)

    training_frame, submission_frame, team_stats = build_datasets(files)

    assert not training_frame.empty
    assert not submission_frame.empty
    assert {"M", "W"} == set(training_frame["gender"].unique())
    assert {"ID", "gender", "season", "team1_id", "team2_id"}.issubset(submission_frame.columns)
    assert set(TEAM_STAT_COLUMNS).issubset(team_stats.columns)


def test_holdout_training_and_submission_generation(tmp_path: Path):
    raw_dir = _prepare_competition_dir(tmp_path)
    files = discover_competition_files(raw_dir)
    training_frame, submission_frame, _ = build_datasets(files)
    candidates = [
        TorchCandidateConfig(
            name="test_linear",
            architecture="linear",
            hidden_dims=(),
            dropout=0.0,
            learning_rate=0.01,
            weight_decay=1e-4,
            batch_size=32,
            epochs=20,
            patience=5,
            label_smoothing=0.0,
        ),
        TorchCandidateConfig(
            name="test_mlp",
            architecture="mlp",
            hidden_dims=(32, 16),
            dropout=0.1,
            learning_rate=0.005,
            weight_decay=5e-4,
            batch_size=32,
            epochs=25,
            patience=6,
            label_smoothing=0.01,
        ),
    ]

    holdout_season = default_holdout_season(training_frame, target_season=2026)
    assert holdout_season == 2025

    _, metrics, _ = fit_and_score_holdout(
        training_frame,
        target_season=2026,
        holdout_season=holdout_season,
        device_preference="cpu",
        seed=123,
        candidate_configs=candidates,
    )
    final_model = fit_final_model(
        training_frame,
        target_season=2026,
        device_preference="cpu",
        seed=123,
        selected_configs=[candidate.__dict__.copy() for candidate in candidates],
    )
    submission = generate_submission(final_model, submission_frame, device_preference="cpu")

    assert 0.0 <= metrics["mse"] <= 1.0
    assert metrics["candidate_metrics"]
    assert final_model["device_used"] == "cpu"
    assert list(submission.columns) == ["ID", "Pred"]
    assert submission["Pred"].between(0.025, 0.975).all()


def test_gender_split_training_and_submission_generation(tmp_path: Path):
    raw_dir = _prepare_competition_dir(tmp_path)
    files = discover_competition_files(raw_dir)
    training_frame, submission_frame, _ = build_datasets(files)
    candidates = [
        TorchCandidateConfig(
            name="test_linear",
            architecture="linear",
            hidden_dims=(),
            dropout=0.0,
            learning_rate=0.01,
            weight_decay=1e-4,
            batch_size=32,
            epochs=20,
            patience=5,
            label_smoothing=0.0,
        ),
        TorchCandidateConfig(
            name="test_mlp",
            architecture="mlp",
            hidden_dims=(32, 16),
            dropout=0.1,
            learning_rate=0.005,
            weight_decay=5e-4,
            batch_size=32,
            epochs=25,
            patience=6,
            label_smoothing=0.01,
        ),
    ]

    holdout_bundle, metrics, _ = fit_and_score_holdout_by_gender(
        training_frame=training_frame,
        target_season=2026,
        holdout_season=2025,
        device_preference="cpu",
        seed=123,
        candidate_configs=candidates,
    )
    final_model = fit_final_model_by_gender(
        training_frame=training_frame,
        target_season=2026,
        device_preference="cpu",
        seed=123,
        selected_configs_by_gender={
            "M": [candidate.__dict__.copy() for candidate in candidates],
            "W": [candidate.__dict__.copy() for candidate in candidates],
        },
    )
    submission = generate_submission(final_model, submission_frame, device_preference="cpu")

    assert holdout_bundle["split_by_gender"] is True
    assert {"M", "W"} == set(holdout_bundle["models_by_gender"].keys())
    assert 0.0 <= metrics["mse"] <= 1.0
    assert {"M", "W"} == set(metrics["candidate_metrics_by_gender"].keys())
    assert final_model["split_by_gender"] is True
    assert list(submission.columns) == ["ID", "Pred"]
    assert submission["Pred"].between(0.025, 0.975).all()
