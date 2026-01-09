"""Feature engineering and baseline modeling for March Machine Learning Mania 2026."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


COMPETITION_SLUG = "march-machine-learning-mania-2026"
COMPETITION_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = COMPETITION_ROOT / "data" / "raw"
DEFAULT_PROCESSED_DIR = COMPETITION_ROOT / "data" / "processed"
DEFAULT_MODEL_PATH = COMPETITION_ROOT / "models" / "march_machine_learning_mania_2026.joblib"
DEFAULT_METRICS_PATH = COMPETITION_ROOT / "models" / "march_machine_learning_mania_2026_metrics.json"
DEFAULT_SUBMISSION_PATH = COMPETITION_ROOT / "submissions" / "submission.csv"

HOME_ADVANTAGE_ELO = 50.0
BASE_ELO = 1500.0
ELO_K_FACTOR = 20.0

TEAM_STAT_COLUMNS = [
    "games_played",
    "wins",
    "win_pct",
    "avg_points_for",
    "avg_points_against",
    "avg_score_diff",
    "score_diff_std",
    "avg_num_ot",
    "home_win_pct",
    "away_win_pct",
    "neutral_win_pct",
    "last10_win_pct",
    "last10_score_diff",
    "opp_win_pct",
    "opp_score_diff",
    "elo_rating",
    "seed_rank",
    "seed_missing",
]


@dataclass(frozen=True)
class CompetitionFiles:
    raw_dir: Path
    men_regular_season_results: Path
    women_regular_season_results: Path
    men_tourney_results: Path
    women_tourney_results: Path
    men_seeds: Path
    women_seeds: Path
    sample_submission: Path
    men_teams: Path | None = None
    women_teams: Path | None = None


def available_csv_files(raw_dir: Path) -> list[Path]:
    return sorted(path for path in raw_dir.rglob("*.csv") if path.is_file())


def _pick_file(csv_files: list[Path], candidates: tuple[str, ...], required: bool = True) -> Path | None:
    by_name = {path.name.lower(): path for path in csv_files}
    for name in candidates:
        match = by_name.get(name.lower())
        if match is not None:
            return match

    if required:
        available = ", ".join(sorted(path.name for path in csv_files))
        raise FileNotFoundError(f"Could not find any of {candidates}. Available CSV files: {available}")
    return None


def discover_competition_files(raw_dir: Path = DEFAULT_RAW_DIR) -> CompetitionFiles:
    csv_files = available_csv_files(raw_dir)
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files were found under {raw_dir}. Run the download script first."
        )

    return CompetitionFiles(
        raw_dir=raw_dir,
        men_regular_season_results=_pick_file(csv_files, ("MRegularSeasonCompactResults.csv",)),
        women_regular_season_results=_pick_file(csv_files, ("WRegularSeasonCompactResults.csv",)),
        men_tourney_results=_pick_file(csv_files, ("MNCAATourneyCompactResults.csv",)),
        women_tourney_results=_pick_file(csv_files, ("WNCAATourneyCompactResults.csv",)),
        men_seeds=_pick_file(csv_files, ("MNCAATourneySeeds.csv",)),
        women_seeds=_pick_file(csv_files, ("WNCAATourneySeeds.csv",)),
        sample_submission=_pick_file(
            csv_files,
            ("SampleSubmissionStage2.csv", "SampleSubmissionStage1.csv", "SampleSubmission.csv"),
        ),
        men_teams=_pick_file(csv_files, ("MTeams.csv",), required=False),
        women_teams=_pick_file(csv_files, ("WTeams.csv",), required=False),
    )


def parse_seed(seed: str | float | int | None) -> float:
    if seed is None or (isinstance(seed, float) and pd.isna(seed)):
        return 17.0
    match = re.search(r"(\d{2})", str(seed))
    if match is None:
        return 17.0
    return float(int(match.group(1)))


def _reverse_location(loc: str) -> str:
    if loc == "H":
        return "A"
    if loc == "A":
        return "H"
    return "N"


def load_compact_results(path: Path, gender: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required_columns = {"Season", "DayNum", "WTeamID", "WScore", "LTeamID", "LScore", "WLoc"}
    missing = sorted(required_columns - set(df.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    out = pd.DataFrame(
        {
            "gender": gender,
            "season": df["Season"].astype(int),
            "day_num": df["DayNum"].astype(int),
            "wteam_id": df["WTeamID"].astype(int),
            "wscore": df["WScore"].astype(float),
            "lteam_id": df["LTeamID"].astype(int),
            "lscore": df["LScore"].astype(float),
            "wloc": df["WLoc"].fillna("N").astype(str),
            "num_ot": pd.to_numeric(df.get("NumOT", 0), errors="coerce").fillna(0).astype(float),
        }
    )
    return out.sort_values(["gender", "season", "day_num", "wteam_id", "lteam_id"]).reset_index(drop=True)


def load_seeds(path: Path, gender: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required_columns = {"Season", "TeamID", "Seed"}
    missing = sorted(required_columns - set(df.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    out = pd.DataFrame(
        {
            "gender": gender,
            "season": df["Season"].astype(int),
            "team_id": df["TeamID"].astype(int),
            "seed_rank": df["Seed"].map(parse_seed).astype(float),
            "seed_raw": df["Seed"].astype(str),
        }
    )
    out["seed_missing"] = 0.0
    return out


def load_teams(path: Path | None, gender: str) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame(columns=["gender", "team_id", "team_name"])

    df = pd.read_csv(path)
    name_column = "TeamName" if "TeamName" in df.columns else "Team"
    if name_column not in df.columns or "TeamID" not in df.columns:
        return pd.DataFrame(columns=["gender", "team_id", "team_name"])

    return pd.DataFrame(
        {
            "gender": gender,
            "team_id": df["TeamID"].astype(int),
            "team_name": df[name_column].astype(str),
        }
    )


def build_team_game_log(results: pd.DataFrame) -> pd.DataFrame:
    winners = pd.DataFrame(
        {
            "gender": results["gender"],
            "season": results["season"],
            "day_num": results["day_num"],
            "team_id": results["wteam_id"],
            "opp_team_id": results["lteam_id"],
            "is_win": 1.0,
            "points_for": results["wscore"],
            "points_against": results["lscore"],
            "score_diff": results["wscore"] - results["lscore"],
            "loc": results["wloc"],
            "num_ot": results["num_ot"],
        }
    )
    losers = pd.DataFrame(
        {
            "gender": results["gender"],
            "season": results["season"],
            "day_num": results["day_num"],
            "team_id": results["lteam_id"],
            "opp_team_id": results["wteam_id"],
            "is_win": 0.0,
            "points_for": results["lscore"],
            "points_against": results["wscore"],
            "score_diff": results["lscore"] - results["wscore"],
            "loc": results["wloc"].map(_reverse_location),
            "num_ot": results["num_ot"],
        }
    )
    team_games = pd.concat([winners, losers], ignore_index=True)
    team_games["is_home"] = (team_games["loc"] == "H").astype(float)
    team_games["is_away"] = (team_games["loc"] == "A").astype(float)
    team_games["is_neutral"] = (team_games["loc"] == "N").astype(float)
    return team_games.sort_values(["gender", "season", "team_id", "day_num"]).reset_index(drop=True)


def compute_elo_ratings(results: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, float | int | str]] = []
    for (gender, season), season_df in results.groupby(["gender", "season"], sort=True):
        ratings: dict[int, float] = {}
        ordered_games = season_df.sort_values(["day_num", "wteam_id", "lteam_id"])
        for row in ordered_games.itertuples(index=False):
            winner_rating = ratings.get(row.wteam_id, BASE_ELO)
            loser_rating = ratings.get(row.lteam_id, BASE_ELO)

            if row.wloc == "H":
                winner_adjustment = HOME_ADVANTAGE_ELO
            elif row.wloc == "A":
                winner_adjustment = -HOME_ADVANTAGE_ELO
            else:
                winner_adjustment = 0.0

            expected_win = 1.0 / (1.0 + 10.0 ** ((loser_rating - (winner_rating + winner_adjustment)) / 400.0))
            delta = ELO_K_FACTOR * (1.0 - expected_win)
            ratings[row.wteam_id] = winner_rating + delta
            ratings[row.lteam_id] = loser_rating - delta

        for team_id, rating in ratings.items():
            records.append(
                {
                    "gender": gender,
                    "season": int(season),
                    "team_id": int(team_id),
                    "elo_rating": float(rating),
                }
            )
    return pd.DataFrame(records)


def build_team_stats(regular_season_results: pd.DataFrame, seeds: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    game_log = build_team_game_log(regular_season_results)
    group_keys = ["gender", "season", "team_id"]

    base = (
        game_log.groupby(group_keys, as_index=False)
        .agg(
            games_played=("is_win", "size"),
            wins=("is_win", "sum"),
            avg_points_for=("points_for", "mean"),
            avg_points_against=("points_against", "mean"),
            avg_score_diff=("score_diff", "mean"),
            score_diff_std=("score_diff", "std"),
            avg_num_ot=("num_ot", "mean"),
        )
        .reset_index(drop=True)
    )
    base["win_pct"] = base["wins"] / base["games_played"].clip(lower=1)
    base["score_diff_std"] = base["score_diff_std"].fillna(0.0)

    location_rates = (
        game_log.groupby(group_keys, as_index=False)
        .agg(
            home_win_pct=("is_win", lambda s: float(s[game_log.loc[s.index, "is_home"] == 1.0].mean())),
            away_win_pct=("is_win", lambda s: float(s[game_log.loc[s.index, "is_away"] == 1.0].mean())),
            neutral_win_pct=("is_win", lambda s: float(s[game_log.loc[s.index, "is_neutral"] == 1.0].mean())),
        )
        .fillna(0.5)
    )

    recent = (
        game_log.sort_values(["gender", "season", "team_id", "day_num"])
        .groupby(group_keys, group_keys=False)
        .tail(10)
        .groupby(group_keys, as_index=False)
        .agg(
            last10_win_pct=("is_win", "mean"),
            last10_score_diff=("score_diff", "mean"),
        )
    )

    opponent_win_pct = base[group_keys + ["win_pct", "avg_score_diff"]].rename(
        columns={
            "team_id": "opp_team_id",
            "win_pct": "opp_win_pct",
            "avg_score_diff": "opp_score_diff",
        }
    )
    opponent_strength = (
        game_log.merge(opponent_win_pct, on=["gender", "season", "opp_team_id"], how="left")
        .groupby(group_keys, as_index=False)
        .agg(
            opp_win_pct=("opp_win_pct", "mean"),
            opp_score_diff=("opp_score_diff", "mean"),
        )
    )

    elo = compute_elo_ratings(regular_season_results)

    team_stats = base.merge(location_rates, on=group_keys, how="left")
    team_stats = team_stats.merge(recent, on=group_keys, how="left")
    team_stats = team_stats.merge(opponent_strength, on=group_keys, how="left")
    team_stats = team_stats.merge(elo, on=group_keys, how="left")
    team_stats = team_stats.merge(seeds[group_keys + ["seed_rank", "seed_missing"]], on=group_keys, how="left")
    team_stats = team_stats.merge(teams, on=["gender", "team_id"], how="left")

    fill_defaults = {
        "home_win_pct": 0.5,
        "away_win_pct": 0.5,
        "neutral_win_pct": 0.5,
        "last10_win_pct": team_stats["win_pct"].fillna(0.5),
        "last10_score_diff": team_stats["avg_score_diff"].fillna(0.0),
        "opp_win_pct": 0.5,
        "opp_score_diff": 0.0,
        "elo_rating": BASE_ELO,
        "seed_rank": 17.0,
        "seed_missing": 1.0,
    }
    for column, default_value in fill_defaults.items():
        team_stats[column] = team_stats[column].fillna(default_value)

    return team_stats.sort_values(["gender", "season", "team_id"]).reset_index(drop=True)


def build_historical_tournament_matchups(tourney_results: pd.DataFrame) -> pd.DataFrame:
    team1 = np.minimum(tourney_results["wteam_id"], tourney_results["lteam_id"])
    team2 = np.maximum(tourney_results["wteam_id"], tourney_results["lteam_id"])
    label = (tourney_results["wteam_id"] == team1).astype(float)

    return pd.DataFrame(
        {
            "season": tourney_results["season"].astype(int),
            "gender": tourney_results["gender"].astype(str),
            "team1_id": team1.astype(int),
            "team2_id": team2.astype(int),
            "label": label.astype(float),
        }
    )


def parse_sample_submission(sample_submission_path: Path) -> pd.DataFrame:
    df = pd.read_csv(sample_submission_path)
    if "ID" not in df.columns:
        raise ValueError(f"{sample_submission_path} must include an ID column.")

    parts = df["ID"].astype(str).str.split("_", expand=True)
    if parts.shape[1] != 3:
        raise ValueError("Sample submission IDs must look like 'Season_Team1_Team2'.")

    out = pd.DataFrame(
        {
            "ID": df["ID"].astype(str),
            "season": parts[0].astype(int),
            "team1_id": parts[1].astype(int),
            "team2_id": parts[2].astype(int),
        }
    )
    return out


def infer_submission_gender(matchups: pd.DataFrame, team_stats: pd.DataFrame) -> pd.Series:
    men_membership = set(
        zip(
            team_stats.loc[team_stats["gender"] == "M", "season"],
            team_stats.loc[team_stats["gender"] == "M", "team_id"],
        )
    )
    women_membership = set(
        zip(
            team_stats.loc[team_stats["gender"] == "W", "season"],
            team_stats.loc[team_stats["gender"] == "W", "team_id"],
        )
    )

    genders: list[str] = []
    for row in matchups.itertuples(index=False):
        men_match = (row.season, row.team1_id) in men_membership and (row.season, row.team2_id) in men_membership
        women_match = (row.season, row.team1_id) in women_membership and (row.season, row.team2_id) in women_membership
        if men_match and not women_match:
            genders.append("M")
        elif women_match and not men_match:
            genders.append("W")
        else:
            raise ValueError(
                f"Could not infer gender for sample submission row {row}. Check that downloaded data is complete."
            )
    return pd.Series(genders, index=matchups.index, dtype="object")


def build_matchup_features(matchups: pd.DataFrame, team_stats: pd.DataFrame) -> pd.DataFrame:
    team1_stats = team_stats[["gender", "season", "team_id", *TEAM_STAT_COLUMNS]].rename(
        columns={"team_id": "team1_id", **{column: f"team1_{column}" for column in TEAM_STAT_COLUMNS}}
    )
    team2_stats = team_stats[["gender", "season", "team_id", *TEAM_STAT_COLUMNS]].rename(
        columns={"team_id": "team2_id", **{column: f"team2_{column}" for column in TEAM_STAT_COLUMNS}}
    )

    feature_frame = matchups.merge(team1_stats, on=["gender", "season", "team1_id"], how="left")
    feature_frame = feature_frame.merge(team2_stats, on=["gender", "season", "team2_id"], how="left")

    missing_team1 = feature_frame.filter(like="team1_").isna().all(axis=1)
    missing_team2 = feature_frame.filter(like="team2_").isna().all(axis=1)
    if missing_team1.any() or missing_team2.any():
        raise ValueError("Some matchup teams are missing engineered stats. Check the raw competition files.")

    for column in TEAM_STAT_COLUMNS:
        feature_frame[f"diff_{column}"] = feature_frame[f"team1_{column}"] - feature_frame[f"team2_{column}"]

    feature_frame["seed_sum"] = feature_frame["team1_seed_rank"] + feature_frame["team2_seed_rank"]
    feature_frame["seed_abs_gap"] = feature_frame["diff_seed_rank"].abs()
    feature_frame["elo_abs_gap"] = feature_frame["diff_elo_rating"].abs()
    feature_frame["gender_is_women"] = (feature_frame["gender"] == "W").astype(float)

    return feature_frame


def build_datasets(files: CompetitionFiles) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    men_regular = load_compact_results(files.men_regular_season_results, gender="M")
    women_regular = load_compact_results(files.women_regular_season_results, gender="W")
    men_tourney = load_compact_results(files.men_tourney_results, gender="M")
    women_tourney = load_compact_results(files.women_tourney_results, gender="W")

    regular_results = pd.concat([men_regular, women_regular], ignore_index=True)
    tourney_results = pd.concat([men_tourney, women_tourney], ignore_index=True)
    seeds = pd.concat([load_seeds(files.men_seeds, "M"), load_seeds(files.women_seeds, "W")], ignore_index=True)
    teams = pd.concat([load_teams(files.men_teams, "M"), load_teams(files.women_teams, "W")], ignore_index=True)

    team_stats = build_team_stats(regular_season_results=regular_results, seeds=seeds, teams=teams)
    historical_matchups = build_historical_tournament_matchups(tourney_results)
    training_frame = build_matchup_features(historical_matchups, team_stats)

    submission_matchups = parse_sample_submission(files.sample_submission)
    submission_matchups["gender"] = infer_submission_gender(submission_matchups, team_stats)
    submission_frame = build_matchup_features(submission_matchups, team_stats)
    return training_frame, submission_frame, team_stats


def feature_columns() -> list[str]:
    columns = ["gender_is_women"]
    columns.extend([f"team1_{column}" for column in TEAM_STAT_COLUMNS])
    columns.extend([f"team2_{column}" for column in TEAM_STAT_COLUMNS])
    columns.extend([f"diff_{column}" for column in TEAM_STAT_COLUMNS])
    columns.extend(["seed_sum", "seed_abs_gap", "elo_abs_gap"])
    return columns


def default_holdout_season(training_frame: pd.DataFrame, target_season: int) -> int:
    valid_seasons = sorted(int(season) for season in training_frame["season"].unique() if int(season) < target_season)
    if not valid_seasons:
        raise ValueError(f"No historical seasons are available before target season {target_season}.")
    return valid_seasons[-1]


def fit_ensemble(training_frame: pd.DataFrame) -> dict[str, Any]:
    columns = feature_columns()
    fill_values = training_frame[columns].median(numeric_only=True)
    x_train = training_frame[columns].fillna(fill_values)
    y_train = training_frame["label"].astype(int)

    logistic_model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=0.15,
                    max_iter=4000,
                    solver="lbfgs",
                    random_state=42,
                ),
            ),
        ]
    )
    boosting_model = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_depth=3,
        max_iter=300,
        min_samples_leaf=20,
        random_state=42,
    )

    logistic_model.fit(x_train, y_train)
    boosting_model.fit(x_train, y_train)

    return {
        "feature_columns": columns,
        "fill_values": fill_values.to_dict(),
        "logistic_model": logistic_model,
        "boosting_model": boosting_model,
    }


def predict_ensemble(model_bundle: dict[str, Any], feature_frame: pd.DataFrame) -> np.ndarray:
    columns = model_bundle["feature_columns"]
    fill_values = pd.Series(model_bundle["fill_values"])
    x_frame = feature_frame[columns].fillna(fill_values)

    logistic_probs = model_bundle["logistic_model"].predict_proba(x_frame)[:, 1]
    boosting_probs = model_bundle["boosting_model"].predict_proba(x_frame)[:, 1]
    ensemble_probs = 0.5 * logistic_probs + 0.5 * boosting_probs
    return np.clip(ensemble_probs, 0.025, 0.975)


def evaluate_predictions(feature_frame: pd.DataFrame, predictions: np.ndarray) -> dict[str, Any]:
    truth = feature_frame["label"].astype(float).to_numpy()
    output: dict[str, Any] = {
        "rows": int(len(feature_frame)),
        "mse": float(mean_squared_error(truth, predictions)),
        "log_loss": float(log_loss(truth, predictions, labels=[0, 1])),
        "accuracy": float(accuracy_score(truth, predictions >= 0.5)),
    }

    by_gender: dict[str, Any] = {}
    for gender, group in feature_frame.groupby("gender"):
        gender_truth = group["label"].astype(float).to_numpy()
        gender_predictions = predictions[group.index.to_numpy()]
        by_gender[gender] = {
            "rows": int(len(group)),
            "mse": float(mean_squared_error(gender_truth, gender_predictions)),
            "log_loss": float(log_loss(gender_truth, gender_predictions, labels=[0, 1])),
            "accuracy": float(accuracy_score(gender_truth, gender_predictions >= 0.5)),
        }
    output["by_gender"] = by_gender
    return output


def fit_and_score_holdout(
    training_frame: pd.DataFrame,
    target_season: int,
    holdout_season: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame]:
    holdout = holdout_season or default_holdout_season(training_frame, target_season)
    train_rows = training_frame[training_frame["season"] < holdout].reset_index(drop=True)
    holdout_rows = training_frame[training_frame["season"] == holdout].reset_index(drop=True)
    if train_rows.empty or holdout_rows.empty:
        raise ValueError(f"Holdout season {holdout} does not produce a usable train/eval split.")

    model_bundle = fit_ensemble(train_rows)
    holdout_predictions = predict_ensemble(model_bundle, holdout_rows)
    metrics = evaluate_predictions(holdout_rows, holdout_predictions)
    scored_holdout = holdout_rows[["season", "gender", "team1_id", "team2_id", "label"]].copy()
    scored_holdout["prediction"] = holdout_predictions
    return model_bundle, metrics, scored_holdout


def fit_final_model(training_frame: pd.DataFrame, target_season: int) -> dict[str, Any]:
    final_training_rows = training_frame[training_frame["season"] < target_season].reset_index(drop=True)
    if final_training_rows.empty:
        raise ValueError(f"No training rows are available before target season {target_season}.")
    model_bundle = fit_ensemble(final_training_rows)
    model_bundle["train_seasons"] = sorted(int(season) for season in final_training_rows["season"].unique())
    model_bundle["target_season"] = int(target_season)
    return model_bundle


def generate_submission(model_bundle: dict[str, Any], submission_frame: pd.DataFrame) -> pd.DataFrame:
    predictions = predict_ensemble(model_bundle, submission_frame)
    return pd.DataFrame({"ID": submission_frame["ID"], "Pred": predictions})


def save_model_bundle(model_bundle: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_bundle, path)


def save_metrics(metrics: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
