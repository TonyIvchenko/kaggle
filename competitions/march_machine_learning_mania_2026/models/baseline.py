"""Feature engineering and baseline modeling for March Machine Learning Mania 2026."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, mean_squared_error
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset


COMPETITION_SLUG = "march-machine-learning-mania-2026"
COMPETITION_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = COMPETITION_ROOT / "data" / "raw"
DEFAULT_PROCESSED_DIR = COMPETITION_ROOT / "data" / "processed"
DEFAULT_MODEL_PATH = COMPETITION_ROOT / "models" / "march_machine_learning_mania_2026.pt"
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


@dataclass(frozen=True)
class TorchCandidateConfig:
    name: str
    architecture: str
    hidden_dims: tuple[int, ...]
    dropout: float
    learning_rate: float
    weight_decay: float
    batch_size: int
    epochs: int
    patience: int
    label_smoothing: float = 0.0


class ResidualBlock(nn.Module):
    def __init__(self, width: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.fc1 = nn.Linear(width, width)
        self.fc2 = nn.Linear(width, width)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.norm(x)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return residual + x


class MatchupNet(nn.Module):
    def __init__(self, input_dim: int, config: TorchCandidateConfig) -> None:
        super().__init__()
        self.architecture = config.architecture

        if config.architecture == "linear":
            self.encoder = nn.Identity()
            self.residual_stack = nn.Identity()
            self.head = nn.Linear(input_dim, 1)
            return

        if not config.hidden_dims:
            raise ValueError(f"Architecture {config.architecture} requires hidden_dims.")

        layers: list[nn.Module] = []
        current_dim = input_dim
        for width in config.hidden_dims:
            layers.extend(
                [
                    nn.Linear(current_dim, width),
                    nn.LayerNorm(width),
                    nn.GELU(),
                    nn.Dropout(config.dropout),
                ]
            )
            current_dim = width
        self.encoder = nn.Sequential(*layers)

        if config.architecture == "residual":
            self.residual_stack = nn.Sequential(*[ResidualBlock(current_dim, config.dropout) for _ in range(2)])
        else:
            self.residual_stack = nn.Identity()

        self.head = nn.Linear(current_dim, 1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.encoder(x)
        x = self.residual_stack(x)
        return self.head(x)


def default_torch_candidates() -> list[TorchCandidateConfig]:
    return [
        TorchCandidateConfig(
            name="linear_l2",
            architecture="linear",
            hidden_dims=(),
            dropout=0.0,
            learning_rate=3e-3,
            weight_decay=5e-4,
            batch_size=256,
            epochs=260,
            patience=35,
            label_smoothing=0.01,
        ),
        TorchCandidateConfig(
            name="mlp_wider",
            architecture="mlp",
            hidden_dims=(320, 160, 64),
            dropout=0.25,
            learning_rate=9e-4,
            weight_decay=1.5e-3,
            batch_size=256,
            epochs=340,
            patience=45,
            label_smoothing=0.02,
        ),
        TorchCandidateConfig(
            name="mlp_deep",
            architecture="mlp",
            hidden_dims=(256, 128, 128, 64),
            dropout=0.30,
            learning_rate=9e-4,
            weight_decay=2e-3,
            batch_size=256,
            epochs=340,
            patience=45,
            label_smoothing=0.03,
        ),
        TorchCandidateConfig(
            name="residual_wide",
            architecture="residual",
            hidden_dims=(256, 256),
            dropout=0.25,
            learning_rate=9e-4,
            weight_decay=1.2e-3,
            batch_size=256,
            epochs=380,
            patience=55,
            label_smoothing=0.02,
        ),
        TorchCandidateConfig(
            name="residual",
            architecture="residual",
            hidden_dims=(192, 192),
            dropout=0.20,
            learning_rate=1.1e-3,
            weight_decay=1e-3,
            batch_size=256,
            epochs=360,
            patience=50,
            label_smoothing=0.02,
        ),
    ]


def resolve_device(preferred: str = "auto") -> torch.device:
    pref = preferred.lower()
    if pref == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if pref == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS device requested but not available on this machine.")
        return torch.device("mps")
    if pref == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unsupported device preference: {preferred}")


def _feature_fill_values(training_frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    return training_frame[columns].median(numeric_only=True)


def _feature_array(frame: pd.DataFrame, columns: list[str], fill_values: pd.Series) -> np.ndarray:
    return frame[columns].fillna(fill_values).to_numpy(dtype=np.float32, copy=True)


def _norm_stats(x_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def _to_tensor(x: np.ndarray, device: torch.device) -> Tensor:
    return torch.tensor(x, dtype=torch.float32, device=device)


def _evaluate_probs(y_true: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    clipped = np.clip(probs, 1e-5, 1 - 1e-5)
    return {
        "mse": float(mean_squared_error(y_true, probs)),
        "log_loss": float(log_loss(y_true, clipped, labels=[0, 1])),
        "accuracy": float(accuracy_score(y_true, probs >= 0.5)),
    }


def _train_torch_candidate(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    config: TorchCandidateConfig,
    device: torch.device,
    seed: int,
    norm_mean: np.ndarray,
    norm_std: np.ndarray,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    if device.type == "mps":
        torch.mps.manual_seed(seed)

    train_x_norm = ((train_x - norm_mean) / norm_std).astype(np.float32)
    val_x_norm = ((val_x - norm_mean) / norm_std).astype(np.float32)

    train_features = torch.tensor(train_x_norm, dtype=torch.float32)
    train_targets = torch.tensor(train_y.reshape(-1, 1), dtype=torch.float32)
    loader = DataLoader(TensorDataset(train_features, train_targets), batch_size=config.batch_size, shuffle=True)

    model = MatchupNet(input_dim=train_x.shape[1], config=config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    pos_count = max(float(train_y.sum()), 1.0)
    neg_count = max(float((1.0 - train_y).sum()), 1.0)
    pos_weight = torch.tensor([neg_count / pos_count], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    val_tensor = _to_tensor(val_x_norm, device=device)
    val_targets = val_y.astype(np.float32)

    best_mse = float("inf")
    best_state: dict[str, Tensor] | None = None
    best_probs: np.ndarray | None = None
    best_epoch = 0
    stale_epochs = 0

    for epoch in range(1, config.epochs + 1):
        model.train()
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            smooth = config.label_smoothing
            if smooth > 0:
                batch_y = batch_y * (1.0 - 2.0 * smooth) + smooth
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.5)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            probs = torch.sigmoid(model(val_tensor)).detach().cpu().numpy().reshape(-1)
        mse = float(mean_squared_error(val_targets, probs))
        if mse < best_mse - 1e-6:
            best_mse = mse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_probs = probs
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= config.patience:
                break

    if best_state is None or best_probs is None:
        raise RuntimeError(f"Training failed for candidate {config.name}.")

    metrics = _evaluate_probs(val_targets, best_probs)
    return {
        "config": config,
        "state_dict": best_state,
        "val_predictions": best_probs.astype(np.float32),
        "val_metrics": metrics,
        "best_epoch": best_epoch,
    }


def _candidate_config_from_dict(raw: dict[str, Any]) -> TorchCandidateConfig:
    return TorchCandidateConfig(
        name=str(raw["name"]),
        architecture=str(raw["architecture"]),
        hidden_dims=tuple(int(x) for x in raw["hidden_dims"]),
        dropout=float(raw["dropout"]),
        learning_rate=float(raw["learning_rate"]),
        weight_decay=float(raw["weight_decay"]),
        batch_size=int(raw["batch_size"]),
        epochs=int(raw["epochs"]),
        patience=int(raw["patience"]),
        label_smoothing=float(raw.get("label_smoothing", 0.0)),
    )


def _candidate_to_dict(config: TorchCandidateConfig) -> dict[str, Any]:
    return {
        "name": config.name,
        "architecture": config.architecture,
        "hidden_dims": list(config.hidden_dims),
        "dropout": config.dropout,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "batch_size": config.batch_size,
        "epochs": config.epochs,
        "patience": config.patience,
        "label_smoothing": config.label_smoothing,
    }


def _select_top_candidates(
    results: list[dict[str, Any]],
    y_true: np.ndarray | None = None,
    max_models: int = 3,
) -> list[dict[str, Any]]:
    if not results:
        raise ValueError("No candidate training results are available.")

    ordered = sorted(results, key=lambda r: (r["val_metrics"]["mse"], r["val_metrics"]["log_loss"]))
    model_count = max(1, min(max_models, len(ordered)))
    if y_true is None or len(ordered) == 1:
        return ordered[:model_count]

    pool = ordered[: min(len(ordered), 8)]
    best_combo: tuple[float, float, tuple[dict[str, Any], ...]] | None = None
    for size in range(1, model_count + 1):
        for combo in itertools.combinations(pool, size):
            blended = np.mean(np.stack([entry["val_predictions"] for entry in combo], axis=0), axis=0)
            blended = np.clip(blended, 0.025, 0.975)
            mse = float(mean_squared_error(y_true, blended))
            ll = float(log_loss(y_true, np.clip(blended, 1e-5, 1.0 - 1e-5), labels=[0, 1]))
            score = (mse, ll, combo)
            if best_combo is None or score[:2] < best_combo[:2]:
                best_combo = score

    if best_combo is None:
        return ordered[:model_count]
    return list(best_combo[2])


def _bundle_predictions(
    model_entries: list[dict[str, Any]],
    x_data: np.ndarray,
    norm_mean: np.ndarray,
    norm_std: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    x_norm = ((x_data - norm_mean) / norm_std).astype(np.float32)
    x_tensor = _to_tensor(x_norm, device=device)

    probs: list[np.ndarray] = []
    for entry in model_entries:
        config = _candidate_config_from_dict(entry["config"])
        model = MatchupNet(input_dim=x_data.shape[1], config=config).to(device)
        model.load_state_dict(entry["state_dict"])
        model.eval()
        with torch.no_grad():
            p = torch.sigmoid(model(x_tensor)).detach().cpu().numpy().reshape(-1)
        probs.append(p)
    blended = np.mean(np.stack(probs, axis=0), axis=0)
    return np.clip(blended, 0.025, 0.975)


def predict_ensemble(model_bundle: dict[str, Any], feature_frame: pd.DataFrame, device_preference: str = "auto") -> np.ndarray:
    if model_bundle.get("split_by_gender") and "models_by_gender" in model_bundle:
        gender_values = feature_frame["gender"].astype(str).to_numpy()
        predictions = np.zeros(len(feature_frame), dtype=np.float32)
        covered = np.zeros(len(feature_frame), dtype=bool)

        for gender, gender_bundle in model_bundle["models_by_gender"].items():
            mask = gender_values == str(gender)
            if not mask.any():
                continue
            covered |= mask
            gender_frame = feature_frame.loc[mask].reset_index(drop=True)
            predictions[mask] = predict_ensemble(
                gender_bundle,
                gender_frame,
                device_preference=device_preference,
            )

        if not covered.all():
            missing_genders = sorted(pd.unique(gender_values[~covered]).tolist())
            raise ValueError(f"No gender-specific model is available for genders: {missing_genders}")
        return np.clip(predictions, 0.025, 0.975)

    columns = model_bundle["feature_columns"]
    fill_values = pd.Series(model_bundle["fill_values"])
    x_data = _feature_array(feature_frame, columns, fill_values)

    norm_mean = np.asarray(model_bundle["norm_mean"], dtype=np.float32)
    norm_std = np.asarray(model_bundle["norm_std"], dtype=np.float32)
    device = resolve_device(device_preference)
    return _bundle_predictions(model_bundle["models"], x_data, norm_mean, norm_std, device)


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
    device_preference: str = "auto",
    seed: int = 42,
    candidate_configs: list[TorchCandidateConfig] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame]:
    holdout = holdout_season or default_holdout_season(training_frame, target_season)
    train_rows = training_frame[training_frame["season"] < holdout].reset_index(drop=True)
    holdout_rows = training_frame[training_frame["season"] == holdout].reset_index(drop=True)
    if train_rows.empty or holdout_rows.empty:
        raise ValueError(f"Holdout season {holdout} does not produce a usable train/eval split.")

    columns = feature_columns()
    fill_values = _feature_fill_values(train_rows, columns)
    train_x = _feature_array(train_rows, columns, fill_values)
    holdout_x = _feature_array(holdout_rows, columns, fill_values)
    train_y = train_rows["label"].to_numpy(dtype=np.float32)
    holdout_y = holdout_rows["label"].to_numpy(dtype=np.float32)

    norm_mean, norm_std = _norm_stats(train_x)
    device = resolve_device(device_preference)
    configs = candidate_configs or default_torch_candidates()
    results = [
        _train_torch_candidate(
            train_x=train_x,
            train_y=train_y,
            val_x=holdout_x,
            val_y=holdout_y,
            config=config,
            device=device,
            seed=seed,
            norm_mean=norm_mean,
            norm_std=norm_std,
        )
        for config in configs
    ]

    selected = _select_top_candidates(results, y_true=holdout_y)
    model_entries = [
        {
            "config": _candidate_to_dict(result["config"]),
            "state_dict": result["state_dict"],
            "best_epoch": result["best_epoch"],
        }
        for result in selected
    ]
    holdout_predictions = _bundle_predictions(model_entries, holdout_x, norm_mean, norm_std, device)

    model_bundle = {
        "framework": "pytorch",
        "feature_columns": columns,
        "fill_values": fill_values.to_dict(),
        "norm_mean": norm_mean.tolist(),
        "norm_std": norm_std.tolist(),
        "models": model_entries,
        "selected_model_names": [entry["config"]["name"] for entry in model_entries],
        "holdout_season": int(holdout),
        "device_used": str(device),
    }
    metrics = evaluate_predictions(holdout_rows, holdout_predictions)
    metrics["candidate_metrics"] = [
        {
            "name": result["config"].name,
            "architecture": result["config"].architecture,
            "best_epoch": int(result["best_epoch"]),
            **result["val_metrics"],
        }
        for result in sorted(results, key=lambda r: (r["val_metrics"]["mse"], r["val_metrics"]["log_loss"]))
    ]
    metrics["selected_models"] = model_bundle["selected_model_names"]

    scored_holdout = holdout_rows[["season", "gender", "team1_id", "team2_id", "label"]].copy()
    scored_holdout["prediction"] = holdout_predictions
    return model_bundle, metrics, scored_holdout


def fit_final_model(
    training_frame: pd.DataFrame,
    target_season: int,
    device_preference: str = "auto",
    seed: int = 42,
    selected_configs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    final_training_rows = training_frame[training_frame["season"] < target_season].reset_index(drop=True)
    if final_training_rows.empty:
        raise ValueError(f"No training rows are available before target season {target_season}.")

    columns = feature_columns()
    fill_values = _feature_fill_values(final_training_rows, columns)
    train_x = _feature_array(final_training_rows, columns, fill_values)
    train_y = final_training_rows["label"].to_numpy(dtype=np.float32)
    norm_mean, norm_std = _norm_stats(train_x)
    device = resolve_device(device_preference)

    if selected_configs is None:
        configs = default_torch_candidates()[:3]
    else:
        configs = [_candidate_config_from_dict(cfg) for cfg in selected_configs]

    model_entries: list[dict[str, Any]] = []
    for i, config in enumerate(configs):
        result = _train_torch_candidate(
            train_x=train_x,
            train_y=train_y,
            val_x=train_x,
            val_y=train_y,
            config=config,
            device=device,
            seed=seed + i,
            norm_mean=norm_mean,
            norm_std=norm_std,
        )
        model_entries.append(
            {
                "config": _candidate_to_dict(config),
                "state_dict": result["state_dict"],
                "best_epoch": result["best_epoch"],
            }
        )

    return {
        "framework": "pytorch",
        "feature_columns": columns,
        "fill_values": fill_values.to_dict(),
        "norm_mean": norm_mean.tolist(),
        "norm_std": norm_std.tolist(),
        "models": model_entries,
        "train_seasons": sorted(int(season) for season in final_training_rows["season"].unique()),
        "target_season": int(target_season),
        "selected_model_names": [entry["config"]["name"] for entry in model_entries],
        "device_used": str(device),
    }


def fit_and_score_holdout_by_gender(
    training_frame: pd.DataFrame,
    target_season: int,
    holdout_season: int | None = None,
    device_preference: str = "auto",
    seed: int = 42,
    candidate_configs: list[TorchCandidateConfig] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame]:
    holdout = holdout_season or default_holdout_season(training_frame, target_season)
    gender_bundles: dict[str, dict[str, Any]] = {}
    gender_metrics: dict[str, dict[str, Any]] = {}
    scored_parts: list[pd.DataFrame] = []

    for gender in sorted(str(value) for value in training_frame["gender"].unique()):
        subset = training_frame[training_frame["gender"] == gender].reset_index(drop=True)
        if subset.empty:
            continue
        bundle, metrics, scored = fit_and_score_holdout(
            training_frame=subset,
            target_season=target_season,
            holdout_season=holdout,
            device_preference=device_preference,
            seed=seed,
            candidate_configs=candidate_configs,
        )
        gender_bundles[gender] = bundle
        gender_metrics[gender] = metrics
        scored_parts.append(scored)

    if not scored_parts:
        raise ValueError("No gender-specific training data is available for holdout evaluation.")

    scored_holdout = pd.concat(scored_parts, ignore_index=True)
    metrics = evaluate_predictions(scored_holdout, scored_holdout["prediction"].to_numpy(dtype=np.float32))
    metrics["candidate_metrics_by_gender"] = {
        gender: gender_metrics[gender]["candidate_metrics"] for gender in sorted(gender_metrics)
    }
    metrics["selected_models"] = {
        gender: gender_bundles[gender]["selected_model_names"] for gender in sorted(gender_bundles)
    }

    model_bundle = {
        "framework": "pytorch",
        "split_by_gender": True,
        "models_by_gender": gender_bundles,
        "selected_model_names_by_gender": metrics["selected_models"],
        "holdout_season": int(holdout),
        "device_used": str(resolve_device(device_preference)),
    }
    return model_bundle, metrics, scored_holdout


def fit_final_model_by_gender(
    training_frame: pd.DataFrame,
    target_season: int,
    device_preference: str = "auto",
    seed: int = 42,
    selected_configs_by_gender: dict[str, list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    gender_bundles: dict[str, dict[str, Any]] = {}
    for gender in sorted(str(value) for value in training_frame["gender"].unique()):
        subset = training_frame[training_frame["gender"] == gender].reset_index(drop=True)
        if subset.empty:
            continue
        selected_configs = None
        if selected_configs_by_gender is not None:
            selected_configs = selected_configs_by_gender.get(gender)
        gender_bundles[gender] = fit_final_model(
            training_frame=subset,
            target_season=target_season,
            device_preference=device_preference,
            seed=seed,
            selected_configs=selected_configs,
        )

    if not gender_bundles:
        raise ValueError("No gender-specific training data is available for final model training.")

    train_seasons = sorted(int(season) for season in training_frame["season"].unique() if int(season) < target_season)
    return {
        "framework": "pytorch",
        "split_by_gender": True,
        "models_by_gender": gender_bundles,
        "target_season": int(target_season),
        "train_seasons": train_seasons,
        "selected_model_names_by_gender": {
            gender: bundle["selected_model_names"] for gender, bundle in sorted(gender_bundles.items())
        },
        "device_used": str(resolve_device(device_preference)),
    }


def generate_submission(
    model_bundle: dict[str, Any], submission_frame: pd.DataFrame, device_preference: str = "auto"
) -> pd.DataFrame:
    predictions = predict_ensemble(model_bundle, submission_frame, device_preference=device_preference)
    return pd.DataFrame({"ID": submission_frame["ID"], "Pred": predictions})


def save_model_bundle(model_bundle: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model_bundle, path)


def save_metrics(metrics: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
