"""Feature preparation and baseline modeling for Deep Past Initiative translation."""

from __future__ import annotations

import itertools
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
import random
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset


COMPETITION_SLUG = "deep-past-initiative-machine-translation"
COMPETITION_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = COMPETITION_ROOT / "data" / "raw"
DEFAULT_PROCESSED_DIR = COMPETITION_ROOT / "data" / "processed"
DEFAULT_MODEL_PATH = COMPETITION_ROOT / "models" / "deep_past_initiative_machine_translation.pt"
DEFAULT_METRICS_PATH = COMPETITION_ROOT / "models" / "deep_past_initiative_machine_translation_metrics.json"
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
    source_columns: tuple[str, ...]
    train_source_texts: list[str]
    test_source_texts: list[str]
    train_targets: list[str]


@dataclass(frozen=True)
class TfidfConfig:
    name: str = "tfidf_char_36"
    analyzer: str = "char_wb"
    ngram_min: int = 3
    ngram_max: int = 6
    min_df: int = 2
    max_features: int = 350_000


@dataclass(frozen=True)
class TorchRetrieverConfig:
    name: str
    embedding_dim: int
    hidden_dim: int
    dropout: float
    learning_rate: float
    weight_decay: float
    batch_size: int
    epochs: int
    patience: int
    max_length: int
    temperature: float = 0.07


def default_torch_configs() -> list[TorchRetrieverConfig]:
    return [
        TorchRetrieverConfig(
            name="torch_small",
            embedding_dim=96,
            hidden_dim=192,
            dropout=0.15,
            learning_rate=1.0e-3,
            weight_decay=8.0e-5,
            batch_size=128,
            epochs=16,
            patience=4,
            max_length=220,
            temperature=0.07,
        ),
        TorchRetrieverConfig(
            name="torch_medium",
            embedding_dim=128,
            hidden_dim=256,
            dropout=0.20,
            learning_rate=8.0e-4,
            weight_decay=1.0e-4,
            batch_size=128,
            epochs=22,
            patience=5,
            max_length=280,
            temperature=0.06,
        ),
    ]


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
        raise FileNotFoundError(f"No CSV files were found under {raw_dir}. Run the download script first.")

    return CompetitionFiles(
        raw_dir=raw_dir,
        train=_pick_file(csv_files, ("train.csv",)),
        test=_pick_file(csv_files, ("test.csv",)),
        sample_submission=_pick_file(csv_files, ("sample_submission.csv", "sampleSubmission.csv")),
    )


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    if text.lower() == "nan":
        return ""
    return " ".join(text.strip().split())


def _choose_id_column(train: pd.DataFrame, test: pd.DataFrame, sample_submission: pd.DataFrame) -> str:
    preferred = ("id", "ID", "Id")
    test_columns = set(test.columns)
    sample_columns = set(sample_submission.columns)
    for column in preferred:
        if column in test_columns and column in sample_columns:
            return column

    shared = [column for column in sample_submission.columns if column in test_columns]
    if not shared:
        raise ValueError("Could not infer id column: no shared columns between sample submission and test set.")

    for column in shared:
        if column in train.columns:
            return str(column)
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
    for column in candidates:
        if column in train.columns:
            return str(column)

    train_only = [column for column in train.columns if column not in test.columns and column != id_column]
    if len(train_only) == 1:
        return str(train_only[0])
    raise ValueError("Could not infer target column from train/test/sample_submission schema.")


def _choose_source_columns(train: pd.DataFrame, test: pd.DataFrame, id_column: str, target_column: str) -> tuple[str, ...]:
    columns = [column for column in test.columns if column != id_column and column in train.columns]
    columns = [column for column in columns if column != target_column]
    if not columns:
        fallback = [column for column in train.columns if column not in {id_column, target_column}]
        if not fallback:
            raise ValueError("Could not infer source columns.")
        columns = fallback
    return tuple(str(column) for column in columns)


def _combine_source_columns(frame: pd.DataFrame, source_columns: tuple[str, ...]) -> list[str]:
    parts = [frame[column].map(_clean_text) for column in source_columns]
    merged: list[str] = []
    for row_values in zip(*parts):
        tokens = [f"{column}:{value}" for column, value in zip(source_columns, row_values, strict=True) if value]
        merged.append(" | ".join(tokens) if tokens else "")
    return merged


def build_datasets(files: CompetitionFiles, target_column: str | None = None) -> DatasetBundle:
    train = pd.read_csv(files.train)
    test = pd.read_csv(files.test)
    sample_submission = pd.read_csv(files.sample_submission)

    id_column = _choose_id_column(train, test, sample_submission)
    target = _choose_target_column(
        train=train,
        test=test,
        sample_submission=sample_submission,
        id_column=id_column,
        target_override=target_column,
    )
    source_columns = _choose_source_columns(train, test, id_column=id_column, target_column=target)

    if target not in train.columns:
        raise ValueError(f"Target column '{target}' is not present in train.csv.")
    if id_column not in train.columns or id_column not in test.columns:
        raise ValueError(f"ID column '{id_column}' must be present in both train.csv and test.csv.")

    train_targets = train[target].map(_clean_text).tolist()
    train_sources = _combine_source_columns(train, source_columns=source_columns)
    test_sources = _combine_source_columns(test, source_columns=source_columns)

    return DatasetBundle(
        train_frame=train.copy(),
        test_frame=test.copy(),
        sample_submission=sample_submission.copy(),
        id_column=id_column,
        target_column=target,
        source_columns=source_columns,
        train_source_texts=train_sources,
        test_source_texts=test_sources,
        train_targets=train_targets,
    )


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


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def _safe_f1(precision: float, recall: float) -> float:
    if precision + recall <= 1e-12:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def character_f1_score(y_true: list[str], y_pred: list[str]) -> float:
    values: list[float] = []
    for truth, pred in zip(y_true, y_pred, strict=True):
        truth_counter = Counter(truth)
        pred_counter = Counter(pred)
        overlap = sum(min(pred_counter[ch], truth_counter[ch]) for ch in set(truth_counter) | set(pred_counter))
        precision = overlap / max(len(pred), 1)
        recall = overlap / max(len(truth), 1)
        values.append(_safe_f1(precision, recall))
    return float(np.mean(values)) if values else 0.0


def sequence_ratio(y_true: list[str], y_pred: list[str]) -> float:
    ratios: list[float] = []
    for truth, pred in zip(y_true, y_pred, strict=True):
        if not truth and not pred:
            ratios.append(1.0)
            continue
        max_len = max(len(truth), len(pred), 1)
        edit = _levenshtein_distance(truth, pred)
        ratios.append(1.0 - (edit / max_len))
    return float(np.mean(ratios)) if ratios else 0.0


def _levenshtein_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current = [i]
        for j, char_b in enumerate(b, start=1):
            delete_cost = prev[j] + 1
            insert_cost = current[j - 1] + 1
            sub_cost = prev[j - 1] + (0 if char_a == char_b else 1)
            current.append(min(delete_cost, insert_cost, sub_cost))
        prev = current
    return prev[-1]


def evaluate_text_predictions(y_true: list[str], y_pred: list[str]) -> dict[str, float]:
    exact = [float(truth == pred) for truth, pred in zip(y_true, y_pred, strict=True)]
    return {
        "rows": float(len(y_true)),
        "char_f1": character_f1_score(y_true, y_pred),
        "sequence_ratio": sequence_ratio(y_true, y_pred),
        "exact_match": float(np.mean(exact)) if exact else 0.0,
    }


def fit_tfidf_retriever(train_sources: list[str], train_targets: list[str], config: TfidfConfig) -> dict[str, Any]:
    vectorizer = TfidfVectorizer(
        analyzer=config.analyzer,
        ngram_range=(config.ngram_min, config.ngram_max),
        min_df=config.min_df,
        max_features=config.max_features,
        lowercase=False,
    )
    train_matrix = vectorizer.fit_transform(train_sources)
    return {
        "model_type": "tfidf",
        "config": asdict(config),
        "vectorizer": vectorizer,
        "train_matrix": train_matrix,
        "train_targets": np.asarray(train_targets, dtype=object),
    }


def tfidf_predict(model: dict[str, Any], source_texts: list[str]) -> tuple[list[str], np.ndarray]:
    query_matrix = model["vectorizer"].transform(source_texts)
    similarities = query_matrix @ model["train_matrix"].T
    max_values = similarities.max(axis=1)
    if hasattr(max_values, "toarray"):
        scores = np.asarray(max_values.toarray()).reshape(-1)
    else:
        scores = np.asarray(max_values).reshape(-1)
    best_indices = np.asarray(similarities.argmax(axis=1)).reshape(-1).astype(int)
    predictions = model["train_targets"][best_indices].tolist()
    return predictions, scores.astype(np.float32)


def _build_char_vocab(texts: list[str], max_size: int = 4096) -> dict[str, int]:
    counts = Counter("".join(texts))
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    tokens = [token for token, _ in ordered[: max_size - 2]]
    mapping = {"<pad>": 0, "<unk>": 1}
    for token in tokens:
        if token in mapping:
            continue
        mapping[token] = len(mapping)
    return mapping


def _encode_texts(texts: list[str], token_to_id: dict[str, int], max_length: int) -> np.ndarray:
    encoded = np.zeros((len(texts), max_length), dtype=np.int64)
    unk = token_to_id["<unk>"]
    for i, text in enumerate(texts):
        symbols = list(text)[:max_length]
        if not symbols:
            encoded[i, 0] = unk
            continue
        for j, symbol in enumerate(symbols):
            encoded[i, j] = token_to_id.get(symbol, unk)
    return encoded


class TextEncoder(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, token_ids: Tensor) -> Tensor:
        mask = (token_ids != 0).unsqueeze(-1)
        emb = self.embedding(token_ids)
        summed = (emb * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1)
        pooled = summed / denom
        vector = self.ffn(pooled)
        vector = self.norm(vector)
        return nn.functional.normalize(vector, dim=1)


class BiEncoderRetriever(nn.Module):
    def __init__(self, vocab_size: int, config: TorchRetrieverConfig) -> None:
        super().__init__()
        self.encoder = TextEncoder(
            vocab_size=vocab_size,
            embedding_dim=config.embedding_dim,
            hidden_dim=config.hidden_dim,
            dropout=config.dropout,
        )

    def encode_source(self, token_ids: Tensor) -> Tensor:
        return self.encoder(token_ids)

    def encode_target(self, token_ids: Tensor) -> Tensor:
        return self.encoder(token_ids)


def _to_tensor(array: np.ndarray, device: torch.device) -> Tensor:
    return torch.tensor(array, dtype=torch.long, device=device)


def _encode_embeddings(
    model: BiEncoderRetriever,
    token_ids: np.ndarray,
    device: torch.device,
    mode: str,
    batch_size: int = 1024,
) -> np.ndarray:
    outputs: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(token_ids), batch_size):
            batch = _to_tensor(token_ids[start : start + batch_size], device=device)
            if mode == "source":
                emb = model.encode_source(batch)
            else:
                emb = model.encode_target(batch)
            outputs.append(emb.detach().cpu().numpy())
    if not outputs:
        return np.zeros((0, model.encoder.norm.normalized_shape[0]), dtype=np.float32)
    return np.vstack(outputs).astype(np.float32)


def _torch_predict_from_arrays(
    model: BiEncoderRetriever,
    source_ids: np.ndarray,
    candidate_target_ids: np.ndarray,
    candidate_targets: list[str],
    device: torch.device,
) -> tuple[list[str], np.ndarray]:
    source_emb = _encode_embeddings(model, source_ids, device=device, mode="source")
    target_emb = _encode_embeddings(model, candidate_target_ids, device=device, mode="target")
    similarities = source_emb @ target_emb.T
    best_indices = similarities.argmax(axis=1)
    confidences = similarities[np.arange(len(similarities)), best_indices]
    predictions = [candidate_targets[index] for index in best_indices.tolist()]
    return predictions, confidences.astype(np.float32)


def fit_torch_retriever(
    train_sources: list[str],
    train_targets: list[str],
    eval_sources: list[str] | None,
    eval_targets: list[str] | None,
    config: TorchRetrieverConfig,
    device: torch.device,
    seed: int,
    forced_epochs: int | None = None,
) -> dict[str, Any]:
    _set_seed(seed)
    token_to_id = _build_char_vocab(train_sources + train_targets)

    train_source_ids = _encode_texts(train_sources, token_to_id=token_to_id, max_length=config.max_length)
    train_target_ids = _encode_texts(train_targets, token_to_id=token_to_id, max_length=config.max_length)
    eval_source_ids = (
        _encode_texts(eval_sources, token_to_id=token_to_id, max_length=config.max_length) if eval_sources else None
    )

    dataset = TensorDataset(torch.tensor(train_source_ids, dtype=torch.long), torch.tensor(train_target_ids, dtype=torch.long))
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, drop_last=False)

    model = BiEncoderRetriever(vocab_size=len(token_to_id), config=config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    epochs = max(1, forced_epochs or config.epochs)
    best_char_f1 = -1.0
    best_state: dict[str, Tensor] | None = None
    best_epoch = 1
    best_eval_predictions: list[str] | None = None
    best_eval_confidences: np.ndarray | None = None
    stale_epochs = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for batch_source, batch_target in loader:
            batch_source = batch_source.to(device)
            batch_target = batch_target.to(device)

            source_emb = model.encode_source(batch_source)
            target_emb = model.encode_target(batch_target)
            logits = (source_emb @ target_emb.T) / max(config.temperature, 1e-4)
            labels = torch.arange(logits.shape[0], device=device)

            loss_source = nn.functional.cross_entropy(logits, labels)
            loss_target = nn.functional.cross_entropy(logits.T, labels)
            loss = 0.5 * (loss_source + loss_target)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        if eval_source_ids is None or eval_targets is None:
            continue

        predictions, confidences = _torch_predict_from_arrays(
            model=model,
            source_ids=eval_source_ids,
            candidate_target_ids=train_target_ids,
            candidate_targets=train_targets,
            device=device,
        )
        metrics = evaluate_text_predictions(eval_targets, predictions)
        char_f1 = metrics["char_f1"]
        if char_f1 > best_char_f1 + 1e-8:
            best_char_f1 = char_f1
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_epoch = epoch
            best_eval_predictions = predictions
            best_eval_confidences = confidences
            stale_epochs = 0
        else:
            stale_epochs += 1
            if forced_epochs is None and stale_epochs >= config.patience:
                break

    if best_state is None:
        best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        best_epoch = epochs

    if eval_targets is not None and best_eval_predictions is None:
        model.load_state_dict(best_state)
        predictions, confidences = _torch_predict_from_arrays(
            model=model,
            source_ids=eval_source_ids if eval_source_ids is not None else train_source_ids,
            candidate_target_ids=train_target_ids,
            candidate_targets=train_targets,
            device=device,
        )
        best_eval_predictions = predictions
        best_eval_confidences = confidences

    eval_metrics = (
        evaluate_text_predictions(eval_targets, best_eval_predictions) if eval_targets is not None and best_eval_predictions else {}
    )

    return {
        "model_type": "torch",
        "config": asdict(config),
        "token_to_id": token_to_id,
        "state_dict": best_state,
        "max_length": int(config.max_length),
        "train_targets": list(train_targets),
        "best_epoch": int(best_epoch),
        "eval_predictions": best_eval_predictions,
        "eval_confidences": best_eval_confidences,
        "metrics": eval_metrics,
    }


def torch_predict(model: dict[str, Any], source_texts: list[str], device: torch.device) -> tuple[list[str], np.ndarray]:
    config = TorchRetrieverConfig(**model["config"])
    token_to_id = model["token_to_id"]
    source_ids = _encode_texts(source_texts, token_to_id=token_to_id, max_length=model["max_length"])
    target_ids = _encode_texts(model["train_targets"], token_to_id=token_to_id, max_length=model["max_length"])

    network = BiEncoderRetriever(vocab_size=len(token_to_id), config=config).to(device)
    network.load_state_dict(model["state_dict"])
    predictions, confidences = _torch_predict_from_arrays(
        model=network,
        source_ids=source_ids,
        candidate_target_ids=target_ids,
        candidate_targets=model["train_targets"],
        device=device,
    )
    return predictions, confidences


def _select_best_hybrid_threshold(
    y_true: list[str],
    tfidf_pred: list[str],
    tfidf_conf: np.ndarray,
    torch_pred: list[str],
) -> tuple[float, dict[str, float], list[str]]:
    thresholds = np.quantile(tfidf_conf, q=np.linspace(0.0, 1.0, 11))
    candidates = sorted(set(float(value) for value in thresholds))

    best_threshold = candidates[0]
    best_metrics: dict[str, float] | None = None
    best_predictions: list[str] | None = None
    for threshold in candidates:
        preds = [
            tfidf if conf >= threshold else torch_value
            for tfidf, conf, torch_value in zip(tfidf_pred, tfidf_conf, torch_pred, strict=True)
        ]
        metrics = evaluate_text_predictions(y_true, preds)
        if best_metrics is None:
            best_metrics = metrics
            best_predictions = preds
            best_threshold = threshold
            continue
        current = (metrics["char_f1"], metrics["sequence_ratio"], metrics["exact_match"])
        best = (best_metrics["char_f1"], best_metrics["sequence_ratio"], best_metrics["exact_match"])
        if current > best:
            best_metrics = metrics
            best_predictions = preds
            best_threshold = threshold
    if best_metrics is None or best_predictions is None:
        raise RuntimeError("Failed to evaluate hybrid threshold candidates.")
    return best_threshold, best_metrics, best_predictions


def _extract_strategy_metrics(result_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in result_rows:
        rows.append(
            {
                "strategy": row["strategy"],
                "char_f1": float(row["metrics"]["char_f1"]),
                "sequence_ratio": float(row["metrics"]["sequence_ratio"]),
                "exact_match": float(row["metrics"]["exact_match"]),
                "details": row["details"],
            }
        )
    return sorted(rows, key=lambda item: (item["char_f1"], item["sequence_ratio"], item["exact_match"]), reverse=True)


def fit_and_score_holdout(
    dataset: DatasetBundle,
    holdout_fraction: float = 0.15,
    seed: int = 42,
    device_preference: str = "auto",
    use_torch: bool = True,
    torch_configs: list[TorchRetrieverConfig] | None = None,
    tfidf_config: TfidfConfig | None = None,
) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame]:
    if not 0.0 < holdout_fraction < 0.5:
        raise ValueError("holdout_fraction must be between 0 and 0.5.")
    row_count = len(dataset.train_frame)
    if row_count < 6:
        raise ValueError("Need at least 6 training rows for holdout evaluation.")

    indices = np.arange(row_count)
    train_idx, holdout_idx = train_test_split(indices, test_size=holdout_fraction, random_state=seed, shuffle=True)
    train_sources = [dataset.train_source_texts[index] for index in train_idx]
    train_targets = [dataset.train_targets[index] for index in train_idx]
    holdout_sources = [dataset.train_source_texts[index] for index in holdout_idx]
    holdout_targets = [dataset.train_targets[index] for index in holdout_idx]

    tfidf_cfg = tfidf_config or TfidfConfig()
    tfidf_model = fit_tfidf_retriever(train_sources=train_sources, train_targets=train_targets, config=tfidf_cfg)
    tfidf_predictions, tfidf_confidences = tfidf_predict(tfidf_model, holdout_sources)
    tfidf_metrics = evaluate_text_predictions(holdout_targets, tfidf_predictions)

    result_rows: list[dict[str, Any]] = [
        {
            "strategy": "tfidf",
            "metrics": tfidf_metrics,
            "predictions": tfidf_predictions,
            "details": {"config": asdict(tfidf_cfg)},
        }
    ]

    best_torch_artifact: dict[str, Any] | None = None
    if use_torch:
        device = resolve_device(device_preference)
        configs = torch_configs or default_torch_configs()
        torch_candidates: list[dict[str, Any]] = []
        for config in configs:
            candidate = fit_torch_retriever(
                train_sources=train_sources,
                train_targets=train_targets,
                eval_sources=holdout_sources,
                eval_targets=holdout_targets,
                config=config,
                device=device,
                seed=seed,
            )
            torch_candidates.append(candidate)

        torch_candidates.sort(key=lambda item: item["metrics"]["char_f1"], reverse=True)
        if torch_candidates:
            best_torch_artifact = torch_candidates[0]
            result_rows.append(
                {
                    "strategy": "torch",
                    "metrics": best_torch_artifact["metrics"],
                    "predictions": best_torch_artifact["eval_predictions"],
                    "details": {
                        "config": best_torch_artifact["config"],
                        "best_epoch": int(best_torch_artifact["best_epoch"]),
                    },
                }
            )
            threshold, hybrid_metrics, hybrid_predictions = _select_best_hybrid_threshold(
                y_true=holdout_targets,
                tfidf_pred=tfidf_predictions,
                tfidf_conf=tfidf_confidences,
                torch_pred=best_torch_artifact["eval_predictions"],
            )
            result_rows.append(
                {
                    "strategy": "hybrid",
                    "metrics": hybrid_metrics,
                    "predictions": hybrid_predictions,
                    "details": {
                        "threshold": float(threshold),
                        "tfidf_config": asdict(tfidf_cfg),
                        "torch_config": best_torch_artifact["config"],
                        "torch_best_epoch": int(best_torch_artifact["best_epoch"]),
                    },
                }
            )

    result_rows.sort(
        key=lambda row: (
            row["metrics"]["char_f1"],
            row["metrics"]["sequence_ratio"],
            row["metrics"]["exact_match"],
        ),
        reverse=True,
    )
    best_row = result_rows[0]

    holdout_frame = dataset.train_frame.iloc[holdout_idx].copy()
    holdout_predictions = holdout_frame[[dataset.id_column, dataset.target_column]].copy()
    holdout_predictions["prediction"] = best_row["predictions"]

    model_selection = {
        "selected_strategy": best_row["strategy"],
        "id_column": dataset.id_column,
        "target_column": dataset.target_column,
        "source_columns": list(dataset.source_columns),
        "tfidf_config": asdict(tfidf_cfg),
        "selected_torch_config": best_torch_artifact["config"] if best_torch_artifact else None,
        "selected_torch_best_epoch": int(best_torch_artifact["best_epoch"]) if best_torch_artifact else None,
        "hybrid_threshold": (
            float(best_row["details"]["threshold"]) if best_row["strategy"] == "hybrid" else None
        ),
    }
    holdout_metrics = {
        "rows": int(len(holdout_targets)),
        "selected_strategy": best_row["strategy"],
        "char_f1": float(best_row["metrics"]["char_f1"]),
        "sequence_ratio": float(best_row["metrics"]["sequence_ratio"]),
        "exact_match": float(best_row["metrics"]["exact_match"]),
        "strategy_metrics": _extract_strategy_metrics(result_rows),
    }

    return model_selection, holdout_metrics, holdout_predictions.reset_index(drop=True)


def _strip_torch_artifact(model: dict[str, Any]) -> dict[str, Any]:
    return {
        "model_type": "torch",
        "config": model["config"],
        "token_to_id": model["token_to_id"],
        "state_dict": model["state_dict"],
        "max_length": model["max_length"],
        "train_targets": model["train_targets"],
        "best_epoch": model["best_epoch"],
    }


def fit_final_model(
    dataset: DatasetBundle,
    selection: dict[str, Any],
    seed: int = 42,
    device_preference: str = "auto",
) -> dict[str, Any]:
    selected_strategy = selection["selected_strategy"]
    device = resolve_device(device_preference)

    tfidf_model: dict[str, Any] | None = None
    torch_model: dict[str, Any] | None = None

    if selected_strategy in {"tfidf", "hybrid"}:
        tfidf_model = fit_tfidf_retriever(
            train_sources=dataset.train_source_texts,
            train_targets=dataset.train_targets,
            config=TfidfConfig(**selection["tfidf_config"]),
        )

    if selected_strategy in {"torch", "hybrid"}:
        torch_config = selection["selected_torch_config"]
        if torch_config is None:
            raise RuntimeError("Torch strategy selected but torch config is missing.")
        best_epoch = max(1, int(selection.get("selected_torch_best_epoch") or torch_config["epochs"]))
        full_torch = fit_torch_retriever(
            train_sources=dataset.train_source_texts,
            train_targets=dataset.train_targets,
            eval_sources=None,
            eval_targets=None,
            config=TorchRetrieverConfig(**torch_config),
            device=device,
            seed=seed,
            forced_epochs=best_epoch,
        )
        torch_model = _strip_torch_artifact(full_torch)

    return {
        "framework": "retrieval",
        "competition": COMPETITION_SLUG,
        "id_column": dataset.id_column,
        "target_column": dataset.target_column,
        "source_columns": list(dataset.source_columns),
        "selected_strategy": selected_strategy,
        "hybrid_threshold": selection.get("hybrid_threshold"),
        "tfidf_model": tfidf_model,
        "torch_model": torch_model,
        "device_used": str(device),
        "training_rows": int(len(dataset.train_frame)),
    }


def predict_texts(model_bundle: dict[str, Any], source_texts: list[str], device_preference: str = "auto") -> list[str]:
    strategy = model_bundle["selected_strategy"]
    if strategy == "tfidf":
        if model_bundle["tfidf_model"] is None:
            raise RuntimeError("Model bundle does not include tfidf model.")
        predictions, _ = tfidf_predict(model_bundle["tfidf_model"], source_texts)
        return predictions
    if strategy == "torch":
        if model_bundle["torch_model"] is None:
            raise RuntimeError("Model bundle does not include torch model.")
        device = resolve_device(device_preference)
        predictions, _ = torch_predict(model_bundle["torch_model"], source_texts, device=device)
        return predictions
    if strategy == "hybrid":
        if model_bundle["tfidf_model"] is None or model_bundle["torch_model"] is None:
            raise RuntimeError("Hybrid strategy requires both tfidf and torch models.")
        tfidf_predictions, tfidf_conf = tfidf_predict(model_bundle["tfidf_model"], source_texts)
        device = resolve_device(device_preference)
        torch_predictions, _ = torch_predict(model_bundle["torch_model"], source_texts, device=device)
        threshold = float(model_bundle.get("hybrid_threshold", 0.0))
        return [
            tfidf if conf >= threshold else torch_value
            for tfidf, conf, torch_value in zip(tfidf_predictions, tfidf_conf, torch_predictions, strict=True)
        ]
    raise ValueError(f"Unsupported strategy: {strategy}")


def generate_submission(
    model_bundle: dict[str, Any],
    dataset: DatasetBundle,
    device_preference: str = "auto",
) -> pd.DataFrame:
    predictions = predict_texts(model_bundle, dataset.test_source_texts, device_preference=device_preference)
    id_column = dataset.id_column
    target_column = dataset.target_column

    prediction_frame = pd.DataFrame(
        {
            id_column: dataset.test_frame[id_column].tolist(),
            target_column: predictions,
        }
    )
    ordered = dataset.sample_submission[[id_column]].merge(prediction_frame, on=id_column, how="left")
    if ordered[target_column].isna().any():
        raise ValueError("Submission contains missing predictions after merge. Check id alignment between test and sample.")
    return ordered


def save_model_bundle(model_bundle: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model_bundle, path)


def save_metrics(metrics: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
