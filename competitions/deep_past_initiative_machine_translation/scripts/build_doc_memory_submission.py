"""Build a document-aware translation memory submission for Deep Past Initiative."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from competitions.deep_past_initiative_machine_translation.models.baseline import (  # noqa: E402
    DEFAULT_RAW_DIR,
    discover_competition_files,
)


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    if text.lower() == "nan":
        return ""
    return " ".join(text.strip().split())


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _choose_id_column(test: pd.DataFrame, sample_submission: pd.DataFrame) -> str:
    preferred = ("id", "ID", "Id")
    test_columns = set(test.columns)
    sample_columns = set(sample_submission.columns)
    for column in preferred:
        if column in test_columns and column in sample_columns:
            return str(column)

    shared = [column for column in sample_submission.columns if column in test_columns]
    if not shared:
        raise ValueError("Could not infer id column: no shared columns between sample submission and test set.")
    return str(shared[0])


def _choose_target_column(train: pd.DataFrame, sample_submission: pd.DataFrame, id_column: str) -> str:
    candidates = [column for column in sample_submission.columns if column != id_column]
    if len(candidates) == 1:
        return str(candidates[0])
    for column in candidates:
        if column in train.columns:
            return str(column)
    raise ValueError("Could not infer target column from sample_submission columns.")


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


def _split_sentences(text: str) -> list[str]:
    normalized = _clean_text(text)
    if not normalized:
        return [""]

    boundaries: list[int] = []
    quote_chars = {'"', "”", "’", "'"}
    for idx, char in enumerate(normalized):
        if char in {".", "!", "?"}:
            end = idx + 1
            while end < len(normalized) and normalized[end] in quote_chars:
                end += 1
            boundaries.append(end)

    if not boundaries:
        return [normalized]

    sentences: list[str] = []
    start = 0
    for end in boundaries:
        segment = normalized[start:end].strip()
        if segment:
            sentences.append(segment)
        start = end
    tail = normalized[start:].strip()
    if tail:
        sentences.append(tail)
    return sentences if sentences else [normalized]


def _best_sentence_partition(sentence_lengths: list[int], source_lengths: list[int]) -> list[tuple[int, int]]:
    m = len(sentence_lengths)
    n = len(source_lengths)
    if m < n:
        raise ValueError("Need at least as many translation sentences as source segments for partitioning.")

    prefix = np.zeros(m + 1, dtype=np.float64)
    for i, length in enumerate(sentence_lengths, start=1):
        prefix[i] = prefix[i - 1] + float(length)

    source = np.asarray(source_lengths, dtype=np.float64)
    source_ratio = source / max(float(source.sum()), 1.0)
    total_sentence_len = max(float(prefix[-1]), 1.0)

    inf = float("inf")
    dp = np.full((m + 1, n + 1), inf, dtype=np.float64)
    prev = np.full((m + 1, n + 1), -1, dtype=np.int64)
    dp[0, 0] = 0.0

    for i in range(1, m + 1):
        for j in range(1, min(i, n) + 1):
            for k in range(j - 1, i):
                group_len = prefix[i] - prefix[k]
                ratio = group_len / total_sentence_len
                cost = (ratio - source_ratio[j - 1]) ** 2
                candidate = dp[k, j - 1] + cost
                if candidate < dp[i, j]:
                    dp[i, j] = candidate
                    prev[i, j] = k

    if not np.isfinite(dp[m, n]):
        raise RuntimeError("Failed to partition translation sentences.")

    ranges: list[tuple[int, int]] = []
    i = m
    j = n
    while j > 0:
        k = int(prev[i, j])
        if k < 0:
            raise RuntimeError("Invalid backtrace while partitioning translation sentences.")
        ranges.append((k, i))
        i = k
        j -= 1
    ranges.reverse()
    return ranges


def _segment_document_translation(full_translation: str, row_source_lengths: list[int], rows_count: int) -> list[str]:
    sentences = _split_sentences(full_translation)
    if rows_count <= 1:
        return [_clean_text(full_translation)]
    if len(sentences) < rows_count:
        text = _clean_text(full_translation)
        boundaries = np.linspace(0, len(text), rows_count + 1).round().astype(np.int64)
        return [text[boundaries[i] : boundaries[i + 1]].strip() for i in range(rows_count)]

    partition = _best_sentence_partition(
        sentence_lengths=[len(sentence) for sentence in sentences],
        source_lengths=row_source_lengths,
    )
    chunks: list[str] = []
    for start, end in partition:
        chunks.append(" ".join(sentences[start:end]).strip())
    return chunks


def _group_test_rows(test: pd.DataFrame) -> list[list[int]]:
    if "text_id" in test.columns and "line_start" in test.columns:
        sorted_frame = test.sort_values(["text_id", "line_start"]).copy()
        groups: list[list[int]] = []
        for _, group in sorted_frame.groupby("text_id", sort=False):
            groups.append(group.index.to_list())
        return groups
    return [[index] for index in test.index.to_list()]


def _build_submission(
    train: pd.DataFrame,
    test: pd.DataFrame,
    sample_submission: pd.DataFrame,
    id_column: str,
    target_column: str,
    source_columns: tuple[str, ...],
    doc_match_threshold: float,
    top_k: int,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    train_sources = _combine_source_columns(train, source_columns)
    test_sources = _combine_source_columns(test, source_columns)
    train_targets = train[target_column].map(_clean_text).tolist()

    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 6),
        min_df=1,
        max_features=350_000,
        lowercase=False,
    )
    train_matrix = vectorizer.fit_transform(train_sources)
    predictions = [""] * len(test)
    report_rows: list[dict[str, Any]] = []

    for group in _group_test_rows(test):
        group_sources = [test_sources[idx] for idx in group]
        combined_source = " ".join(group_sources)
        combined_sims = linear_kernel(vectorizer.transform([combined_source]), train_matrix).reshape(-1)

        ranked = np.argsort(combined_sims)[::-1]
        top_idx = int(ranked[0])
        top_score = float(combined_sims[top_idx])

        used_strategy = "row_retrieval"
        top_doc_ids = train.iloc[ranked[: max(1, top_k)]].index.to_list()
        group_predictions: list[str] = []

        if len(group) > 1 and top_score >= doc_match_threshold:
            used_strategy = "document_partition"
            segmented = _segment_document_translation(
                full_translation=train_targets[top_idx],
                row_source_lengths=[len(source) for source in group_sources],
                rows_count=len(group),
            )
            if len(segmented) == len(group):
                group_predictions = segmented

        if used_strategy == "row_retrieval":
            row_sims = linear_kernel(vectorizer.transform(group_sources), train_matrix)
            for row_position, row_index in enumerate(group):
                row_best = int(np.argmax(row_sims[row_position]))
                if row_position < len(group_predictions):
                    group_predictions[row_position] = train_targets[row_best]
                else:
                    group_predictions.append(train_targets[row_best])

        for row_index, prediction in zip(group, group_predictions, strict=True):
            predictions[row_index] = _clean_text(prediction)

        report_rows.append(
            {
                "rows": [_to_jsonable(test.loc[idx, id_column]) for idx in group],
                "strategy": used_strategy,
                "top_match_index": top_idx,
                "top_match_score": top_score,
                "top_match_source_len": int(len(train_sources[top_idx])),
                "top_match_target_len": int(len(train_targets[top_idx])),
                "top_k_match_indices": [int(value) for value in ranked[: max(1, top_k)]],
            }
        )

    prediction_frame = pd.DataFrame(
        {
            id_column: test[id_column].tolist(),
            target_column: predictions,
        }
    )
    ordered = sample_submission[[id_column]].merge(prediction_frame, on=id_column, how="left")
    if ordered[target_column].isna().any():
        raise ValueError("Submission contains missing predictions after merge. Check id alignment.")
    return ordered, report_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build document-aware translation memory submission.")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR, help="Directory containing train/test CSV files.")
    parser.add_argument(
        "--submission-path",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "submissions" / "submission_doc_memory.csv",
        help="Path to write submission CSV.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "models" / "submission_doc_memory_report.json",
        help="Path to write diagnostic JSON report.",
    )
    parser.add_argument(
        "--doc-match-threshold",
        type=float,
        default=0.72,
        help="Minimum concatenated similarity to activate document-level partition strategy.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of nearest document indices to include in report.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = discover_competition_files(args.raw_dir)
    train = pd.read_csv(files.train)
    test = pd.read_csv(files.test)
    sample_submission = pd.read_csv(files.sample_submission)

    id_column = _choose_id_column(test=test, sample_submission=sample_submission)
    target_column = _choose_target_column(train=train, sample_submission=sample_submission, id_column=id_column)
    source_columns = _choose_source_columns(
        train=train,
        test=test,
        id_column=id_column,
        target_column=target_column,
    )

    submission, report = _build_submission(
        train=train,
        test=test,
        sample_submission=sample_submission,
        id_column=id_column,
        target_column=target_column,
        source_columns=source_columns,
        doc_match_threshold=args.doc_match_threshold,
        top_k=max(1, int(args.top_k)),
    )

    args.submission_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.submission_path, index=False)
    args.report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Saved submission to: {args.submission_path}")
    print(f"Saved report to: {args.report_path}")
    print("Submission preview:")
    print(submission.head().to_string(index=False))


if __name__ == "__main__":
    main()
