"""Build an advanced document-aware submission for Deep Past Initiative."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from competitions.deep_past_initiative_machine_translation.models.baseline import (  # noqa: E402
    DEFAULT_RAW_DIR,
    build_datasets,
    discover_competition_files,
)


SENTENCE_DATA_FILE = "Sentences_Oare_FirstWord_LinNum.csv"
PUBLISHED_TEXTS_FILE = "published_texts.csv"


ALIGNMENT_VECTORIZER = HashingVectorizer(
    analyzer="char_wb",
    ngram_range=(2, 6),
    n_features=2**20,
    alternate_sign=False,
    norm="l2",
    lowercase=False,
)


@dataclass
class SentenceDoc:
    text_uuid: str
    source_sentences: list[str]
    target_sentences: list[str]
    concatenated_source: str
    span_indices: list[tuple[int, int]] | None = None
    span_sources: list[str] | None = None
    span_targets: list[str] | None = None
    span_source_token_counts: list[int] | None = None
    span_lookup: dict[tuple[int, int], int] | None = None

    def ensure_spans(self) -> None:
        if self.span_indices is not None:
            return
        indices: list[tuple[int, int]] = []
        span_sources: list[str] = []
        span_targets: list[str] = []
        span_source_token_counts: list[int] = []
        lookup: dict[tuple[int, int], int] = {}
        m = len(self.source_sentences)
        for start in range(m):
            source_acc: list[str] = []
            target_acc: list[str] = []
            for end in range(start + 1, m + 1):
                source_acc.append(self.source_sentences[end - 1])
                target_acc.append(self.target_sentences[end - 1])
                source_text = " ".join(source_acc).strip()
                target_text = " ".join(target_acc).strip()
                span_idx = len(indices)
                key = (start, end)
                indices.append(key)
                lookup[key] = span_idx
                span_sources.append(source_text)
                span_targets.append(target_text)
                span_source_token_counts.append(max(1, len(source_text.split())))

        self.span_indices = indices
        self.span_sources = span_sources
        self.span_targets = span_targets
        self.span_source_token_counts = span_source_token_counts
        self.span_lookup = lookup


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


def _build_sentence_documents(raw_dir: Path) -> dict[str, SentenceDoc]:
    sentences_path = raw_dir / SENTENCE_DATA_FILE
    published_texts_path = raw_dir / PUBLISHED_TEXTS_FILE
    if not sentences_path.exists() or not published_texts_path.exists():
        raise FileNotFoundError(
            "Advanced submission requires supplemental files. Missing one of: "
            f"{sentences_path.name}, {published_texts_path.name}. "
            "Run download_data.py --all-files first."
        )

    sentences = pd.read_csv(sentences_path)
    published_texts = pd.read_csv(published_texts_path, usecols=["oare_id", "transliteration"])
    transliteration_by_id = {
        _clean_text(row.oare_id): _clean_text(row.transliteration)
        for row in published_texts.itertuples(index=False)
    }

    docs: dict[str, SentenceDoc] = {}
    for text_uuid, group in sentences.groupby("text_uuid"):
        text_uuid_clean = _clean_text(text_uuid)
        doc_source = transliteration_by_id.get(text_uuid_clean, "")
        if not doc_source:
            continue
        tokens = doc_source.split()
        if not tokens:
            continue

        ordered = group.sort_values("first_word_obj_in_text")
        starts = ordered["first_word_obj_in_text"].tolist()
        targets = [_clean_text(value) for value in ordered["translation"].tolist()]

        source_sentences: list[str] = []
        target_sentences: list[str] = []
        for idx, start in enumerate(starts):
            if pd.isna(start):
                continue
            start_index = max(int(start) - 1, 0)
            if start_index >= len(tokens):
                continue
            if idx + 1 < len(starts) and not pd.isna(starts[idx + 1]):
                end_index = max(int(starts[idx + 1]) - 1, start_index + 1)
            else:
                end_index = len(tokens)
            end_index = min(end_index, len(tokens))

            source_piece = " ".join(tokens[start_index:end_index]).strip()
            target_piece = targets[idx]
            if source_piece and target_piece:
                source_sentences.append(source_piece)
                target_sentences.append(target_piece)

        if len(source_sentences) < 2:
            continue
        docs[text_uuid_clean] = SentenceDoc(
            text_uuid=text_uuid_clean,
            source_sentences=source_sentences,
            target_sentences=target_sentences,
            concatenated_source=" ".join(source_sentences).strip(),
        )
    return docs


def _build_row_memory(
    sentence_docs: dict[str, SentenceDoc],
    train_sources: list[str],
    train_targets: list[str],
) -> tuple[list[str], list[str]]:
    memory_rows: list[tuple[str, str]] = []
    for doc in sentence_docs.values():
        memory_rows.extend(zip(doc.source_sentences, doc.target_sentences, strict=True))
    memory_rows.extend(zip(train_sources, train_targets, strict=True))

    deduped: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for source, target in memory_rows:
        key = (_clean_text(source), _clean_text(target))
        if not key[0] or not key[1]:
            continue
        if key in seen:
            continue
        seen.add(key)
        deduped.append(key)
    if not deduped:
        raise ValueError("Row memory is empty. Check input data files.")
    sources = [row[0] for row in deduped]
    targets = [row[1] for row in deduped]
    return sources, targets


def _build_grouped_test_indices(test: pd.DataFrame) -> list[list[int]]:
    if "text_id" in test.columns and "line_start" in test.columns:
        sorted_frame = test.sort_values(["text_id", "line_start"]).copy()
        return [group.index.to_list() for _, group in sorted_frame.groupby("text_id", sort=False)]
    return [[index] for index in test.index.to_list()]


def _align_rows_to_doc(
    row_sources: list[str],
    doc: SentenceDoc,
    length_penalty: float,
) -> tuple[float, list[str], list[tuple[int, int]]] | None:
    doc.ensure_spans()
    assert doc.span_indices is not None
    assert doc.span_sources is not None
    assert doc.span_targets is not None
    assert doc.span_source_token_counts is not None
    assert doc.span_lookup is not None

    n_rows = len(row_sources)
    n_sentences = len(doc.source_sentences)
    if n_rows <= 0 or n_sentences <= 0 or n_sentences < n_rows:
        return None

    matrix = ALIGNMENT_VECTORIZER.transform(row_sources + doc.span_sources)
    row_matrix = matrix[:n_rows]
    span_matrix = matrix[n_rows:]
    similarities = (row_matrix @ span_matrix.T).toarray()
    row_token_counts = [max(1, len(text.split())) for text in row_sources]

    negative_inf = -1.0e12
    dp = np.full((n_rows + 1, n_sentences + 1), negative_inf, dtype=np.float64)
    prev = np.full((n_rows + 1, n_sentences + 1), -1, dtype=np.int64)

    # Allow skipping document prefix for partial matches.
    dp[0, :] = 0.0

    for row_idx in range(1, n_rows + 1):
        for end in range(row_idx, n_sentences + 1):
            best_score = negative_inf
            best_start = -1
            for start in range(row_idx - 1, end):
                prev_score = dp[row_idx - 1, start]
                if prev_score <= negative_inf / 2:
                    continue
                span_idx = doc.span_lookup.get((start, end))
                if span_idx is None:
                    continue
                similarity = float(similarities[row_idx - 1, span_idx])
                token_ratio_penalty = abs(
                    math.log(doc.span_source_token_counts[span_idx] / row_token_counts[row_idx - 1])
                )
                score = prev_score + similarity - (length_penalty * token_ratio_penalty)
                if score > best_score:
                    best_score = score
                    best_start = start
            dp[row_idx, end] = best_score
            prev[row_idx, end] = best_start

    best_end = int(np.argmax(dp[n_rows, :]))
    best_total = float(dp[n_rows, best_end])
    if best_total <= negative_inf / 2:
        return None

    predictions = [""] * n_rows
    spans: list[tuple[int, int]] = [(-1, -1)] * n_rows
    end = best_end
    for row_idx in range(n_rows, 0, -1):
        start = int(prev[row_idx, end])
        if start < 0:
            return None
        span_idx = doc.span_lookup[(start, end)]
        predictions[row_idx - 1] = doc.span_targets[span_idx]
        spans[row_idx - 1] = (start, end)
        end = start

    normalized_score = best_total / max(n_rows, 1)
    return normalized_score, predictions, spans


def _build_submission(
    dataset: Any,
    sentence_docs: dict[str, SentenceDoc],
    top_k_docs: int,
    length_penalty: float,
    doc_score_weight: float,
    min_doc_blend_score: float,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    row_memory_sources, row_memory_targets = _build_row_memory(
        sentence_docs=sentence_docs,
        train_sources=dataset.train_source_texts,
        train_targets=dataset.train_targets,
    )
    row_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 7),
        min_df=1,
        max_features=500_000,
        lowercase=False,
    )
    row_matrix = row_vectorizer.fit_transform(row_memory_sources)

    docs_list = list(sentence_docs.values())
    doc_sources = [doc.concatenated_source for doc in docs_list]
    doc_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 7),
        min_df=1,
        max_features=500_000,
        lowercase=False,
    )
    doc_matrix = doc_vectorizer.fit_transform(doc_sources)

    grouped_indices = _build_grouped_test_indices(dataset.test_frame)
    predictions = [""] * len(dataset.test_frame)
    report_rows: list[dict[str, Any]] = []

    for group_indices in grouped_indices:
        row_sources = [dataset.test_source_texts[index] for index in group_indices]

        row_similarities = linear_kernel(row_vectorizer.transform(row_sources), row_matrix)
        row_best_indices = np.argmax(row_similarities, axis=1)
        fallback_predictions = [row_memory_targets[int(index)] for index in row_best_indices]
        fallback_scores = [
            float(row_similarities[row_position, int(index)])
            for row_position, index in enumerate(row_best_indices.tolist())
        ]

        combined_source = " ".join(row_sources).strip()
        doc_scores = linear_kernel(doc_vectorizer.transform([combined_source]), doc_matrix).ravel()
        top_doc_indices = np.argsort(doc_scores)[::-1][: max(1, top_k_docs)]

        best_doc_score = -1.0e12
        best_doc_predictions: list[str] | None = None
        best_doc_key = None
        best_alignment_spans: list[tuple[int, int]] | None = None
        best_alignment_component = None
        for doc_index in top_doc_indices.tolist():
            doc = docs_list[int(doc_index)]
            alignment = _align_rows_to_doc(
                row_sources=row_sources,
                doc=doc,
                length_penalty=length_penalty,
            )
            if alignment is None:
                continue
            alignment_score, aligned_predictions, aligned_spans = alignment
            blended = (doc_score_weight * alignment_score) + ((1.0 - doc_score_weight) * float(doc_scores[doc_index]))
            if blended > best_doc_score:
                best_doc_score = blended
                best_doc_predictions = aligned_predictions
                best_doc_key = doc.text_uuid
                best_alignment_spans = aligned_spans
                best_alignment_component = alignment_score

        strategy = "row_retrieval"
        final_predictions = fallback_predictions
        if best_doc_predictions is not None and best_doc_score >= min_doc_blend_score:
            strategy = "document_alignment"
            final_predictions = best_doc_predictions

        for row_index, prediction in zip(group_indices, final_predictions, strict=True):
            predictions[row_index] = _clean_text(prediction)

        report_rows.append(
            {
                "rows": [_to_jsonable(dataset.test_frame.loc[index, dataset.id_column]) for index in group_indices],
                "strategy": strategy,
                "doc_blend_score": float(best_doc_score),
                "doc_alignment_score": (
                    float(best_alignment_component) if best_alignment_component is not None else None
                ),
                "selected_doc_uuid": best_doc_key,
                "selected_doc_spans": (
                    [[int(start), int(end)] for start, end in best_alignment_spans]
                    if best_alignment_spans is not None
                    else None
                ),
                "row_fallback_scores": fallback_scores,
                "top_doc_scores": [
                    {
                        "doc_uuid": docs_list[int(index)].text_uuid,
                        "score": float(doc_scores[int(index)]),
                    }
                    for index in top_doc_indices.tolist()
                ],
            }
        )

    prediction_frame = pd.DataFrame(
        {
            dataset.id_column: dataset.test_frame[dataset.id_column].tolist(),
            dataset.target_column: predictions,
        }
    )
    ordered = dataset.sample_submission[[dataset.id_column]].merge(
        prediction_frame,
        on=dataset.id_column,
        how="left",
    )
    if ordered[dataset.target_column].isna().any():
        raise ValueError("Submission contains missing predictions after merge. Check id alignment.")
    return ordered, report_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build advanced document-aware submission.")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR, help="Directory containing competition CSV files.")
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
        help="Path to write diagnostic report JSON.",
    )
    parser.add_argument("--top-k-docs", type=int, default=10, help="Number of candidate docs to align per test group.")
    parser.add_argument(
        "--length-penalty",
        type=float,
        default=0.12,
        help="Penalty applied when aligned span length diverges from source row length.",
    )
    parser.add_argument(
        "--doc-score-weight",
        type=float,
        default=0.80,
        help="Blend weight for alignment score vs document retrieval score.",
    )
    parser.add_argument(
        "--min-doc-blend-score",
        type=float,
        default=0.06,
        help="Minimum blended score to trust document alignment over row fallback.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not (0.0 <= args.doc_score_weight <= 1.0):
        raise ValueError("--doc-score-weight must be in [0, 1].")
    if args.top_k_docs < 1:
        raise ValueError("--top-k-docs must be >= 1.")

    files = discover_competition_files(args.raw_dir)
    dataset = build_datasets(files)
    sentence_docs = _build_sentence_documents(args.raw_dir)
    if not sentence_docs:
        raise RuntimeError("No aligned sentence documents could be built from supplemental files.")

    submission, report = _build_submission(
        dataset=dataset,
        sentence_docs=sentence_docs,
        top_k_docs=int(args.top_k_docs),
        length_penalty=float(args.length_penalty),
        doc_score_weight=float(args.doc_score_weight),
        min_doc_blend_score=float(args.min_doc_blend_score),
    )

    args.submission_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.submission_path, index=False)
    args.report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Loaded sentence docs: {len(sentence_docs)}")
    print(f"Saved submission to: {args.submission_path}")
    print(f"Saved report to: {args.report_path}")
    print("Submission preview:")
    print(submission.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
