"""Kaggle notebook source for Deep Past Initiative all-files submission."""

from __future__ import annotations

import json
import math
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


SENTENCE_FILE = "Sentences_Oare_FirstWord_LinNum.csv"
PUBLISHED_TEXTS_FILE = "published_texts.csv"
LEXICON_FILE = "OA_Lexicon_eBL.csv"
DICTIONARY_FILE = "eBL_Dictionary.csv"
PUBLICATIONS_FILE = "publications.csv"
BIBLIOGRAPHY_FILE = "bibliography.csv"
RESOURCES_FILE = "resources.csv"


@dataclass(frozen=True)
class DatasetBundle:
    train_frame: pd.DataFrame
    test_frame: pd.DataFrame
    sample_submission: pd.DataFrame
    train_id_column: str
    id_column: str
    target_column: str
    source_columns: tuple[str, ...]
    train_source_texts: list[str]
    test_source_texts: list[str]
    train_targets: list[str]


@dataclass(frozen=True)
class SentenceDoc:
    text_uuid: str
    source_sentences_raw: list[str]
    source_sentences_norm: list[str]
    sentence_translations: list[str]
    line_numbers: list[float]
    full_source_norm: str


@dataclass(frozen=True)
class TrainStyleMapping:
    b_segments: list[str]
    a_to_b: list[int]


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


def _normalize_english(text: str) -> str:
    value = _clean_text(text).lower()
    value = unicodedata.normalize("NFKD", value)
    value = "".join(char for char in value if not unicodedata.combining(char))
    value = re.sub(r"[^a-z0-9\\s]", " ", value)
    return " ".join(value.split())


def _split_translation_sentences(text: str) -> list[str]:
    normalized = _clean_text(text)
    if not normalized:
        return [""]
    parts: list[str] = []
    start = 0
    for match in re.finditer(r'[.!?](?:"|”)?', normalized):
        end = match.end()
        piece = normalized[start:end].strip()
        if piece:
            parts.append(piece)
        start = end
    tail = normalized[start:].strip()
    if tail:
        parts.append(tail)
    return parts if parts else [normalized]


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
) -> str:
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


def _choose_source_columns(
    train: pd.DataFrame,
    test: pd.DataFrame,
    id_column: str,
    target_column: str,
) -> tuple[str, ...]:
    columns = [column for column in test.columns if column != id_column and column in train.columns]
    columns = [column for column in columns if column != target_column]
    if not columns:
        fallback = [column for column in train.columns if column not in {id_column, target_column}]
        if not fallback:
            raise ValueError("Could not infer source columns.")
        columns = fallback
    return tuple(str(column) for column in columns)


def _choose_train_id_column(train: pd.DataFrame, submission_id_column: str, target_column: str) -> str | None:
    if submission_id_column in train.columns and submission_id_column != target_column:
        return submission_id_column

    preferred = ("oare_id", "train_id", "id", "ID", "Id")
    for column in preferred:
        if column in train.columns and column != target_column:
            return str(column)

    id_like = [
        column
        for column in train.columns
        if column != target_column and ("_id" in str(column).lower() or str(column).lower().endswith("id"))
    ]
    if id_like:
        return str(id_like[0])
    return None


def _combine_source_columns(frame: pd.DataFrame, source_columns: tuple[str, ...]) -> list[str]:
    parts = [frame[column].map(_clean_text) for column in source_columns]
    merged: list[str] = []
    for row_values in zip(*parts):
        tokens = [f"{column}:{value}" for column, value in zip(source_columns, row_values, strict=True) if value]
        merged.append(" | ".join(tokens) if tokens else "")
    return merged


def _build_dataset(raw_dir: Path) -> DatasetBundle:
    train = pd.read_csv(raw_dir / "train.csv")
    test = pd.read_csv(raw_dir / "test.csv")
    sample_submission = pd.read_csv(raw_dir / "sample_submission.csv")

    id_column = _choose_id_column(train, test, sample_submission)
    target_column = _choose_target_column(train, test, sample_submission, id_column=id_column)
    source_columns = _choose_source_columns(train, test, id_column=id_column, target_column=target_column)

    train_work = train.copy()
    train_id_column = _choose_train_id_column(train_work, submission_id_column=id_column, target_column=target_column)
    if train_id_column is None:
        train_id_column = "_row_id"
        train_work[train_id_column] = np.arange(len(train_work), dtype=np.int64)

    train_targets = train_work[target_column].map(_clean_text).tolist()
    train_sources = _combine_source_columns(train_work, source_columns=source_columns)
    test_sources = _combine_source_columns(test, source_columns=source_columns)

    return DatasetBundle(
        train_frame=train_work,
        test_frame=test.copy(),
        sample_submission=sample_submission.copy(),
        train_id_column=train_id_column,
        id_column=id_column,
        target_column=target_column,
        source_columns=source_columns,
        train_source_texts=train_sources,
        test_source_texts=test_sources,
        train_targets=train_targets,
    )


def _discover_data_dir() -> Path:
    kaggle_root = Path("/kaggle/input")
    if kaggle_root.exists():
        candidates: list[Path] = []
        for train_path in kaggle_root.rglob("train.csv"):
            parent = train_path.parent
            if (parent / "test.csv").exists() and (parent / "sample_submission.csv").exists():
                candidates.append(parent)
        if candidates:
            return sorted(candidates, key=lambda path: str(path))[0]

    local_candidates = [
        Path.cwd() / "competitions" / "deep_past_initiative_machine_translation" / "data" / "raw",
        Path.cwd() / "data" / "raw",
        Path.cwd(),
    ]
    for candidate in local_candidates:
        if (candidate / "train.csv").exists() and (candidate / "test.csv").exists() and (candidate / "sample_submission.csv").exists():
            return candidate

    raise FileNotFoundError("Could not find train/test/sample_submission in /kaggle/input or local paths.")


def _load_lexicon_normalizer(raw_dir: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    lex_path = raw_dir / LEXICON_FILE
    if lex_path.exists():
        lexicon = pd.read_csv(lex_path)
        if {"form", "norm"}.issubset(set(lexicon.columns)):
            for row in lexicon.itertuples(index=False):
                form = _clean_text(getattr(row, "form", ""))
                norm = _clean_text(getattr(row, "norm", ""))
                if form and norm:
                    mapping.setdefault(form, norm)
    return mapping


def _normalize_transliteration(text: str, token_normalizer: dict[str, str]) -> str:
    tokens = [_clean_text(token) for token in _clean_text(text).split()]
    normalized = [token_normalizer.get(token, token) for token in tokens if token]
    return " ".join(normalized)


def _select_sentence_indices_for_line_range(
    line_numbers: list[float],
    line_start: int | float,
    line_end: int | float,
) -> list[int]:
    selected: list[int] = []
    start = float(line_start)
    end = float(line_end)
    for idx, line_number in enumerate(line_numbers):
        if np.isnan(line_number):
            continue
        if start <= float(line_number) < end:
            selected.append(idx)
    return selected


def _build_sentence_docs(
    raw_dir: Path,
    token_normalizer: dict[str, str],
    train_sources_by_id: dict[str, str],
) -> dict[str, SentenceDoc]:
    sentences_path = raw_dir / SENTENCE_FILE
    published_path = raw_dir / PUBLISHED_TEXTS_FILE
    if not sentences_path.exists() or not published_path.exists():
        raise FileNotFoundError(
            "Supplemental files missing. Expected Sentences_Oare_FirstWord_LinNum.csv and published_texts.csv."
        )

    sentence_frame = pd.read_csv(sentences_path)
    published = pd.read_csv(published_path, usecols=["oare_id", "transliteration"])
    published_sources = {_clean_text(row.oare_id): _clean_text(row.transliteration) for row in published.itertuples(index=False)}

    docs: dict[str, SentenceDoc] = {}
    for text_uuid, group in sentence_frame.groupby("text_uuid"):
        text_uuid_clean = _clean_text(text_uuid)
        full_source = train_sources_by_id.get(text_uuid_clean) or published_sources.get(text_uuid_clean) or ""
        if not full_source:
            continue
        tokens = _clean_text(full_source).split()
        if not tokens:
            continue

        ordered = group.sort_values("first_word_obj_in_text")
        starts = ordered["first_word_obj_in_text"].tolist()
        line_numbers = ordered["line_number"].tolist()
        sentence_targets = [_clean_text(value) for value in ordered["translation"].tolist()]

        source_sentences_raw: list[str] = []
        source_sentences_norm: list[str] = []
        target_sentences: list[str] = []
        aligned_line_numbers: list[float] = []

        for idx, start in enumerate(starts):
            if pd.isna(start):
                continue
            start_idx = max(int(start) - 1, 0)
            if idx + 1 < len(starts) and not pd.isna(starts[idx + 1]):
                end_idx = max(int(starts[idx + 1]) - 1, start_idx + 1)
            else:
                end_idx = len(tokens)
            end_idx = min(max(end_idx, start_idx + 1), len(tokens))

            if start_idx >= len(tokens):
                start_idx = max(0, len(tokens) - 1)
                end_idx = len(tokens)

            source_piece = " ".join(tokens[start_idx:end_idx]).strip()
            if not source_piece:
                source_piece = _clean_text(full_source)
            target_piece = sentence_targets[idx]
            if not source_piece or not target_piece:
                continue

            source_sentences_raw.append(source_piece)
            source_sentences_norm.append(_normalize_transliteration(source_piece, token_normalizer))
            target_sentences.append(target_piece)
            aligned_line_numbers.append(float(line_numbers[idx]) if not pd.isna(line_numbers[idx]) else np.nan)

        if len(source_sentences_raw) < 2:
            continue
        docs[text_uuid_clean] = SentenceDoc(
            text_uuid=text_uuid_clean,
            source_sentences_raw=source_sentences_raw,
            source_sentences_norm=source_sentences_norm,
            sentence_translations=target_sentences,
            line_numbers=aligned_line_numbers,
            full_source_norm=" ".join(source_sentences_norm).strip(),
        )
    return docs


def _align_a_to_b_groups(
    sentence_targets_a: list[str],
    sentence_targets_b: list[str],
) -> list[tuple[int, int]] | None:
    n = len(sentence_targets_a)
    m = len(sentence_targets_b)
    if n <= 0 or m <= 0:
        return None
    if n < m:
        return None

    a_norm = [_normalize_english(text) for text in sentence_targets_a]
    b_norm = [_normalize_english(text) for text in sentence_targets_b]
    negative_inf = -1.0e12
    dp = np.full((n + 1, m + 1), negative_inf, dtype=np.float64)
    prev = np.full((n + 1, m + 1), -1, dtype=np.int64)
    dp[0, 0] = 0.0

    for i in range(1, n + 1):
        for k in range(1, min(i, m) + 1):
            best = negative_inf
            best_start = -1
            for start in range(k - 1, i):
                base = dp[start, k - 1]
                if base <= negative_inf / 2:
                    continue
                a_join = " ".join(a_norm[start:i]).strip()
                b_text = b_norm[k - 1]
                similarity = _sequence_ratio(a_join, b_text)
                length_penalty = abs(math.log((len(a_join.split()) + 1) / (len(b_text.split()) + 1)))
                score = base + similarity - (0.05 * length_penalty)
                if score > best:
                    best = score
                    best_start = start
            dp[i, k] = best
            prev[i, k] = best_start

    if dp[n, m] <= negative_inf / 2:
        return None

    groups: list[tuple[int, int]] = [(-1, -1)] * m
    i = n
    k = m
    while k > 0:
        start = int(prev[i, k])
        if start < 0:
            return None
        groups[k - 1] = (start, i)
        i = start
        k -= 1
    return groups


def _compress_segments_to_count(segments: list[str], target_count: int) -> list[str]:
    merged = list(segments)
    while len(merged) > target_count:
        lengths = [len(_normalize_english(value)) for value in merged]
        best_idx = min(range(len(merged) - 1), key=lambda idx: lengths[idx] + lengths[idx + 1])
        merged = merged[:best_idx] + [f"{merged[best_idx]} {merged[best_idx + 1]}"] + merged[best_idx + 2 :]
    return merged


def _build_train_style_mappings(
    sentence_docs: dict[str, SentenceDoc],
    train_targets_by_id: dict[str, str],
) -> tuple[dict[str, TrainStyleMapping], list[tuple[str, str]]]:
    mappings: dict[str, TrainStyleMapping] = {}
    style_pairs: list[tuple[str, str]] = []

    for text_uuid, doc in sentence_docs.items():
        train_target = train_targets_by_id.get(text_uuid)
        if not train_target:
            continue
        a_sentences = doc.sentence_translations
        b_segments = _split_translation_sentences(train_target)
        if len(a_sentences) < len(b_segments):
            b_segments = _compress_segments_to_count(b_segments, target_count=len(a_sentences))
        groups = _align_a_to_b_groups(a_sentences, b_segments)
        if groups is None:
            continue

        a_to_b = [0] * len(a_sentences)
        for b_idx, (start, end) in enumerate(groups):
            joined_a = " ".join(a_sentences[start:end]).strip()
            style_pairs.append((joined_a, b_segments[b_idx]))
            for sentence_idx in range(start, end):
                a_to_b[sentence_idx] = b_idx

        mappings[text_uuid] = TrainStyleMapping(
            b_segments=list(b_segments),
            a_to_b=a_to_b,
        )
    return mappings, style_pairs


def _build_style_converter(
    style_pairs: list[tuple[str, str]],
) -> tuple[Callable[[str], tuple[str, float]], TfidfVectorizer | None]:
    if not style_pairs:
        return (lambda text: (_clean_text(text), 0.0)), None

    source_texts = [_clean_text(pair[0]) for pair in style_pairs]
    target_texts = [_clean_text(pair[1]) for pair in style_pairs]
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 6),
        min_df=1,
        max_features=250_000,
        lowercase=False,
    )
    matrix = vectorizer.fit_transform(source_texts)

    def convert(text: str) -> tuple[str, float]:
        query = vectorizer.transform([_clean_text(text)])
        similarities = (query @ matrix.T).toarray().reshape(-1)
        if similarities.size == 0:
            return _clean_text(text), 0.0
        best_idx = int(np.argmax(similarities))
        best_score = float(similarities[best_idx])
        if best_score >= 0.52:
            return target_texts[best_idx], best_score
        return _clean_text(text), best_score

    return convert, vectorizer


def _build_auxiliary_english_vocab(raw_dir: Path, train_targets: list[str]) -> set[str]:
    corpus: list[str] = []
    corpus.extend(_normalize_english(text) for text in train_targets if _clean_text(text))

    dictionary_path = raw_dir / DICTIONARY_FILE
    if dictionary_path.exists():
        dictionary = pd.read_csv(dictionary_path)
        for column in ("definition", "gloss", "sense"):
            if column in dictionary.columns:
                corpus.extend(_normalize_english(value) for value in dictionary[column].astype(str).tolist())

    publications_path = raw_dir / PUBLICATIONS_FILE
    if publications_path.exists():
        publications = pd.read_csv(publications_path, nrows=4000)
        if "page_text" in publications.columns:
            if "has_akkadian" in publications.columns:
                page_texts = publications.loc[~publications["has_akkadian"].fillna(False), "page_text"].astype(str).tolist()
            else:
                page_texts = publications["page_text"].astype(str).tolist()
            corpus.extend(_normalize_english(text) for text in page_texts)

    bibliography_path = raw_dir / BIBLIOGRAPHY_FILE
    if bibliography_path.exists():
        bibliography = pd.read_csv(bibliography_path)
        for column in ("title", "Title"):
            if column in bibliography.columns:
                corpus.extend(_normalize_english(value) for value in bibliography[column].astype(str).tolist())

    resources_path = raw_dir / RESOURCES_FILE
    if resources_path.exists():
        resources = pd.read_csv(resources_path)
        for column in ("Title", "English abstract for non-English or N/A papers"):
            if column in resources.columns:
                corpus.extend(_normalize_english(value) for value in resources[column].astype(str).tolist())

    vocab: set[str] = set()
    for text in corpus:
        for token in text.split():
            if len(token) >= 3:
                vocab.add(token)
    return vocab


def _rare_word_ratio(text: str, english_vocab: set[str]) -> float:
    tokens = _normalize_english(text).split()
    long_tokens = [token for token in tokens if len(token) >= 3]
    if not long_tokens:
        return 0.0
    rare = sum(1 for token in long_tokens if token not in english_vocab)
    return float(rare) / float(len(long_tokens))


def _sequence_ratio(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return _difflib_ratio(a, b)


def _difflib_ratio(a: str, b: str) -> float:
    import difflib

    return float(difflib.SequenceMatcher(None, a, b).ratio())


def _build_row_fallback_memory(
    dataset: DatasetBundle,
    token_normalizer: dict[str, str],
    train_style_mappings: dict[str, TrainStyleMapping],
    sentence_docs: dict[str, SentenceDoc],
) -> tuple[TfidfVectorizer, Any, list[str]]:
    pairs: list[tuple[str, str]] = []
    for _, row in dataset.train_frame.iterrows():
        pairs.append(
            (
                _normalize_transliteration(_clean_text(row[dataset.source_columns[0]]), token_normalizer)
                if len(dataset.source_columns) == 1
                else _normalize_transliteration(
                    " | ".join(
                        f"{column}:{_clean_text(row[column])}"
                        for column in dataset.source_columns
                        if _clean_text(row[column])
                    ),
                    token_normalizer,
                ),
                _clean_text(row[dataset.target_column]),
            )
        )

    for text_uuid, mapping in train_style_mappings.items():
        doc = sentence_docs[text_uuid]
        for idx, source_sentence in enumerate(doc.source_sentences_norm):
            segment = mapping.b_segments[mapping.a_to_b[idx]]
            pairs.append((source_sentence, _clean_text(segment)))

    deduped: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for source, target in pairs:
        key = (_clean_text(source), _clean_text(target))
        if not key[0] or not key[1]:
            continue
        if key in seen:
            continue
        seen.add(key)
        deduped.append(key)

    sources = [row[0] for row in deduped]
    targets = [row[1] for row in deduped]
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 7),
        min_df=1,
        max_features=500_000,
        lowercase=False,
    )
    matrix = vectorizer.fit_transform(sources)
    return vectorizer, matrix, targets


def _normalize_dataset_source_rows(dataset: DatasetBundle, token_normalizer: dict[str, str]) -> tuple[list[str], list[str]]:
    train_norm = [_normalize_transliteration(text, token_normalizer) for text in dataset.train_source_texts]
    test_norm = [_normalize_transliteration(text, token_normalizer) for text in dataset.test_source_texts]
    return train_norm, test_norm


def _build_grouped_indices(test_frame: pd.DataFrame) -> list[list[int]]:
    if "text_id" in test_frame.columns and "line_start" in test_frame.columns:
        sorted_frame = test_frame.sort_values(["text_id", "line_start"]).copy()
        return [group.index.to_list() for _, group in sorted_frame.groupby("text_id", sort=False)]
    return [[index] for index in test_frame.index.to_list()]


def _predict_for_candidate_doc(
    row_sources_norm: list[str],
    row_ranges: list[tuple[int | float | None, int | float | None]],
    doc: SentenceDoc,
    train_style_mapping: TrainStyleMapping | None,
    style_convert: Callable[[str], tuple[str, float]],
) -> tuple[list[str], float, float]:
    predictions: list[str] = []
    covered_rows = 0
    local_similarity_sum = 0.0

    sentence_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 7),
        min_df=1,
        max_features=150_000,
        lowercase=False,
    )
    sentence_matrix = sentence_vectorizer.fit_transform(doc.source_sentences_norm)

    for row_source, (line_start, line_end) in zip(row_sources_norm, row_ranges, strict=True):
        sentence_indices: list[int] = []
        if line_start is not None and line_end is not None:
            sentence_indices = _select_sentence_indices_for_line_range(
                line_numbers=doc.line_numbers,
                line_start=float(line_start),
                line_end=float(line_end),
            )

        if sentence_indices:
            covered_rows += 1
            if train_style_mapping is not None:
                selected_segments = sorted(
                    {
                        train_style_mapping.a_to_b[idx]
                        for idx in sentence_indices
                        if idx < len(train_style_mapping.a_to_b)
                    }
                )
                prediction = " ".join(train_style_mapping.b_segments[idx] for idx in selected_segments).strip()
            else:
                sentence_text = " ".join(doc.sentence_translations[idx] for idx in sentence_indices).strip()
                prediction, _ = style_convert(sentence_text)
        else:
            query = sentence_vectorizer.transform([row_source])
            similarities = (query @ sentence_matrix.T).toarray().reshape(-1)
            best_idx = int(np.argmax(similarities))
            local_similarity_sum += float(similarities[best_idx])
            if train_style_mapping is not None and best_idx < len(train_style_mapping.a_to_b):
                prediction = train_style_mapping.b_segments[train_style_mapping.a_to_b[best_idx]]
            else:
                prediction, _ = style_convert(doc.sentence_translations[best_idx])

        predictions.append(_clean_text(prediction))

    coverage_ratio = float(covered_rows) / max(1, len(row_sources_norm))
    local_similarity = local_similarity_sum / max(1, len(row_sources_norm))
    return predictions, coverage_ratio, local_similarity


def _build_submission(
    dataset: DatasetBundle,
    raw_dir: Path,
    top_k_docs: int,
    token_normalizer: dict[str, str],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    train_sources_by_id = {
        _clean_text(row[dataset.train_id_column]): _clean_text(row[dataset.source_columns[0]])
        if len(dataset.source_columns) == 1
        else " | ".join(
            f"{column}:{_clean_text(row[column])}"
            for column in dataset.source_columns
            if _clean_text(row[column])
        )
        for _, row in dataset.train_frame.iterrows()
    }
    train_targets_by_id = {
        _clean_text(row[dataset.train_id_column]): _clean_text(row[dataset.target_column])
        for _, row in dataset.train_frame.iterrows()
    }
    sentence_docs = _build_sentence_docs(
        raw_dir=raw_dir,
        token_normalizer=token_normalizer,
        train_sources_by_id=train_sources_by_id,
    )
    if not sentence_docs:
        raise RuntimeError("No sentence documents available. Cannot run line-aware pipeline.")

    train_style_mappings, style_pairs = _build_train_style_mappings(
        sentence_docs=sentence_docs,
        train_targets_by_id=train_targets_by_id,
    )
    style_convert, _ = _build_style_converter(style_pairs)

    _, test_source_norm = _normalize_dataset_source_rows(dataset, token_normalizer=token_normalizer)
    row_fallback_vec, row_fallback_matrix, row_fallback_targets = _build_row_fallback_memory(
        dataset=dataset,
        token_normalizer=token_normalizer,
        train_style_mappings=train_style_mappings,
        sentence_docs=sentence_docs,
    )

    doc_keys = list(sentence_docs.keys())
    doc_corpus = [sentence_docs[key].full_source_norm for key in doc_keys]
    doc_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 7),
        min_df=1,
        max_features=500_000,
        lowercase=False,
    )
    doc_matrix = doc_vectorizer.fit_transform(doc_corpus)

    english_vocab = _build_auxiliary_english_vocab(raw_dir=raw_dir, train_targets=dataset.train_targets)

    grouped_indices = _build_grouped_indices(dataset.test_frame)
    predictions = [""] * len(dataset.test_frame)
    report_rows: list[dict[str, Any]] = []

    line_columns_present = {"line_start", "line_end"}.issubset(set(dataset.test_frame.columns))

    for group_indices in grouped_indices:
        row_sources_norm = [test_source_norm[idx] for idx in group_indices]
        row_ranges: list[tuple[int | float | None, int | float | None]] = []
        for idx in group_indices:
            if line_columns_present:
                row_ranges.append((dataset.test_frame.loc[idx, "line_start"], dataset.test_frame.loc[idx, "line_end"]))
            else:
                row_ranges.append((None, None))

        fallback_similarities = (
            row_fallback_vec.transform(row_sources_norm) @ row_fallback_matrix.T
        ).toarray()
        fallback_best = np.argmax(fallback_similarities, axis=1)
        fallback_predictions = [row_fallback_targets[int(index)] for index in fallback_best.tolist()]
        fallback_scores = [
            float(fallback_similarities[row_idx, int(best_idx)])
            for row_idx, best_idx in enumerate(fallback_best.tolist())
        ]

        combined_source = " ".join(row_sources_norm).strip()
        doc_scores = (doc_vectorizer.transform([combined_source]) @ doc_matrix.T).toarray().reshape(-1)
        candidate_doc_indices = np.argsort(doc_scores)[::-1][: max(1, top_k_docs)]

        best_score = -1.0e12
        best_predictions: list[str] | None = None
        best_doc_key = None
        best_coverage = 0.0
        best_local_sim = 0.0

        for candidate_idx in candidate_doc_indices.tolist():
            text_uuid = doc_keys[int(candidate_idx)]
            doc = sentence_docs[text_uuid]
            candidate_predictions, coverage_ratio, local_similarity = _predict_for_candidate_doc(
                row_sources_norm=row_sources_norm,
                row_ranges=row_ranges,
                doc=doc,
                train_style_mapping=train_style_mappings.get(text_uuid),
                style_convert=style_convert,
            )
            rare_penalty = float(np.mean([_rare_word_ratio(text, english_vocab) for text in candidate_predictions]))
            score = (
                (0.62 * float(doc_scores[int(candidate_idx)]))
                + (0.25 * coverage_ratio)
                + (0.13 * local_similarity)
                - (0.05 * rare_penalty)
            )
            if score > best_score:
                best_score = score
                best_predictions = candidate_predictions
                best_doc_key = text_uuid
                best_coverage = coverage_ratio
                best_local_sim = local_similarity

        strategy = "row_fallback"
        final_predictions = fallback_predictions
        if best_predictions is not None:
            strategy = "line_aware_doc_alignment"
            final_predictions = best_predictions

        for row_idx, prediction in zip(group_indices, final_predictions, strict=True):
            predictions[row_idx] = _clean_text(prediction)

        report_rows.append(
            {
                "rows": [_to_jsonable(dataset.test_frame.loc[idx, dataset.id_column]) for idx in group_indices],
                "strategy": strategy,
                "selected_doc_uuid": best_doc_key,
                "selected_score": float(best_score),
                "selected_coverage_ratio": float(best_coverage),
                "selected_local_similarity": float(best_local_sim),
                "fallback_scores": fallback_scores,
                "top_doc_scores": [
                    {
                        "doc_uuid": doc_keys[int(idx)],
                        "score": float(doc_scores[int(idx)]),
                    }
                    for idx in candidate_doc_indices.tolist()
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


def main() -> None:
    data_dir = _discover_data_dir()
    expected_files = [
        "sample_submission.csv",
        BIBLIOGRAPHY_FILE,
        PUBLICATIONS_FILE,
        SENTENCE_FILE,
        LEXICON_FILE,
        DICTIONARY_FILE,
        "train.csv",
        "test.csv",
        PUBLISHED_TEXTS_FILE,
        RESOURCES_FILE,
    ]

    print(f"Using data dir: {data_dir}")
    print("Input file check:")
    for name in expected_files:
        print(f"  {name}: {'ok' if (data_dir / name).exists() else 'missing'}")

    dataset = _build_dataset(data_dir)
    token_normalizer = _load_lexicon_normalizer(data_dir)
    submission, report = _build_submission(
        dataset=dataset,
        raw_dir=data_dir,
        top_k_docs=8,
        token_normalizer=token_normalizer,
    )

    if Path("/kaggle/working").exists():
        output_dir = Path("/kaggle/working")
    else:
        output_dir = Path.cwd()
    output_dir.mkdir(parents=True, exist_ok=True)

    submission_path = output_dir / "submission.csv"
    report_path = output_dir / "submission_report.json"
    submission.to_csv(submission_path, index=False)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    strategies = {}
    for row in report:
        strategy = str(row["strategy"])
        strategies[strategy] = strategies.get(strategy, 0) + 1

    print("\nSchema:")
    print("  id column:", dataset.id_column)
    print("  target column:", dataset.target_column)
    print("  source columns:", list(dataset.source_columns))
    print("  train rows:", len(dataset.train_frame), "test rows:", len(dataset.test_frame))
    print("  grouped texts:", len(report))
    print("  strategies:", strategies)
    print("\nSubmission preview:")
    print(submission.head(20).to_string(index=False))
    print(f"\nSaved submission: {submission_path}")
    print(f"Saved report: {report_path}")


main()
