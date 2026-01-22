from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from competitions.deep_past_initiative_machine_translation.scripts.build_doc_memory_submission import (
    SentenceDoc,
    _align_rows_to_doc,
    _build_grouped_test_indices,
    _build_submission,
)


@dataclass
class _MockDataset:
    train_source_texts: list[str]
    train_targets: list[str]
    test_source_texts: list[str]
    test_frame: pd.DataFrame
    sample_submission: pd.DataFrame
    id_column: str = "id"
    target_column: str = "translation"


def test_build_grouped_test_indices_uses_text_id_order():
    frame = pd.DataFrame(
        [
            {"id": 2, "text_id": "b", "line_start": 4},
            {"id": 1, "text_id": "a", "line_start": 7},
            {"id": 0, "text_id": "a", "line_start": 1},
        ]
    )
    groups = _build_grouped_test_indices(frame)
    assert groups == [[2, 1], [0]]


def test_align_rows_to_doc_allows_partial_doc_alignment():
    doc = SentenceDoc(
        text_uuid="doc-main",
        source_sentences=["tokA tokB", "tokC tokD", "tokE tokF", "tokG tokH"],
        target_sentences=["A", "B", "C", "D"],
        concatenated_source="tokA tokB tokC tokD tokE tokF tokG tokH",
    )
    rows = ["tokC tokD tokE tokF", "tokG tokH"]
    aligned = _align_rows_to_doc(row_sources=rows, doc=doc, length_penalty=0.12)
    assert aligned is not None
    _, predictions, spans = aligned
    assert predictions == ["B C", "D"]
    assert spans == [(1, 3), (3, 4)]


def test_build_submission_prefers_document_alignment_when_confident():
    test = pd.DataFrame(
        [
            {"id": 1, "text_id": "x", "line_start": 1, "line_end": 2, "transliteration": "tokA tokB"},
            {"id": 2, "text_id": "x", "line_start": 2, "line_end": 3, "transliteration": "tokC tokD tokE"},
        ]
    )
    dataset = _MockDataset(
        train_source_texts=["fallback source"],
        train_targets=["fallback translation"],
        test_source_texts=test["transliteration"].tolist(),
        test_frame=test,
        sample_submission=pd.DataFrame({"id": [1, 2], "translation": ["", ""]}),
    )
    sentence_docs = {
        "doc-main": SentenceDoc(
            text_uuid="doc-main",
            source_sentences=["tokA tokB", "tokC tokD tokE", "tokF tokG"],
            target_sentences=["Alpha", "Beta", "Gamma"],
            concatenated_source="tokA tokB tokC tokD tokE tokF tokG",
        )
    }

    submission, report = _build_submission(
        dataset=dataset,
        sentence_docs=sentence_docs,
        top_k_docs=4,
        length_penalty=0.12,
        doc_score_weight=0.8,
        min_doc_blend_score=-1.0,
    )

    assert report[0]["strategy"] == "document_alignment"
    assert list(submission["translation"]) == ["Alpha", "Beta"]


def test_build_submission_uses_row_fallback_when_doc_score_is_below_threshold():
    test = pd.DataFrame(
        [
            {"id": 1, "text_id": "x", "line_start": 1, "line_end": 2, "transliteration": "rare source term"},
            {"id": 2, "text_id": "x", "line_start": 2, "line_end": 3, "transliteration": "unknown source term"},
        ]
    )
    dataset = _MockDataset(
        train_source_texts=["rare source term", "unknown source term"],
        train_targets=["fallback one", "fallback two"],
        test_source_texts=test["transliteration"].tolist(),
        test_frame=test,
        sample_submission=pd.DataFrame({"id": [1, 2], "translation": ["", ""]}),
    )
    sentence_docs = {
        "doc-main": SentenceDoc(
            text_uuid="doc-main",
            source_sentences=["tokA tokB", "tokC tokD"],
            target_sentences=["Alpha", "Beta"],
            concatenated_source="tokA tokB tokC tokD",
        )
    }

    submission, report = _build_submission(
        dataset=dataset,
        sentence_docs=sentence_docs,
        top_k_docs=2,
        length_penalty=0.12,
        doc_score_weight=0.8,
        min_doc_blend_score=1_000.0,
    )

    assert report[0]["strategy"] == "row_retrieval"
    assert list(submission["translation"]) == ["fallback one", "fallback two"]
