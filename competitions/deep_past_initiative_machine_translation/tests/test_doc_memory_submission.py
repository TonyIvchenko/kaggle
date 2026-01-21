from __future__ import annotations

import pandas as pd

from competitions.deep_past_initiative_machine_translation.scripts.build_doc_memory_submission import (
    _best_sentence_partition,
    _build_submission,
    _group_test_rows,
    _split_sentences,
)


def test_split_sentences_preserves_quote_boundaries():
    text = 'One sentence. Another sentence?" Final fragment.'
    sentences = _split_sentences(text)
    assert sentences == ["One sentence.", 'Another sentence?"', "Final fragment."]


def test_best_sentence_partition_returns_contiguous_spans():
    spans = _best_sentence_partition(
        sentence_lengths=[10, 12, 30, 7],
        source_lengths=[11, 13, 35],
    )
    assert spans == [(0, 1), (1, 2), (2, 4)]


def test_build_submission_uses_document_partition_for_grouped_rows():
    train = pd.DataFrame(
        [
            {
                "oare_id": "doc-main",
                "transliteration": "tokA tokB tokC tokD tokE tokF tokG tokH tokI tokJ",
                "translation": "Alpha sentence. Beta sentence. Gamma sentence. Delta sentence.",
            },
            {
                "oare_id": "noise",
                "transliteration": "completely unrelated source row",
                "translation": "noise translation",
            },
        ]
    )
    test = pd.DataFrame(
        [
            {"id": 1, "text_id": "txt", "line_start": 1, "line_end": 2, "transliteration": "tokA tokB tokC"},
            {"id": 2, "text_id": "txt", "line_start": 2, "line_end": 4, "transliteration": "tokD tokE tokF tokG"},
            {"id": 3, "text_id": "txt", "line_start": 4, "line_end": 6, "transliteration": "tokH tokI tokJ"},
        ]
    )
    sample_submission = pd.DataFrame({"id": [1, 2, 3], "translation": ["", "", ""]})

    submission, report = _build_submission(
        train=train,
        test=test,
        sample_submission=sample_submission,
        id_column="id",
        target_column="translation",
        source_columns=("transliteration",),
        doc_match_threshold=0.2,
        top_k=2,
    )

    assert report[0]["strategy"] == "document_partition"
    assert list(submission["id"]) == [1, 2, 3]
    assert submission.loc[0, "translation"] == "Alpha sentence."
    assert "Beta sentence." in submission.loc[1, "translation"]
    assert "Delta sentence." in submission.loc[2, "translation"]


def test_group_test_rows_without_text_id_returns_singletons():
    frame = pd.DataFrame([{"id": 1}, {"id": 2}, {"id": 3}])
    groups = _group_test_rows(frame)
    assert groups == [[0], [1], [2]]
