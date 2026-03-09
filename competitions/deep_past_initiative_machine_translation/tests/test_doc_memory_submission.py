from __future__ import annotations

from competitions.deep_past_initiative_machine_translation.scripts.build_doc_memory_submission import (
    _align_a_to_b_groups,
    _normalize_transliteration,
    _select_sentence_indices_for_line_range,
    _split_translation_sentences,
)


def test_split_translation_sentences_handles_basic_punctuation():
    text = 'One sentence. Another sentence?" Final sentence.'
    parts = _split_translation_sentences(text)
    assert parts == ["One sentence.", 'Another sentence?"', "Final sentence."]


def test_normalize_transliteration_uses_token_map():
    token_map = {
        "kà-ru-um": "kārum",
        "a-lim(ki)": "ālim",
    }
    value = _normalize_transliteration("kà-ru-um a-lim(ki) unknown", token_map)
    assert value == "kārum ālim unknown"


def test_select_sentence_indices_for_line_range_uses_half_open_interval():
    line_numbers = [1.0, 6.0, 7.0, 8.0, 14.0, 25.0, 28.0]
    first = _select_sentence_indices_for_line_range(line_numbers, line_start=1, line_end=7)
    second = _select_sentence_indices_for_line_range(line_numbers, line_start=7, line_end=14)
    last = _select_sentence_indices_for_line_range(line_numbers, line_start=25, line_end=30)

    assert first == [0, 1]
    assert second == [2, 3]
    assert last == [5, 6]


def test_align_a_to_b_groups_prefers_monotonic_partition():
    a_sentences = [
        "Thus karum Kanesh says...",
        "A letter of the City has arrived.",
        "In the letter of the City it is written:",
        "From this day on whoever buys meteoric iron...",
        "As soon as you have heard our letter...",
        "Send a copy of this letter to every colony.",
        "Even when somebody sold meteoric iron via agent.",
    ]
    b_sentences = [
        "Thus Kanesh says... A letter of the City has arrived.",
        "In the letter of the City it is written: From this day on whoever buys meteoric iron...",
        "As soon as you have heard our letter...",
        "Send a copy of this letter to every colony. Even when somebody sold meteoric iron via agent.",
    ]

    groups = _align_a_to_b_groups(a_sentences, b_sentences)
    assert groups == [(0, 2), (2, 4), (4, 5), (5, 7)]
