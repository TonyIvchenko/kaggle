from __future__ import annotations

from competitions.deep_past_initiative_machine_translation.scripts.train_transformer_editor import (
    build_editor_prompt,
    fit_retrieval_memory,
    retrieve_drafts,
)


def test_build_editor_prompt_includes_source_and_draft():
    prompt = build_editor_prompt("a-na be-lí", "to my lord")
    assert "Source: a-na be-lí" in prompt
    assert "Draft: to my lord" in prompt
    assert prompt.endswith("Final translation:")


def test_retrieve_drafts_respects_blocked_indices_for_leave_one_out():
    sources = ["alpha transliteration", "beta transliteration"]
    targets = ["alpha translation", "beta translation"]
    vectorizer, matrix, target_lookup = fit_retrieval_memory(sources, targets)

    drafts, _ = retrieve_drafts(
        query_sources=sources,
        vectorizer=vectorizer,
        memory_matrix=matrix,
        memory_targets=target_lookup,
        blocked_memory_indices=[0, 1],
        search_top_k=2,
    )

    assert drafts == ["beta translation", "alpha translation"]
