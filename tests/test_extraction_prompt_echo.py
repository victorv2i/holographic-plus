"""Guards against the extractor echoing its own prompt examples as facts.

Regression for facts 10811/10858/10899 (2026-07-21): qwen3:30b copied the
system prompt's few-shot examples ("Avery now works at Acme." etc.) into the
store verbatim, attached to unrelated evidence spans.
"""
from __future__ import annotations

from enfold.extraction_contract import (
    PROMPT_EXAMPLE_CONTENTS,
    PROMPT_IDENTITY,
    SYSTEM_PROMPT,
    normalize_proposal_document,
)
from enfold.extraction_spans import transcript_spans


def _proposal(content: str, span_id: str) -> dict:
    return {
        "content": content,
        "category": "state",
        "tags": "test",
        "evidence_span_id": span_id,
        "sensitivity": "normal",
    }


def test_prompt_examples_never_use_the_real_user() -> None:
    assert "Avery" not in SYSTEM_PROMPT
    assert "avery" not in SYSTEM_PROMPT


def test_prompt_identity_bumped_for_new_prompt() -> None:
    assert PROMPT_IDENTITY == "durable-memory-v3"


def test_prompt_example_contents_appear_in_prompt() -> None:
    assert PROMPT_EXAMPLE_CONTENTS
    for example in PROMPT_EXAMPLE_CONTENTS:
        assert example in SYSTEM_PROMPT


def test_normalize_drops_prompt_example_echoes() -> None:
    spans = transcript_spans("A friend of the user moved to Queens for a new job.")
    span_id = spans[0].span_id
    example = next(iter(PROMPT_EXAMPLE_CONTENTS))
    document = {
        "proposals": [
            _proposal(example, span_id),
            _proposal("The user's friend moved to Queens for a new job.", span_id),
        ]
    }
    normalized = normalize_proposal_document(document, spans)
    contents = [item["content"] for item in normalized["proposals"]]
    assert contents == ["The user's friend moved to Queens for a new job."]


def test_normalize_drops_echoes_ignoring_case_and_punctuation() -> None:
    spans = transcript_spans("Some unrelated transcript text about tooling.")
    span_id = spans[0].span_id
    document = {
        "proposals": [_proposal('  "Dana now works at Northwind."  ', span_id)]
    }
    normalized = normalize_proposal_document(document, spans)
    assert normalized["proposals"] == []


def test_normalize_still_keeps_prompt_example_paraphrases() -> None:
    """Commit 8e84240 only drops the three verbatim example sentences.

    A near-echo that a small model actually emits is stored by the contract
    normalizer. The processor must apply the real echo check.
    """
    spans = transcript_spans("A friend of the user moved to Queens for a new job.")
    span_id = spans[0].span_id
    document = {
        "proposals": [_proposal("Dana currently works at Northwind.", span_id)]
    }
    normalized = normalize_proposal_document(document, spans)
    assert [item["content"] for item in normalized["proposals"]] == [
        "Dana currently works at Northwind."
    ]
