from __future__ import annotations

import pytest

from enfold.extraction_contract import (
    ExtractionContractError,
    SYSTEM_PROMPT,
    normalize_proposal_document,
)
from enfold.extraction_spans import transcript_spans


def test_confidence_only_typed_output_is_rejected():
    transcript = "USER: Mara now works at Northwind."
    spans = transcript_spans(transcript)
    proposal = {
        "content": "Mara works at Northwind.",
        "category": "status",
        "tags": "mara,employment",
        "evidence_span_id": spans[0].span_id,
        "sensitivity": "sensitive",
        "confidence": 0.98,
    }

    with pytest.raises(ExtractionContractError, match="incomplete typed fields"):
        normalize_proposal_document({"proposals": [proposal]}, spans)


def test_prompt_requires_typed_fields_as_an_all_or_nothing_group():
    prompt = SYSTEM_PROMPT.lower()

    assert "typed fields are all-or-nothing" in prompt
    assert "never emit confidence alone" in prompt


def test_model_typed_date_must_be_grounded_in_the_evidence_span():
    transcript = "The DRC quiz-extension skill was created and validated."
    spans = transcript_spans(transcript)
    proposal = {
        "content": transcript,
        "category": "event",
        "tags": "drc,validated",
        "evidence_span_id": spans[0].span_id,
        "sensitivity": "normal",
        "kind": "event",
        "subject": "skill:drc_quiz_extension",
        "predicate": "validated",
        "object": "created and validated",
        "occurred_at": "2023-10-05T14:30:00Z",
        "confidence": 0.95,
    }

    with pytest.raises(ExtractionContractError, match="typed date is not grounded"):
        normalize_proposal_document({"proposals": [proposal]}, spans)
