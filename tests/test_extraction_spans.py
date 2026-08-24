from __future__ import annotations

import pytest

from enfold.extraction_spans import MAX_EVIDENCE_CHARS, transcript_spans


def test_transcript_spans_are_deterministic_bounded_and_exact():
    transcript = (
        "USER: Avery prefers local-first tools.\n\n"
        "ASSISTANT: Understood.\n\n"
        "USER: " + "é" * (MAX_EVIDENCE_CHARS + 20)
    )

    first = transcript_spans(transcript)
    second = transcript_spans(transcript)

    assert first == second
    assert len(first) >= 2
    assert all(span.text == transcript[span.start:span.end] for span in first)
    assert all(0 < len(span.text) <= MAX_EVIDENCE_CHARS for span in first)
    assert all(
        span.span_id == f"span-{span.start:06d}-{span.end:06d}"
        for span in first
    )
    assert [span.start for span in first] == sorted(span.start for span in first)


def test_transcript_spans_skip_only_separating_whitespace():
    transcript = "  USER: one.\n\n\tASSISTANT: two.  "

    spans = transcript_spans(transcript, max_chars=64)

    assert [span.text for span in spans] == [
        "USER: one.",
        "ASSISTANT: two.",
    ]
    assert all(span.text == transcript[span.start:span.end] for span in spans)


def test_short_hermes_turns_do_not_collapse_into_one_broad_evidence_span():
    transcript = (
        "USER: Avery prefers local tools.\n\n"
        "ASSISTANT: I will remember that.\n\n"
        "USER: Avery's backups run every Friday."
    )

    spans = transcript_spans(transcript)

    assert [span.text for span in spans] == [
        "USER: Avery prefers local tools.",
        "ASSISTANT: I will remember that.",
        "USER: Avery's backups run every Friday.",
    ]


@pytest.mark.parametrize("max_chars", [0, -1, True, MAX_EVIDENCE_CHARS + 1])
def test_transcript_span_limit_is_strict(max_chars):
    with pytest.raises(ValueError, match="evidence limit"):
        transcript_spans("transcript", max_chars=max_chars)
