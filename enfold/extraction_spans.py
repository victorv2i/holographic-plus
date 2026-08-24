"""Deterministic exact-source spans for grounded memory extraction."""

from __future__ import annotations

from dataclasses import dataclass
import re


MAX_EVIDENCE_CHARS = 2_000
_MIN_BOUNDARY_SEARCH_CHARS = 64
_BLANK_LINE = re.compile(r"\r?\n[ \t]*\r?\n")


@dataclass(frozen=True, slots=True)
class TranscriptSpan:
    """One bounded, exact substring of a transcript."""

    span_id: str
    start: int
    end: int
    text: str

    def __post_init__(self) -> None:
        if self.start < 0 or self.end <= self.start:
            raise ValueError("transcript span offsets are invalid")
        if not self.text or len(self.text) != self.end - self.start:
            raise ValueError("transcript span text does not match its offsets")
        if len(self.text) > MAX_EVIDENCE_CHARS:
            raise ValueError("transcript span exceeds the evidence limit")
        expected_id = f"span-{self.start:06d}-{self.end:06d}"
        if self.span_id != expected_id:
            raise ValueError("transcript span id does not match its offsets")

    def as_model_input(self) -> dict[str, str]:
        return {"id": self.span_id, "text": self.text}


def _preferred_end(
    transcript: str, start: int, hard_end: int, block_end: int
) -> int:
    """Prefer a readable boundary without ever exceeding ``hard_end``."""

    if hard_end == block_end:
        return hard_end
    minimum = min(hard_end, start + _MIN_BOUNDARY_SEARCH_CHARS)
    for separator in ("\n\n", "\n", ". ", "? ", "! "):
        boundary = transcript.rfind(separator, minimum, hard_end)
        if boundary >= minimum:
            return boundary + (1 if separator[-1].isspace() else len(separator))
    return hard_end


def transcript_spans(
    transcript: str, *, max_chars: int = MAX_EVIDENCE_CHARS
) -> tuple[TranscriptSpan, ...]:
    """Return ordered, bounded spans with deterministic offset-derived IDs.

    Blank-line-separated turns or paragraphs remain separate spans. Oversized
    blocks are split at a readable boundary when possible. Separating
    whitespace is not included in evidence spans. Every returned span remains
    an exact substring of ``transcript``; no normalization or model-produced
    text participates in evidence construction.
    """

    if not isinstance(transcript, str):
        raise TypeError("transcript must be text")
    if (
        isinstance(max_chars, bool)
        or not isinstance(max_chars, int)
        or max_chars <= 0
        or max_chars > MAX_EVIDENCE_CHARS
    ):
        raise ValueError("max_chars must be within the evidence limit")

    spans: list[TranscriptSpan] = []
    cursor = 0
    length = len(transcript)
    while cursor < length:
        while cursor < length and transcript[cursor].isspace():
            cursor += 1
        if cursor >= length:
            break
        separator = _BLANK_LINE.search(transcript, cursor)
        block_end = separator.start() if separator is not None else length
        while block_end > cursor and transcript[block_end - 1].isspace():
            block_end -= 1

        block_cursor = cursor
        while block_cursor < block_end:
            hard_end = min(block_end, block_cursor + max_chars)
            end = _preferred_end(transcript, block_cursor, hard_end, block_end)
            while end > block_cursor and transcript[end - 1].isspace():
                end -= 1
            if end <= block_cursor:
                end = hard_end
            text = transcript[block_cursor:end]
            spans.append(
                TranscriptSpan(
                    span_id=f"span-{block_cursor:06d}-{end:06d}",
                    start=block_cursor,
                    end=end,
                    text=text,
                )
            )
            block_cursor = end
            while block_cursor < block_end and transcript[block_cursor].isspace():
                block_cursor += 1
        cursor = separator.end() if separator is not None else length
    return tuple(spans)
