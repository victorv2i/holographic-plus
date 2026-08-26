"""Prompt-boundary classifiers shared by context packing and extraction."""

from __future__ import annotations

import re
import unicodedata


_PROMPT_CONTROL_RE = re.compile(
    r"(?:"
    r"(?:^|\n)[ \t]*(?:[#>*-]+[ \t]*)?(?:system|developer|assistant)\s*"
    r"(?:(?:message|prompt|instructions?)\s*)?:"
    r"|\[(?:/?inst|system)\]"
    r"|<\/?(?:system|developer|assistant)(?:\s|>)"
    r"|\b(?:ignore|disregard|forget|override|bypass|follow|obey)\b"
    r".{0,96}\b(?:instructions?|prompt|rules?|policy|memory)\b"
    r"|\b(?:reveal|expose|exfiltrate|disclose)\b"
    r".{0,64}\b(?:secrets?|private\s+data|hidden\s+instructions?)\b"
    r")",
    re.IGNORECASE,
)
_STRUCTURAL_INJECTION_RE = re.compile(
    r"(?:"
    r"(?:^|\n)[ \t]*(?:[#>*-]+[ \t]*)?(?:system|developer|assistant)\s*"
    r"(?:(?:message|prompt|instructions?)\s*)?:"
    r"|\[(?:/?inst|system)\]"
    r"|<\/?(?:system|developer|assistant)(?:\s|>)"
    r")",
    re.IGNORECASE,
)
_LEADING_CONTROL_RE = re.compile(
    r"(?i)^\s*(?:ignore|disregard|forget|override|bypass|follow|obey)\b"
    r".{0,96}\b(?:instructions?|prompt|rules?|policy|memory)\b"
)
_EXFIL_RE = re.compile(
    r"(?i)\b(?:reveal|expose|exfiltrate|disclose)\b"
    r".{0,64}\b(?:secrets?|private\s+data|hidden\s+instructions?)\b"
)
_TASK_SHAPED_RE = re.compile(
    r"(?is)^\s*(?:"
    r"pitch\b|"
    r"todo\b|"
    r"action items?:"
    r")"
)
_HOMOGLYPH_FOLD = str.maketrans(
    {
        "\u0430": "a",
        "\u0435": "e",
        "\u043e": "o",
        "\u0440": "p",
        "\u0441": "c",
        "\u0443": "y",
        "\u0445": "x",
        "\u0456": "i",
        "\u0410": "a",
        "\u0415": "e",
        "\u041e": "o",
        "\u0420": "p",
        "\u0421": "c",
        "\u0423": "y",
        "\u0425": "x",
        "\u0406": "i",
    }
)
_ECHO_STOPWORDS = frozenset(
    {"a", "an", "as", "at", "in", "longer", "no", "now", "the"}
)
_ECHO_JACCARD = 0.65
_ECHO_MIN_SHARED = 3


def normalized_prompt_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value)
    return "".join(
        character
        for character in normalized
        if unicodedata.category(character) != "Cf"
    )


def instruction_shaped(content: str) -> bool:
    text = normalized_prompt_text(content)
    return not text.strip() or _PROMPT_CONTROL_RE.search(text) is not None


def _echo_folded_text(content: str) -> str:
    return normalized_prompt_text(content).translate(_HOMOGLYPH_FOLD)


def _echo_alnum_key(content: str) -> str:
    return "".join(ch for ch in _echo_folded_text(content).lower() if ch.isalnum())


def _echo_tokens(content: str) -> frozenset[str]:
    tokens: list[str] = []
    current: list[str] = []
    for character in _echo_folded_text(content).lower():
        if character.isalnum():
            current.append(character)
            continue
        if current:
            tokens.append("".join(current))
            current = []
    if current:
        tokens.append("".join(current))
    return frozenset(tokens)


def prompt_example_echo_reason(content: str) -> str | None:
    """Return why a proposal is a near-echo of a few-shot prompt example."""

    from .extraction_contract import PROMPT_EXAMPLE_CONTENTS

    key = _echo_alnum_key(content)
    if not key:
        return None
    content_tokens = _echo_tokens(content) - _ECHO_STOPWORDS
    for example in PROMPT_EXAMPLE_CONTENTS:
        example_key = _echo_alnum_key(example)
        if not example_key:
            continue
        if key == example_key or example_key in key or key in example_key:
            return "prompt_example_echo"
        example_tokens = _echo_tokens(example) - _ECHO_STOPWORDS
        if not example_tokens:
            continue
        shared = content_tokens & example_tokens
        union = content_tokens | example_tokens
        if len(shared) >= _ECHO_MIN_SHARED and len(shared) / len(union) >= _ECHO_JACCARD:
            return "prompt_example_echo"
    return None


def extraction_instruction_shaped(content: str) -> bool:
    """True only for leading or structural injection, not mid-sentence verbs."""

    text = normalized_prompt_text(content)
    if not text.strip():
        return True
    return (
        _STRUCTURAL_INJECTION_RE.search(text) is not None
        or _LEADING_CONTROL_RE.search(text) is not None
        or _EXFIL_RE.search(text) is not None
    )


def task_shaped(content: str) -> bool:
    return _TASK_SHAPED_RE.search(normalized_prompt_text(content)) is not None


def ephemeral_extraction_reason(content: str) -> str | None:
    """Return why an automatic proposal is not durable memory, if it isn't."""

    if prompt_example_echo_reason(content) is not None:
        return "prompt_example_echo"
    if extraction_instruction_shaped(content):
        return "instruction_shaped_content"
    if task_shaped(content):
        return "task_shaped_content"
    return None
