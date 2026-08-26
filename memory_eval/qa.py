"""Local reader and judge helpers for conversational memory QA.

The reader never receives the gold answer. The judge prompt is strict and is
not comparable to vendor GPT-4o-J.
"""

from __future__ import annotations

from collections import Counter
import re
from typing import Protocol, Sequence


_ARTICLE = frozenset({"a", "an", "the"})
_NON_ALNUM = re.compile(r"[^a-z0-9\s]+")

LOCAL_JUDGE_PROMPT = """You are a strict factual judge.
Compare the predicted answer to the gold answer.
Score CORRECT only when the predicted answer contains the same facts as the gold
answer after ignoring case, punctuation, and small wording changes.
Score INCORRECT when the prediction is on a related topic but misses the fact,
adds a different fact, or abstains when a gold answer exists.
Do not reward topical overlap. Reply with CORRECT or INCORRECT only.
"""


class AnswerProvider(Protocol):
    identity: str

    def answer(self, question: str, context: str, *, now: str | None = None) -> str:
        """Produce a short answer from question plus packed context only."""


def normalize_answer_tokens(text: str) -> list[str]:
    lowered = _NON_ALNUM.sub(" ", str(text).lower())
    return [token for token in lowered.split() if token and token not in _ARTICLE]


def token_f1(predicted: str, gold: str) -> float:
    """ACL-style token-overlap F1 after article and punctuation stripping."""

    pred = normalize_answer_tokens(predicted)
    exp = normalize_answer_tokens(gold)
    if not pred and not exp:
        return 1.0
    if not pred or not exp:
        return 0.0
    overlap = sum((Counter(pred) & Counter(exp)).values())
    precision = overlap / len(pred)
    recall = overlap / len(exp)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


class PackedContextAnswerer:
    """Join retrieved fact strings and call a reader that cannot see gold."""

    def __init__(self, reader: AnswerProvider):
        self._reader = reader
        self.identity = reader.identity

    def answer(
        self,
        question: str,
        facts: Sequence[str],
        *,
        now: str | None = None,
    ) -> str:
        context = "\n".join(str(fact) for fact in facts)
        return self._reader.answer(question, context, now=now)
