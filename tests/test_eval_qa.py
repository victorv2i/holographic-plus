from __future__ import annotations

import pytest

from memory_eval.qa import (
    LOCAL_JUDGE_PROMPT,
    AnswerProvider,
    PackedContextAnswerer,
    token_f1,
)


def test_token_f1_is_acl_style_overlap_after_normalization():
    assert token_f1("The Local-First Databases", "local first databases") == pytest.approx(1.0)
    assert token_f1("Contoso Research", "Northwind Labs") == pytest.approx(0.0)
    assert token_f1("", "something") == pytest.approx(0.0)


def test_local_judge_prompt_is_not_generous():
    assert "be generous" not in LOCAL_JUDGE_PROMPT.lower()
    assert "same topic" not in LOCAL_JUDGE_PROMPT.lower()


def test_answer_provider_is_not_given_the_gold_answer():
    seen: list[dict[str, object]] = []

    class Probe:
        identity = "probe-reader"

        def answer(self, question: str, context: str, *, now: str | None = None) -> str:
            seen.append({"question": question, "context": context, "now": now})
            return "Local-first databases"

    packed = PackedContextAnswerer(Probe())
    text = packed.answer("What did Riv study?", ["Riv researched local-first databases."])

    assert text == "Local-first databases"
    assert "Local-first databases" not in seen[0]["context"] or "researched" in seen[0]["context"]
    assert "gold" not in seen[0]
    assert not hasattr(AnswerProvider, "gold")
