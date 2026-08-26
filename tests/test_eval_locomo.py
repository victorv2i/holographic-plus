from __future__ import annotations

import pytest

from memory_eval.locomo_adapter import (
    LOCOMO_ACQUISITION,
    LOCOMO_SMOKE_PATH,
    load_locomo,
    locomo_questions,
    locomo_sessions,
    run_locomo,
    score_locomo_retrieval,
)
from memory_eval.protocol import LOCOMO_CATEGORY_TABLE, LOCOMO_DATASET


def test_locomo_smoke_fixture_keeps_category_5_and_walks_sessions_in_order():
    dataset = load_locomo(LOCOMO_SMOKE_PATH)

    sessions = locomo_sessions(dataset)
    questions = locomo_questions(dataset)

    assert [session.session_key for session in sessions] == ["session_1", "session_2"]
    assert sessions[0].observed_at == "1:56 pm on 8 May, 2023"
    assert "Riv:" in sessions[0].transcript
    assert "Nia:" in sessions[0].transcript
    assert {item.category for item in questions} == {1, 2, 5}
    assert any(item.category == 5 and item.adversarial_answer for item in questions)
    assert questions[0].evidence == ("D1:2",)
    assert LOCOMO_CATEGORY_TABLE[5]["name"] == "adversarial"


def test_locomo_adapter_refuses_to_drop_category_5():
    dataset = load_locomo(LOCOMO_SMOKE_PATH)

    with pytest.raises(ValueError, match="never drop"):
        locomo_questions(dataset, drop_category_5=True)


def test_locomo_retrieval_recall_uses_evidence_dialog_ids():
    dataset = load_locomo(LOCOMO_SMOKE_PATH)
    questions = locomo_questions(dataset)
    retrieved = [
        ["D1:2", "D1:1"],
        ["D2:1"],
        ["D9:9"],
    ]

    scores = score_locomo_retrieval(questions, retrieved, k_values=(1, 3))

    assert scores["kind"] == "retrieval_only"
    assert scores["by_category"]["5"]["cases"] == 1
    assert scores["recall@1"] == pytest.approx(2 / 3)
    assert scores["recall@3"] == pytest.approx(2 / 3)


def test_missing_locomo_dataset_is_blocked_not_scored(tmp_path):
    missing = tmp_path / "locomo10.json"

    with pytest.raises(FileNotFoundError, match="locomo10.json"):
        load_locomo(missing)
    report = run_locomo(missing)

    assert report["status"] == "blocked"
    assert report["scores"] is None
    assert LOCOMO_DATASET["sha256"] in report["acquisition"]
    assert "curl" in LOCOMO_ACQUISITION


def test_full_locomo_hash_is_checked_when_present(tmp_path):
    path = tmp_path / "locomo10.json"
    path.write_text("[]\n", encoding="utf-8")

    with pytest.raises(ValueError, match="sha256"):
        load_locomo(path, require_published_hash=True)
