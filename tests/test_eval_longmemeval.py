from __future__ import annotations

import json

import pytest

from memory_eval.longmemeval_adapter import (
    LME_ACQUISITION,
    LME_SMOKE_PATH,
    load_longmemeval,
    lme_questions,
    lme_sessions,
    run_longmemeval,
)
from memory_eval.protocol import LONGMEMEVAL_QUESTION_TYPES


def test_longmemeval_smoke_passes_question_date_as_now_and_splits_types():
    dataset = load_longmemeval(LME_SMOKE_PATH, split="S")
    questions = lme_questions(dataset)
    sessions = lme_sessions(dataset)

    assert {item.question_type for item in questions} == {"knowledge-update", "abstention"}
    assert questions[0].reference_time == "2026-07-20"
    assert sessions[0].observed_at == "2025-04-01"
    assert "Mara works at Northwind Labs" in sessions[0].transcript
    assert all(item.question_type in LONGMEMEVAL_QUESTION_TYPES for item in questions)


def test_longmemeval_refuses_to_label_oracle_as_s(tmp_path):
    payload = json.loads(LME_SMOKE_PATH.read_text(encoding="utf-8"))
    path = tmp_path / "longmemeval_oracle.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="oracle"):
        load_longmemeval(path, split="S")


def test_missing_longmemeval_dataset_is_blocked_not_scored(tmp_path):
    missing = tmp_path / "longmemeval_s_cleaned.json"

    with pytest.raises(FileNotFoundError, match="longmemeval"):
        load_longmemeval(missing, split="S")
    report = run_longmemeval(missing, split="S")

    assert report["status"] == "blocked"
    assert report["scores"] is None
    assert "huggingface" in LME_ACQUISITION.lower()
    assert "do not report oracle" in LME_ACQUISITION.lower()
