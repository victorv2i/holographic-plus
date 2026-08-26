from __future__ import annotations

import json
from dataclasses import replace

import pytest

from memory_eval.extraction_arena import DEFAULT_CASES_PATH as SEED_CASES_PATH
from memory_eval.transcript_gate import (
    DEFAULT_CASES_PATH,
    PRODUCTION_FAILURE_CASE_ID,
    PRODUCTION_JUNK_OUTPUTS_PATH,
    ROLE_GOLD_OUTPUTS_PATH,
    GATE_THRESHOLDS,
    load_transcript_gate,
    load_predicted_outputs,
    score_transcript_gate,
    evaluate_capture_gate,
    main,
)


def test_seed_arena_is_not_the_capture_ship_gate():
    bank = load_transcript_gate()

    assert DEFAULT_CASES_PATH != SEED_CASES_PATH
    assert DEFAULT_CASES_PATH.name != "extraction_arena_seed.jsonl"
    assert len(bank.cases) >= 45
    assert "seed-7-is-not-the-gate" in bank.tags or any(
        "not-seed" in case.tags or "real-transcript" in case.tags for case in bank.cases
    )


def test_bank_cases_are_role_structured_with_hand_labels():
    bank = load_transcript_gate()

    kinds = set()
    speakers = set()
    forbidden_speakers = set()
    for case in bank.cases:
        assert case.turns, case.case_id
        assert {turn.role for turn in case.turns} <= {"user", "assistant", "tool"}
        assert any(turn.role == "user" for turn in case.turns) or case.expected_decision in {
            "abstain",
            "reject",
        }
        for fact in case.expected_facts:
            assert fact.kind in {"typed_slot", "untyped_durable"}
            assert fact.speaker == "user"
            kinds.add(fact.kind)
            speakers.add(fact.speaker)
        for span in case.forbidden_spans:
            assert span.speaker in {"assistant", "tool"}
            forbidden_speakers.add(span.speaker)

    assert kinds == {"typed_slot", "untyped_durable"}
    assert speakers == {"user"}
    assert forbidden_speakers == {"assistant", "tool"}


def test_production_replay_case_requires_zero_assistant_and_tool_facts():
    bank = load_transcript_gate()
    replay = next(case for case in bank.cases if case.case_id == PRODUCTION_FAILURE_CASE_ID)

    assert "production-failure" in replay.tags
    assert any(turn.role == "assistant" for turn in replay.turns)
    assert any(turn.role == "tool" for turn in replay.turns)
    assert any("METIS_FLEET_OK" in turn.content for turn in replay.turns if turn.role == "tool")
    assert any(
        "Brick" in turn.content or "brick" in turn.content.lower()
        for turn in replay.turns
        if turn.role == "assistant"
    )
    assert any(
        "implement the concise style" in turn.content.lower()
        for turn in replay.turns
        if turn.role == "assistant"
    )
    assert replay.expected_facts
    assert all(fact.speaker == "user" for fact in replay.expected_facts)
    assert any(
        span.class_name == "assistant_authored" for span in replay.forbidden_spans
    )
    assert any(span.class_name == "tool_banner" for span in replay.forbidden_spans)


def test_four_metrics_are_scored_separately_and_fail_independently():
    bank = load_transcript_gate()
    gold = load_predicted_outputs(ROLE_GOLD_OUTPUTS_PATH)
    gold_score = score_transcript_gate(bank, gold)

    assert set(gold_score.metrics) >= {
        "speaker_attribution",
        "typed_slot_completeness",
        "incidental_durable_recall",
        "forbidden_assistant_tool",
    }
    assert gold_score.metrics["forbidden_assistant_tool"]["leaks"] == 0
    assert gold_score.metrics["forbidden_assistant_tool"]["rate"] == 0.0

    junk = load_predicted_outputs(PRODUCTION_JUNK_OUTPUTS_PATH)
    junk_score = score_transcript_gate(bank, junk)
    replay = next(
        case for case in junk_score.cases if case.case_id == PRODUCTION_FAILURE_CASE_ID
    )

    assert junk_score.metrics["forbidden_assistant_tool"]["leaks"] >= 2
    assert replay.assistant_authored_facts >= 1
    assert replay.tool_banner_facts >= 1
    assert replay.forbidden_leaks
    assert junk_score.metrics["speaker_attribution"]["accuracy"] < 1.0
    assert junk_score.passed is False
    assert gold_score.metrics["forbidden_assistant_tool"]["leaks"] == 0


def test_scorer_penalizes_assistant_evidence_labeled_as_user():
    bank = load_transcript_gate()
    gold = list(load_predicted_outputs(ROLE_GOLD_OUTPUTS_PATH))
    replay = next(output for output in gold if output.case_id == PRODUCTION_FAILURE_CASE_ID)
    poisoned = replace(
        replay,
        facts=replay.facts
        + (
            replace(
                replay.facts[0],
                content="I'm Brick, your Minecraft server setup assistant.",
                asserted_by="user",
                evidence_role="assistant",
                evidence_excerpt="I'm Brick, your Minecraft server setup assistant.",
                state=None,
            ),
        ),
    )
    by_id = {output.case_id: output for output in gold}
    by_id[replay.case_id] = poisoned

    score = score_transcript_gate(bank, by_id.values())
    case = next(item for item in score.cases if item.case_id == replay.case_id)

    assert case.speaker_misattributions >= 1
    assert case.assistant_authored_facts >= 1
    assert "speaker misattribution" in " ".join(case.failures).lower() or case.forbidden_leaks
    assert score.metrics["forbidden_assistant_tool"]["leaks"] >= 1


def test_typed_slot_completeness_requires_a_full_state_group():
    bank = load_transcript_gate()
    gold = list(load_predicted_outputs(ROLE_GOLD_OUTPUTS_PATH))
    typed_case = next(
        case
        for case in bank.cases
        if any(fact.kind == "typed_slot" for fact in case.expected_facts)
        and case.case_id != PRODUCTION_FAILURE_CASE_ID
    )
    output = next(item for item in gold if item.case_id == typed_case.case_id)
    demoted = replace(
        output,
        facts=tuple(replace(fact, state=None) for fact in output.facts),
    )
    by_id = {item.case_id: item for item in gold}
    by_id[output.case_id] = demoted

    score = score_transcript_gate(bank, by_id.values())

    assert score.metrics["typed_slot_completeness"]["silent_demotions"] >= 1
    assert score.metrics["typed_slot_completeness"]["recall"] < 1.0


def test_capture_gate_fails_closed_on_production_junk_and_threshold_is_stricter_than_seed():
    bank = load_transcript_gate()
    junk_gate = evaluate_capture_gate(
        score_transcript_gate(bank, load_predicted_outputs(PRODUCTION_JUNK_OUTPUTS_PATH))
    )
    gold_gate = evaluate_capture_gate(
        score_transcript_gate(bank, load_predicted_outputs(ROLE_GOLD_OUTPUTS_PATH))
    )

    assert GATE_THRESHOLDS["forbidden_assistant_tool_rate"] == 0.0
    assert GATE_THRESHOLDS["speaker_misattribution_rate"] == 0.0
    assert GATE_THRESHOLDS["typed_slot_precision_min"] == 1.0
    assert GATE_THRESHOLDS["silent_demotion_rate"] == 0.0
    assert junk_gate["ship"] is False
    assert "forbidden_assistant_tool" in junk_gate["failed_checks"]
    assert gold_gate["ship"] is True
    assert gold_gate["failed_checks"] == []


def test_cli_scores_saved_outputs_offline(tmp_path, capsys):
    report = tmp_path / "gate.json"
    exit_code = main(
        [
            "--cases",
            str(DEFAULT_CASES_PATH),
            "--outputs",
            str(ROLE_GOLD_OUTPUTS_PATH),
            "--report",
            str(report),
            "--require-ship",
        ]
    )
    printed = json.loads(capsys.readouterr().out)
    saved = json.loads(report.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert printed["metrics"]["forbidden_assistant_tool"]["leaks"] == 0
    assert saved["metadata"]["offline"] is True
    assert saved["metadata"]["provider_calls"] == 0
    assert saved["gate"]["ship"] is True

    fail_code = main(
        [
            "--cases",
            str(DEFAULT_CASES_PATH),
            "--outputs",
            str(PRODUCTION_JUNK_OUTPUTS_PATH),
            "--require-ship",
        ]
    )
    assert fail_code == 1
