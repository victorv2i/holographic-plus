from __future__ import annotations

from dataclasses import replace

from memory_eval.extraction_arena import (
    DEFAULT_OUTPUTS_PATH,
    DECISIONS,
    load_candidate_outputs,
    load_extraction_arena,
)
from memory_eval.extraction_runtime_arena import score_extraction_runtime


def test_golden_outputs_replay_to_authoritative_runtime_decisions():
    arena = load_extraction_arena()
    score = score_extraction_runtime(
        arena,
        load_candidate_outputs(DEFAULT_OUTPUTS_PATH),
    )

    assert score.passed is True
    assert score.summary == {
        "total": 7,
        "passed": 7,
        "failed": 0,
        "decision_accuracy": 1.0,
        "reported_decisions_ignored": True,
        "isolated_temporary_databases": 7,
        "live_database_writes": 0,
    }
    assert {case.actual_decision for case in score.cases} == set(DECISIONS)
    assert next(
        case for case in score.cases if case.case_id == "reject-secret"
    ).processor_outcome is None


def test_runtime_score_ignores_candidate_self_reported_decisions():
    arena = load_extraction_arena()
    outputs = tuple(
        replace(output, decision="add" if output.decision != "add" else "reject")
        for output in load_candidate_outputs(DEFAULT_OUTPUTS_PATH)
    )

    score = score_extraction_runtime(arena, outputs)

    assert score.passed is True
    assert all(
        case.reported_decision != case.actual_decision for case in score.cases
    )


def test_runtime_score_exposes_reported_dedup_that_authority_resolves_as_add():
    arena = load_extraction_arena()
    outputs = list(load_candidate_outputs(DEFAULT_OUTPUTS_PATH))
    index = next(
        index
        for index, output in enumerate(outputs)
        if output.case_id == "dedup-existing-constraint"
    )
    output = outputs[index]
    outputs[index] = replace(
        output,
        facts=(replace(output.facts[0], content="Mara backs up on Fridays manually."),),
    )

    score = score_extraction_runtime(arena, outputs)
    case = next(
        case for case in score.cases if case.case_id == output.case_id
    )

    assert case.reported_decision == "dedup"
    assert case.actual_decision == "add"
    assert case.write_outcomes == ("inserted",)
    assert case.passed is False


def test_runtime_score_fails_mixed_multi_proposal_outcomes_without_precedence():
    arena = load_extraction_arena()
    outputs = list(load_candidate_outputs(DEFAULT_OUTPUTS_PATH))
    index = next(
        index
        for index, output in enumerate(outputs)
        if output.case_id == "add-grounded-preference"
    )
    output = outputs[index]
    outputs[index] = replace(output, facts=(output.facts[0], output.facts[0]))

    score = score_extraction_runtime(arena, outputs)
    case = next(
        case for case in score.cases if case.case_id == output.case_id
    )

    assert case.write_outcomes == ("inserted", "dedup")
    assert case.actual_decision is None
    assert case.passed is False
