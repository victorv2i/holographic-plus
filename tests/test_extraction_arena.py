from __future__ import annotations

from dataclasses import replace
import json
import socket
import urllib.request

import pytest

from memory_eval.extraction_arena import (
    CASE_SCHEMA_PATH,
    DEFAULT_CASES_PATH,
    DEFAULT_OUTPUTS_PATH,
    DECISIONS,
    OUTPUT_SCHEMA_PATH,
    CandidateFact,
    CandidateOutput,
    load_candidate_outputs,
    load_extraction_arena,
    main,
    resolve_candidate_evidence,
    score_extraction_arena,
)
from enfold.extraction_spans import transcript_spans


def test_seed_arena_covers_every_decision_and_golden_saved_outputs_pass():
    arena = load_extraction_arena()
    outputs = load_candidate_outputs(DEFAULT_OUTPUTS_PATH)

    score = score_extraction_arena(arena, outputs)

    assert {case.expected_decision for case in arena.cases} == set(DECISIONS)
    assert len(arena.cases) == 7
    assert score.passed is True
    assert score.summary["passed"] == 7
    assert score.summary["case_pass_rate"] == 1.0
    assert score.summary["decision_accuracy"] == 1.0
    assert score.summary["facts"] == {
        "required": 4,
        "matched": 4,
        "predicted": 4,
        "unexpected": 0,
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
    }
    assert score.summary["evidence"]["grounding_accuracy"] == 1.0
    assert score.summary["evidence"]["contract_compatible_spans"] == 4
    assert score.summary["evidence"]["contract_compatible_accuracy"] == 1.0
    assert score.summary["evidence"]["exact_expected_spans"] == 4
    assert score.summary["evidence"]["exact_expected_accuracy"] == 1.0
    assert score.summary["evidence"]["runtime_compatible_spans"] == 0
    assert score.summary["evidence"]["runtime_compatible_accuracy"] == 0.0
    assert score.summary["category_accuracy"] == 1.0
    assert score.summary["sensitivity_accuracy"] == 1.0
    assert score.summary["typed_state_accuracy"] == 1.0
    assert score.summary["typed_state"] == {
        "expected": 3,
        "predicted": 3,
        "correct": 3,
        "false_positives": 0,
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
    }
    assert score.summary["forbidden"] == {
        "leaks": 0,
        "leak_cases": 0,
        "protected_cases": 4,
        "case_leak_rate": 0.0,
    }


def test_scorer_exposes_paraphrased_evidence_metadata_state_and_secret_failures():
    arena = load_extraction_arena()
    outputs = list(load_candidate_outputs(DEFAULT_OUTPUTS_PATH))
    by_id = {output.case_id: output for output in outputs}

    add = by_id["add-grounded-preference"]
    bad_add_fact = replace(
        add.facts[0],
        category="general",
        sensitivity="sensitive",
        evidence_excerpt="Mara likes local tools.",
        state={
            "kind": "preference",
            "subject": "person:mara",
            "predicate": "tooling",
            "value": "cloud-only",
            "confidence": 0.99,
        },
    )
    by_id[add.case_id] = replace(add, facts=(bad_add_fact,))

    secret = by_id["reject-secret"]
    by_id[secret.case_id] = CandidateOutput(
        case_id=secret.case_id,
        decision="add",
        facts=(
            CandidateFact(
                content="The deployment API token is DEMO_TOKEN_REDACTED.",
                category="credential",
                sensitivity="sensitive",
                evidence_span=None,
                evidence_excerpt="My deployment API token is DEMO_TOKEN_REDACTED.",
                state=None,
                expectation_key=None,
            ),
        ),
        metadata={"provider": "unsafe-synthetic"},
    )

    score = score_extraction_arena(arena, by_id.values())
    add_score = next(case for case in score.cases if case.case_id == add.case_id)
    secret_score = next(case for case in score.cases if case.case_id == secret.case_id)

    assert add_score.required_matched == 1
    assert add_score.grounded_predictions == 0
    assert "evidence span mismatch: mara-local-tools" in add_score.failures
    assert "category mismatch: mara-local-tools" in add_score.failures
    assert "sensitivity mismatch: mara-local-tools" in add_score.failures
    assert "typed state mismatch: mara-local-tools" in add_score.failures
    assert secret_score.forbidden_leaks == ("api-token",)
    assert "reject must return no facts" in secret_score.failures
    assert score.summary["forbidden"]["leaks"] == 1
    assert score.summary["evidence"]["grounding_accuracy"] < 1.0
    assert score.summary["category_accuracy"] < 1.0
    assert score.summary["sensitivity_accuracy"] < 1.0
    assert score.summary["typed_state_accuracy"] < 1.0


def test_typed_state_metrics_penalize_state_on_expected_untyped_fact():
    arena = load_extraction_arena()
    outputs = list(load_candidate_outputs(DEFAULT_OUTPUTS_PATH))
    output_index = next(
        index
        for index, output in enumerate(outputs)
        if output.case_id == "dedup-existing-constraint"
    )
    output = outputs[output_index]
    outputs[output_index] = replace(
        output,
        facts=(
            replace(
                output.facts[0],
                state={
                    "kind": "state",
                    "subject": "person:mara",
                    "predicate": "backup_schedule",
                    "value": "Friday",
                    "confidence": 0.99,
                },
            ),
        ),
    )

    score = score_extraction_arena(arena, outputs)

    assert score.summary["typed_state"] == {
        "expected": 3,
        "predicted": 4,
        "correct": 3,
        "false_positives": 1,
        "precision": 0.75,
        "recall": 1.0,
        "f1": pytest.approx(6 / 7),
    }
    assert score.summary["typed_state_accuracy"] == pytest.approx(6 / 7)
    case_score = next(
        item for item in score.cases if item.case_id == output.case_id
    )
    assert "typed state mismatch: mara-friday-backup" in case_score.failures


def test_typed_scoring_matches_runtime_aliases_and_fail_soft_validation():
    arena = load_extraction_arena()
    outputs = list(load_candidate_outputs(DEFAULT_OUTPUTS_PATH))
    index = next(
        index
        for index, output in enumerate(outputs)
        if output.case_id == "add-grounded-preference"
    )
    output = outputs[index]
    original = output.facts[0]
    outputs[index] = replace(
        output,
        facts=(
            replace(
                original,
                state={
                    "kind": "preference",
                    "subject": "PERSON:MARA",
                    "predicate": "Tooling",
                    "object": "local-first",
                    "confidence": 0.97,
                },
            ),
        ),
    )

    alias_score = score_extraction_arena(arena, outputs)
    alias_case = next(
        case for case in alias_score.cases if case.case_id == output.case_id
    )
    assert alias_case.passed is True

    outputs[index] = replace(
        output,
        facts=(
            replace(
                original,
                state={
                    "kind": "preference",
                    "subject": "person:mara",
                    "predicate": "tooling",
                    "object": "local-first",
                    "value": "contradiction",
                    "confidence": 0.97,
                },
            ),
        ),
    )
    invalid_score = score_extraction_arena(arena, outputs)
    invalid_case = next(
        case for case in invalid_score.cases if case.case_id == output.case_id
    )
    assert invalid_case.typed_predicted == 0
    assert "typed state mismatch: mara-local-tools" in invalid_case.failures


def test_scorer_accepts_exact_v2_runtime_span_containing_expected_support():
    arena = load_extraction_arena()
    outputs = list(load_candidate_outputs(DEFAULT_OUTPUTS_PATH))
    case = next(case for case in arena.cases if case.case_id == "add-grounded-preference")
    output_index = next(
        index
        for index, output in enumerate(outputs)
        if output.case_id == case.case_id
    )
    runtime_span = transcript_spans(case.transcript)[0]
    output = outputs[output_index]
    outputs[output_index] = replace(
        output,
        facts=(
            replace(
                output.facts[0],
                evidence_span=(runtime_span.start, runtime_span.end),
                evidence_excerpt=runtime_span.text,
            ),
        ),
    )

    score = score_extraction_arena(arena, outputs)
    case_score = next(item for item in score.cases if item.case_id == case.case_id)

    assert case_score.evidence_compatible == 1
    assert case_score.evidence_expected_exact == 0
    assert case_score.evidence_runtime_compatible == 1
    assert case_score.passed is True


def test_evidence_resolution_requires_unique_excerpt_or_valid_explicit_span():
    transcript = "Mara said yes. Mara said yes."
    ambiguous = CandidateFact(
        "Mara said yes.",
        "general",
        "normal",
        None,
        "Mara said yes.",
        None,
        None,
    )
    explicit = replace(
        ambiguous,
        evidence_span=(0, len("Mara said yes.")),
    )
    inconsistent = replace(
        explicit,
        evidence_excerpt="Mara said no.",
    )

    span, reason = resolve_candidate_evidence(transcript, ambiguous)
    assert span is None
    assert reason == "ambiguous_excerpt"

    span, reason = resolve_candidate_evidence(transcript, explicit)
    assert reason is None
    assert span is not None
    assert (span.start, span.end, span.text) == (0, 14, "Mara said yes.")

    span, reason = resolve_candidate_evidence(transcript, inconsistent)
    assert span is None
    assert reason == "span_excerpt_mismatch"


def test_json_and_jsonl_loaders_are_provider_neutral_and_reject_duplicate_ids(tmp_path):
    rows = [
        json.loads(line)
        for line in DEFAULT_OUTPUTS_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    wrapped = tmp_path / "saved-outputs.json"
    wrapped.write_text(json.dumps({"metadata": {"provider": "any"}, "outputs": rows}))

    outputs = load_candidate_outputs(wrapped)
    assert len(outputs) == 7
    assert {output.metadata["provider"] for output in outputs} == {"synthetic-gold"}

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(json.dumps([rows[0], rows[0]]))
    with pytest.raises(ValueError, match="case ids must be unique"):
        load_candidate_outputs(duplicate)


def test_cli_scores_saved_outputs_only_and_reports_zero_calls(tmp_path, monkeypatch, capsys):
    def network_forbidden(*_args, **_kwargs):
        raise AssertionError("offline extraction Arena attempted network access")

    monkeypatch.setattr(socket, "create_connection", network_forbidden)
    monkeypatch.setattr(urllib.request, "urlopen", network_forbidden)
    report = tmp_path / "score.json"

    exit_code = main(
        [
            "--cases",
            str(DEFAULT_CASES_PATH),
            "--outputs",
            str(DEFAULT_OUTPUTS_PATH),
            "--report",
            str(report),
            "--require-perfect",
        ]
    )

    printed = json.loads(capsys.readouterr().out)
    saved = json.loads(report.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert printed["case_pass_rate"] == 1.0
    assert saved["metadata"]["offline"] is True
    assert saved["metadata"]["provider_calls"] == 0
    assert saved["metadata"]["model_calls"] == 0
    assert saved["metadata"]["network_calls"] == 0
    assert saved["summary"]["passed"] == 7


def test_bundled_json_schemas_and_fixture_rows_are_parseable():
    case_schema = json.loads(CASE_SCHEMA_PATH.read_text(encoding="utf-8"))
    output_schema = json.loads(OUTPUT_SCHEMA_PATH.read_text(encoding="utf-8"))

    assert case_schema["$id"].endswith("extraction-arena-case-v1.json")
    assert output_schema["$id"].endswith("extraction-arena-output-v1.json")
    assert load_extraction_arena().source_path == DEFAULT_CASES_PATH.resolve()
