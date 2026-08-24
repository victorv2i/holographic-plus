from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import stat

import pytest

from enfold.extraction_processor import ExtractedMemory
from memory_eval.extraction_arena import (
    DEFAULT_OUTPUTS_PATH,
    load_candidate_outputs,
    load_extraction_arena,
)
from memory_eval.extraction_benchmark import (
    ADAPTER_CONFIG_VERSION,
    BENCHMARK_REPORT_VERSION,
    PROPOSAL_ARTIFACT_VERSION,
    _write,
    benchmark_report,
    dry_run_plan,
    load_benchmark_adapter,
    main,
    proposal_artifact,
    run_extraction_benchmark,
)


class _StepClock:
    def __init__(self, step: int = 10):
        self.value = 0
        self.step = step

    def __call__(self) -> int:
        value = self.value
        self.value += self.step
        return value


class _GoldenExtractor:
    identity = "fake:golden:durable-memory-v3"

    def __init__(self):
        arena = load_extraction_arena()
        cases = {case.case_id: case for case in arena.cases}
        self._outputs = {
            cases[output.case_id].transcript: tuple(
                ExtractedMemory(
                    content=fact.content,
                    category=fact.category,
                    evidence_excerpt=fact.evidence_excerpt,
                    sensitivity=fact.sensitivity,
                    state=fact.state,
                )
                for fact in output.facts
            )
            for output in load_candidate_outputs(DEFAULT_OUTPUTS_PATH)
        }
        self.calls: list[str] = []

    def extract(self, envelope):
        self.calls.append(envelope.transcript)
        return self._outputs[envelope.transcript]


def _adapter_config(*, recipe=None, environment=None):
    return {
        "schema_version": ADAPTER_CONFIG_VERSION,
        "host": {
            "type": "subprocess",
            "argv": ["/does/not/exist", "--model", "fixture"],
            "model_identity": "fixture-model",
            "prompt_identity": "durable-memory-v3",
            "environment": environment or {},
        },
        "recipe": recipe or {
            "decoder": {"seed": 7, "temperature": 0},
            "model_artifact_digest": "sha256:" + "a" * 64,
            "runtime": "fixture-runtime-1",
        },
    }


def _contains_key(value, key: str) -> bool:
    if isinstance(value, dict):
        return key in value or any(_contains_key(item, key) for item in value.values())
    if isinstance(value, list):
        return any(_contains_key(item, key) for item in value)
    return False


def test_fake_extractor_saves_proposals_and_uses_runtime_decisions():
    arena = load_extraction_arena()
    extractor = _GoldenExtractor()

    result = run_extraction_benchmark(
        arena,
        extractor,
        adapter_recipe={"decoder": {"temperature": 0}},
        repetitions=2,
        timing_class="warm",
        clock_ns=_StepClock(),
    )

    assert result.passed is True
    assert len(extractor.calls) == 12
    assert result.summary == {
        "adapter_calls": 12,
        "adapter_errors": 0,
        "case_run_pass_rate": 1.0,
        "case_runs": 14,
        "failed": 0,
        "latency_ns": {
            "maximum": 10,
            "minimum": 10,
            "p50": 10,
            "p95": 10,
        },
        "passed": 14,
        "policy_rejections": 2,
        "reported_model_decisions": 0,
        "runtime_decisions_authoritative": True,
    }
    assert {case.actual_decision for case in result.cases} == {
        "abstain",
        "add",
        "conflict",
        "dedup",
        "reject",
        "supersede",
    }
    rejected = [
        run for run in result.proposal_runs if run.adapter_outcome == "policy_rejected"
    ]
    assert len(rejected) == 2
    assert all(run.elapsed_ns is None and not run.proposals for run in rejected)

    artifact = proposal_artifact(result)
    assert artifact["schema_version"] == PROPOSAL_ARTIFACT_VERSION
    assert _contains_key(artifact, "decision") is False
    assert _contains_key(artifact, "expected_decision") is False
    assert _contains_key(artifact, "actual_decision") is False
    report = benchmark_report(result)
    assert report["schema_version"] == BENCHMARK_REPORT_VERSION
    assert report["metadata"]["authoritative_lifecycle"] == (
        "extraction_runtime_arena"
    )


def test_adapter_failures_are_redacted_and_cannot_pass_as_abstention():
    arena = load_extraction_arena()
    extractor = _GoldenExtractor()
    target = next(
        case.transcript
        for case in arena.cases
        if case.case_id == "abstain-ephemeral-chatter"
    )
    original_extract = extractor.extract

    def fail_one(envelope):
        if envelope.transcript == target:
            raise RuntimeError("private provider body must not enter artifacts")
        return original_extract(envelope)

    extractor.extract = fail_one
    result = run_extraction_benchmark(
        arena,
        extractor,
        adapter_recipe={},
        clock_ns=_StepClock(),
    )

    case = next(item for item in result.cases if item.case_id == "abstain-ephemeral-chatter")
    assert case.actual_decision == "abstain"
    assert case.runtime_passed is True
    assert case.offline_passed is True
    assert case.error_code == "extractor_failed"
    assert case.passed is False
    assert result.summary["adapter_errors"] == 1
    rendered = json.dumps(proposal_artifact(result))
    assert "private provider body" not in rendered


def test_nonstandard_policy_fields_fail_instead_of_being_dropped():
    arena = load_extraction_arena()
    extractor = _GoldenExtractor()
    target = next(
        case.transcript
        for case in arena.cases
        if case.case_id == "add-grounded-preference"
    )
    original_extract = extractor.extract

    def change_authority(envelope):
        proposals = original_extract(envelope)
        if envelope.transcript == target:
            return (replace(proposals[0], source_authority=0.9),)
        return proposals

    extractor.extract = change_authority
    result = run_extraction_benchmark(
        arena,
        extractor,
        adapter_recipe={},
        clock_ns=_StepClock(),
    )

    case = next(item for item in result.cases if item.case_id == "add-grounded-preference")
    assert case.error_code == "adapter_invalid_output"
    assert case.actual_decision == "abstain"
    assert case.passed is False


def test_safe_evidence_identity_metadata_is_retained():
    arena = load_extraction_arena()
    extractor = _GoldenExtractor()
    target = next(
        case.transcript
        for case in arena.cases
        if case.case_id == "add-grounded-preference"
    )
    original_extract = extractor.extract

    def add_span_identity(envelope):
        proposals = original_extract(envelope)
        if envelope.transcript == target:
            return (
                replace(
                    proposals[0],
                    metadata={"evidence_span_id": "turn-0001-user"},
                ),
            )
        return proposals

    extractor.extract = add_span_identity
    result = run_extraction_benchmark(
        arena,
        extractor,
        adapter_recipe={},
        clock_ns=_StepClock(),
    )

    run = next(item for item in result.proposal_runs if item.case_id == "add-grounded-preference")
    assert run.proposals[0].metadata == {"evidence_span_id": "turn-0001-user"}
    assert result.passed is True


def test_adapter_config_records_environment_names_but_not_values(tmp_path):
    config = tmp_path / "adapter.json"
    config.write_text(
        json.dumps(
            _adapter_config(
                environment={
                    "PATH": "/private/fixture/path",
                    "OPENAI_API_KEY": "credential-value-must-not-appear",
                }
            )
        )
    )

    loaded = load_benchmark_adapter(config)
    rendered = json.dumps(dict(loaded.recipe), sort_keys=True)

    assert loaded.extractor.identity == (
        "subprocess:fixture-model:durable-memory-v3"
    )
    assert "OPENAI_API_KEY" in rendered
    assert "credential-value-must-not-appear" not in rendered
    assert "/private/fixture/path" not in rendered
    assert loaded.recipe["adapter"]["environment_value_digests"] == [
        {
            "name": "OPENAI_API_KEY",
            "value_digest": "sha256:"
            + hashlib.sha256(b"credential-value-must-not-appear").hexdigest(),
        },
        {
            "name": "PATH",
            "value_digest": "sha256:"
            + hashlib.sha256(b"/private/fixture/path").hexdigest(),
        },
    ]
    assert loaded.recipe["adapter"]["executable"] == {
        "digest": None,
        "status": "absent",
    }


def test_adapter_recipe_digest_attests_executable_environment_and_sources(tmp_path):
    executable = tmp_path / "adapter"
    executable.write_bytes(b"adapter-v1")
    first_value = _adapter_config(environment={"RUNTIME_MODE": "first"})
    first_value["host"]["argv"][0] = str(executable)
    config = tmp_path / "adapter.json"
    config.write_text(json.dumps(first_value))

    first = load_benchmark_adapter(config)
    first_plan = dry_run_plan(
        load_extraction_arena(), first, repetitions=1, timing_class="warm"
    )

    assert first.recipe["adapter"]["executable"] == {
        "digest": "sha256:" + hashlib.sha256(b"adapter-v1").hexdigest(),
        "status": "present",
    }
    root = Path(__file__).resolve().parents[1]
    expected_sources = {
        "offline_scorer_source": root / "memory_eval" / "extraction_arena.py",
        "prompt_source": root / "enfold" / "extraction_contract.py",
        "runtime_scorer_source": root / "memory_eval" / "extraction_runtime_arena.py",
    }
    assert first_plan["recipe"]["source_digests"] == {
        name: "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in expected_sources.items()
    }

    executable.write_bytes(b"adapter-v2")
    second_value = _adapter_config(environment={"RUNTIME_MODE": "second"})
    second_value["host"]["argv"][0] = str(executable)
    config.write_text(json.dumps(second_value))
    second_plan = dry_run_plan(
        load_extraction_arena(),
        load_benchmark_adapter(config),
        repetitions=1,
        timing_class="warm",
    )

    assert first_plan["recipe_digest"] != second_plan["recipe_digest"]


@pytest.mark.parametrize(
    "field",
    [
        "context_tokens",
        "output_tokens",
        "max_tokens",
        "prompt_tokens",
        "max_output_tokens",
    ],
)
def test_adapter_config_accepts_documented_token_count_fields(tmp_path, field):
    config = tmp_path / "adapter.json"
    config.write_text(json.dumps(_adapter_config(recipe={
        "decoder": {field: 4_096},
    })))

    loaded = load_benchmark_adapter(config)

    assert loaded.recipe["controls"]["decoder"] == {field: 4_096}


def test_adapter_config_rejects_credentials_in_reported_argv(tmp_path):
    value = _adapter_config()
    value["host"]["argv"].extend(["--api-key", "sk-not-safe"])
    config = tmp_path / "adapter.json"
    config.write_text(json.dumps(value))

    with pytest.raises(ValueError, match="argv must not contain credentials"):
        load_benchmark_adapter(config)


@pytest.mark.parametrize(
    "field",
    [
        "api_key",
        "api_tokens",
        "password",
        "passwords",
        "access-token",
        "access_keys",
        "client_secrets",
        "openai_api_key",
        "database_password",
        "service_token",
        "auth_token",
    ],
)
def test_adapter_config_rejects_credential_fields_in_public_recipe(tmp_path, field):
    config = tmp_path / "adapter.json"
    config.write_text(json.dumps(_adapter_config(recipe={field: "not-public"})))

    with pytest.raises(ValueError, match="credential fields"):
        load_benchmark_adapter(config)


@pytest.mark.parametrize("value", ["sk-not-public", "Bearer not-public"])
def test_adapter_config_rejects_credential_values_in_public_recipe(tmp_path, value):
    config = tmp_path / "adapter.json"
    config.write_text(json.dumps(_adapter_config(recipe={"runtime": value})))

    with pytest.raises(ValueError, match="credential values"):
        load_benchmark_adapter(config)


def test_benchmark_report_revalidates_public_recipe():
    result = run_extraction_benchmark(
        load_extraction_arena(),
        _GoldenExtractor(),
        adapter_recipe={},
        clock_ns=_StepClock(),
    )

    with pytest.raises(ValueError, match="credential fields"):
        benchmark_report(replace(result, recipe={"service_token": "not-public"}))


def test_cli_dry_run_validates_recipe_without_launching_adapter(
    tmp_path, capsys
):
    config = tmp_path / "adapter.json"
    config.write_text(json.dumps(_adapter_config()))

    exit_code = main(
        [
            "--adapter-config",
            str(config),
            "--repetitions",
            "3",
            "--timing-class",
            "cold",
            "--dry-run",
        ]
    )

    plan = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert plan["dry_run"] is True
    assert plan["adapter_calls"] == 0
    assert plan["recipe"]["repetitions"] == 3
    assert plan["recipe"]["timing_class"] == "cold"
    assert len(plan["case_ids"]) == 7


def test_artifact_writer_is_private_atomic_and_does_not_follow_symlink(tmp_path):
    target = tmp_path / "unrelated.txt"
    target.write_text("do not replace")
    artifact = tmp_path / "proposals.json"
    artifact.symlink_to(target)

    _write(artifact, b'{"private":true}\n')

    assert target.read_text() == "do not replace"
    assert artifact.is_symlink() is False
    assert artifact.read_bytes() == b'{"private":true}\n'
    assert stat.S_IMODE(artifact.stat().st_mode) == 0o600
