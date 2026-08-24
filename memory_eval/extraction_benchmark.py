"""Reproducible model runner for Enfold's extraction Arenas.

The runner calls one explicitly configured :class:`~enfold.extraction_processor.Extractor`
and saves its proposals without asking the model to classify write lifecycle.
Lifecycle decisions come from :mod:`memory_eval.extraction_runtime_arena`, which
uses disposable migrated databases and Enfold's authoritative write path.

The proposal artifact and evaluation report are intentionally separate.  The
artifact is safe to rescore: it contains no expected or derived decision.  The
report records scoring, stable recipe identity, bounded error codes, and
monotonic elapsed time.  No live Enfold database or daemon configuration is
opened by this module.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
import time
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

from enfold.extraction_processor import ExtractedMemory, ExtractionEnvelope, Extractor
from enfold.host_extractor import (
    HostExtractorConfig,
    HostExtractorError,
    SubprocessHostExtractor,
)
from enfold.protocol import ClientContext

from .extraction_arena import (
    DEFAULT_CASES_PATH,
    CandidateFact,
    CandidateOutput,
    ExtractionArena,
    ExtractionCase,
    load_extraction_arena,
    score_extraction_arena,
)
from .extraction_runtime_arena import (
    RuntimeCaseScore,
    score_extraction_runtime,
)


ADAPTER_CONFIG_VERSION = "enfold-extraction-benchmark-adapter-v1"
PROPOSAL_ARTIFACT_VERSION = "enfold-extraction-proposals-v1"
BENCHMARK_REPORT_VERSION = "enfold-extraction-benchmark-report-v1"
RUNNER_IDENTITY = "enfold-extraction-benchmark-v1"

_CLIENT_ID = "enfold-extraction-benchmark"
_TIMING_CLASSES = frozenset({"cold", "warm", "unspecified"})
_SAFE_ERROR = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_SECRET_METADATA_KEY_PARTS = (
    "key",
    "token",
    "secret",
    "password",
    "credential",
    "auth",
)
_TOKEN_COUNT_QUALIFIERS = frozenset({
    "completion",
    "context",
    "input",
    "max",
    "max_input",
    "max_output",
    "output",
    "prompt",
})
_SECRET_VALUE = re.compile(r"(?i)(?:^sk-|^bearer\s|-----BEGIN [A-Z ]*PRIVATE KEY-----)")
_PROMPT_SOURCE_PATH = Path(__file__).parents[1] / "enfold" / "extraction_contract.py"
_OFFLINE_SCORER_SOURCE_PATH = Path(__file__).with_name("extraction_arena.py")
_RUNTIME_SCORER_SOURCE_PATH = Path(__file__).with_name("extraction_runtime_arena.py")
_HOST_FIELDS = frozenset(
    {
        "type",
        "argv",
        "model_identity",
        "prompt_identity",
        "timeout_seconds",
        "terminate_grace_seconds",
        "max_input_bytes",
        "max_output_bytes",
        "max_error_bytes",
        "environment",
    }
)


@dataclass(frozen=True, slots=True)
class LoadedBenchmarkAdapter:
    """A subprocess extractor plus its credential-free public recipe."""

    extractor: SubprocessHostExtractor
    recipe: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class ProposalRecord:
    """A serializable, proposal-only view of one adapter result."""

    content: str
    category: str
    tags: str
    trust_score: float
    source_authority: float
    evidence_excerpt: str | None
    scope: str | None
    sensitivity: str
    state: Mapping[str, Any] | None
    metadata: Mapping[str, Any]

    def candidate(self) -> CandidateFact:
        return CandidateFact(
            content=self.content,
            category=self.category,
            sensitivity=self.sensitivity,
            evidence_span=None,
            evidence_excerpt=self.evidence_excerpt,
            state=self.state,
            expectation_key=None,
        )


@dataclass(frozen=True, slots=True)
class ProposalRun:
    """One case/repetition adapter observation, without a lifecycle label."""

    case_id: str
    repetition: int
    adapter_outcome: str
    error_code: str | None
    elapsed_ns: int | None
    proposals: tuple[ProposalRecord, ...]


@dataclass(frozen=True, slots=True)
class BenchmarkCaseResult:
    case_id: str
    repetition: int
    adapter_outcome: str
    error_code: str | None
    elapsed_ns: int | None
    proposal_count: int
    expected_decision: str
    actual_decision: str | None
    enqueue_outcome: str
    processor_outcome: str | None
    write_outcomes: tuple[str, ...]
    offline_passed: bool
    runtime_passed: bool
    passed: bool


@dataclass(frozen=True, slots=True)
class ExtractionBenchmarkResult:
    arena: ExtractionArena
    extractor_identity: str
    recipe: Mapping[str, Any]
    recipe_digest: str
    corpus_digest: str
    repetitions: int
    timing_class: str
    proposal_runs: tuple[ProposalRun, ...]
    cases: tuple[BenchmarkCaseResult, ...]
    summary: Mapping[str, Any]

    @property
    def passed(self) -> bool:
        return all(case.passed for case in self.cases)


class _InvalidAdapterOutput(ValueError):
    pass


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, RecursionError) as exc:
        raise ValueError("benchmark metadata must be canonical JSON") from exc


def _sha256(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def _metadata_key_parts(key: str) -> tuple[str, ...]:
    separated = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", key)
    normalized = re.sub(r"[^a-z0-9]+", "_", separated.casefold())
    return tuple(part for part in normalized.split("_") if part)


def _contains_secret_key_part(key: str, *, allow_token_count: bool = False) -> bool:
    parts = _metadata_key_parts(key)
    if (
        allow_token_count
        and parts[-1:] == ("tokens",)
        and "_".join(parts[:-1]) in _TOKEN_COUNT_QUALIFIERS
    ):
        return False
    return any(
        part.removesuffix("s") in _SECRET_METADATA_KEY_PARTS for part in parts
    )


def _validate_public_metadata(value: Any, label: str = "recipe") -> Any:
    """Validate JSON metadata and reject obvious credential-bearing fields."""

    if isinstance(value, Mapping):
        output: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise ValueError(f"{label} keys must be non-empty strings")
            if _contains_secret_key_part(
                key, allow_token_count=isinstance(item, int) and not isinstance(item, bool)
            ):
                raise ValueError(f"{label} must not contain credential fields")
            output[key] = _validate_public_metadata(item, f"{label}.{key}")
        _canonical_bytes(output)
        return output
    if isinstance(value, list):
        output = [
            _validate_public_metadata(item, f"{label}[]") for item in value
        ]
        _canonical_bytes(output)
        return output
    if isinstance(value, str) and _SECRET_VALUE.search(value):
        raise ValueError(f"{label} must not contain credential values")
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise ValueError(f"{label} must contain finite JSON values")


def _validate_public_argv(argv: Sequence[str]) -> None:
    for item in argv:
        option = item.split("=", 1)[0].lstrip("-").casefold().replace("-", "_")
        if _contains_secret_key_part(
            option, allow_token_count=True
        ) or _SECRET_VALUE.search(item):
            raise ValueError("adapter argv must not contain credentials")


def _strict_keys(
    value: Mapping[str, Any],
    label: str,
    *,
    required: set[str],
    optional: set[str] = frozenset(),
) -> None:
    missing = required - set(value)
    extra = set(value) - required - optional
    if missing:
        raise ValueError(f"{label} is missing fields: {sorted(missing)}")
    if extra:
        raise ValueError(f"{label} has unknown fields: {sorted(extra)}")


def _source_digest(path: Path, label: str) -> str:
    try:
        return _sha256(path.read_bytes())
    except OSError as exc:
        raise ValueError(f"could not hash {label}: {path}") from exc


def _executable_identity(path: Path) -> dict[str, Any]:
    try:
        digest = _sha256(path.read_bytes())
    except FileNotFoundError:
        return {"digest": None, "status": "absent"}
    except OSError as exc:
        raise ValueError(f"could not hash adapter executable: {path}") from exc
    return {"digest": digest, "status": "present"}


def load_benchmark_adapter(path: str | Path) -> LoadedBenchmarkAdapter:
    """Load an explicit subprocess adapter without launching it.

    Environment values are used by the child but never copied to the public
    recipe.  The optional ``recipe`` object is public metadata and therefore
    rejects obvious credential-shaped fields.
    """

    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not load benchmark adapter config: {source}") from exc
    if not isinstance(value, dict):
        raise ValueError("benchmark adapter config must be an object")
    _strict_keys(
        value,
        "adapter config",
        required={"schema_version", "host", "recipe"},
    )
    if value["schema_version"] != ADAPTER_CONFIG_VERSION:
        raise ValueError(f"schema_version must be {ADAPTER_CONFIG_VERSION!r}")
    host = value["host"]
    if not isinstance(host, dict):
        raise ValueError("adapter config host must be an object")
    _strict_keys(
        host,
        "adapter config host",
        required={
            "type",
            "argv",
            "model_identity",
            "prompt_identity",
        },
        optional=set(_HOST_FIELDS)
        - {"type", "argv", "model_identity", "prompt_identity"},
    )
    if host["type"] != "subprocess":
        raise ValueError("adapter config host.type must be 'subprocess'")
    argv = host["argv"]
    if (
        not isinstance(argv, list)
        or not argv
        or not all(isinstance(item, str) and item for item in argv)
        or not Path(argv[0]).is_absolute()
    ):
        raise ValueError("adapter config host.argv must start with an absolute path")
    _validate_public_argv(argv)
    environment = host.get("environment", {})
    if not isinstance(environment, dict):
        raise ValueError("adapter config host.environment must be an object")
    config = HostExtractorConfig(
        argv=tuple(argv),
        model_identity=host["model_identity"],
        prompt_identity=host["prompt_identity"],
        timeout_seconds=host.get("timeout_seconds", 180.0),
        terminate_grace_seconds=host.get("terminate_grace_seconds", 2.0),
        max_input_bytes=host.get("max_input_bytes", 16 * 1024),
        max_output_bytes=host.get("max_output_bytes", 64 * 1024),
        max_error_bytes=host.get("max_error_bytes", 16 * 1024),
        environment=environment,
    )
    controls = _validate_public_metadata(value["recipe"])
    recipe = {
        "adapter": {
            "argv": list(config.argv),
            "environment_value_digests": [
                {
                    "name": name,
                    "value_digest": _sha256(config.environment[name].encode("utf-8")),
                }
                for name in sorted(config.environment)
            ],
            "environment_names": sorted(config.environment),
            "executable": _executable_identity(Path(config.argv[0])),
            "max_error_bytes": config.max_error_bytes,
            "max_input_bytes": config.max_input_bytes,
            "max_output_bytes": config.max_output_bytes,
            "terminate_grace_seconds": float(config.terminate_grace_seconds),
            "timeout_seconds": float(config.timeout_seconds),
            "type": "subprocess",
        },
        "controls": controls,
        "model_identity": config.model_identity,
        "prompt_identity": config.prompt_identity,
    }
    return LoadedBenchmarkAdapter(
        SubprocessHostExtractor(config), MappingProxyType(recipe)
    )


def _corpus_digest(arena: ExtractionArena) -> str:
    try:
        return _sha256(arena.source_path.read_bytes())
    except OSError as exc:
        raise ValueError(f"could not hash Arena corpus: {arena.source_path}") from exc


def _benchmark_recipe(
    *,
    arena: ExtractionArena,
    extractor_identity: str,
    adapter_recipe: Mapping[str, Any],
    repetitions: int,
    timing_class: str,
) -> tuple[Mapping[str, Any], str, str]:
    public_recipe = _validate_public_metadata(adapter_recipe, "adapter_recipe")
    corpus_digest = _corpus_digest(arena)
    recipe = {
        "adapter_recipe": public_recipe,
        "corpus": {
            "digest": corpus_digest,
        },
        "extractor_identity": extractor_identity,
        "repetitions": repetitions,
        "runner": RUNNER_IDENTITY,
        "source_digests": {
            "offline_scorer_source": _source_digest(
                _OFFLINE_SCORER_SOURCE_PATH, "offline scorer source"
            ),
            "prompt_source": _source_digest(_PROMPT_SOURCE_PATH, "prompt source"),
            "runtime_scorer_source": _source_digest(
                _RUNTIME_SCORER_SOURCE_PATH, "runtime scorer source"
            ),
        },
        "timing_class": timing_class,
    }
    return MappingProxyType(recipe), _sha256(_canonical_bytes(recipe)), corpus_digest


def _context(case: ExtractionCase, repetition: int) -> ClientContext:
    return ClientContext(
        client_id=_CLIENT_ID,
        surface="evaluation",
        agent_id="extraction-benchmark",
        session_id=f"benchmark-{case.case_id}-{repetition}",
        repository="enfold",
        branch="evaluation",
        access_scopes=("private",),
    )


def _envelope(case: ExtractionCase, repetition: int) -> ExtractionEnvelope:
    return ExtractionEnvelope(
        transcript=case.transcript,
        source="extraction_benchmark",
        scope="private",
        context=_context(case, repetition),
        metadata={"case_id": case.case_id, "repetition": repetition},
    )


def _proposal_record(value: Any) -> ProposalRecord:
    if not isinstance(value, ExtractedMemory):
        raise _InvalidAdapterOutput
    if (
        not isinstance(value.content, str)
        or not value.content.strip()
        or not isinstance(value.category, str)
        or not value.category.strip()
        or not isinstance(value.tags, str)
        or not isinstance(value.sensitivity, str)
        or value.sensitivity not in {"normal", "sensitive"}
        or (
            value.evidence_excerpt is not None
            and not isinstance(value.evidence_excerpt, str)
        )
        or (value.state is not None and not isinstance(value.state, Mapping))
        or not isinstance(value.metadata, Mapping)
    ):
        raise _InvalidAdapterOutput
    for number in (value.trust_score, value.source_authority):
        if (
            isinstance(number, bool)
            or not isinstance(number, (int, float))
            or not math.isfinite(float(number))
        ):
            raise _InvalidAdapterOutput
    # Runtime Arena's recorded adapter deliberately represents the bundled v2
    # model contract.  Reject custom policy controls instead of silently
    # dropping fields that could alter an authoritative write outcome.
    if (
        float(value.trust_score) != 0.5
        or float(value.source_authority) != 0.5
        or value.scope is not None
    ):
        raise _InvalidAdapterOutput
    state = dict(value.state) if value.state is not None else None
    try:
        metadata = _validate_public_metadata(value.metadata, "proposal.metadata")
    except ValueError as exc:
        raise _InvalidAdapterOutput from exc
    if not isinstance(metadata, dict):  # guarded by the Mapping check above
        raise _InvalidAdapterOutput
    _canonical_bytes(state)
    _canonical_bytes(metadata)
    return ProposalRecord(
        content=value.content,
        category=value.category,
        tags=value.tags,
        trust_score=float(value.trust_score),
        source_authority=float(value.source_authority),
        evidence_excerpt=value.evidence_excerpt,
        scope=value.scope,
        sensitivity=value.sensitivity,
        state=state,
        metadata=metadata,
    )


def _safe_error_code(exc: Exception) -> str:
    if isinstance(exc, HostExtractorError) and _SAFE_ERROR.fullmatch(
        exc.error_code
    ):
        return exc.error_code
    if isinstance(exc, _InvalidAdapterOutput):
        return "adapter_invalid_output"
    return "extractor_failed"


def _empty_outputs(arena: ExtractionArena) -> tuple[CandidateOutput, ...]:
    return tuple(
        CandidateOutput(case.case_id, "abstain", (), {"preflight": True})
        for case in arena.cases
    )


def _invoke(
    extractor: Extractor,
    case: ExtractionCase,
    repetition: int,
    *,
    clock_ns: Callable[[], int],
) -> ProposalRun:
    start = clock_ns()
    try:
        raw = extractor.extract(_envelope(case, repetition))
        if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
            raise _InvalidAdapterOutput
        proposals = tuple(_proposal_record(item) for item in raw)
        outcome = "completed"
        error_code = None
    except Exception as exc:
        proposals = ()
        outcome = "error"
        error_code = _safe_error_code(exc)
    end = clock_ns()
    if (
        isinstance(start, bool)
        or isinstance(end, bool)
        or not isinstance(start, int)
        or not isinstance(end, int)
        or start < 0
        or end < start
    ):
        raise ValueError("clock_ns must return monotonic non-negative integers")
    return ProposalRun(
        case.case_id,
        repetition,
        outcome,
        error_code,
        end - start,
        proposals,
    )


def _candidate_output(run: ProposalRun, *, decision: str = "abstain") -> CandidateOutput:
    return CandidateOutput(
        case_id=run.case_id,
        decision=decision,
        facts=tuple(proposal.candidate() for proposal in run.proposals),
        metadata={
            "adapter_outcome": run.adapter_outcome,
            "error_code": run.error_code,
            "repetition": run.repetition,
        },
    )


def _percentile(values: Sequence[int], percentile: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def _result_for_case(
    run: ProposalRun,
    runtime: RuntimeCaseScore,
    offline_passed: bool,
) -> BenchmarkCaseResult:
    adapter_ok = run.adapter_outcome in {"completed", "policy_rejected"}
    passed = adapter_ok and runtime.passed and offline_passed
    return BenchmarkCaseResult(
        case_id=run.case_id,
        repetition=run.repetition,
        adapter_outcome=run.adapter_outcome,
        error_code=run.error_code,
        elapsed_ns=run.elapsed_ns,
        proposal_count=len(run.proposals),
        expected_decision=runtime.expected_decision,
        actual_decision=runtime.actual_decision,
        enqueue_outcome=runtime.enqueue_outcome,
        processor_outcome=runtime.processor_outcome,
        write_outcomes=runtime.write_outcomes,
        offline_passed=offline_passed,
        runtime_passed=runtime.passed,
        passed=passed,
    )


def run_extraction_benchmark(
    arena: ExtractionArena,
    extractor: Extractor,
    *,
    adapter_recipe: Mapping[str, Any],
    repetitions: int = 1,
    timing_class: str = "unspecified",
    clock_ns: Callable[[], int] = time.perf_counter_ns,
) -> ExtractionBenchmarkResult:
    """Invoke an extractor and evaluate proposals through both Arenas.

    A zero-proposal runtime replay happens first.  Cases rejected by Enfold's
    enqueue policy are recorded as ``policy_rejected`` and are never given to
    the extractor.  All other cases are invoked once per repetition.
    """

    if isinstance(repetitions, bool) or not isinstance(repetitions, int) or repetitions <= 0:
        raise ValueError("repetitions must be a positive integer")
    if timing_class not in _TIMING_CLASSES:
        raise ValueError(f"timing_class must be one of {sorted(_TIMING_CLASSES)}")
    identity = getattr(extractor, "identity", None)
    if not isinstance(identity, str) or not identity.strip():
        raise ValueError("extractor identity must be a non-empty string")
    recipe, recipe_digest, corpus_digest = _benchmark_recipe(
        arena=arena,
        extractor_identity=identity,
        adapter_recipe=adapter_recipe,
        repetitions=repetitions,
        timing_class=timing_class,
    )

    preflight = score_extraction_runtime(arena, _empty_outputs(arena))
    preflight_by_id = {case.case_id: case for case in preflight.cases}
    proposal_runs: list[ProposalRun] = []
    case_results: list[BenchmarkCaseResult] = []
    for repetition in range(1, repetitions + 1):
        current: list[ProposalRun] = []
        for case in arena.cases:
            enqueue_outcome = preflight_by_id[case.case_id].enqueue_outcome
            if enqueue_outcome == "rejected":
                run = ProposalRun(
                    case.case_id,
                    repetition,
                    "policy_rejected",
                    None,
                    None,
                    (),
                )
            elif enqueue_outcome == "queued":
                run = _invoke(
                    extractor,
                    case,
                    repetition,
                    clock_ns=clock_ns,
                )
            else:
                raise RuntimeError(
                    f"unsupported extraction preflight outcome: {enqueue_outcome}"
                )
            current.append(run)
            proposal_runs.append(run)

        raw_outputs = tuple(_candidate_output(run) for run in current)
        runtime_score = score_extraction_runtime(arena, raw_outputs)
        runtime_by_id = {case.case_id: case for case in runtime_score.cases}
        derived_outputs = tuple(
            _candidate_output(
                run,
                decision=runtime_by_id[run.case_id].actual_decision or "abstain",
            )
            for run in current
        )
        offline_score = score_extraction_arena(arena, derived_outputs)
        offline_by_id = {case.case_id: case for case in offline_score.cases}
        case_results.extend(
            _result_for_case(
                run,
                runtime_by_id[run.case_id],
                offline_by_id[run.case_id].passed,
            )
            for run in current
        )

    elapsed = [
        run.elapsed_ns
        for run in proposal_runs
        if run.elapsed_ns is not None
    ]
    passed = sum(case.passed for case in case_results)
    total = len(case_results)
    summary = {
        "adapter_calls": len(elapsed),
        "adapter_errors": sum(run.adapter_outcome == "error" for run in proposal_runs),
        "case_run_pass_rate": passed / total if total else 0.0,
        "case_runs": total,
        "failed": total - passed,
        "latency_ns": {
            "maximum": max(elapsed) if elapsed else None,
            "minimum": min(elapsed) if elapsed else None,
            "p50": _percentile(elapsed, 0.50),
            "p95": _percentile(elapsed, 0.95),
        },
        "passed": passed,
        "policy_rejections": sum(
            run.adapter_outcome == "policy_rejected" for run in proposal_runs
        ),
        "reported_model_decisions": 0,
        "runtime_decisions_authoritative": True,
    }
    return ExtractionBenchmarkResult(
        arena=arena,
        extractor_identity=identity,
        recipe=recipe,
        recipe_digest=recipe_digest,
        corpus_digest=corpus_digest,
        repetitions=repetitions,
        timing_class=timing_class,
        proposal_runs=tuple(proposal_runs),
        cases=tuple(case_results),
        summary=MappingProxyType(summary),
    )


def proposal_artifact(result: ExtractionBenchmarkResult) -> dict[str, Any]:
    """Return proposal-only output suitable for later rescoring."""

    return {
        "schema_version": PROPOSAL_ARTIFACT_VERSION,
        "metadata": {
            "corpus_digest": result.corpus_digest,
            "extractor_identity": result.extractor_identity,
            "recipe_digest": result.recipe_digest,
            "repetitions": result.repetitions,
            "timing_class": result.timing_class,
        },
        "runs": [asdict(run) for run in result.proposal_runs],
    }


def benchmark_report(
    result: ExtractionBenchmarkResult,
    *,
    proposal_artifact_digest: str | None = None,
) -> dict[str, Any]:
    recipe = _validate_public_metadata(result.recipe, "benchmark recipe")
    metadata: dict[str, Any] = {
        "authoritative_lifecycle": "extraction_runtime_arena",
        "corpus_digest": result.corpus_digest,
        "cases_path": str(result.arena.source_path),
        "extractor_identity": result.extractor_identity,
        "live_database_writes": 0,
        "proposal_artifact_schema": PROPOSAL_ARTIFACT_VERSION,
        "recipe": recipe,
        "recipe_digest": result.recipe_digest,
        "reported_model_decisions": 0,
        "timing_clock": "monotonic_ns",
    }
    if proposal_artifact_digest is not None:
        metadata["proposal_artifact_digest"] = proposal_artifact_digest
    return {
        "schema_version": BENCHMARK_REPORT_VERSION,
        "metadata": metadata,
        "summary": dict(result.summary),
        "cases": [asdict(case) for case in result.cases],
    }


def dry_run_plan(
    arena: ExtractionArena,
    loaded: LoadedBenchmarkAdapter,
    *,
    repetitions: int,
    timing_class: str,
) -> dict[str, Any]:
    recipe, digest, corpus_digest = _benchmark_recipe(
        arena=arena,
        extractor_identity=loaded.extractor.identity,
        adapter_recipe=loaded.recipe,
        repetitions=repetitions,
        timing_class=timing_class,
    )
    return {
        "schema_version": BENCHMARK_REPORT_VERSION,
        "dry_run": True,
        "adapter_calls": 0,
        "case_ids": [case.case_id for case in arena.cases],
        "corpus_digest": corpus_digest,
        "recipe": dict(recipe),
        "recipe_digest": digest,
    }


def _render(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8") + b"\n"


def _write(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary)
    try:
        offset = 0
        while offset < len(value):
            written = os.write(descriptor, value[offset:])
            if written <= 0:
                raise OSError("benchmark artifact write made no progress")
            offset += written
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        os.replace(temporary_path, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run an explicit extraction adapter, save proposal-only artifacts, "
            "and derive lifecycle through isolated Enfold runtime replay."
        )
    )
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    parser.add_argument("--adapter-config", type=Path, required=True)
    parser.add_argument("--proposals", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument(
        "--timing-class", choices=sorted(_TIMING_CLASSES), default="unspecified"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--require-perfect", action="store_true")
    args = parser.parse_args(argv)

    arena = load_extraction_arena(args.cases)
    loaded = load_benchmark_adapter(args.adapter_config)
    if args.dry_run:
        plan = dry_run_plan(
            arena,
            loaded,
            repetitions=args.repetitions,
            timing_class=args.timing_class,
        )
        print(_render(plan).decode("utf-8"), end="")
        return 0
    if args.proposals is None or args.report is None:
        parser.error("--proposals and --report are required unless --dry-run is used")

    result = run_extraction_benchmark(
        arena,
        loaded.extractor,
        adapter_recipe=loaded.recipe,
        repetitions=args.repetitions,
        timing_class=args.timing_class,
    )
    proposal_bytes = _render(proposal_artifact(result))
    report = benchmark_report(
        result,
        proposal_artifact_digest=_sha256(proposal_bytes),
    )
    report_bytes = _render(report)
    _write(args.proposals, proposal_bytes)
    _write(args.report, report_bytes)
    print(report_bytes.decode("utf-8"), end="")
    return 1 if args.require_perfect and not result.passed else 0


if __name__ == "__main__":
    raise SystemExit(main())
