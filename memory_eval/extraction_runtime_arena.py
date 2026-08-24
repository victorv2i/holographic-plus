"""Isolated runtime replay for saved extraction proposals.

Unlike :mod:`memory_eval.extraction_arena`, this evaluator intentionally runs
Enfold's service, durable queue, extraction processor, and write-state
machinery.  Every case receives a fresh temporary SQLite database.  Candidate
``decision`` labels are diagnostic only: the authoritative decision is
derived from enqueue policy and ``memory_write_log`` outcomes.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
from typing import Any, Iterable, Mapping, Sequence

from enfold.extraction_enqueue import ExtractionEnqueuer
from enfold.extraction_processor import (
    EvidenceVerification,
    ExtractedMemory,
    ExtractionEnvelope,
    ExtractionProcessor,
)
from enfold.extraction_spans import transcript_spans
from enfold.policy import MemoryPolicy
from enfold.protocol import ClientContext, Request
from enfold.schema import migrate
from enfold.service import EnfoldService

from .extraction_arena import (
    DEFAULT_CASES_PATH,
    DEFAULT_OUTPUTS_PATH,
    CandidateFact,
    CandidateOutput,
    ExtractionArena,
    ExtractionCase,
    PriorFact,
    load_candidate_outputs,
    load_extraction_arena,
)


_CLIENT_ID = "enfold-extraction-runtime-arena"
_EXTRACTOR_IDENTITY = "enfold-extraction-runtime-arena:recorded-v1"
_WRITE_OUTCOMES = frozenset(
    {"inserted", "add", "dedup", "supersede", "conflict"}
)


@dataclass(frozen=True, slots=True)
class RuntimeCaseScore:
    case_id: str
    expected_decision: str
    reported_decision: str
    actual_decision: str | None
    enqueue_outcome: str
    processor_outcome: str | None
    processor_error: str | None
    write_outcomes: tuple[str, ...]
    writes: int
    decision_correct: bool
    passed: bool


@dataclass(frozen=True, slots=True)
class ExtractionRuntimeScore:
    arena: ExtractionArena
    cases: tuple[RuntimeCaseScore, ...]
    summary: Mapping[str, Any]

    @property
    def passed(self) -> bool:
        return all(case.passed for case in self.cases)


class _RecordedExtractor:
    identity = _EXTRACTOR_IDENTITY

    def __init__(self, transcript: str, facts: Sequence[CandidateFact]):
        self._transcript = transcript
        self._facts = tuple(facts)

    def extract(self, envelope: ExtractionEnvelope) -> tuple[ExtractedMemory, ...]:
        if envelope.transcript != self._transcript:
            raise RuntimeError("runtime Arena replay received the wrong transcript")
        return tuple(self._proposal(fact) for fact in self._facts)

    def _proposal(self, fact: CandidateFact) -> ExtractedMemory:
        excerpt = fact.evidence_excerpt
        if excerpt is None and fact.evidence_span is not None:
            start, end = fact.evidence_span
            if 0 <= start < end <= len(self._transcript):
                excerpt = self._transcript[start:end]
        matching = (
            tuple(
                span
                for span in transcript_spans(self._transcript)
                if excerpt is not None and excerpt in span.text
            )
            if excerpt is not None
            else ()
        )
        # The persisted evidence is the complete deterministic span, not a
        # fixture substring. This makes the Arena exercise the same
        # span-identity boundary as a real child adapter.
        evidence = matching[0] if len(matching) == 1 else None
        return ExtractedMemory(
            content=fact.content,
            category=fact.category,
            evidence_excerpt=None if evidence is None else evidence.text,
            sensitivity=fact.sensitivity,
            state=fact.state,
            metadata=(
                {}
                if evidence is None
                else {"evidence_span_id": evidence.span_id}
            ),
        )


class _FixtureEvidenceVerifier:
    """Explicitly reviewed fixture boundary for isolated, non-live Arenas."""

    identity = "runtime-arena-reviewed-fixture:v1"

    def verify(
        self,
        _proposal: ExtractedMemory,
        *,
        evidence_excerpt: str,
        envelope: ExtractionEnvelope,
    ) -> EvidenceVerification:
        if evidence_excerpt not in envelope.transcript:
            return EvidenceVerification("needs_review", self.identity)
        return EvidenceVerification("verified", self.identity)


def _context(case_id: str) -> ClientContext:
    return ClientContext(
        client_id=_CLIENT_ID,
        surface="evaluation",
        agent_id="runtime-arena",
        session_id=f"runtime-{case_id}",
        repository="enfold",
        branch="evaluation",
        access_scopes=("private", "sensitive"),
    )


def _state_params(state: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if state is None or state.get("kind") != "state":
        return None
    object_value = state.get("object", state.get("value"))
    params: dict[str, Any] = {
        "subject_key": state.get("subject"),
        "predicate_key": state.get("predicate"),
        "object_value": object_value,
    }
    valid_from = state.get("valid_from", state.get("occurred_at"))
    if valid_from is not None:
        params["valid_from"] = valid_from
    return params


def _seed_prior(
    service: EnfoldService,
    context: ClientContext,
    case: ExtractionCase,
    prior: PriorFact,
) -> None:
    params: dict[str, Any] = {
        "idempotency_key": f"arena-seed:{case.case_id}:{prior.key}",
        "content": prior.content,
        "source_type": "arena_fixture",
        "category": prior.category,
        "source_authority": 0.5,
        "scope": "private",
        "sensitivity": prior.sensitivity,
    }
    state = _state_params(prior.state)
    if state is not None:
        params["state"] = state
    response = service.handle(
        context,
        Request(f"seed-{case.case_id}-{prior.key}", "memory.write", params),
    )
    if response["outcome"] not in _WRITE_OUTCOMES:
        raise RuntimeError(
            f"fixture seed {case.case_id}/{prior.key} was not accepted: "
            f"{response['outcome']}"
        )


def _derive_write_decision(outcomes: Sequence[str]) -> str | None:
    normalized = ("add" if outcome == "inserted" else outcome for outcome in outcomes)
    distinct = tuple(dict.fromkeys(normalized))
    if not distinct:
        return "abstain"
    if len(distinct) == 1 and distinct[0] in {
        "add", "dedup", "supersede", "conflict",
    }:
        return distinct[0]
    return None


def _replay_case(
    directory: Path,
    case: ExtractionCase,
    output: CandidateOutput,
) -> RuntimeCaseScore:
    database = directory / f"{case.case_id}.sqlite3"
    conn = sqlite3.connect(database)
    try:
        conn.execute("PRAGMA foreign_keys=ON")
        migrate(conn)
        context = _context(case.case_id)
        service = EnfoldService(
            conn,
            MemoryPolicy({_CLIENT_ID: ("private", "sensitive")}),
            extraction_enqueuer=ExtractionEnqueuer(conn),
            extraction_processing_mode="daemon-supervised",
            near_dedup_enabled=False,
        )
        for prior in case.memory_before:
            _seed_prior(service, context, case, prior)
        write_log_watermark = int(
            conn.execute(
                "SELECT COALESCE(MAX(rowid), 0) FROM memory_write_log"
            ).fetchone()[0]
        )

        enqueue = service.handle(
            context,
            Request(
                f"enqueue-{case.case_id}",
                "memory.extraction.enqueue",
                {
                    "transcript": case.transcript,
                    "source": "extraction_runtime_arena",
                    "scope": "private",
                    "metadata": {"case_id": case.case_id},
                },
            ),
        )
        enqueue_outcome = str(enqueue["outcome"])
        if enqueue_outcome == "rejected":
            actual = "reject"
            processor_outcome = None
            processor_error = None
            writes = 0
            write_outcomes: tuple[str, ...] = ()
        else:
            if enqueue_outcome != "queued":
                raise RuntimeError(
                    f"unexpected runtime Arena enqueue outcome: {enqueue_outcome}"
                )
            processor = ExtractionProcessor(
                conn,
                service,
                _RecordedExtractor(case.transcript, output.facts),
                max_attempts=1,
                retry_delay_seconds=0,
                evidence_verifier=_FixtureEvidenceVerifier(),
            )
            process_result = processor.process_one()
            processor_outcome = process_result.outcome
            processor_error = process_result.error
            writes = process_result.writes
            rows = conn.execute(
                "SELECT outcome FROM memory_write_log "
                "WHERE rowid > ? ORDER BY rowid",
                (write_log_watermark,),
            ).fetchall()
            write_outcomes = tuple(str(row[0]) for row in rows)
            actual = (
                _derive_write_decision(write_outcomes)
                if processor_outcome == "completed"
                else None
            )
        correct = actual == case.expected_decision
        return RuntimeCaseScore(
            case_id=case.case_id,
            expected_decision=case.expected_decision,
            reported_decision=output.decision,
            actual_decision=actual,
            enqueue_outcome=enqueue_outcome,
            processor_outcome=processor_outcome,
            processor_error=processor_error,
            write_outcomes=write_outcomes,
            writes=writes,
            decision_correct=correct,
            passed=correct,
        )
    finally:
        conn.close()


def score_extraction_runtime(
    arena: ExtractionArena,
    outputs: Iterable[CandidateOutput],
) -> ExtractionRuntimeScore:
    """Replay saved proposals through isolated authoritative Enfold writes."""

    output_rows = tuple(outputs)
    output_by_id = {output.case_id: output for output in output_rows}
    if len(output_by_id) != len(output_rows):
        raise ValueError("candidate output case ids must be unique")
    case_ids = {case.case_id for case in arena.cases}
    missing = sorted(case_ids - output_by_id.keys())
    unknown = sorted(output_by_id.keys() - case_ids)
    if missing:
        raise ValueError(f"candidate outputs are missing cases: {missing}")
    if unknown:
        raise ValueError(f"candidate outputs contain unknown cases: {unknown}")

    with TemporaryDirectory(prefix="enfold-extraction-runtime-arena-") as temp:
        directory = Path(temp)
        cases = tuple(
            _replay_case(directory, case, output_by_id[case.case_id])
            for case in arena.cases
        )
    passed = sum(case.passed for case in cases)
    total = len(cases)
    summary = {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "decision_accuracy": passed / total if total else 0.0,
        "reported_decisions_ignored": True,
        "isolated_temporary_databases": total,
        "live_database_writes": 0,
    }
    return ExtractionRuntimeScore(arena, cases, summary)


def _report(score: ExtractionRuntimeScore) -> dict[str, Any]:
    return {
        "metadata": {
            "arena": "enfold-extraction-runtime-arena",
            "cases_path": str(score.arena.source_path),
            "provider_calls": 0,
            "model_calls": 0,
            "network_calls": 0,
            "reported_decisions_ignored": True,
        },
        "summary": dict(score.summary),
        "cases": [asdict(case) for case in score.cases],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Replay saved extraction proposals through isolated Enfold databases."
    )
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    parser.add_argument("--outputs", type=Path, default=DEFAULT_OUTPUTS_PATH)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--require-perfect", action="store_true")
    args = parser.parse_args(argv)

    score = score_extraction_runtime(
        load_extraction_arena(args.cases),
        load_candidate_outputs(args.outputs),
    )
    report = _report(score)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 1 if args.require_perfect and not score.passed else 0


if __name__ == "__main__":
    raise SystemExit(main())
