from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import replace

import pytest

from enfold.extraction_enqueue import ExtractionEnqueuer
from enfold.extraction_processor import (
    EvidenceVerification,
    ExtractedMemory,
    ExtractionProcessor,
    ExtractionProcessorUnavailable,
)
from enfold.extraction_spans import MAX_EVIDENCE_CHARS
from enfold.extraction_spans import transcript_spans
from enfold.policy import MemoryPolicy, UnknownMemoryClient
from enfold.provenance import ConnectionContext
from enfold.schema import migrate
from enfold.service import EnfoldService


def _setup(tmp_path):
    conn = sqlite3.connect(tmp_path / "processor.db")
    conn.execute("PRAGMA foreign_keys=ON")
    migrate(conn)
    context = ConnectionContext(
        client_id="hermes-install",
        surface="hermes",
        agent_id="avery",
        session_id="hermes-session-1",
        parent_agent_id="orchestrator",
        repository="enfold",
        branch="processor",
        access_scopes=("private", "work"),
    )
    service = EnfoldService(conn, MemoryPolicy({"hermes-install": ("private", "work")}))
    return conn, context, service


class FakeExtractor:
    identity = "fake-extractor:v1"

    def __init__(self, proposals=(), failures=0):
        self.proposals = tuple(proposals)
        self.failures = failures
        self.calls = 0

    def extract(self, envelope):
        self.calls += 1
        if self.calls <= self.failures:
            raise RuntimeError("temporary fake model failure")
        assert envelope.context.agent_id == "avery"
        spans = transcript_spans(envelope.transcript)
        normalized = []
        for proposal in self.proposals:
            if proposal.metadata.get("evidence_span_id") is not None:
                normalized.append(proposal)
                continue
            matching = next(
                (span for span in spans if span.text == proposal.evidence_excerpt),
                None,
            )
            normalized.append(
                proposal
                if matching is None
                else replace(
                    proposal,
                    metadata={
                        **proposal.metadata,
                        "evidence_span_id": matching.span_id,
                    },
                )
            )
        return tuple(normalized)


class VerifiedTestEvidence:
    """An explicit stand-in for an independently reviewed test fixture."""

    def verify(self, _proposal, *, evidence_excerpt, envelope):
        assert evidence_excerpt in envelope.transcript
        return EvidenceVerification("verified", "test-evidence-v1")


DEFAULT_TRANSCRIPT = "Avery uses Enfold."


def _grounded(content, *, evidence_excerpt=DEFAULT_TRANSCRIPT, **kwargs):
    return ExtractedMemory(content, evidence_excerpt=evidence_excerpt, **kwargs)


def _enqueue(conn, context, *, scope="private", transcript=DEFAULT_TRANSCRIPT):
    return ExtractionEnqueuer(conn).enqueue_after_commit(
        context, transcript, source="session_end", scope=scope
    )


def test_fake_extraction_applies_authoritative_attributed_writes(tmp_path):
    conn, context, service = _setup(tmp_path)
    _enqueue(conn, context, scope="work")
    extractor = FakeExtractor(
        [
            _grounded(
                "Avery uses Enfold as a shared second brain.",
                category="preference",
                tags="enfold,second-brain",
                scope="work",
                evidence_excerpt="Avery uses Enfold.",
            )
        ]
    )

    result = ExtractionProcessor(
        conn, service, extractor, evidence_verifier=VerifiedTestEvidence()
    ).process_one()

    assert result.outcome == "completed"
    assert result.writes == 1
    assert conn.execute("SELECT count(*) FROM extract_queue").fetchone()[0] == 0
    row = conn.execute(
        "SELECT client_id, session_id, performed_by, asserted_by, scope, metadata_json "
        "FROM observations"
    ).fetchone()
    assert tuple(row[:5]) == (
        "hermes-install",
        "hermes-session-1",
        "avery",
        "fake-extractor:v1",
        "work",
    )
    assert '"extractor_identity":"fake-extractor:v1"' in row[5]
    session = conn.execute(
        "SELECT agent_id, parent_agent_id FROM memory_sessions"
    ).fetchone()
    assert tuple(session) == ("avery", "orchestrator")
    conn.close()


def test_default_evidence_boundary_quarantines_unrelated_but_valid_evidence(tmp_path):
    """An exact span is provenance, not proof that it supports a claim."""

    conn, context, service = _setup(tmp_path)
    transcript = "Avery prefers tea."
    _enqueue(conn, context, transcript=transcript)
    extractor = FakeExtractor(
        [
            _grounded(
                "Avery is the chief executive.",
                evidence_excerpt=transcript,
                metadata={
                    "evidence_span_id": transcript_spans(transcript)[0].span_id,
                },
            )
        ]
    )

    result = ExtractionProcessor(conn, service, extractor).process_one()

    assert (result.outcome, result.error) == ("dead", "proposal_support_unverified")
    assert conn.execute("SELECT count(*) FROM facts").fetchone()[0] == 0
    assert conn.execute(
        "SELECT last_error FROM extract_queue WHERE id = ?", (result.queue_id,)
    ).fetchone()[0] == "proposal_support_unverified"
    conn.close()


def test_processor_health_reports_whether_evidence_verification_is_configured(tmp_path):
    conn, _context, service = _setup(tmp_path)
    default = ExtractionProcessor(conn, service, FakeExtractor())

    assert default.health["evidence_verifier"] == {
        "configured": False,
        "verifier_id": "unconfigured",
    }

    class ConfiguredVerifier:
        identity = "independent-review:v1"

        def verify(self, _proposal, *, evidence_excerpt, envelope):
            assert evidence_excerpt in envelope.transcript
            return EvidenceVerification("verified", self.identity)

    configured = ExtractionProcessor(
        conn,
        service,
        FakeExtractor(),
        evidence_verifier=ConfiguredVerifier(),
    )
    assert configured.health["evidence_verifier"] == {
        "configured": True,
        "verifier_id": "independent-review:v1",
    }
    conn.close()


def test_retry_then_success_and_exhaustion_dead_letters(tmp_path):
    conn, context, service = _setup(tmp_path)
    _enqueue(conn, context)
    clock = [100.0]
    flaky = FakeExtractor([_grounded("A durable preference.")], failures=1)
    worker = ExtractionProcessor(
        conn,
        service,
        flaky,
        max_attempts=3,
        retry_delay_seconds=5,
        clock=lambda: clock[0],
        evidence_verifier=VerifiedTestEvidence(),
    )

    first = worker.process_one()
    assert first.outcome == "retry"
    assert worker.process_one().outcome == "idle"
    clock[0] += 5
    assert worker.process_one().outcome == "completed"

    _enqueue(conn, context, transcript="A second transcript.")
    broken = ExtractionProcessor(
        conn,
        service,
        FakeExtractor(failures=99),
        max_attempts=2,
        retry_delay_seconds=0,
        clock=lambda: clock[0],
    )
    assert broken.process_one().outcome == "retry"
    result = broken.process_one()
    assert result.outcome == "dead"
    assert result.error == "extractor_failed"
    row = conn.execute(
        "SELECT status, attempts, last_error FROM extract_queue"
    ).fetchone()
    assert tuple(row[:2]) == ("dead", 2)
    assert row[2] == "extractor_failed"
    conn.close()


def test_transient_failures_use_capped_exponential_backoff(tmp_path):
    conn, context, service = _setup(tmp_path)
    _enqueue(conn, context)
    clock = [100.0]
    worker = ExtractionProcessor(
        conn,
        service,
        FakeExtractor(failures=99),
        max_attempts=4,
        retry_delay_seconds=2,
        clock=lambda: clock[0],
    )

    assert worker.process_one().outcome == "retry"
    assert conn.execute("SELECT not_before FROM extract_queue").fetchone()[0] == 102.0
    clock[0] = 102.0
    assert worker.process_one().outcome == "retry"
    assert conn.execute("SELECT not_before FROM extract_queue").fetchone()[0] == 106.0
    clock[0] = 105.0
    assert worker.process_one().outcome == "idle"
    clock[0] = 106.0
    assert worker.process_one().outcome == "retry"
    assert conn.execute("SELECT not_before FROM extract_queue").fetchone()[0] == 114.0
    conn.close()


def test_transient_retry_backoff_is_capped(tmp_path):
    conn, context, service = _setup(tmp_path)
    _enqueue(conn, context)
    clock = [100.0]
    worker = ExtractionProcessor(
        conn,
        service,
        FakeExtractor(failures=99),
        max_attempts=10,
        retry_delay_seconds=200,
        clock=lambda: clock[0],
    )

    assert worker.process_one().outcome == "retry"
    assert conn.execute("SELECT not_before FROM extract_queue").fetchone()[0] == 300.0
    clock[0] = 300.0
    assert worker.process_one().outcome == "retry"
    assert conn.execute("SELECT not_before FROM extract_queue").fetchone()[0] == 600.0
    conn.close()


def test_rate_limit_hint_reschedules_without_consuming_attempt_budget(tmp_path):
    conn, context, service = _setup(tmp_path)
    _enqueue(conn, context)
    clock = [100.0]

    class RateLimited(RuntimeError):
        error_code = "adapter_rate_limited"
        retryable = True
        consumes_attempt = False

        def __init__(self, retry_after_seconds):
            super().__init__("redacted")
            self.retry_after_seconds = retry_after_seconds

    class RateLimitedExtractor:
        identity = "rate-limited:v1"

        def __init__(self):
            self.calls = 0

        def extract(self, _envelope):
            self.calls += 1
            if self.calls == 1:
                raise RateLimited(120)
            if self.calls == 2:
                raise RateLimited(1_000)
            return ()

    worker = ExtractionProcessor(
        conn,
        service,
        RateLimitedExtractor(),
        max_attempts=1,
        retry_delay_seconds=1,
        clock=lambda: clock[0],
    )

    first = worker.process_one()
    assert (first.outcome, first.attempts) == ("retry", 0)
    assert tuple(
        conn.execute(
            "SELECT status, attempts, not_before, last_error FROM extract_queue"
        ).fetchone()
    ) == ("pending", 0, 220.0, "adapter_rate_limited")

    clock[0] = 220.0
    second = worker.process_one()
    assert (second.outcome, second.attempts) == ("retry", 0)
    assert conn.execute("SELECT not_before FROM extract_queue").fetchone()[0] == 1220.0

    clock[0] = 1220.0
    assert worker.process_one().outcome == "completed"
    conn.close()


def test_rate_limit_retries_stop_after_queue_row_age_cap(tmp_path):
    conn, context, service = _setup(tmp_path)
    _enqueue(conn, context)
    clock = [200_000.0]
    conn.execute(
        "UPDATE extract_queue SET created_at = datetime(?, 'unixepoch')",
        (clock[0] - 49 * 3600,),
    )
    conn.commit()

    class RateLimited(RuntimeError):
        error_code = "adapter_rate_limited"
        retryable = True
        consumes_attempt = False
        retry_after_seconds = 3600

    class RateLimitedExtractor:
        identity = "rate-limited:v1"

        def extract(self, _envelope):
            raise RateLimited("redacted")

    result = ExtractionProcessor(
        conn,
        service,
        RateLimitedExtractor(),
        max_attempts=1,
        clock=lambda: clock[0],
    ).process_one()

    assert (result.outcome, result.attempts) == ("dead", 0)
    assert tuple(
        conn.execute(
            "SELECT status, attempts, not_before FROM extract_queue"
        ).fetchone()
    ) == ("dead", 0, None)
    conn.close()


def test_lease_renewal_failure_cancels_and_joins_active_extraction(tmp_path):
    conn, context, service = _setup(tmp_path)
    _enqueue(conn, context)

    class BlockingExtractor:
        identity = "blocking:v1"

        def __init__(self):
            self.started = threading.Event()
            self.release = threading.Event()
            self.finished = threading.Event()
            self.handle = object()
            self.cancelled = None

        def extract(self, _envelope, *, register_invocation=None):
            assert register_invocation is not None
            register_invocation(self.handle)
            self.started.set()
            self.release.wait(2)
            self.finished.set()
            return ()

        def cancel(self, invocation):
            self.cancelled = invocation
            assert self.started.wait(1)
            self.release.set()

    extractor = BlockingExtractor()
    processor = ExtractionProcessor(
        conn,
        service,
        extractor,
        lease_seconds=1,
        heartbeat_seconds=0.01,
    )
    claimed = processor._claim()
    assert claimed is not None
    envelope = processor._decode_envelope(claimed[1])
    processor._renew = lambda _row_id, _token: (_ for _ in ()).throw(
        RuntimeError("extraction lease was lost before renewal")
    )

    with pytest.raises(RuntimeError, match="lost before renewal"):
        processor._extract_with_heartbeat(envelope, claimed[0], claimed[4])

    assert extractor.finished.is_set()
    assert extractor.cancelled is extractor.handle
    assert not any(
        thread.name == "enfold-extraction-call" and thread.is_alive()
        for thread in threading.enumerate()
    )
    conn.close()


def test_legacy_raw_transcript_row_is_quarantined_without_calling_a_model(tmp_path):
    conn, _context, service = _setup(tmp_path)
    transcript = "USER: Avery prefers local tools."
    conn.execute(
        "INSERT INTO extract_queue(payload, status) VALUES (?, 'pending')",
        (transcript,),
    )
    conn.commit()

    class LegacyExtractor:
        identity = "legacy-test:v1"

        def __init__(self):
            self.calls = 0

        def extract(self, envelope):
            del envelope
            self.calls += 1
            return ()

    extractor = LegacyExtractor()
    result = ExtractionProcessor(conn, service, extractor).process_one()

    assert (result.outcome, result.error) == ("dead", "legacy_extraction_quarantined")
    assert extractor.calls == 0
    assert tuple(
        conn.execute("SELECT status, last_error FROM extract_queue").fetchone()
    ) == ("dead", "legacy_extraction_quarantined")
    conn.close()


def test_legacy_raw_transcript_row_never_commits_a_proposal(tmp_path):
    conn, _context, service = _setup(tmp_path)
    transcript = "USER: Avery prefers local tools."
    conn.execute(
        "INSERT INTO extract_queue(payload, status) VALUES (?, 'pending')",
        (transcript,),
    )
    conn.commit()
    legacy_context = ConnectionContext(
        client_id="legacy-extract-queue",
        surface="legacy",
        agent_id="legacy",
        session_id="legacy-extract-queue",
        access_scopes=("private",),
    )
    with pytest.raises(UnknownMemoryClient):
        MemoryPolicy({}).authorize_context(legacy_context)

    class LegacyExtractor:
        identity = "legacy-test:v1"

        def extract(self, _envelope):
            return (
                ExtractedMemory(
                    "Avery prefers local tools.",
                    evidence_excerpt="Avery prefers local tools.",
                ),
            )

    result = ExtractionProcessor(conn, service, LegacyExtractor()).process_one()

    assert (result.outcome, result.writes, result.error) == (
        "dead",
        0,
        "legacy_extraction_quarantined",
    )
    assert conn.execute("SELECT count(*) FROM observations").fetchone()[0] == 0
    assert conn.execute("SELECT count(*) FROM facts").fetchone()[0] == 0
    conn.close()


def test_structured_envelope_cannot_claim_reserved_legacy_client(tmp_path):
    conn, _context, service = _setup(tmp_path)
    forged_context = ConnectionContext(
        client_id="legacy-extract-queue",
        surface="legacy",
        agent_id="legacy",
        session_id="legacy-extract-queue",
        access_scopes=("private",),
    )
    _enqueue(conn, forged_context, transcript="Avery prefers local tools.")
    extractor = FakeExtractor(
        [_grounded("Avery prefers local tools.", evidence_excerpt="Avery prefers local tools.")]
    )

    result = ExtractionProcessor(conn, service, extractor).process_one()

    assert (result.outcome, result.error) == ("dead", "invalid_envelope")
    assert extractor.calls == 0
    assert conn.execute("SELECT count(*) FROM facts").fetchone()[0] == 0
    conn.close()


def test_malformed_json_queue_row_remains_permanent_invalid_envelope(tmp_path):
    conn, _context, service = _setup(tmp_path)
    conn.execute(
        "INSERT INTO extract_queue(payload, status) VALUES (?, 'pending')",
        ('{"version":1',),
    )
    conn.commit()
    extractor = FakeExtractor()

    result = ExtractionProcessor(conn, service, extractor).process_one()

    assert (result.outcome, result.error) == ("dead", "invalid_envelope")
    assert extractor.calls == 0
    conn.close()


def test_explicit_nonretryable_adapter_error_dead_letters_once(tmp_path):
    conn, context, service = _setup(tmp_path)
    _enqueue(conn, context)

    class InvalidAdapterOutput(RuntimeError):
        error_code = "adapter_invalid_output"
        retryable = False

    class InvalidAdapter:
        identity = "invalid-adapter:v1"

        def extract(self, _envelope):
            raise InvalidAdapterOutput("private model detail")

    result = ExtractionProcessor(
        conn, service, InvalidAdapter(), max_attempts=3
    ).process_one()

    assert result.outcome == "dead"
    assert result.attempts == 1
    assert result.error == "adapter_invalid_output"
    assert tuple(
        conn.execute(
            "SELECT status, attempts, last_error FROM extract_queue"
        ).fetchone()
    ) == ("dead", 1, "adapter_invalid_output")
    conn.close()


def test_secret_output_is_dead_lettered_before_any_write(tmp_path):
    conn, context, service = _setup(tmp_path)
    _enqueue(conn, context)
    extractor = FakeExtractor([_grounded("api_key = abcdefghijklmnopqrstuv")])

    result = ExtractionProcessor(conn, service, extractor).process_one()

    assert result.outcome == "dead"
    assert conn.execute("SELECT count(*) FROM facts").fetchone()[0] == 0
    assert tuple(
        conn.execute("SELECT status, attempts FROM extract_queue").fetchone()
    ) == ("dead", 1)
    assert conn.execute("SELECT proposal_json FROM extract_queue").fetchone()[0] is None
    conn.close()


def test_legacy_bare_list_snapshot_is_quarantined_without_recall_or_writes(tmp_path):
    conn, _context, service = _setup(tmp_path)
    transcript = "USER: Avery prefers local tools."
    legacy_snapshot = json.dumps(
        [
            {
                "content": "Avery prefers local tools.",
                "category": "user_pref",
                "tags": "local,tools",
            }
        ],
        sort_keys=True,
        separators=(",", ":"),
    )
    conn.execute(
        "INSERT INTO extract_queue(payload, status, proposal_json) "
        "VALUES (?, 'pending', ?)",
        (transcript, legacy_snapshot),
    )
    conn.commit()

    class NoRecallExtractor:
        identity = "takeover:v1"
        calls = 0

        def extract(self, _envelope):
            self.calls += 1
            return ()

    extractor = NoRecallExtractor()
    first = ExtractionProcessor(
        conn, service, extractor, retry_delay_seconds=0
    ).process_one()

    assert (first.outcome, first.error) == ("dead", "legacy_extraction_quarantined")
    assert extractor.calls == 0
    assert conn.execute("SELECT count(*) FROM facts").fetchone()[0] == 0
    conn.close()


def test_failure_update_reports_lost_lease_when_fenced_update_changes_nothing(
    tmp_path,
):
    conn, context, service = _setup(tmp_path)
    _enqueue(conn, context)
    processor = ExtractionProcessor(conn, service, FakeExtractor())
    claimed = processor._claim()
    assert claimed is not None
    conn.execute(
        "CREATE TRIGGER ignore_extraction_failure "
        "BEFORE UPDATE OF last_error ON extract_queue "
        "BEGIN SELECT RAISE(IGNORE); END"
    )
    conn.commit()

    with pytest.raises(RuntimeError, match="lease was lost while recording failure"):
        processor._fail(claimed[0], claimed[4], "extractor_failed", permanent=False)

    assert conn.execute("SELECT status FROM extract_queue").fetchone()[0] == (
        "processing"
    )
    conn.close()


@pytest.mark.parametrize("excerpt", [None, "not present in the transcript"])
def test_ungrounded_output_is_permanently_rejected_before_snapshot(tmp_path, excerpt):
    conn, context, service = _setup(tmp_path)
    _enqueue(conn, context)
    extractor = FakeExtractor(
        [ExtractedMemory("An unsupported proposal.", evidence_excerpt=excerpt)]
    )

    result = ExtractionProcessor(conn, service, extractor).process_one()

    assert result.outcome == "dead"
    assert result.attempts == 1
    assert result.error == "proposal_grounding_rejected"
    assert conn.execute("SELECT count(*) FROM facts").fetchone()[0] == 0
    assert conn.execute("SELECT proposal_json FROM extract_queue").fetchone()[0] is None
    conn.close()


def test_oversized_exact_evidence_is_permanently_rejected(tmp_path):
    conn, context, service = _setup(tmp_path)
    transcript = "x" * (MAX_EVIDENCE_CHARS + 1)
    _enqueue(conn, context, transcript=transcript)
    extractor = FakeExtractor(
        [ExtractedMemory("An oversized proposal.", evidence_excerpt=transcript)]
    )

    result = ExtractionProcessor(conn, service, extractor).process_one()

    assert result.outcome == "dead"
    assert result.error == "proposal_grounding_rejected"
    assert conn.execute("SELECT count(*) FROM facts").fetchone()[0] == 0
    conn.close()


def test_evidence_span_id_is_validated_and_preserved_with_duplicate_text(tmp_path):
    conn, context, service = _setup(tmp_path)
    transcript = "Avery uses Enfold.\n\nAvery uses Enfold."
    spans = transcript_spans(transcript)
    assert len(spans) == 2 and spans[0].text == spans[1].text
    _enqueue(conn, context, transcript=transcript)
    proposal = ExtractedMemory(
        "Avery uses Enfold as durable memory.",
        evidence_excerpt=spans[1].text,
        metadata={"evidence_span_id": spans[1].span_id},
    )

    result = ExtractionProcessor(
        conn,
        service,
        FakeExtractor([proposal]),
        evidence_verifier=VerifiedTestEvidence(),
    ).process_one()

    assert result.outcome == "completed"
    metadata = json.loads(
        conn.execute(
            "SELECT metadata_json FROM observations "
            "WHERE source_type = 'automatic_extraction'"
        ).fetchone()[0]
    )
    assert metadata["evidence_span_id"] == spans[1].span_id
    conn.close()


def test_invalid_evidence_span_id_is_rejected_before_write(tmp_path):
    conn, context, service = _setup(tmp_path)
    _enqueue(conn, context)
    proposal = _grounded(
        "Avery uses Enfold as durable memory.",
        metadata={"evidence_span_id": "span-999999-999999"},
    )

    result = ExtractionProcessor(conn, service, FakeExtractor([proposal])).process_one()

    assert result.outcome == "dead"
    assert result.error == "proposal_grounding_rejected"
    assert conn.execute("SELECT count(*) FROM facts").fetchone()[0] == 0
    conn.close()


def test_missing_evidence_span_id_is_rejected_before_verification(tmp_path):
    conn, context, service = _setup(tmp_path)
    _enqueue(conn, context)
    proposal = _grounded("Avery uses Enfold as durable memory.")

    class NoSpanExtractor:
        identity = "missing-span-test:v1"

        def extract(self, _envelope):
            return (proposal,)

    result = ExtractionProcessor(
        conn,
        service,
        NoSpanExtractor(),
        evidence_verifier=VerifiedTestEvidence(),
    ).process_one()

    assert (result.outcome, result.error) == ("dead", "proposal_grounding_rejected")
    assert conn.execute("SELECT count(*) FROM facts").fetchone()[0] == 0
    conn.close()


def test_crash_before_atomic_batch_commit_rolls_back_and_reuses_snapshot(tmp_path):
    conn, context, service = _setup(tmp_path)
    _enqueue(conn, context)
    clock = [100.0]
    extractor = FakeExtractor(
        [
            _grounded("First crash-safe extracted fact."),
            _grounded("Second crash-safe extracted fact."),
        ]
    )
    crashed = ExtractionProcessor(
        conn,
        service,
        extractor,
        worker_id="worker-a",
        lease_seconds=10,
        clock=lambda: clock[0],
        evidence_verifier=VerifiedTestEvidence(),
    )
    complete_in_transaction = crashed._complete_in_transaction

    def interrupt_before_commit(row_id, token):
        complete_in_transaction(row_id, token)
        raise KeyboardInterrupt()

    crashed._complete_in_transaction = interrupt_before_commit

    with pytest.raises(KeyboardInterrupt):
        crashed.process_one()
    assert conn.execute("SELECT count(*) FROM facts").fetchone()[0] == 0
    assert conn.execute("SELECT count(*) FROM observations").fetchone()[0] == 0
    assert conn.execute("SELECT count(*) FROM memory_write_log").fetchone()[0] == 0
    assert (
        conn.execute("SELECT status FROM extract_queue").fetchone()[0] == "processing"
    )
    assert (
        conn.execute("SELECT proposal_json IS NOT NULL FROM extract_queue").fetchone()[
            0
        ]
        == 1
    )

    clock[0] += 11
    recovered = ExtractionProcessor(
        conn,
        service,
        extractor,
        worker_id="worker-b",
        clock=lambda: clock[0],
        evidence_verifier=VerifiedTestEvidence(),
    )
    assert recovered.process_one().outcome == "completed"
    assert conn.execute("SELECT count(*) FROM facts").fetchone()[0] == 2
    assert conn.execute("SELECT count(*) FROM observations").fetchone()[0] == 1
    assert conn.execute("SELECT count(*) FROM fact_provenance").fetchone()[0] == 2
    assert conn.execute("SELECT count(*) FROM memory_write_log").fetchone()[0] == 2
    assert conn.execute("SELECT count(*) FROM extract_queue").fetchone()[0] == 0
    assert extractor.calls == 1
    conn.close()


def test_processor_refuses_minimal_enqueue_only_schema(tmp_path):
    conn = sqlite3.connect(tmp_path / "minimal.db")
    conn.execute(
        "CREATE TABLE extract_queue (id INTEGER PRIMARY KEY, payload TEXT, "
        "status TEXT, payload_hash TEXT)"
    )
    with pytest.raises(ExtractionProcessorUnavailable, match="attempts"):
        ExtractionProcessor(conn, object(), FakeExtractor())
    conn.close()


def test_fencing_token_blocks_stale_completion_with_stable_worker_id(tmp_path):
    conn, context, service = _setup(tmp_path)
    _enqueue(conn, context)
    clock = [100.0]
    first = ExtractionProcessor(
        conn,
        service,
        FakeExtractor(),
        worker_id="stable-worker",
        lease_seconds=5,
        clock=lambda: clock[0],
        max_attempts=2,
    )
    claimed = first._claim()
    assert claimed is not None
    row_id, _payload, _digest, attempts, stale_token = claimed
    assert attempts == 1

    clock[0] = 106.0
    second = ExtractionProcessor(
        conn,
        service,
        FakeExtractor(),
        worker_id="stable-worker",
        lease_seconds=5,
        clock=lambda: clock[0],
        max_attempts=2,
    )
    reclaimed = second._claim()
    assert reclaimed is not None
    assert reclaimed[3] == 2
    assert reclaimed[4] != stale_token
    with pytest.raises(RuntimeError, match="lease was lost"):
        first._complete(row_id, stale_token)
    second._fail(row_id, reclaimed[4], "failed", permanent=False)
    assert tuple(
        conn.execute(
            "SELECT status, attempts FROM extract_queue WHERE id = ?", (row_id,)
        ).fetchone()
    ) == ("dead", 2)


def test_multi_proposal_failure_rolls_back_batch_and_reuses_snapshot(tmp_path):
    conn, context, service = _setup(tmp_path)
    _enqueue(conn, context)
    extractor = FakeExtractor(
        [
            _grounded("First durable proposal."),
            _grounded("Second durable proposal."),
        ]
    )

    original_writer = service._writes._fact_writer
    calls = 0

    def fail_second(connection, request, observation_id):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("synthetic service interruption")
        return original_writer(connection, request, observation_id)

    service._writes._fact_writer = fail_second

    partial = ExtractionProcessor(
        conn,
        service,
        extractor,
        retry_delay_seconds=0,
        evidence_verifier=VerifiedTestEvidence(),
    ).process_one()
    assert partial.outcome == "retry"
    assert partial.writes == 0
    for table in (
        "facts",
        "observations",
        "fact_provenance",
        "memory_write_log",
    ):
        assert conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0] == 0
    assert (
        conn.execute("SELECT proposal_json IS NOT NULL FROM extract_queue").fetchone()[
            0
        ]
        == 1
    )

    service._writes._fact_writer = original_writer
    resumed = ExtractionProcessor(
        conn,
        service,
        extractor,
        retry_delay_seconds=0,
        evidence_verifier=VerifiedTestEvidence(),
    ).process_one()
    assert resumed.outcome == "completed"
    assert extractor.calls == 1
    assert conn.execute("SELECT count(*) FROM facts").fetchone()[0] == 2
    assert conn.execute("SELECT count(*) FROM observations").fetchone()[0] == 1
    assert conn.execute("SELECT count(*) FROM fact_provenance").fetchone()[0] == 2
