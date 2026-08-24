from __future__ import annotations

import sqlite3
import time

import pytest

from enfold.extraction_enqueue import ExtractionEnqueuer
from enfold.extraction_processor import (
    EvidenceVerification,
    ExtractedMemory,
    ExtractionProcessor,
)
from enfold.extraction_spans import transcript_spans
from enfold.llm_extract import ExtractionParseError, _parse_response
from enfold.policy import MemoryPolicy, default_credential_screen
from enfold.provenance import ConnectionContext, WriteRequest
from enfold.schema import MigrationError, migrate
from enfold.service import EnfoldService


def _setup(tmp_path):
    conn = sqlite3.connect(tmp_path / "extraction-safety.db")
    conn.execute("PRAGMA foreign_keys=ON")
    migrate(conn)
    context = ConnectionContext(
        client_id="hermes-install",
        surface="hermes",
        agent_id="avery",
        session_id="safety-session",
        access_scopes=("private", "work"),
    )
    service = EnfoldService(
        conn, MemoryPolicy({"hermes-install": ("private", "work")})
    )
    ExtractionEnqueuer(conn).enqueue_after_commit(
        context, "Avery uses durable shared memory.", source="session_end"
    )
    return conn, context, service


class ChangingExtractor:
    identity = "changing-extractor:v1"

    def __init__(self):
        self.calls = 0

    def extract(self, envelope):
        self.calls += 1
        span_id = transcript_spans(envelope.transcript)[0].span_id
        return (
            ExtractedMemory(
                f"First proposal from model call {self.calls}.",
                evidence_excerpt="Avery uses durable shared memory.",
                metadata={"evidence_span_id": span_id},
            ),
            ExtractedMemory(
                f"Second proposal from model call {self.calls}.",
                evidence_excerpt="Avery uses durable shared memory.",
                metadata={"evidence_span_id": span_id},
            ),
        )


class VerifiedTestEvidence:
    def verify(self, _proposal, *, evidence_excerpt, envelope):
        assert evidence_excerpt in envelope.transcript
        return EvidenceVerification("verified", "test-evidence-v1")


def test_partial_write_replays_persisted_snapshot_without_recalling_model(tmp_path):
    conn, _context, service = _setup(tmp_path)
    extractor = ChangingExtractor()

    class FailSecondWrite:
        calls = 0

        def handle(self, client_context, request):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("simulated write interruption")
            return service.handle(client_context, request)

    partial = ExtractionProcessor(
        conn,
        FailSecondWrite(),
        extractor,
        retry_delay_seconds=0,
        evidence_verifier=VerifiedTestEvidence(),
    ).process_one()
    assert partial.outcome == "retry"
    assert extractor.calls == 1
    snapshot = conn.execute(
        "SELECT proposal_json, proposal_hash FROM extract_queue"
    ).fetchone()
    assert snapshot[0] and snapshot[1]

    resumed = ExtractionProcessor(
        conn,
        service,
        extractor,
        retry_delay_seconds=0,
        evidence_verifier=VerifiedTestEvidence(),
    ).process_one()
    assert resumed.outcome == "completed"
    assert extractor.calls == 1
    assert [row[0] for row in conn.execute("SELECT content FROM facts ORDER BY fact_id")] == [
        "First proposal from model call 1.",
        "Second proposal from model call 1.",
    ]
    conn.close()


def test_heartbeat_renews_active_lease_during_slow_extraction(tmp_path):
    conn, _context, service = _setup(tmp_path)

    class SlowExtractor:
        identity = "slow-extractor:v1"

        def extract(self, envelope):
            time.sleep(0.06)
            return (
                ExtractedMemory(
                    "A fact after a long extraction.",
                    evidence_excerpt="Avery uses durable shared memory.",
                    metadata={
                        "evidence_span_id": transcript_spans(
                            envelope.transcript
                        )[0].span_id,
                    },
                ),
            )

    worker = ExtractionProcessor(
        conn,
        service,
        SlowExtractor(),
        lease_seconds=1,
        heartbeat_seconds=0.01,
        evidence_verifier=VerifiedTestEvidence(),
    )
    renewals = []
    renew = worker._renew

    def record_renew(row_id, token):
        renewals.append((row_id, token))
        return renew(row_id, token)

    worker._renew = record_renew
    assert worker.process_one().outcome == "completed"
    assert renewals
    conn.close()


def test_stale_worker_cannot_renew_a_reclaimed_fence(tmp_path):
    conn, _context, service = _setup(tmp_path)
    clock = [100.0]
    first = ExtractionProcessor(
        conn,
        service,
        ChangingExtractor(),
        worker_id="same-worker",
        lease_seconds=5,
        heartbeat_seconds=1,
        clock=lambda: clock[0],
    )
    claimed = first._claim()
    assert claimed is not None
    row_id, _payload, _digest, _attempts, old_token = claimed
    clock[0] = 106.0
    second = ExtractionProcessor(
        conn,
        service,
        ChangingExtractor(),
        worker_id="same-worker",
        lease_seconds=5,
        heartbeat_seconds=1,
        clock=lambda: clock[0],
    )
    assert second._claim() is not None
    with pytest.raises(RuntimeError, match="lease was lost before renewal"):
        first._renew(row_id, old_token)
    conn.close()


def test_automatic_proposals_cannot_escalate_scope(tmp_path):
    conn, _context, service = _setup(tmp_path)
    proposal = ExtractedMemory(
        "A scoped proposal.",
        scope="work",
        evidence_excerpt="Avery uses durable shared memory.",
    )

    class UnsafeExtractor:
        identity = "unsafe-extractor:v1"

        def extract(self, _envelope):
            return (proposal,)

    result = ExtractionProcessor(conn, service, UnsafeExtractor()).process_one()
    assert result.outcome == "dead"
    assert result.error == "proposal_scope_rejected"
    assert conn.execute("SELECT count(*) FROM facts").fetchone()[0] == 0
    assert conn.execute("SELECT last_error FROM extract_queue").fetchone()[0] == (
        "proposal_scope_rejected"
    )
    conn.close()


def test_automatic_trust_and_authority_are_server_fixed(tmp_path):
    conn, _context, service = _setup(tmp_path)

    class ElevatingExtractor:
        identity = "elevating-extractor:v1"

        def extract(self, envelope):
            return (
                ExtractedMemory(
                    "Model tried to self-elevate authority.",
                    trust_score=1.0,
                    source_authority=1.0,
                    evidence_excerpt="Avery uses durable shared memory.",
                    metadata={
                        "evidence_span_id": transcript_spans(
                            envelope.transcript
                        )[0].span_id,
                    },
                ),
            )

    result = ExtractionProcessor(
        conn,
        service,
        ElevatingExtractor(),
        evidence_verifier=VerifiedTestEvidence(),
    ).process_one()
    assert result.outcome == "completed"
    row = conn.execute("SELECT trust_score, source_authority FROM facts").fetchone()
    assert tuple(row) == (0.5, 0.5)
    conn.close()


def test_model_exception_text_is_never_persisted(tmp_path):
    conn, _context, service = _setup(tmp_path)
    secret = "sk-" + "super-secret-model-error"

    class FailingExtractor:
        identity = "failing-extractor:v1"

        def extract(self, _envelope):
            raise RuntimeError(secret)

    result = ExtractionProcessor(conn, service, FailingExtractor()).process_one()
    assert result.outcome == "retry"
    assert result.error == "extractor_failed"
    stored = conn.execute("SELECT last_error FROM extract_queue").fetchone()[0]
    assert stored == "extractor_failed"
    assert secret not in stored
    conn.close()


@pytest.mark.parametrize(
    "header",
    [
        "-----BEGIN ENCRYPTED PRIVATE KEY-----",
        "-----BEGIN DSA PRIVATE KEY-----",
    ],
)
def test_standard_pem_private_key_headers_are_screened(header):
    request = WriteRequest(
        idempotency_key="pem-screen",
        content=f"Stored key: {header}",
        source_type="automatic_extraction",
    )

    assert default_credential_screen(request) is not None


def test_parse_error_never_contains_raw_model_output():
    secret = "api_key = abcdefghijklmnopqrstuv"
    raw = f"model transcript fragment with {secret} and no array"

    with pytest.raises(ExtractionParseError) as captured:
        _parse_response(raw)

    message = str(captured.value)
    assert "length=" in message
    assert "parse_error_pos=" in message
    assert secret not in message
    assert raw[:20] not in message


def test_explicit_migration_adds_snapshot_columns_to_a_pre_snapshot_v1_store(tmp_path):
    conn = sqlite3.connect(tmp_path / "old-v1.db")
    migrate(conn)
    conn.execute("ALTER TABLE extract_queue DROP COLUMN proposal_json")
    conn.commit()
    assert "proposal_json" not in {
        row[1] for row in conn.execute("PRAGMA table_info(extract_queue)")
    }
    assert migrate(conn) == 1
    assert {"proposal_json", "proposal_hash"} <= {
        row[1] for row in conn.execute("PRAGMA table_info(extract_queue)")
    }
    conn.close()


def test_migration_refuses_a_partial_proposal_snapshot(tmp_path):
    conn = sqlite3.connect(tmp_path / "partial-snapshot.db")
    migrate(conn)
    conn.execute(
        "INSERT INTO extract_queue(payload, status, proposal_hash) "
        "VALUES ('legacy transcript', 'pending', 'orphaned-hash')"
    )
    conn.commit()
    with pytest.raises(MigrationError, match="proposal snapshot columns are inconsistent"):
        migrate(conn)
    conn.close()
