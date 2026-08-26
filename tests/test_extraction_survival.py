"""Headline survival cases for automatic extraction.

A credential-shaped line in the same transcript must not erase durable facts.
"""

from __future__ import annotations

import sqlite3

from enfold.extraction_enqueue import ExtractionEnqueuer
from enfold.extraction_processor import (
    EvidenceVerification,
    ExtractedMemory,
    ExtractionProcessor,
)
from enfold.extraction_spans import transcript_spans
from enfold.policy import MemoryPolicy
from enfold.provenance import ConnectionContext
from enfold.schema import migrate
from enfold.service import EnfoldService
from enfold.state_slots import current_state_facts, list_state_conflicts, read_current_state


SURVIVAL_TRANSCRIPT = "\n\n".join(
    (
        "Ada prefers Terra for daily briefing work.",
        "The home lab keeps Enfold on a local SQLite store.",
        "Weekly reviews happen on Monday evenings.",
        "The coffee order is a medium oat latte.",
        "api_key = exampletest",
        "Deployments stay on the private worktree until review.",
    )
)

DURABLE_FACTS = (
    "Ada prefers Terra for daily briefing work.",
    "The home lab keeps Enfold on a local SQLite store.",
    "Weekly reviews happen on Monday evenings.",
    "The coffee order is a medium oat latte.",
    "Deployments stay on the private worktree until review.",
)
CREDENTIAL_LINE = "api_key = exampletest"


class SurvivalExtractor:
    identity = "survival-extractor:v1"

    def extract(self, envelope):
        spans = {span.text: span for span in transcript_spans(envelope.transcript)}
        proposals = []
        for content in (*DURABLE_FACTS, CREDENTIAL_LINE):
            span = spans[content]
            proposals.append(
                ExtractedMemory(
                    content,
                    evidence_excerpt=content,
                    metadata={"evidence_span_id": span.span_id},
                )
            )
        return tuple(proposals)


class VerifiedTestEvidence:
    identity = "test-evidence-v1"

    def verify(self, _proposal, *, evidence_excerpt, envelope):
        assert evidence_excerpt in envelope.transcript
        return EvidenceVerification("verified", self.identity)


def _setup(tmp_path, transcript=SURVIVAL_TRANSCRIPT):
    conn = sqlite3.connect(tmp_path / "survival.db")
    conn.execute("PRAGMA foreign_keys=ON")
    migrate(conn)
    context = ConnectionContext(
        client_id="survival-tests",
        surface="client-a",
        agent_id="client-a",
        session_id="survival-session",
        access_scopes=("private",),
    )
    service = EnfoldService(conn, MemoryPolicy({"survival-tests": ("private",)}))
    ExtractionEnqueuer(conn).enqueue_after_commit(
        context,
        [{"role": "user", "content": transcript}],
        source="session_end",
        scope="private",
    )
    return conn, context, service


def test_durable_facts_survive_one_credential_shaped_line_in_the_same_transcript(
    tmp_path,
):
    conn, _context, service = _setup(tmp_path)

    result = ExtractionProcessor(
        conn,
        service,
        SurvivalExtractor(),
        evidence_verifier=VerifiedTestEvidence(),
    ).process_one()

    assert result.outcome == "completed"
    contents = {
        row[0] for row in conn.execute("SELECT content FROM facts").fetchall()
    }
    assert contents == set(DURABLE_FACTS)
    assert CREDENTIAL_LINE not in contents
    assert conn.execute("SELECT count(*) FROM extract_queue").fetchone()[0] == 0
    conn.close()


def test_same_batch_duplicate_slots_keep_typed_state_and_open_a_conflict(tmp_path):
    transcript = "briefing uses Terra\n\nbriefing uses Grok"
    conn, _context, service = _setup(tmp_path, transcript)
    spans = transcript_spans(transcript)

    class SlotExtractor:
        identity = "slot-survival:v1"

        def extract(self, envelope):
            del envelope
            return (
                ExtractedMemory(
                    "briefing uses Terra",
                    evidence_excerpt="briefing uses Terra",
                    state={
                        "kind": "state",
                        "subject": "briefing",
                        "predicate": "model",
                        "value": "Terra",
                        "confidence": 0.99,
                    },
                    metadata={"evidence_span_id": spans[0].span_id},
                ),
                ExtractedMemory(
                    "briefing uses Grok",
                    evidence_excerpt="briefing uses Grok",
                    state={
                        "kind": "state",
                        "subject": "briefing",
                        "predicate": "model",
                        "value": "Grok",
                        "confidence": 0.99,
                    },
                    metadata={"evidence_span_id": spans[1].span_id},
                ),
            )

    result = ExtractionProcessor(
        conn,
        service,
        SlotExtractor(),
        evidence_verifier=VerifiedTestEvidence(),
    ).process_one()

    assert result.outcome == "completed"
    assert len(list_state_conflicts(conn)) == 1
    current = current_state_facts(conn, "briefing", "model")
    assert {fact.object_value for fact in current} == {"Terra", "Grok"}
    assert read_current_state(conn, "briefing", "model") is None
    assert {
        row[0] for row in conn.execute("SELECT memory_kind FROM facts").fetchall()
    } == {"state"}
    conn.close()
