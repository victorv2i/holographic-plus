"""Replay stability for reconnects and factless first outcomes."""

from __future__ import annotations

import sqlite3

from enfold.policy import MemoryPolicy
from enfold.provenance import ConnectionContext, WriteRequest, ensure_provenance_schema
from enfold.write_service import FactWriteResult, MemoryWriteService


def _connection() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute(
        """CREATE TABLE facts (
            fact_id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL UNIQUE,
            category TEXT,
            tags TEXT,
            trust_score REAL,
            source_authority REAL,
            scope TEXT NOT NULL DEFAULT 'private',
            correction_status TEXT,
            invalid_at TEXT,
            superseded_by INTEGER
        )"""
    )
    ensure_provenance_schema(conn)
    conn.commit()
    return conn


def _writer(conn, request, observation_id):
    cursor = conn.execute(
        """INSERT INTO facts (
               content, category, tags, trust_score, source_authority,
               correction_status, scope
           ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (
            request.content,
            request.category,
            request.tags,
            request.trust_score,
            request.source_authority,
            request.correction_status,
            request.scope,
        ),
    )
    return FactWriteResult(cursor.lastrowid)


def _context(**changes):
    values = {
        "client_id": "client-a-install-1",
        "surface": "client-a",
        "agent_id": "client-a",
        "session_id": "thread-123",
        "repository": "enfold",
    }
    values.update(changes)
    return ConnectionContext(**values)


def _request(**changes):
    values = {
        "idempotency_key": "write-123",
        "content": "Client A implemented Enfold provenance.",
        "source_type": "agent_report",
        "performed_by": "client-a",
        "evidence_excerpt": "Focused tests passed.",
    }
    values.update(changes)
    return WriteRequest(**values)


def _service(conn):
    return MemoryWriteService(
        conn,
        _writer,
        MemoryPolicy({"client-a-install-1": ("private", "work")}),
    )


def test_same_key_and_body_replay_after_session_reconnect():
    conn = _connection()
    service = _service(conn)
    first = service.write(_context(session_id="thread-123"), _request())

    replay = service.write(_context(session_id="thread-reconnect"), _request())

    assert first.replayed is False
    assert replay.replayed is True
    assert replay.write_id == first.write_id
    assert replay.fact_id == first.fact_id
    assert conn.execute("SELECT count(*) FROM facts").fetchone()[0] == 1
    assert conn.execute("SELECT count(*) FROM memory_write_log").fetchone()[0] == 1


def test_missing_supersede_target_does_not_burn_key_across_reconnect():
    conn = _connection()
    service = _service(conn)
    request = _request(
        idempotency_key="supersede-later",
        content="replacement claim",
        supersede_fact_id=7,
    )

    first = service.write(_context(session_id="thread-1"), request)

    assert first.outcome == "needs_review"
    assert first.replayed is False
    assert conn.execute("SELECT count(*) FROM memory_write_log").fetchone()[0] == 0

    conn.execute("INSERT INTO facts(fact_id, content) VALUES (7, 'now visible')")
    conn.commit()

    second = service.write(_context(session_id="thread-2"), request)

    assert second.outcome == "inserted"
    assert second.replayed is False
    assert second.fact_id is not None
    assert conn.execute(
        "SELECT superseded_by FROM facts WHERE fact_id = 7"
    ).fetchone()[0] == second.fact_id
    assert conn.execute("SELECT count(*) FROM memory_write_log").fetchone()[0] == 1


def test_unauthorized_scope_does_not_burn_key_after_reconnect_with_grant():
    conn = _connection()
    service = _service(conn)
    request = _request(
        idempotency_key="scope-later",
        content="work-scoped claim",
        scope="work",
    )

    first = service.write(
        _context(session_id="thread-1", access_scopes=("private",)),
        request,
    )

    assert first.outcome == "rejected"
    assert first.replayed is False
    assert conn.execute("SELECT count(*) FROM facts").fetchone()[0] == 0
    assert conn.execute("SELECT count(*) FROM memory_write_log").fetchone()[0] == 0

    second = service.write(
        _context(session_id="thread-2", access_scopes=("private", "work")),
        request,
    )

    assert second.outcome == "inserted"
    assert second.replayed is False
    assert conn.execute("SELECT scope FROM facts").fetchone()[0] == "work"
    assert conn.execute("SELECT count(*) FROM memory_write_log").fetchone()[0] == 1
