from __future__ import annotations

import sqlite3

from enfold.policy import MemoryPolicy
from enfold.protocol import ClientContext, Request
from enfold.provenance import ConnectionContext, WriteRequest
from enfold.schema import migrate
from enfold.service import EnfoldService
from enfold.state_slots import (
    StateCandidate,
    current_state_facts,
    list_state_conflicts,
    read_current_state,
)
from enfold.write_service import FactWriteResult, MemoryWriteService


def _store() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys = ON")
    migrate(conn)
    return conn


def _writer(conn, request, observation_id):
    cursor = conn.execute(
        """
        INSERT INTO facts (
            content, category, tags, trust_score, source_authority,
            scope, sensitivity, correction_status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            request.content,
            request.category,
            request.tags,
            request.trust_score,
            request.source_authority,
            request.scope,
            request.sensitivity,
            request.correction_status,
        ),
    )
    return FactWriteResult(int(cursor.lastrowid))


def _context(client_id: str, agent_id: str) -> ConnectionContext:
    return ConnectionContext(
        client_id=client_id,
        surface=agent_id,
        agent_id=agent_id,
        session_id=f"{agent_id}-session",
        access_scopes=("private", "work"),
    )


def _service(conn) -> MemoryWriteService:
    return MemoryWriteService(
        conn,
        _writer,
        MemoryPolicy(
            {
                "agent-a-install-1": ("private", "work"),
                "agent-b-install-1": ("private", "work"),
                "agent-c-install-1": ("private", "work"),
            }
        ),
    )


def _state_write(
    service: MemoryWriteService,
    context: ConnectionContext,
    key: str,
    value: str,
    *,
    authority: float = 0.5,
    valid_from: str | None = None,
    observed_at: str | None = None,
):
    content = f"The dashboard port is {value}"
    request = WriteRequest(
        idempotency_key=key,
        content=content,
        source_type="agent_report",
        source_authority=authority,
        observed_at=observed_at,
        performed_by=context.agent_id,
    )
    candidate = StateCandidate(
        content=content,
        subject_key="env:dashboard",
        predicate_key="port",
        object_value=value,
        source_authority=authority,
        valid_from=valid_from,
    )
    return service.write(context, request, state_candidate=candidate)


def test_equal_authority_undated_agent_writes_open_a_conflict():
    conn = _store()
    service = _service(conn)
    first = _state_write(
        service, _context("agent-a-install-1", "agent-a"), "slot-a", "3100"
    )
    second = _state_write(
        service, _context("agent-b-install-1", "agent-b"), "slot-b", "3200"
    )

    assert first.outcome == "add"
    assert second.outcome == "conflict"
    assert read_current_state(conn, "env:dashboard", "port") is None
    current = current_state_facts(conn, "env:dashboard", "port")
    assert {fact.object_value for fact in current} == {"3100", "3200"}
    assert all(fact.conflict_group for fact in current)
    conflicts = list_state_conflicts(conn)
    assert len(conflicts) == 1
    assert set(conflicts[0].member_fact_ids) == {first.fact_id, second.fact_id}


def test_undated_state_keeps_transaction_time_off_valid_from():
    conn = _store()
    service = _service(conn)
    outcome = _state_write(
        service, _context("agent-a-install-1", "agent-a"), "clock-1", "3100"
    )

    valid_from, = conn.execute(
        "SELECT valid_from FROM facts WHERE fact_id = ?", (outcome.fact_id,)
    ).fetchone()
    recorded_at, = conn.execute(
        "SELECT recorded_at FROM memory_write_log WHERE fact_id = ?",
        (outcome.fact_id,),
    ).fetchone()
    observed_recorded_at, = conn.execute(
        "SELECT recorded_at FROM observations WHERE observation_id = ?",
        (outcome.observation_id,),
    ).fetchone()
    assert valid_from is None
    assert recorded_at is not None
    assert observed_recorded_at == recorded_at


def test_caller_observed_at_is_used_as_valid_from():
    conn = _store()
    service = _service(conn)
    first = _state_write(
        service,
        _context("agent-a-install-1", "agent-a"),
        "obs-1",
        "3100",
        observed_at="2026-07-11T10:00:00Z",
    )
    second = _state_write(
        service,
        _context("agent-b-install-1", "agent-b"),
        "obs-2",
        "3200",
        observed_at="2026-07-12T10:00:00Z",
    )

    assert first.outcome == "add"
    assert second.outcome == "supersede"
    assert (
        conn.execute(
            "SELECT valid_from FROM facts WHERE fact_id = ?", (second.fact_id,)
        ).fetchone()[0]
        == "2026-07-12T10:00:00Z"
    )
    assert (
        read_current_state(conn, "env:dashboard", "port").fact_id == second.fact_id
    )


def test_higher_authority_still_supersedes_undated_current():
    conn = _store()
    service = _service(conn)
    first = _state_write(
        service, _context("agent-a-install-1", "agent-a"), "auth-1", "3100"
    )
    second = _state_write(
        service,
        _context("agent-b-install-1", "agent-b"),
        "auth-2",
        "3200",
        authority=0.9,
    )

    assert first.outcome == "add"
    assert second.outcome == "supersede"
    assert (
        read_current_state(conn, "env:dashboard", "port").fact_id == second.fact_id
    )


def test_caller_supplied_newer_valid_from_still_supersedes():
    conn = _store()
    service = _service(conn)
    first = _state_write(
        service,
        _context("agent-a-install-1", "agent-a"),
        "vf-1",
        "3100",
        valid_from="2026-07-11T12:00:00Z",
    )
    second = _state_write(
        service,
        _context("agent-b-install-1", "agent-b"),
        "vf-2",
        "3200",
        valid_from="2026-07-12T12:00:00Z",
    )

    assert first.outcome == "add"
    assert second.outcome == "supersede"
    assert (
        read_current_state(conn, "env:dashboard", "port").fact_id == second.fact_id
    )


def test_memory_conflicts_shows_both_undated_equal_authority_values(tmp_path):
    conn = sqlite3.connect(tmp_path / "state-truth.db")
    conn.execute("PRAGMA foreign_keys = ON")
    migrate(conn)
    grants = {
        "agent-a-install": ("private", "work"),
        "agent-b-install": ("private", "work"),
    }
    service = EnfoldService(conn, MemoryPolicy(grants))
    slot = {"subject_key": "env:dashboard", "predicate_key": "port"}
    contexts = {
        "a": ClientContext(
            client_id="agent-a-install",
            surface="agent-a",
            agent_id="agent-a",
            session_id="agent-a-session",
            access_scopes=("private", "work"),
        ),
        "b": ClientContext(
            client_id="agent-b-install",
            surface="agent-b",
            agent_id="agent-b",
            session_id="agent-b-session",
            access_scopes=("private", "work"),
        ),
    }

    first = service.handle(
        contexts["a"],
        Request(
            "req-a",
            "memory.write",
            {
                "idempotency_key": "truth-a",
                "content": "The dashboard port is 3100",
                "source_type": "agent_report",
                "state": {**slot, "object_value": "3100"},
            },
        ),
    )
    second = service.handle(
        contexts["b"],
        Request(
            "req-b",
            "memory.write",
            {
                "idempotency_key": "truth-b",
                "content": "The dashboard port is 3200",
                "source_type": "agent_report",
                "state": {**slot, "object_value": "3200"},
            },
        ),
    )

    assert first["outcome"] == "add"
    assert second["outcome"] == "conflict"
    listed = service.handle(
        contexts["a"],
        Request("req-conflicts", "memory.conflicts", {}),
    )
    assert len(listed["conflicts"]) == 1
    member_ids = set(listed["conflicts"][0]["member_fact_ids"])
    assert member_ids == {first["fact_id"], second["fact_id"]}
    contents = {member["content"] for member in listed["conflicts"][0]["members"]}
    assert contents == {
        "The dashboard port is 3100",
        "The dashboard port is 3200",
    }
    conn.close()
