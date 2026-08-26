"""Untyped contradiction opens a visible, resolvable conflict by default."""

from __future__ import annotations

import sqlite3

import numpy as np
import pytest

from enfold.embeddings import embedding_to_bytes
from enfold.policy import MemoryPolicy
from enfold.protocol import ClientContext, Request
from enfold.schema import migrate
from enfold.service import EnfoldService


def _store(tmp_path) -> sqlite3.Connection:
    conn = sqlite3.connect(tmp_path / "enfold-untyped-conflict.db")
    conn.execute("PRAGMA foreign_keys=ON")
    migrate(conn)
    return conn


def _context(client: str, surface: str, agent: str) -> ClientContext:
    return ClientContext(
        client_id=client,
        surface=surface,
        agent_id=agent,
        session_id=f"{agent}-session",
        repository="enfold",
        branch="untyped-conflict",
        commit_sha="abc123",
        access_scopes=("private", "work"),
    )


def _request(request_id: str, method: str, **params) -> Request:
    return Request(request_id, method, params)


def _write(service: EnfoldService, context: ClientContext, key: str, content: str, **params):
    return service.handle(
        context,
        _request(
            f"req-{key}",
            "memory.write",
            idempotency_key=key,
            content=content,
            source_type="agent_report",
            **params,
        ),
    )


def _service(conn: sqlite3.Connection, **kwargs) -> EnfoldService:
    grants = {
        "codex-install": ("private", "work"),
        "claude-install": ("private", "work"),
        "cursor-install": ("private", "work"),
        "hermes-install": ("private", "work"),
    }
    return EnfoldService(
        conn,
        MemoryPolicy(
            grants,
            correction_authorities=("hermes-install",),
            conflict_resolution_authorities=("hermes-install",),
        ),
        **kwargs,
    )


def test_two_agents_untyped_contradiction_opens_conflict_and_third_agent_sees_receipt(
    tmp_path,
):
    conn = _store(tmp_path)
    service = _service(conn)
    codex = _context("codex-install", "codex", "codex")
    claude = _context("claude-install", "claude", "claude")
    cursor = _context("cursor-install", "cursor", "cursor")

    first = _write(
        service,
        codex,
        "codex-backend",
        "Enfold retrieval backend is sqlite-vec.",
    )
    second = _write(
        service,
        claude,
        "claude-backend",
        "Enfold retrieval backend is brute.",
    )

    assert first["outcome"] == "inserted"
    assert second["outcome"] == "conflict"
    assert second["fact_id"] != first["fact_id"]
    assert second["existing_fact_id"] == first["fact_id"]
    conflict_id = second["detail"]["conflict_id"]
    assert conflict_id
    assert {first["fact_id"], second["fact_id"]} <= set(
        second["detail"]["member_fact_ids"]
    )

    listed = service.handle(cursor, _request("conflicts", "memory.conflicts"))
    assert len(listed["conflicts"]) == 1
    assert listed["conflicts"][0]["conflict_id"] == conflict_id
    assert set(listed["conflicts"][0]["member_fact_ids"]) == {
        first["fact_id"],
        second["fact_id"],
    }
    member_text = " ".join(
        member["content"] for member in listed["conflicts"][0]["members"]
    )
    assert "sqlite-vec" in member_text
    assert "brute" in member_text

    search = service.handle(
        cursor,
        _request("search-backend", "memory.search", query="Enfold retrieval backend"),
    )
    assert search["facts"] == []
    assert search["open_conflicts"]
    receipt = search["open_conflicts"][0]
    assert receipt["conflict_id"] == conflict_id
    assert first["fact_id"] in receipt["member_fact_ids"]
    assert second["fact_id"] in receipt["member_fact_ids"]
    assert "do not treat either as current" in receipt["summary"]

    pack = service.handle(
        cursor,
        _request(
            "context-backend",
            "memory.context",
            query="what vector backend does Enfold use",
            token_budget=256,
        ),
    )
    assert f"[conflict:{conflict_id}" in pack["markdown"]
    assert "do not treat either as current" in pack["markdown"]
    assert "sqlite-vec" not in pack["markdown"]
    assert "brute" not in pack["markdown"]
    assert pack["open_conflicts"][0]["conflict_id"] == conflict_id
    assert all(fact.get("fact_id") for fact in pack["facts"])
    conn.close()


@pytest.mark.parametrize(
    "first_content, second_content",
    [
        (
            "Enfold retrieval backend is enabled.",
            "Enfold retrieval backend is disabled.",
        ),
        (
            "Enfold uses sqlite-vec for retrieval.",
            "Enfold does not use sqlite-vec for retrieval.",
        ),
        (
            "The dashboard port is 3100.",
            "The dashboard port is 3200.",
        ),
        (
            "Avery prefers tea.",
            "Avery prefers coffee.",
        ),
        (
            "Alice is in Boston.",
            "Alice is in Seattle.",
        ),
    ],
)
def test_untyped_polarity_and_value_flips_open_a_conflict(
    tmp_path, first_content, second_content
):
    conn = _store(tmp_path)
    service = _service(conn)
    first = _write(
        service,
        _context("codex-install", "codex", "codex"),
        "first",
        first_content,
    )
    second = _write(
        service,
        _context("claude-install", "claude", "claude"),
        "second",
        second_content,
    )

    assert first["outcome"] == "inserted"
    assert second["outcome"] == "conflict"
    assert second["detail"]["conflict_id"]
    conn.close()


@pytest.mark.parametrize(
    "first_content, second_content",
    [
        (
            "Avery prefers tea.",
            "Avery prefers tea with honey.",
        ),
        (
            "Client A implemented Enfold provenance.",
            "The Cedar registry stores the incident runbook.",
        ),
        (
            "Enfold retrieval backend is sqlite-vec.",
            "The Atlas backup schedule runs Tuesday.",
        ),
        (
            "Project Atlas is not deployed.",
            "Project Atlas uses Python.",
        ),
        (
            "Project Atlas is deployed.",
            "Project Atlas uses Python.",
        ),
        (
            "Project Alpha service is enabled.",
            "Project Beta feature is disabled.",
        ),
        (
            "Avery is a doctor.",
            "Avery is in Boston.",
        ),
    ],
)
def test_untyped_elaboration_and_unrelated_facts_do_not_open_a_conflict(
    tmp_path, first_content, second_content
):
    conn = _store(tmp_path)
    service = _service(conn)
    first = _write(
        service,
        _context("codex-install", "codex", "codex"),
        "first",
        first_content,
    )
    second = _write(
        service,
        _context("claude-install", "claude", "claude"),
        "second",
        second_content,
    )

    assert first["outcome"] == "inserted"
    assert second["outcome"] == "inserted"
    assert "conflict_id" not in second["detail"]
    assert (
        conn.execute(
            "SELECT COUNT(*) FROM fact_conflicts WHERE resolved_at IS NULL"
        ).fetchone()[0]
        == 0
    )
    conn.close()


def test_untyped_contradiction_still_detected_after_oldest_256_active_facts(tmp_path):
    conn = _store(tmp_path)
    service = _service(conn)
    conn.executemany(
        """
        INSERT INTO facts (content, category, tags, trust_score, scope)
        VALUES (?, 'general', '', 0.5, 'private')
        """,
        [
            (f"Standalone notebook entry {index} records a unique errand.",)
            for index in range(256)
        ],
    )
    conn.commit()

    first = _write(
        service,
        _context("codex-install", "codex", "codex"),
        "late-port-3100",
        "The dashboard port is 3100.",
    )
    second = _write(
        service,
        _context("claude-install", "claude", "claude"),
        "late-port-3200",
        "The dashboard port is 3200.",
    )

    assert first["outcome"] == "inserted"
    assert second["outcome"] == "conflict"
    assert second["detail"]["conflict_id"]
    assert first["fact_id"] in second["detail"]["member_fact_ids"]
    assert second["fact_id"] in second["detail"]["member_fact_ids"]
    conn.close()


def test_untyped_numbered_event_series_does_not_open_a_conflict(tmp_path):
    conn = _store(tmp_path)
    service = _service(conn)
    context = _context("codex-install", "codex", "codex")
    first = _write(service, context, "load-1", "Crash durability load memory 1.")
    second = _write(service, context, "load-2", "Crash durability load memory 2.")

    assert first["outcome"] == "inserted"
    assert second["outcome"] == "inserted"
    assert "conflict_id" not in second["detail"]
    conn.close()


def test_untyped_cross_scope_contradiction_does_not_open_a_conflict(tmp_path):
    conn = _store(tmp_path)
    service = _service(conn)
    first = _write(
        service,
        _context("codex-install", "codex", "codex"),
        "private-backend",
        "Enfold retrieval backend is sqlite-vec.",
        scope="private",
    )
    second = _write(
        service,
        _context("claude-install", "claude", "claude"),
        "work-backend",
        "Enfold retrieval backend is brute.",
        scope="work",
    )

    assert first["outcome"] == "inserted"
    assert second["outcome"] == "inserted"
    assert "conflict_id" not in second["detail"]
    conn.close()


def test_untyped_agreeing_near_duplicate_still_merges(tmp_path):
    conn = _store(tmp_path)
    identity = "fake:untyped:document:none:v1"
    service = _service(
        conn,
        embedding_identity=identity,
        query_embedder=lambda _content: np.asarray((1.0, 0.0), dtype=np.float32),
        near_dedup_enabled=True,
    )
    context = _context("codex-install", "codex", "codex")
    existing = _write(
        service, context, "existing-port", "The build uses port 3100.", trust_score=0.8
    )
    conn.execute(
        "INSERT INTO fact_embeddings(fact_id, embedding, dim, embedding_identity) "
        "VALUES (?, ?, 2, ?)",
        (
            existing["fact_id"],
            embedding_to_bytes(np.asarray((1.0, 0.0), dtype=np.float32)),
            identity,
        ),
    )
    conn.execute("INSERT INTO facts_fts(facts_fts) VALUES ('rebuild')")
    conn.commit()

    result = _write(
        service,
        context,
        "paraphrase-port",
        "Build service listens on port 3100.",
        trust_score=0.4,
    )

    assert result["outcome"] == "near_dedup"
    assert result["fact_id"] == existing["fact_id"]
    assert (
        conn.execute(
            "SELECT COUNT(*) FROM fact_conflicts WHERE resolved_at IS NULL"
        ).fetchone()[0]
        == 0
    )
    conn.close()


def test_untyped_contradiction_is_not_near_dedup_merged(tmp_path):
    conn = _store(tmp_path)
    identity = "fake:untyped:document:none:v1"
    service = _service(
        conn,
        embedding_identity=identity,
        query_embedder=lambda _content: np.asarray((1.0, 0.0), dtype=np.float32),
        near_dedup_enabled=True,
    )
    first = _write(
        service,
        _context("codex-install", "codex", "codex"),
        "vec",
        "Enfold retrieval backend is sqlite-vec.",
        trust_score=0.4,
    )
    conn.execute(
        "INSERT INTO fact_embeddings(fact_id, embedding, dim, embedding_identity) "
        "VALUES (?, ?, 2, ?)",
        (
            first["fact_id"],
            embedding_to_bytes(np.asarray((1.0, 0.0), dtype=np.float32)),
            identity,
        ),
    )
    conn.execute("INSERT INTO facts_fts(facts_fts) VALUES ('rebuild')")
    conn.commit()

    second = _write(
        service,
        _context("claude-install", "claude", "claude"),
        "brute",
        "Enfold retrieval backend is brute.",
        trust_score=0.9,
    )

    assert second["outcome"] == "conflict"
    assert second["fact_id"] != first["fact_id"]
    assert (
        conn.execute(
            "SELECT COUNT(*) FROM facts WHERE invalid_at IS NULL"
        ).fetchone()[0]
        == 2
    )
    conn.close()


def test_untyped_conflict_is_resolvable_through_review_flow(tmp_path):
    conn = _store(tmp_path)
    service = _service(conn)
    first = _write(
        service,
        _context("codex-install", "codex", "codex"),
        "codex-backend",
        "Enfold retrieval backend is sqlite-vec.",
    )
    second = _write(
        service,
        _context("claude-install", "claude", "claude"),
        "claude-backend",
        "Enfold retrieval backend is brute.",
    )
    conflict_id = second["detail"]["conflict_id"]
    hermes = _context("hermes-install", "hermes", "hermes")
    cursor = _context("cursor-install", "cursor", "cursor")

    with pytest.raises(Exception) as denied:
        service.handle(
            cursor,
            _request(
                "resolve-denied",
                "memory.resolve_conflict",
                conflict_id=conflict_id,
                resolution_fact_id=first["fact_id"],
                reason="cursor is not a resolution authority",
            ),
        )
    assert denied.value.code == "access_denied"

    resolved = service.handle(
        hermes,
        _request(
            "resolve-ok",
            "memory.resolve_conflict",
            conflict_id=conflict_id,
            resolution_fact_id=first["fact_id"],
            reason="inspected config still uses sqlite-vec",
        ),
    )["resolution"]
    assert resolved["resolution_fact_id"] == first["fact_id"]
    assert resolved["superseded_fact_ids"] == [second["fact_id"]]

    search = service.handle(
        cursor,
        _request("settled-search", "memory.search", query="Enfold retrieval backend"),
    )
    assert [fact["fact_id"] for fact in search["facts"]] == [first["fact_id"]]
    assert search["open_conflicts"] == []
    conn.close()
