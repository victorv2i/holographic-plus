"""Conservative write-time embedding near-duplicate handling."""

from __future__ import annotations

import sqlite3
import urllib.error

import numpy as np
import pytest

from enfold import _is_semantic_duplicate
from enfold.hybrid_retrieval import StoredEmbeddingError
from enfold.policy import MemoryPolicy
from enfold.provenance import ConnectionContext, WriteRequest, ensure_provenance_schema
from enfold.temporal import fact_history
from enfold.write_service import (
    FactWriteResult,
    MemoryWriteService,
    NearDedupConfig,
)


_IDENTITY = "test:model:document:none:v1"


def _connection() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(
        """
        CREATE TABLE facts (
            fact_id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL,
            category TEXT NOT NULL DEFAULT 'general',
            tags TEXT NOT NULL DEFAULT '',
            trust_score REAL NOT NULL DEFAULT 0.5,
            source_authority REAL NOT NULL DEFAULT 0.5,
            scope TEXT NOT NULL DEFAULT 'private',
            correction_status TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            invalid_at TEXT,
            superseded_by INTEGER,
            conflict_group TEXT
        );
        CREATE VIRTUAL TABLE facts_fts
        USING fts5(content, tags, content=facts, content_rowid=fact_id);
        """
    )
    ensure_provenance_schema(conn)
    conn.commit()
    return conn


def _writer(conn, request, observation_id):
    cursor = conn.execute(
        """INSERT INTO facts (content, category, tags, trust_score,
                               source_authority, scope, correction_status)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (
            request.content,
            request.category,
            request.tags,
            request.trust_score,
            request.source_authority,
            request.scope,
            request.correction_status,
        ),
    )
    return FactWriteResult(int(cursor.lastrowid))


def _context() -> ConnectionContext:
    return ConnectionContext(
        client_id="client-a-install-1",
        surface="client-a",
        agent_id="client-a",
        session_id="near-dedup",
        access_scopes=("private",),
    )


def _request(content: str, **changes) -> WriteRequest:
    values = {
        "idempotency_key": "near-dedup-write",
        "content": content,
        "source_type": "agent_report",
    }
    values.update(changes)
    return WriteRequest(**values)


def _embed(conn: sqlite3.Connection, fact_id: int, vector: tuple[float, ...]) -> None:
    values = np.asarray(vector, dtype="<f4")
    conn.execute(
        """INSERT INTO fact_embeddings (fact_id, embedding, dim, embedding_identity)
           VALUES (?, ?, ?, ?)""",
        (fact_id, values.tobytes(), len(values), _IDENTITY),
    )
    conn.commit()


def _refresh_fts(conn: sqlite3.Connection) -> None:
    conn.execute("INSERT INTO facts_fts(facts_fts) VALUES ('rebuild')")
    conn.commit()


def _service(
    conn: sqlite3.Connection, *, config: NearDedupConfig
) -> MemoryWriteService:
    return MemoryWriteService(
        conn,
        _writer,
        MemoryPolicy({"client-a-install-1": ("private",)}),
        near_dedup=config,
        query_embedder=lambda content: np.asarray((1.0, 0.0), dtype=np.float32),
    )


@pytest.mark.parametrize(
    "existing, incoming",
    [
        ("Avery prefers tea", "Avery prefers tea with honey"),
        ("Avery enjoys tea", "Avery enjoys fresh hot tea"),
        ("Avery prefers tea with honey", "Avery prefers tea"),
    ],
)
def test_semantic_duplicate_guard_keeps_inserted_or_deleted_content(
    existing, incoming
):
    assert _is_semantic_duplicate(incoming, existing, 1.0, 0.92) is False


def test_semantic_duplicate_guard_allows_stopword_only_insertion():
    assert _is_semantic_duplicate(
        "Avery prefers the tea", "Avery prefers tea", 1.0, 0.92
    ) is True


@pytest.mark.parametrize(
    "existing, incoming",
    [
        ("Alice manages Bob", "Bob manages Alice"),
        ("Avery prefers Earl Grey tea", "Avery prefers coffee"),
        ("Avery likes tea", "Avery really loves coffee with honey"),
    ],
)
def test_semantic_duplicate_guard_keeps_reordered_and_replaced_content(
    existing, incoming
):
    assert _is_semantic_duplicate(incoming, existing, 1.0, 0.92) is False


def test_near_duplicate_merges_evidence_and_keeps_history_on_higher_trust_survivor():
    conn = _connection()
    survivor = _writer(conn, _request("The build uses port 3100."), 0).fact_id
    conn.commit()
    _embed(conn, survivor, (1.0, 0.0))
    _refresh_fts(conn)

    result = _service(conn, config=NearDedupConfig(embedding_identity=_IDENTITY)).write(
        _context(),
        _request(
            "Build service listens on port 3100.",
            idempotency_key="paraphrase",
            trust_score=0.4,
        ),
    )

    assert result.outcome == "near_dedup"
    assert result.fact_id == survivor
    assert (
        conn.execute(
            "SELECT COUNT(*) FROM fact_provenance WHERE fact_id = ?", (survivor,)
        ).fetchone()[0]
        == 1
    )
    history = fact_history(conn, survivor)
    assert [row["fact_id"] for row in history] == [survivor, 2]
    assert history[1]["superseded_by"] == survivor
    assert tuple(
        conn.execute("SELECT outcome, fact_id FROM memory_write_log").fetchone()
    ) == ("near_dedup", survivor)


def test_batch_query_embeddings_are_computed_before_transaction_ownership():
    conn = _connection()
    survivor = _writer(conn, _request("The build uses port 3100."), 0).fact_id
    conn.commit()
    _embed(conn, survivor, (1.0, 0.0))
    _refresh_fts(conn)
    transaction_states: list[bool] = []

    def query_embedder(_content):
        transaction_states.append(conn.in_transaction)
        return np.asarray((1.0, 0.0), dtype=np.float32)

    service = MemoryWriteService(
        conn,
        _writer,
        MemoryPolicy({"client-a-install-1": ("private",)}),
        near_dedup=NearDedupConfig(embedding_identity=_IDENTITY),
        query_embedder=query_embedder,
    )
    writes = (
        (
            _request(
                "Build service listens on port 3100.",
                idempotency_key="batch-near-1",
                trust_score=0.4,
            ),
            None,
        ),
        (
            _request(
                "Build server listens on port 3100.",
                idempotency_key="batch-near-2",
                trust_score=0.4,
            ),
            None,
        ),
    )

    result = service.write_batch(_context(), writes)

    assert result.committed is True
    assert transaction_states == [False, False]


def test_rejected_batch_item_is_not_sent_to_query_embedder():
    conn = _connection()
    embedded: list[str] = []
    service = MemoryWriteService(
        conn,
        _writer,
        MemoryPolicy({"client-a-install-1": ("private",)}),
        near_dedup=NearDedupConfig(embedding_identity=_IDENTITY),
        query_embedder=lambda content: (
            embedded.append(content) or np.asarray((1.0, 0.0), dtype=np.float32)
        ),
    )
    valid = _request("A valid first fact.", idempotency_key="batch-policy-1")
    rejected = _request(
        "api_key = abcdefghijklmnopqrstuv",
        idempotency_key="batch-policy-2",
    )

    result = service.write_batch(
        _context(),
        ((valid, None), (rejected, None)),
        rollback_if=lambda outcome: outcome.outcome == "rejected",
    )

    assert result.committed is False
    assert embedded == [valid.content]
    assert conn.execute("SELECT count(*) FROM facts").fetchone()[0] == 0


@pytest.mark.parametrize(
    "existing, incoming",
    [
        ("The service runs on port 3100.", "The service runs on port 3200."),
        ("The deployment is active.", "The deployment is archived."),
        ("The feature is enabled.", "The feature is not enabled."),
        ("The release is scheduled for March.", "The release is scheduled for April."),
        ("The application is deployed.", "The application is unavailable."),
    ],
)
def test_near_duplicate_guards_keep_value_and_state_changes(existing, incoming):
    conn = _connection()
    existing_id = _writer(conn, _request(existing), 0).fact_id
    conn.commit()
    _embed(conn, existing_id, (1.0, 0.0))
    _refresh_fts(conn)

    result = _service(conn, config=NearDedupConfig(embedding_identity=_IDENTITY)).write(
        _context(), _request(incoming, idempotency_key="changed")
    )

    assert result.outcome == "inserted"
    assert (
        conn.execute("SELECT COUNT(*) FROM facts WHERE invalid_at IS NULL").fetchone()[
            0
        ]
        == 2
    )


@pytest.mark.parametrize(
    "exc",
    [
        StoredEmbeddingError("production query embedding is unavailable"),
        urllib.error.URLError("connection refused"),
        TimeoutError("ollama embed timed out"),
        OSError("network is unreachable"),
    ],
)
def test_embedder_outage_skips_near_dedup_and_still_writes(exc):
    conn = _connection()
    existing_id = _writer(
        conn, _request("Build service listens on port 3100."), 0
    ).fact_id
    conn.commit()
    _embed(conn, existing_id, (1.0, 0.0))
    _refresh_fts(conn)

    def boom(_content):
        raise exc

    service = MemoryWriteService(
        conn,
        _writer,
        MemoryPolicy({"client-a-install-1": ("private",)}),
        near_dedup=NearDedupConfig(embedding_identity=_IDENTITY),
        query_embedder=boom,
    )

    result = service.write(
        _context(),
        _request("The build uses port 3100.", idempotency_key="embedder-outage"),
    )

    assert result.outcome == "inserted"
    assert result.fact_id != existing_id
    existing = conn.execute(
        "SELECT invalid_at, superseded_by FROM facts WHERE fact_id = ?",
        (existing_id,),
    ).fetchone()
    assert tuple(existing) == (None, None)


def test_near_duplicate_is_disabled_or_unavailable_without_a_stored_embedding():
    conn = _connection()
    existing_id = _writer(
        conn, _request("Build service listens on port 3100."), 0
    ).fact_id
    conn.commit()

    result = _service(
        conn,
        config=NearDedupConfig(enabled=False, embedding_identity=_IDENTITY),
    ).write(
        _context(),
        _request("The build uses port 3100.", idempotency_key="disabled"),
    )

    assert result.outcome == "inserted"
    assert existing_id != result.fact_id


def test_implicit_near_dedup_does_not_retire_typed_state():
    conn = _connection()
    conn.execute("ALTER TABLE facts ADD COLUMN memory_kind TEXT")
    existing_id = conn.execute(
        """INSERT INTO facts (content, trust_score, source_authority, memory_kind)
           VALUES ('Build service listens on port 3100.', 0.5, 0.5, 'state')"""
    ).lastrowid
    conn.commit()
    _embed(conn, existing_id, (1.0, 0.0))
    _refresh_fts(conn)

    result = _service(conn, config=NearDedupConfig(embedding_identity=_IDENTITY)).write(
        _context(),
        _request(
            "The build uses port 3100.",
            idempotency_key="keep-state",
            trust_score=0.5,
        ),
    )

    assert result.outcome == "near_dedup"
    assert result.fact_id == existing_id
    kept = conn.execute(
        "SELECT invalid_at, superseded_by, memory_kind FROM facts WHERE fact_id = ?",
        (existing_id,),
    ).fetchone()
    assert tuple(kept) == (None, None, "state")


def test_unreviewed_write_skips_near_dedup_in_both_directions():
    conn = _connection()
    existing_id = _writer(
        conn, _request("Build service listens on port 3100."), 0
    ).fact_id
    conn.commit()
    _embed(conn, existing_id, (1.0, 0.0))
    _refresh_fts(conn)

    result = _service(conn, config=NearDedupConfig(embedding_identity=_IDENTITY)).write(
        _context(),
        _request(
            "The build uses port 3100.",
            idempotency_key="unreviewed-skip",
            correction_status="unreviewed",
            trust_score=0.9,
        ),
    )

    assert result.outcome == "inserted"
    assert result.fact_id != existing_id
    existing = conn.execute(
        "SELECT invalid_at, superseded_by FROM facts WHERE fact_id = ?",
        (existing_id,),
    ).fetchone()
    incoming = conn.execute(
        "SELECT invalid_at, superseded_by FROM facts WHERE fact_id = ?",
        (result.fact_id,),
    ).fetchone()
    assert tuple(existing) == (None, None)
    assert tuple(incoming) == (None, None)


def test_near_duplicate_prefers_the_newer_fact_when_trust_ties():
    conn = _connection()
    existing_id = _writer(
        conn, _request("Build service listens on port 3100."), 0
    ).fact_id
    conn.commit()
    _embed(conn, existing_id, (1.0, 0.0))
    _refresh_fts(conn)

    result = _service(conn, config=NearDedupConfig(embedding_identity=_IDENTITY)).write(
        _context(),
        _request("The build uses port 3100.", idempotency_key="newer", trust_score=0.5),
    )

    assert result.outcome == "near_dedup"
    assert result.fact_id != existing_id
    assert (
        conn.execute(
            "SELECT superseded_by FROM facts WHERE fact_id = ?", (existing_id,)
        ).fetchone()[0]
        == result.fact_id
    )


def test_missing_embedding_keeps_the_exact_dedup_fallback():
    conn = _connection()
    existing_id = _writer(
        conn, _request("Build service listens on port 3100."), 0
    ).fact_id
    conn.commit()

    result = _service(conn, config=NearDedupConfig(embedding_identity=_IDENTITY)).write(
        _context(),
        _request("Build service listens on port 3100.", idempotency_key="exact"),
    )

    assert result.outcome == "dedup"
    assert result.fact_id == existing_id
