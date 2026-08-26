from __future__ import annotations

import json
import sqlite3

import pytest

from enfold.extraction_enqueue import (
    CaptureDisabled,
    CaptureEnableError,
    ExtractionEnqueuer,
    ExtractionQueueUnavailable,
    capture_status,
    enable_capture,
)
from enfold.provenance import ConnectionContext
from enfold.schema import migrate


def _connection(tmp_path):
    conn = sqlite3.connect(tmp_path / "queue.db")
    conn.execute(
        """
        CREATE TABLE extract_queue (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            payload TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            payload_hash TEXT
        )
        """
    )
    conn.execute(
        "CREATE UNIQUE INDEX active_payload_hash ON extract_queue(payload_hash) "
        "WHERE status IN ('pending', 'processing')"
    )
    conn.commit()
    return conn


def _context():
    return ConnectionContext(
        client_id="client-a-install",
        surface="client-a",
        agent_id="client-a",
        session_id="session-1",
        repository="enfold",
        branch="retrieval",
        access_scopes=("private",),
    )


def test_attributed_enqueue_is_after_commit_model_free_and_idempotent(tmp_path):
    conn = _connection(tmp_path)
    queue = ExtractionEnqueuer(conn)

    first = queue.enqueue_after_commit(
        _context(), "Avery prefers local memory.", source="session_end"
    )
    second = queue.enqueue_after_commit(
        _context(), "Avery prefers local memory.", source="session_end"
    )

    assert first.replayed is False
    assert second.replayed is True
    assert second.queue_id == first.queue_id
    row = conn.execute("SELECT payload, payload_hash FROM extract_queue").fetchone()
    payload = json.loads(row[0])
    assert payload["scope"] == "private"
    assert payload["provenance"]["agent_id"] == "client-a"
    assert payload["provenance"]["repository"] == "enfold"
    assert row[1] == first.payload_sha256
    conn.close()


def test_enqueue_rejects_open_transaction_and_unprovisioned_queue(tmp_path):
    conn = _connection(tmp_path)
    queue = ExtractionEnqueuer(conn)
    conn.execute("BEGIN")
    with pytest.raises(RuntimeError, match="after commit"):
        queue.enqueue_after_commit(_context(), "transcript", source="session_end")
    conn.rollback()
    conn.close()

    empty = sqlite3.connect(tmp_path / "empty.db")
    with pytest.raises(ExtractionQueueUnavailable, match="not provisioned"):
        ExtractionEnqueuer(empty)
    empty.close()


def _v1_connection(tmp_path):
    conn = sqlite3.connect(tmp_path / "v1.db")
    conn.execute("PRAGMA foreign_keys=ON")
    migrate(conn)
    return conn


def test_session_capture_is_disabled_by_default_and_does_not_write_facts(tmp_path):
    conn = _v1_connection(tmp_path)
    queue = ExtractionEnqueuer(conn)

    status = capture_status(conn)
    assert status["enabled"] is False
    assert status["visible_to_default_recall"] is False

    with pytest.raises(CaptureDisabled, match="disabled"):
        queue.enqueue_session_capture(_context(), "Avery prefers local memory.")

    assert conn.execute("SELECT count(*) FROM extract_queue").fetchone()[0] == 0
    assert conn.execute("SELECT count(*) FROM facts").fetchone()[0] == 0
    conn.close()


def test_capture_enable_refuses_silent_invisible_rows(tmp_path):
    conn = _v1_connection(tmp_path)

    with pytest.raises(CaptureEnableError, match="allow-unreviewed"):
        enable_capture(conn)

    assert capture_status(conn)["enabled"] is False
    conn.close()


def test_session_capture_enqueues_without_memory_write_after_opt_in(tmp_path):
    conn = _v1_connection(tmp_path)
    queue = ExtractionEnqueuer(conn)

    enable_capture(conn, allow_unreviewed=True)
    status = capture_status(conn)
    assert status["enabled"] is True
    assert status["allow_unreviewed"] is True
    assert status["visible_to_default_recall"] is False

    result = queue.enqueue_session_capture(
        _context(), "Avery prefers local memory."
    )

    assert result.replayed is False
    assert conn.execute("SELECT count(*) FROM facts").fetchone()[0] == 0
    payload = json.loads(
        conn.execute("SELECT payload FROM extract_queue").fetchone()[0]
    )
    assert payload["source"] == "session_capture"
    assert payload["transcript"] == "Avery prefers local memory."
    assert payload["metadata"]["capture"] is True
    conn.close()
