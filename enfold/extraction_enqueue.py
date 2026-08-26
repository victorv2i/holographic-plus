"""Daemon-owned, model-free extraction queue boundary.

This module deliberately does not create or migrate the queue and never calls
an extraction model.  An explicit migration/adapter must provide the durable
``extract_queue`` table.  Enqueue happens only after the caller's fact write
transaction has committed, preserving the write-path latency and rollback
contract while retaining payload-hash idempotency.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import sqlite3
from typing import Any, Mapping

from .extraction_spans import TranscriptInput, normalize_transcript
from .policy import scope_authorized
from .provenance import ConnectionContext


MAX_EXTRACTION_PAYLOAD_BYTES = 12 * 1024
_REQUIRED_COLUMNS = frozenset({"id", "payload", "status", "payload_hash"})
CAPTURE_ENABLED_KEY = "capture.enabled"
CAPTURE_ALLOW_UNREVIEWED_KEY = "capture.allow_unreviewed"
CAPTURE_VERIFIER_READY_KEY = "capture.verifier_ready"


class ExtractionQueueUnavailable(RuntimeError):
    """The explicitly provisioned durable extraction queue is unavailable."""


class CaptureDisabled(RuntimeError):
    """Session capture is off until an operator opts in."""


class CaptureEnableError(ValueError):
    """Capture was refused because it would hide rows from default recall."""


def _meta_value(conn: sqlite3.Connection, key: str) -> str | None:
    try:
        row = conn.execute(
            "SELECT value FROM enfold_meta WHERE key = ?", (key,)
        ).fetchone()
    except sqlite3.OperationalError:
        return None
    return None if row is None else str(row[0])


def capture_status(conn: sqlite3.Connection) -> dict[str, bool]:
    """Report whether session capture is opted in, and whether it can be seen."""

    enabled = _meta_value(conn, CAPTURE_ENABLED_KEY) == "1"
    allow_unreviewed = _meta_value(conn, CAPTURE_ALLOW_UNREVIEWED_KEY) == "1"
    verifier_ready = _meta_value(conn, CAPTURE_VERIFIER_READY_KEY) == "1"
    return {
        "enabled": enabled,
        "allow_unreviewed": allow_unreviewed,
        "verifier_ready": verifier_ready,
        "visible_to_default_recall": enabled and verifier_ready,
    }


def enable_capture(
    conn: sqlite3.Connection,
    *,
    allow_unreviewed: bool = False,
    verifier_ready: bool = False,
) -> dict[str, bool]:
    """Opt in to session capture. Refuses a silent invisible-row default."""

    if not allow_unreviewed and not verifier_ready:
        raise CaptureEnableError(
            "capture enable requires a configured evidence verifier "
            "(local model). Pass --allow-unreviewed to enqueue anyway; "
            "those rows stay excluded from default recall until reviewed"
        )
    if conn.in_transaction:
        raise RuntimeError("capture enable must run after commit")
    try:
        conn.execute("BEGIN IMMEDIATE")
        for key, value in (
            (CAPTURE_ENABLED_KEY, "1"),
            (CAPTURE_ALLOW_UNREVIEWED_KEY, "1" if allow_unreviewed else "0"),
            (CAPTURE_VERIFIER_READY_KEY, "1" if verifier_ready else "0"),
        ):
            conn.execute(
                "INSERT INTO enfold_meta(key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value),
            )
        conn.commit()
    except sqlite3.OperationalError as exc:
        if conn.in_transaction:
            conn.rollback()
        raise CaptureEnableError(
            "capture enable requires a schema-v1 store with enfold_meta"
        ) from exc
    except BaseException:
        if conn.in_transaction:
            conn.rollback()
        raise
    return capture_status(conn)


@dataclass(frozen=True, slots=True)
class ExtractionEnqueueResult:
    queue_id: int
    payload_sha256: str
    replayed: bool


class ExtractionEnqueuer:
    """Append canonical attributed transcripts to an existing durable queue."""

    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn
        columns = {
            str(row[1]) for row in conn.execute("PRAGMA table_info(extract_queue)")
        }
        missing = sorted(_REQUIRED_COLUMNS - columns)
        if missing:
            raise ExtractionQueueUnavailable(
                "extract_queue is not provisioned with required columns: "
                + ", ".join(missing)
            )

    def enqueue_session_capture(
        self,
        context: ConnectionContext,
        transcript: TranscriptInput,
        *,
        source: str = "session_capture",
        scope: str = "private",
        metadata: Mapping[str, Any] | None = None,
    ) -> ExtractionEnqueueResult:
        """Enqueue a session transcript without writing a fact.

        This is the opt-in capture path. It is off until
        :func:`enable_capture` succeeds. It never inserts facts; the
        processor still fail-closes unverified proposals out of default
        recall.
        """

        status = capture_status(self._conn)
        if not status["enabled"]:
            raise CaptureDisabled(
                "session capture is disabled; run capture enable first"
            )
        extra = {
            "capture": True,
            "visible_to_default_recall": status["visible_to_default_recall"],
        }
        extra.update(dict(metadata or {}))
        return self.enqueue_after_commit(
            context,
            transcript,
            source=source,
            scope=scope,
            metadata=extra,
        )

    def enqueue_after_commit(
        self,
        context: ConnectionContext,
        transcript: TranscriptInput,
        *,
        source: str,
        scope: str = "private",
        metadata: Mapping[str, Any] | None = None,
    ) -> ExtractionEnqueueResult:
        """Enqueue one transcript without running a model.

        The connection must be idle.  This makes ordering explicit: the
        authoritative write commits first; queue insertion is a separate,
        idempotent transaction and can be retried safely after a crash.
        """

        if self._conn.in_transaction:
            raise RuntimeError("extraction enqueue must run after commit")
        transcript_text, turns = normalize_transcript(transcript)
        source = source.strip()
        scope = scope.strip()
        if not transcript_text or not source or not scope:
            raise ValueError("transcript, source, and scope must be non-empty")
        if not scope_authorized(scope, context.access_scopes):
            raise ValueError("extraction scope must be present in context access scopes")
        envelope = {
            "version": 1,
            "source": source,
            "scope": scope,
            "provenance": {
                "client_id": context.client_id,
                "surface": context.surface,
                "agent_id": context.agent_id,
                "session_id": context.session_id,
                "parent_agent_id": context.parent_agent_id,
                "project_root": context.project_root,
                "repository": context.repository,
                "branch": context.branch,
                "commit_sha": context.commit_sha,
                "access_scopes": list(context.access_scopes),
            },
            "metadata": dict(metadata or {}),
        }
        if turns is None:
            envelope["transcript"] = transcript_text
        else:
            envelope["turns"] = list(turns)
        try:
            payload = json.dumps(
                envelope,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
        except (TypeError, ValueError, RecursionError) as exc:
            raise ValueError("extraction metadata must contain JSON values") from exc
        if len(payload.encode("utf-8")) > MAX_EXTRACTION_PAYLOAD_BYTES:
            raise ValueError("canonical extraction payload exceeds size limit")
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        try:
            self._conn.execute("BEGIN IMMEDIATE")
            existing = self._conn.execute(
                """
                SELECT id FROM extract_queue
                WHERE payload_hash = ? AND status IN ('pending', 'processing')
                ORDER BY id LIMIT 1
                """,
                (digest,),
            ).fetchone()
            if existing is not None:
                self._conn.commit()
                return ExtractionEnqueueResult(int(existing[0]), digest, True)
            cursor = self._conn.execute(
                "INSERT INTO extract_queue(payload, payload_hash, status) "
                "VALUES (?, ?, 'pending')",
                (payload, digest),
            )
            queue_id = int(cursor.lastrowid)
            self._conn.commit()
            return ExtractionEnqueueResult(queue_id, digest, False)
        except BaseException:
            if self._conn.in_transaction:
                self._conn.rollback()
            raise
