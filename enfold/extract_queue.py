"""Persistent queue for LLM fact extraction.

Rows live in the same SQLite database as the fact store, so queued work
survives gateway restarts and crashes. The provider enqueues an already
formatted transcript at session end and before context compression, and a
single daemon worker drains the queue with retry and backoff. Rows that keep
failing are kept with status 'dead' for inspection instead of being silently
dropped.

Provider quota errors (plan-limit windows, e.g. a 429 with
``resets_in_seconds``) are special: they reset on the provider's schedule,
not ours, so they reschedule the row via ``not_before`` without consuming a
retry attempt. The 48h age cap (``created_at``) is the only thing that kills
a quota-limited row.

The table is created lazily on first use, mirroring EmbedStore's migration
style: the parent plugin stays unaware of it and no parent schema changes
are needed.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import re
import sqlite3
import threading
import time
import uuid
from typing import Any, Dict, Iterable, Optional

from .extraction_spans import TranscriptInput, normalize_transcript
from .policy import default_credential_screen
from .provenance import WriteRequest

logger = logging.getLogger(__name__)

# Transcripts are capped before storage; the formatter already truncates to
# roughly this size, the cap here is a hard safety bound on row size.
MAX_PAYLOAD_BYTES = 12 * 1024

STATUS_PENDING = "pending"
STATUS_PROCESSING = "processing"
STATUS_DEAD = "dead"

# Failure messages matching any of these substrings (case-insensitive) are
# classified as provider quota / rate-limit errors and rescheduled instead of
# consuming retry attempts.
QUOTA_ERROR_PATTERNS = (
    "429",
    "usage_limit_reached",
    "usage_limit",
    "rate limit",
    "quota",
)

# Jitter added on top of a parsed reset window so retries do not land exactly
# on the reset boundary, and the fallback delay when a quota error does not
# say when its window reopens.
QUOTA_JITTER_MIN = 60.0        # seconds
QUOTA_JITTER_MAX = 300.0       # seconds
QUOTA_FALLBACK_DELAY = 1800.0  # seconds

# Rows older than this (by created_at) are marked dead on their next failure
# regardless of error classification; quota-limited rows therefore retry
# patiently for up to 48h and no longer.
MAX_ROW_AGE_SECONDS = 48 * 3600

_AGE_CAP_NOTE = " (dead: 48h age cap)"

_RESETS_IN_RE = re.compile(r"resets_in_seconds\D{0,12}(\d+)", re.IGNORECASE)
_RESETS_AT_RE = re.compile(r"resets_at\D{0,12}(\d{9,12})", re.IGNORECASE)


class _ClaimedLeaseOwner(str):
    """String-compatible owner carrying the exact lease incarnation fence."""

    def __new__(cls, owner: str, lease_until: float):
        value = super().__new__(cls, owner)
        value.lease_until = lease_until
        return value


def is_quota_error(error: Optional[str]) -> bool:
    """True when a failure message looks like a provider quota / rate limit."""
    text = (error or "").lower()
    return any(pattern in text for pattern in QUOTA_ERROR_PATTERNS)


def quota_retry_delay(error: Optional[str], now: Optional[float] = None) -> float:
    """Seconds to wait before retrying a quota-limited row.

    Parses ``resets_in_seconds`` (or a ``resets_at`` epoch) from the error
    text and adds a small jitter. Falls back to QUOTA_FALLBACK_DELAY when the
    error does not say when the window reopens.
    """
    text = error or ""
    match = _RESETS_IN_RE.search(text)
    if match:
        return float(match.group(1)) + random.uniform(QUOTA_JITTER_MIN, QUOTA_JITTER_MAX)
    match = _RESETS_AT_RE.search(text)
    if match:
        current = time.time() if now is None else now
        remaining = max(0.0, float(match.group(1)) - current)
        return remaining + random.uniform(QUOTA_JITTER_MIN, QUOTA_JITTER_MAX)
    return QUOTA_FALLBACK_DELAY


_SCHEMA = """
CREATE TABLE IF NOT EXISTS extract_queue (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    payload    TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    attempts   INTEGER NOT NULL DEFAULT 0,
    last_error TEXT,
    status     TEXT NOT NULL DEFAULT 'pending',
    not_before REAL,
    lease_owner TEXT,
    lease_until REAL,
    payload_hash TEXT,
    proposal_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_extract_queue_status
    ON extract_queue(status, id);
"""


class ExtractQueue:
    """CRUD for the extract_queue table on a shared SQLite connection."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        lock: Optional["threading.RLock"] = None,
    ) -> None:
        self._conn = conn
        # Share the parent store's lock when provided so queue writes and the
        # parent's fact writes serialize on the same connection.
        self._lock = lock if lock is not None else threading.RLock()
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._ensure_columns()
            self._conn.commit()

    def _ensure_columns(self) -> None:
        """Lazy migration for additive queue columns.

        Mirrors EmbedStore's migration style: a PRAGMA table_info guard, then
        a single ALTER TABLE ... ADD COLUMN. Idempotent and safe on every
        startup; new tables already have the column from _SCHEMA.
        """
        info = self._conn.execute("PRAGMA table_info(extract_queue)").fetchall()
        cols = {row[1] for row in info}
        for name, decl in (
            ("not_before", "REAL"),
            ("lease_owner", "TEXT"),
            ("lease_until", "REAL"),
            ("payload_hash", "TEXT"),
            ("proposal_json", "TEXT"),
        ):
            if name in cols:
                continue
            try:
                self._conn.execute(f"ALTER TABLE extract_queue ADD COLUMN {name} {decl}")
            except sqlite3.OperationalError as exc:
                # Two processes racing this same check-then-add on a fresh db
                # (e.g. two MCP server instances starting at once): the
                # loser's ALTER TABLE is a no-op, not a real failure.
                if "duplicate column name" not in str(exc).lower():
                    raise

        # The hash is queue idempotency metadata, not a change to the parent
        # fact schema.  A partial unique index prevents overlapping pending or
        # processing copies while allowing a completed/dead payload to be
        # deliberately queued again later.  Legacy active duplicates are
        # preserved: the oldest gets the hash and later copies remain NULL.
        has_index = self._conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='index' "
            "AND name='uq_extract_queue_active_payload_hash'"
        ).fetchone()
        if has_index is None:
            rows = self._conn.execute(
                "SELECT id, payload, status FROM extract_queue ORDER BY id"
            ).fetchall()
            active_hashes: set[str] = set()
            for row_id, payload, status in rows:
                digest = self._payload_hash(str(payload))
                if status in (STATUS_PENDING, STATUS_PROCESSING):
                    if digest in active_hashes:
                        continue
                    active_hashes.add(digest)
                self._conn.execute(
                    "UPDATE extract_queue SET payload_hash = ? WHERE id = ?",
                    (digest, int(row_id)),
                )
            self._conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS "
                "uq_extract_queue_active_payload_hash "
                "ON extract_queue(payload_hash) "
                "WHERE payload_hash IS NOT NULL "
                "AND status IN ('pending', 'processing')"
            )

    @staticmethod
    def _payload_hash(payload: str, *, role_structured: bool = False) -> str:
        prefix = b"role-structured-v1\0" if role_structured else b""
        return hashlib.sha256(prefix + payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _age_exceeded(created_epoch) -> bool:
        """True when a row's created_at (epoch string/float) is past the age cap."""
        if created_epoch is None:
            return False
        return time.time() - float(created_epoch) >= MAX_ROW_AGE_SECONDS

    @staticmethod
    def _lease_fence(
        lease_owner: Optional[str],
    ) -> tuple[Optional[str], float | None]:
        """Return owner and incarnation fence carried by a claimed owner.

        A plain owner string remains supported as unfenced legacy
        compatibility; callers should pass back the value returned by
        ``next_pending()`` to fence same-owner lease reclamation.
        """
        if lease_owner is None:
            return None, None
        lease_until = getattr(lease_owner, "lease_until", None)
        return str(lease_owner), float(lease_until) if lease_until is not None else None

    @staticmethod
    def _credential_shaped_proposal(proposal: dict[str, str]) -> bool:
        encoded = json.dumps(
            proposal,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        request = WriteRequest(
            idempotency_key="legacy-extraction-snapshot-screen",
            content=str(proposal.get("content", "")).strip() or "invalid proposal",
            source_type="automatic_extraction",
            category=str(proposal.get("category", "general")),
            tags=str(proposal.get("tags", "")),
        )
        return default_credential_screen(request, (encoded,)) is not None

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def enqueue(self, payload: TranscriptInput) -> int:
        """Insert a pending row and return its id.

        Opaque payloads above MAX_PAYLOAD_BYTES keep their tail. Structured
        turns fail closed because truncation would destroy role attribution.
        """
        role_structured = not isinstance(payload, str)
        if role_structured:
            _transcript_text, turns = normalize_transcript(payload)
            payload = json.dumps(
                list(turns or ()),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        encoded = payload.encode("utf-8")
        if len(encoded) > MAX_PAYLOAD_BYTES:
            if role_structured:
                raise ValueError("role-structured extraction payload exceeds size limit")
            payload = encoded[-MAX_PAYLOAD_BYTES:].decode("utf-8", errors="ignore")
        payload_hash = self._payload_hash(payload, role_structured=role_structured)
        with self._lock:
            existing = self._conn.execute(
                "SELECT id FROM extract_queue "
                "WHERE status IN (?, ?) "
                "AND (payload_hash = ? OR (payload_hash IS NULL AND payload = ?)) "
                "ORDER BY id LIMIT 1",
                (STATUS_PENDING, STATUS_PROCESSING, payload_hash, payload),
            ).fetchone()
            if existing is not None:
                return int(existing[0])
            cur = self._conn.execute(
                "INSERT OR IGNORE INTO extract_queue (payload, payload_hash) "
                "VALUES (?, ?)",
                (payload, payload_hash),
            )
            if int(cur.rowcount) == 0:
                row = self._conn.execute(
                    "SELECT id FROM extract_queue WHERE payload_hash = ? "
                    "AND status IN (?, ?) ORDER BY id LIMIT 1",
                    (payload_hash, STATUS_PENDING, STATUS_PROCESSING),
                ).fetchone()
                if row is None:  # pragma: no cover - defensive constraint guard
                    self._conn.rollback()
                    raise sqlite3.IntegrityError(
                        "extract queue idempotency conflict had no active row"
                    )
                self._conn.commit()
                return int(row[0])
            self._conn.commit()
            return int(cur.lastrowid)

    def mark_done(self, row_id: int, lease_owner: Optional[str] = None) -> bool:
        """Delete a successfully processed row.

        Claimed rows may only be completed by their current lease owner.
        A plain owner string is accepted as unfenced legacy compatibility.
        """
        owner, lease_until = self._lease_fence(lease_owner)
        with self._lock:
            cur = self._conn.execute(
                """
                DELETE FROM extract_queue
                WHERE id = ? AND status = ? AND lease_owner = ?
                  AND (? IS NULL OR lease_until = ?)
                """,
                (row_id, STATUS_PROCESSING, owner, lease_until, lease_until),
            )
            self._conn.commit()
            return int(cur.rowcount) == 1

    def save_proposals(
        self,
        row_id: int,
        proposals: list[dict[str, str]],
        lease_owner: Optional[str] = None,
    ) -> bool:
        """Persist the parsed model output before any proposal is inserted.

        A plain owner string is accepted as unfenced legacy compatibility.
        """
        safe_proposals = [
            proposal
            for proposal in proposals
            if not self._credential_shaped_proposal(proposal)
        ]
        encoded = json.dumps(safe_proposals, sort_keys=True, separators=(",", ":"))
        owner, lease_until = self._lease_fence(lease_owner)
        with self._lock:
            cur = self._conn.execute(
                """
                UPDATE extract_queue
                SET proposal_json = COALESCE(proposal_json, ?)
                WHERE id = ? AND status = ? AND lease_owner = ?
                  AND (? IS NULL OR lease_until = ?)
                """,
                (
                    encoded,
                    row_id,
                    STATUS_PROCESSING,
                    owner,
                    lease_until,
                    lease_until,
                ),
            )
            self._conn.commit()
            return int(cur.rowcount) == 1

    def mark_failed(
        self,
        row_id: int,
        error: str,
        max_attempts: int,
        lease_owner: Optional[str] = None,
    ) -> int:
        """Record a failed attempt; mark the row dead once attempts reach the cap.

        Rows past MAX_ROW_AGE_SECONDS are marked dead regardless of the
        attempt count, with last_error noting the age cap.

        A plain owner string is accepted as unfenced legacy compatibility.
        Returns the new attempt count (0 if the row no longer exists).
        """
        owner, lease_until = self._lease_fence(lease_owner)
        with self._lock:
            row = self._conn.execute(
                """
                SELECT attempts, strftime('%s', created_at), status, lease_owner,
                       lease_until
                FROM extract_queue WHERE id = ?
                """,
                (row_id,),
            ).fetchone()
            if row is None:
                return 0
            if (
                row[2] != STATUS_PROCESSING
                or row[3] != owner
                or (lease_until is not None and row[4] != lease_until)
            ):
                return 0
            attempts = int(row[0]) + 1
            message = (error or "")[:500]
            if self._age_exceeded(row[1]):
                status = STATUS_DEAD
                message += _AGE_CAP_NOTE
            else:
                status = STATUS_DEAD if attempts >= max_attempts else STATUS_PENDING
            cur = self._conn.execute(
                """
                UPDATE extract_queue
                SET attempts = ?, last_error = ?, status = ?,
                    lease_owner = NULL, lease_until = NULL
                WHERE id = ? AND status = ? AND lease_owner = ? AND attempts = ?
                  AND (? IS NULL OR lease_until = ?)
                """,
                (
                    attempts,
                    message,
                    status,
                    row_id,
                    STATUS_PROCESSING,
                    owner,
                    int(row[0]),
                    lease_until,
                    lease_until,
                ),
            )
            self._conn.commit()
            return attempts if int(cur.rowcount) == 1 else 0

    def mark_quota_failed(
        self,
        row_id: int,
        error: str,
        not_before: float,
        lease_owner: Optional[str] = None,
    ) -> bool:
        """Reschedule a quota-limited row without consuming a retry attempt.

        The row stays pending and next_pending() skips it until *not_before*
        (epoch seconds). Rows past MAX_ROW_AGE_SECONDS are marked dead with
        last_error noting the age cap.

        A plain owner string is accepted as unfenced legacy compatibility.
        Returns True when the row was rescheduled, False when it went dead by
        the age cap or no longer exists.
        """
        owner, lease_until = self._lease_fence(lease_owner)
        with self._lock:
            row = self._conn.execute(
                """
                SELECT strftime('%s', created_at), status, lease_owner, attempts,
                       lease_until
                FROM extract_queue WHERE id = ?
                """,
                (row_id,),
            ).fetchone()
            if row is None:
                return False
            if (
                row[1] != STATUS_PROCESSING
                or row[2] != owner
                or (lease_until is not None and row[4] != lease_until)
            ):
                return False
            message = (error or "")[:500]
            if self._age_exceeded(row[0]):
                cur = self._conn.execute(
                    """
                    UPDATE extract_queue
                    SET last_error = ?, status = ?, not_before = NULL,
                        lease_owner = NULL, lease_until = NULL
                    WHERE id = ? AND status = ? AND lease_owner = ? AND attempts = ?
                      AND (? IS NULL OR lease_until = ?)
                    """,
                    (
                        message + _AGE_CAP_NOTE,
                        STATUS_DEAD,
                        row_id,
                        STATUS_PROCESSING,
                        owner,
                        int(row[3]),
                        lease_until,
                        lease_until,
                    ),
                )
                self._conn.commit()
                return False
            cur = self._conn.execute(
                """
                UPDATE extract_queue
                SET last_error = ?, not_before = ?, status = ?,
                    lease_owner = NULL, lease_until = NULL
                WHERE id = ? AND status = ? AND lease_owner = ? AND attempts = ?
                  AND (? IS NULL OR lease_until = ?)
                """,
                (
                    message,
                    float(not_before),
                    STATUS_PENDING,
                    row_id,
                    STATUS_PROCESSING,
                    owner,
                    int(row[3]),
                    lease_until,
                    lease_until,
                ),
            )
            self._conn.commit()
            return int(cur.rowcount) == 1

    def release_claim(self, row_id: int, lease_owner: str) -> bool:
        """Return a claimed row to pending without consuming an attempt.

        A plain owner string is accepted as unfenced legacy compatibility.
        """
        owner, lease_until = self._lease_fence(lease_owner)
        with self._lock:
            cur = self._conn.execute(
                """
                UPDATE extract_queue
                SET status = ?, lease_owner = NULL, lease_until = NULL
                WHERE id = ? AND status = ? AND lease_owner = ?
                  AND (? IS NULL OR lease_until = ?)
                """,
                (
                    STATUS_PENDING,
                    row_id,
                    STATUS_PROCESSING,
                    owner,
                    lease_until,
                    lease_until,
                ),
            )
            self._conn.commit()
            return int(cur.rowcount) == 1

    def revive_dead(self, ids: Optional[Iterable[int]] = None) -> int:
        """Reset dead rows to pending so the worker retries them.

        Attempts go back to 0 and not_before is cleared; the last error is
        kept with ' (revived)' appended for inspection. With *ids* of None
        every dead row is revived, otherwise only the listed ids.

        Returns the number of rows revived.
        """
        sql = """
            UPDATE extract_queue AS candidate
            SET status = ?, attempts = 0, not_before = NULL,
                lease_owner = NULL, lease_until = NULL,
                last_error = COALESCE(last_error, '') || ' (revived)'
            WHERE status = ?
              AND (
                    payload_hash IS NULL
                    OR (
                        candidate.id = (
                            SELECT MIN(peer.id) FROM extract_queue AS peer
                            WHERE peer.status = ?
                              AND peer.payload_hash = candidate.payload_hash
                        )
                        AND NOT EXISTS (
                            SELECT 1 FROM extract_queue AS active
                            WHERE active.payload_hash = candidate.payload_hash
                              AND active.status IN (?, ?)
                        )
                    )
                  )
        """
        params: list = [
            STATUS_PENDING,
            STATUS_DEAD,
            STATUS_DEAD,
            STATUS_PENDING,
            STATUS_PROCESSING,
        ]
        if ids is not None:
            id_list = [int(i) for i in ids]
            if not id_list:
                return 0
            placeholders = ",".join("?" * len(id_list))
            sql += f" AND id IN ({placeholders})"
            params.extend(id_list)
        with self._lock:
            cur = self._conn.execute(sql, params)
            self._conn.commit()
            return int(cur.rowcount)

    def revive_recent_quota_dead(self) -> int:
        """Revive dead rows that were killed by quota errors and are still young.

        One-shot recovery for rows dead-lettered before quota errors stopped
        consuming attempts: any dead row younger than MAX_ROW_AGE_SECONDS
        whose last_error matches QUOTA_ERROR_PATTERNS goes back to pending.

        Returns the number of rows revived.
        """
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT id, last_error FROM extract_queue
                WHERE status = ?
                  AND (strftime('%s', 'now') - strftime('%s', created_at)) < ?
                """,
                (STATUS_DEAD, int(MAX_ROW_AGE_SECONDS)),
            ).fetchall()
        quota_ids = [int(row[0]) for row in rows if is_quota_error(row[1])]
        if not quota_ids:
            return 0
        return self.revive_dead(quota_ids)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def next_pending(
        self,
        max_attempts: int,
        exclude_ids=(),
        lease_owner: Optional[str] = None,
        lease_seconds: float = 600.0,
    ) -> Optional[Dict[str, Any]]:
        """Claim and return the oldest due row, or None when drained.

        Rows whose not_before is in the future (quota reschedules) are
        skipped until due. Rows whose id is in *exclude_ids* are skipped; the
        worker uses this as an in-memory bound when recording failures in the
        DB is broken.

        Expired processing leases are eligible to be claimed by a new owner.
        The returned ``lease_owner`` is a string-compatible claim handle that
        carries the lease incarnation; pass it back to mutation methods for
        same-owner ABA fencing.

        Index-based row access so this works with any row_factory.
        """
        exclude = [int(i) for i in exclude_ids]
        owner = lease_owner or f"{threading.get_ident()}-{uuid.uuid4().hex}"
        now = time.time()
        lease_until = now + float(lease_seconds)
        sql = """
            SELECT id FROM extract_queue
            WHERE attempts < ?
              AND (not_before IS NULL OR not_before <= ?)
              AND (
                    status = ?
                    OR (status = ? AND lease_until IS NOT NULL AND lease_until <= ?)
                  )
        """
        params: list = [max_attempts, now, STATUS_PENDING, STATUS_PROCESSING, now]
        if exclude:
            placeholders = ",".join("?" * len(exclude))
            sql += f" AND id NOT IN ({placeholders})"
            params.extend(exclude)
        sql += " ORDER BY id LIMIT 1"
        with self._lock:
            row = self._conn.execute(sql, params).fetchone()
            if row is None:
                return None
            row_id = int(row[0])
            cur = self._conn.execute(
                """
                UPDATE extract_queue
                SET status = ?, lease_owner = ?, lease_until = ?
                WHERE id = ?
                  AND attempts < ?
                  AND (not_before IS NULL OR not_before <= ?)
                  AND (
                        status = ?
                        OR (status = ? AND lease_until IS NOT NULL AND lease_until <= ?)
                      )
                """,
                (
                    STATUS_PROCESSING,
                    owner,
                    lease_until,
                    row_id,
                    max_attempts,
                    now,
                    STATUS_PENDING,
                    STATUS_PROCESSING,
                    now,
                ),
            )
            if int(cur.rowcount) != 1:
                self._conn.rollback()
                return None
            claimed = self._conn.execute(
                """
                SELECT id, payload, attempts, lease_owner, lease_until,
                       proposal_json, payload_hash
                FROM extract_queue WHERE id = ?
                """,
                (row_id,),
            ).fetchone()
            self._conn.commit()
        if claimed is None:
            return None
        return {
            "id": int(claimed[0]),
            "payload": claimed[1],
            "attempts": int(claimed[2]),
            "lease_owner": _ClaimedLeaseOwner(str(claimed[3]), float(claimed[4])),
            "lease_until": float(claimed[4]),
            "proposal_json": claimed[5],
            "role_structured": claimed[6]
            == self._payload_hash(claimed[1], role_structured=True),
        }

    def pending_count(self) -> int:
        """Number of rows still not terminal."""
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM extract_queue WHERE status IN (?, ?)",
                (STATUS_PENDING, STATUS_PROCESSING),
            ).fetchone()
        return int(row[0])

    def dead_count(self) -> int:
        """Number of rows that exhausted their retries (kept for inspection)."""
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM extract_queue WHERE status = ?",
                (STATUS_DEAD,),
            ).fetchone()
        return int(row[0])
