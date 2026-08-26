"""Fail-closed, model-agnostic processing for attributed extraction jobs.

The processor owns no model or persistent worker.  A host supplies an
``Extractor`` and explicitly calls :meth:`ExtractionProcessor.process_one` or
:meth:`~ExtractionProcessor.drain`.  Queue leases make claims crash-safe; a
validated proposal snapshot is persisted before any fact write, so replay
never asks a nondeterministic model to regenerate an already-applied batch.
Every non-empty proposal batch and deletion of its leased queue row commit in
one SQLite transaction; policy rejection or any late side-effect failure rolls
back facts, state transitions, provenance, write logs, and embedding jobs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import logging
import math
import sqlite3
import threading
import time
from typing import Any, Callable, Literal, Mapping, Protocol, Sequence
import uuid

from .extraction_spans import (
    MAX_EVIDENCE_CHARS,
    TranscriptSpan,
    eligible_transcript_spans,
    normalize_transcript,
    transcript_spans,
)
from .prompt_safety import ephemeral_extraction_reason
from .policy import (
    LEGACY_EXTRACTION_CLIENT_ID,
    default_credential_screen,
    extracted_proposal_credential_decision,
    scope_authorized,
    validate_scope,
)
from .protocol import ClientContext, Request
from .provenance import WriteRequest
from .service import EnfoldService
from .state_slots import (
    StateCandidate,
    canonical_slot_registry,
    resolve_extracted_predicate_key,
    resolve_stored_subject_key,
)

logger = logging.getLogger(__name__)


MAX_EXTRACTED_MEMORIES = 32
AUTOMATIC_TRUST_SCORE = 0.5
AUTOMATIC_SOURCE_AUTHORITY = {
    "user": 0.5,
    "assistant": 0.2,
    "tool": 0.4,
}
MIN_TYPED_CONFIDENCE = 0.8
MAX_RETRY_DELAY_SECONDS = 300.0
MAX_RETRY_AFTER_SECONDS = 3600.0
MAX_RATE_LIMIT_AGE_SECONDS = 48 * 3600
DEFAULT_RATE_LIMIT_RETRY_SECONDS = 60.0
EXTRACTION_CANCEL_JOIN_SECONDS = 5.0
_TYPED_KINDS = frozenset({"state", "preference", "commitment", "event"})
_TYPED_FIELDS = frozenset(
    {
        "kind",
        "subject",
        "predicate",
        "object",
        "value",
        "occurred_at",
        "valid_from",
        "negation",
        "confidence",
    }
)
_REQUIRED_QUEUE_COLUMNS = frozenset(
    {
        "id",
        "created_at",
        "payload",
        "status",
        "payload_hash",
        "attempts",
        "last_error",
        "not_before",
        "lease_owner",
        "lease_until",
        "lease_token",
        "proposal_json",
        "proposal_hash",
    }
)
_SAFE_ERROR_CODES = frozenset(
    {
        "adapter_exit",
        "adapter_cleanup_failed",
        "adapter_input_too_large",
        "adapter_invalid_config",
        "adapter_invalid_input",
        "adapter_invalid_output",
        "adapter_output_too_large",
        "adapter_rate_limited",
        "adapter_timeout",
        "adapter_unavailable",
        "extractor_failed",
        "invalid_envelope",
        "invalid_proposal",
        "invalid_snapshot",
        "legacy_extraction_quarantined",
        "proposal_credential_rejected",
        "proposal_grounding_rejected",
        "proposal_support_unverified",
        "proposal_limit",
        "proposal_scope_rejected",
        "proposal_sensitivity_rejected",
        "snapshot_hash_mismatch",
        "write_policy_rejected",
        "invalid_params",
        "access_denied",
        "idempotency_conflict",
    }
)


class ExtractionProcessorUnavailable(RuntimeError):
    """The durable queue does not support safe claimed processing."""


class PermanentExtractionError(ValueError):
    """A job is unsafe or malformed and must go directly to dead letter."""

    def __init__(self, message: str, *, error_code: str = "invalid_proposal") -> None:
        super().__init__(message)
        self.error_code = error_code


@dataclass(frozen=True, slots=True)
class EvidenceVerification:
    """A verifier's decision about one claim/evidence pair.

    Exact transcript-span identity establishes provenance only.  It does not
    establish that a span supports the proposed claim, so automatic writes
    require an explicit verification decision from a boundary outside the
    extractor itself.
    """

    status: Literal["verified", "needs_review"]
    verifier_id: str

    def __post_init__(self) -> None:
        if self.status not in {"verified", "needs_review"}:
            raise ValueError("evidence verification status is invalid")
        if not isinstance(self.verifier_id, str) or not self.verifier_id.strip():
            raise ValueError("evidence verifier id must be non-empty")


class EvidenceVerifier(Protocol):
    """Independent claim-to-evidence verifier supplied by a trusted host."""

    def verify(
        self,
        proposal: "ExtractedMemory",
        *,
        evidence_excerpt: str,
        envelope: "ExtractionEnvelope",
    ) -> EvidenceVerification:
        """Return ``verified`` only when the excerpt supports the whole claim."""


class ReviewRequiredEvidenceVerifier:
    """Safe default: do not turn an extractor assertion into canonical memory."""

    identity = "unconfigured"

    def verify(
        self,
        proposal: "ExtractedMemory",
        *,
        evidence_excerpt: str,
        envelope: "ExtractionEnvelope",
    ) -> EvidenceVerification:
        del proposal, evidence_excerpt, envelope
        return EvidenceVerification("needs_review", "unconfigured")


@dataclass(frozen=True, slots=True)
class ExtractionEnvelope:
    transcript: str
    source: str
    scope: str
    context: ClientContext
    metadata: Mapping[str, Any] = field(default_factory=dict)
    turns: tuple[dict[str, str], ...] | None = None
    legacy_payload: bool = False


def _envelope_spans(envelope: ExtractionEnvelope) -> tuple[TranscriptSpan, ...]:
    source = envelope.turns if envelope.turns is not None else envelope.transcript
    return transcript_spans(source)


def _eligible_envelope_spans(
    envelope: ExtractionEnvelope,
) -> tuple[TranscriptSpan, ...]:
    return eligible_transcript_spans(_envelope_spans(envelope))


@dataclass(frozen=True, slots=True)
class ExtractedMemory:
    """One model-produced proposal; the authoritative service still decides."""

    content: str
    category: str = "general"
    tags: str = ""
    trust_score: float = 0.5
    source_authority: float = 0.5
    evidence_excerpt: str | None = None
    scope: str | None = None
    sensitivity: str = "normal"
    state: Mapping[str, Any] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


class Extractor(Protocol):
    """Host-provided model adapter. Implementations may call a model."""

    @property
    def identity(self) -> str:
        """Stable, non-secret extractor/model identity for provenance."""

    def extract(
        self,
        envelope: ExtractionEnvelope,
        *,
        register_invocation: Callable[[object], None] | None = None,
    ) -> Sequence[ExtractedMemory]:
        """Return proposals and optionally register this call's opaque handle."""

    def cancel(self, invocation: object) -> None:
        """Cancel only the extraction identified by ``invocation``."""


@dataclass(frozen=True, slots=True)
class ExtractionProcessResult:
    outcome: str
    queue_id: int | None
    writes: int = 0
    attempts: int = 0
    error: str | None = None


class ExtractionProcessor:
    """Claim durable jobs and apply proposals through ``EnfoldService``."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        service: EnfoldService,
        extractor: Extractor,
        *,
        worker_id: str | None = None,
        max_attempts: int = 3,
        lease_seconds: float = 300.0,
        heartbeat_seconds: float | None = None,
        retry_delay_seconds: float = 1.0,
        clock: Callable[[], float] = time.time,
        evidence_verifier: EvidenceVerifier | None = None,
    ):
        columns = {
            str(row[1]) for row in conn.execute("PRAGMA table_info(extract_queue)")
        }
        missing = sorted(_REQUIRED_QUEUE_COLUMNS - columns)
        if missing:
            raise ExtractionProcessorUnavailable(
                "extract_queue lacks claimed-processing columns: " + ", ".join(missing)
            )
        identity = getattr(extractor, "identity", None)
        if not isinstance(identity, str) or not identity.strip():
            raise ValueError("extractor identity must be a non-empty string")
        if max_attempts <= 0 or lease_seconds <= 0 or retry_delay_seconds < 0:
            raise ValueError("invalid extraction retry/lease configuration")
        effective_heartbeat = (
            min(30.0, lease_seconds / 3.0)
            if heartbeat_seconds is None
            else heartbeat_seconds
        )
        if effective_heartbeat <= 0 or effective_heartbeat >= lease_seconds:
            raise ValueError(
                "heartbeat_seconds must be positive and shorter than lease_seconds"
            )
        if conn.in_transaction:
            raise RuntimeError("ExtractionProcessor requires an idle connection")
        self._conn = conn
        self._service = service
        self._extractor = extractor
        self._worker_id = worker_id or f"extractor-{uuid.uuid4().hex}"
        self._max_attempts = max_attempts
        self._lease_seconds = lease_seconds
        self._heartbeat_seconds = float(effective_heartbeat)
        self._retry_delay = retry_delay_seconds
        self._clock = clock
        self._evidence_verifier = evidence_verifier or ReviewRequiredEvidenceVerifier()
        self._evidence_verifier_configured = evidence_verifier is not None

    def process_one(self) -> ExtractionProcessResult:
        """Process one due job, or return ``idle`` without model activity."""

        row = self._claim()
        if row is None:
            return ExtractionProcessResult("idle", None)
        row_id, payload, digest, attempts, lease_token = row
        writes = 0
        try:
            observed_at = self._queue_observed_at(row_id, lease_token)
            envelope = self._decode_envelope(payload)
            snapshot_json, snapshot_hash = self._load_snapshot(
                row_id, lease_token, envelope
            )
            if snapshot_json is None:
                if _eligible_envelope_spans(envelope):
                    proposals = self._extract_with_heartbeat(
                        envelope, row_id, lease_token
                    )
                else:
                    proposals = ()
                snapshot_json, snapshot_hash = self._make_snapshot(proposals, envelope)
                self._persist_snapshot(
                    row_id, lease_token, snapshot_json, snapshot_hash
                )
            prepared = self._prepare_snapshot(
                snapshot_json,
                snapshot_hash,
                envelope,
                row_id,
                digest,
                observed_at,
            )
            requests = tuple(
                Request(
                    f"extract-{row_id}-{index}",
                    "memory.write",
                    params,
                )
                for index, params in enumerate(prepared)
            )
            if requests:
                pending = list(requests)
                while pending:
                    batch = self._service.handle_write_batch(
                        envelope.context,
                        tuple(pending),
                        before_commit=lambda: self._complete_in_transaction(
                            row_id, lease_token
                        ),
                    )
                    if batch.committed:
                        writes = len(batch.responses)
                        break
                    survivors = [
                        request
                        for request, response in zip(pending, batch.responses)
                        if response["outcome"] not in {"rejected", "needs_review"}
                    ]
                    if len(survivors) == len(pending):
                        raise RuntimeError(
                            "write batch rolled back without policy rejection"
                        )
                    pending = survivors
                else:
                    self._complete(row_id, lease_token)
            else:
                self._complete(row_id, lease_token)
            return ExtractionProcessResult("completed", row_id, writes, attempts)
        except PermanentExtractionError as exc:
            error_code = self._safe_error_code(exc)
            attempts = self._fail(row_id, lease_token, error_code, permanent=True)
            return ExtractionProcessResult("dead", row_id, writes, attempts, error_code)
        except Exception as exc:
            error_code = self._safe_error_code(exc)
            permanent = getattr(exc, "retryable", None) is False
            attempts = self._fail(
                row_id,
                lease_token,
                error_code,
                permanent=permanent,
                retry_after_seconds=getattr(exc, "retry_after_seconds", None),
                consumes_attempt=getattr(exc, "consumes_attempt", True),
            )
            outcome = (
                "dead"
                if permanent
                or attempts >= self._max_attempts
                or self._job_is_dead(row_id)
                else "retry"
            )
            return ExtractionProcessResult(
                outcome, row_id, writes, attempts, error_code
            )

    def drain(self, *, limit: int = 10) -> tuple[ExtractionProcessResult, ...]:
        """Process at most ``limit`` jobs; never loops indefinitely."""

        if isinstance(limit, bool) or limit <= 0:
            raise ValueError("limit must be positive")
        results: list[ExtractionProcessResult] = []
        for _ in range(limit):
            result = self.process_one()
            if result.outcome == "idle":
                break
            results.append(result)
        return tuple(results)

    @property
    def health(self) -> dict[str, Any]:
        rows = self._conn.execute(
            "SELECT status, count(*) FROM extract_queue GROUP BY status"
        ).fetchall()
        counts = {str(row[0]): int(row[1]) for row in rows}
        return {
            "configured": True,
            "mode": "explicit_host_driven",
            "extractor": self._extractor.identity,
            "evidence_verifier": {
                "configured": self._evidence_verifier_configured,
                "verifier_id": getattr(
                    self._evidence_verifier,
                    "identity",
                    None,
                ),
            },
            "pending": counts.get("pending", 0) + counts.get("processing", 0),
            "dead": counts.get("dead", 0),
        }

    def _claim(self) -> tuple[int, str, str, int, str] | None:
        now = self._clock()
        try:
            self._conn.execute("BEGIN IMMEDIATE")
            self._conn.execute(
                """
                UPDATE extract_queue
                SET status = 'dead', last_error = 'attempts_exhausted',
                    lease_owner = NULL, lease_until = NULL, lease_token = NULL
                WHERE status = 'processing' AND lease_until IS NOT NULL
                  AND lease_until <= ? AND attempts >= ?
                """,
                (now, self._max_attempts),
            )
            row = self._conn.execute(
                """
                SELECT id, payload, payload_hash, attempts
                FROM extract_queue
                WHERE attempts < ?
                  AND (not_before IS NULL OR not_before <= ?)
                  AND (status = 'pending' OR
                       (status = 'processing' AND lease_until IS NOT NULL
                        AND lease_until <= ?))
                ORDER BY id LIMIT 1
                """,
                (self._max_attempts, now, now),
            ).fetchone()
            if row is None:
                self._conn.commit()
                return None
            row_id = int(row[0])
            lease_token = uuid.uuid4().hex
            attempts = int(row[3]) + 1
            cursor = self._conn.execute(
                """
                UPDATE extract_queue
                SET status = 'processing', attempts = ?, lease_owner = ?,
                    lease_until = ?, lease_token = ?
                WHERE id = ? AND attempts < ?
                """,
                (
                    attempts,
                    self._worker_id,
                    now + self._lease_seconds,
                    lease_token,
                    row_id,
                    self._max_attempts,
                ),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("extraction claim was lost")
            self._conn.commit()
            payload = str(row[1])
            digest = str(row[2] or hashlib.sha256(payload.encode()).hexdigest())
            return row_id, payload, digest, attempts, lease_token
        except BaseException:
            if self._conn.in_transaction:
                self._conn.rollback()
            raise

    def _complete(self, row_id: int, lease_token: str) -> None:
        self._delete_claimed_row(row_id, lease_token)
        self._conn.commit()

    def _complete_in_transaction(self, row_id: int, lease_token: str) -> None:
        """Delete the leased job inside the authoritative write transaction."""

        if not self._conn.in_transaction:
            raise RuntimeError("atomic extraction completion requires a transaction")
        self._delete_claimed_row(row_id, lease_token)

    def _delete_claimed_row(self, row_id: int, lease_token: str) -> None:
        cursor = self._conn.execute(
            "DELETE FROM extract_queue WHERE id = ? AND status = 'processing' "
            "AND lease_owner = ? AND lease_token = ?",
            (row_id, self._worker_id, lease_token),
        )
        if cursor.rowcount != 1:
            raise RuntimeError("extraction lease was lost before completion")

    def _renew(self, row_id: int, lease_token: str) -> None:
        """Extend one live lease without permitting a stale worker to revive it."""

        now = self._clock()
        cursor = self._conn.execute(
            """
            UPDATE extract_queue
            SET lease_until = ?
            WHERE id = ? AND status = 'processing' AND lease_owner = ?
              AND lease_token = ? AND lease_until IS NOT NULL AND lease_until > ?
            """,
            (
                now + self._lease_seconds,
                row_id,
                self._worker_id,
                lease_token,
                now,
            ),
        )
        self._conn.commit()
        if cursor.rowcount != 1:
            raise RuntimeError("extraction lease was lost before renewal")

    def _fail(
        self,
        row_id: int,
        lease_token: str,
        error_code: str,
        *,
        permanent: bool,
        retry_after_seconds: object = None,
        consumes_attempt: bool = True,
    ) -> int:
        row = self._conn.execute(
            "SELECT attempts, strftime('%s', created_at) FROM extract_queue "
            "WHERE id = ? AND status = 'processing' "
            "AND lease_owner = ? AND lease_token = ?",
            (row_id, self._worker_id, lease_token),
        ).fetchone()
        if row is None:
            raise RuntimeError("extraction lease was lost while recording failure")
        attempts = int(row[0])
        if not consumes_attempt and not permanent:
            attempts = max(0, attempts - 1)
        rate_limit_age_exceeded = (
            not consumes_attempt
            and not permanent
            and row[1] is not None
            and self._clock() - float(row[1]) >= MAX_RATE_LIMIT_AGE_SECONDS
        )
        dead = permanent or attempts >= self._max_attempts or rate_limit_age_exceeded
        retry_delay = min(
            self._retry_delay * (2 ** max(0, attempts - 1)),
            MAX_RETRY_DELAY_SECONDS,
        )
        if not consumes_attempt and not permanent:
            retry_delay = max(retry_delay, DEFAULT_RATE_LIMIT_RETRY_SECONDS)
        if (
            not isinstance(retry_after_seconds, bool)
            and isinstance(retry_after_seconds, (int, float))
            and math.isfinite(float(retry_after_seconds))
            and retry_after_seconds >= 0
        ):
            retry_delay = max(
                retry_delay,
                min(float(retry_after_seconds), MAX_RETRY_AFTER_SECONDS),
            )
        safe_error_code = self._safe_error_code(error_code)
        cursor = self._conn.execute(
            """
            UPDATE extract_queue
            SET attempts = ?, last_error = ?, status = ?, not_before = ?,
                lease_owner = NULL, lease_until = NULL, lease_token = NULL,
                proposal_json = CASE WHEN ? = 'proposal_credential_rejected'
                    THEN NULL ELSE proposal_json END,
                proposal_hash = CASE WHEN ? = 'proposal_credential_rejected'
                    THEN NULL ELSE proposal_hash END
            WHERE id = ? AND lease_owner = ? AND lease_token = ?
            """,
            (
                attempts,
                safe_error_code,
                "dead" if dead else "pending",
                None if dead else self._clock() + retry_delay,
                safe_error_code,
                safe_error_code,
                row_id,
                self._worker_id,
                lease_token,
            ),
        )
        self._conn.commit()
        if cursor.rowcount != 1:
            raise RuntimeError("extraction lease was lost while recording failure")
        return attempts

    def _job_is_dead(self, row_id: int) -> bool:
        row = self._conn.execute(
            "SELECT status FROM extract_queue WHERE id = ?", (row_id,)
        ).fetchone()
        return row is not None and row[0] == "dead"

    def _extract_with_heartbeat(
        self,
        envelope: ExtractionEnvelope,
        row_id: int,
        lease_token: str,
    ) -> tuple[ExtractedMemory, ...]:
        """Run a model outside SQLite while renewing only the current fence."""

        done = threading.Event()
        invocation_ready = threading.Event()
        result: dict[str, Any] = {}
        invocation: list[object] = []
        cancel = getattr(self._extractor, "cancel", None)

        def register_invocation(handle: object) -> None:
            invocation.append(handle)
            invocation_ready.set()

        def invoke() -> None:
            try:
                if callable(cancel):
                    proposals = self._extractor.extract(
                        envelope, register_invocation=register_invocation
                    )
                else:
                    proposals = self._extractor.extract(envelope)
                result["proposals"] = tuple(proposals)
            except BaseException as exc:  # relay the original model failure
                result["error"] = exc
            finally:
                done.set()

        thread = threading.Thread(
            target=invoke,
            name="enfold-extraction-call",
            daemon=True,
        )
        thread.start()
        while not done.wait(self._heartbeat_seconds):
            try:
                self._renew(row_id, lease_token)
            except BaseException:
                if callable(cancel):
                    invocation_ready.wait(EXTRACTION_CANCEL_JOIN_SECONDS)
                if callable(cancel) and invocation:
                    try:
                        cancel(invocation[0])
                    except Exception:
                        pass
                thread.join(EXTRACTION_CANCEL_JOIN_SECONDS)
                raise
        error = result.get("error")
        if error is not None:
            raise error
        proposals = result.get("proposals")
        if not isinstance(proposals, tuple):
            raise RuntimeError("extractor did not return proposals")
        return proposals

    def _load_snapshot(
        self,
        row_id: int,
        lease_token: str,
        envelope: ExtractionEnvelope,
    ) -> tuple[str | None, str | None]:
        row = self._conn.execute(
            """
            SELECT proposal_json, proposal_hash FROM extract_queue
            WHERE id = ? AND status = 'processing' AND lease_owner = ?
              AND lease_token = ?
            """,
            (row_id, self._worker_id, lease_token),
        ).fetchone()
        if row is None:
            raise RuntimeError("extraction lease was lost before snapshot load")
        proposal_json = row[0]
        proposal_hash = row[1]
        if proposal_json is not None and proposal_hash is None:
            raise PermanentExtractionError(
                "legacy extraction snapshots are quarantined",
                error_code="legacy_extraction_quarantined",
            )
        if (proposal_json is None) != (proposal_hash is None):
            raise PermanentExtractionError(
                "proposal snapshot is inconsistent", error_code="invalid_snapshot"
            )
        if proposal_json is None:
            return None, None
        if not isinstance(proposal_json, str) or not isinstance(proposal_hash, str):
            raise PermanentExtractionError(
                "proposal snapshot is malformed", error_code="invalid_snapshot"
            )
        return proposal_json, proposal_hash

    def _queue_observed_at(self, row_id: int, lease_token: str) -> str:
        """Return the stable enqueue timestamp while the current lease is live."""

        row = self._conn.execute(
            "SELECT created_at FROM extract_queue "
            "WHERE id = ? AND status = 'processing' AND lease_owner = ? "
            "AND lease_token = ?",
            (row_id, self._worker_id, lease_token),
        ).fetchone()
        if row is None:
            raise RuntimeError("extraction lease was lost before timestamp load")
        observed_at = row[0]
        if not isinstance(observed_at, str) or not observed_at.strip():
            raise PermanentExtractionError(
                "queue timestamp is malformed", error_code="invalid_envelope"
            )
        return observed_at.strip()

    def _persist_snapshot(
        self,
        row_id: int,
        lease_token: str,
        proposal_json: str,
        proposal_hash: str,
    ) -> None:
        try:
            self._conn.execute("BEGIN IMMEDIATE")
            cursor = self._conn.execute(
                """
                UPDATE extract_queue
                SET proposal_json = ?, proposal_hash = ?
                WHERE id = ? AND status = 'processing' AND lease_owner = ?
                  AND lease_token = ? AND proposal_json IS NULL AND proposal_hash IS NULL
                """,
                (proposal_json, proposal_hash, row_id, self._worker_id, lease_token),
            )
            if cursor.rowcount != 1:
                raise RuntimeError(
                    "extraction lease was lost before snapshot persistence"
                )
            self._conn.commit()
        except BaseException:
            if self._conn.in_transaction:
                self._conn.rollback()
            raise

    def _convert_legacy_snapshot(
        self,
        row_id: int,
        lease_token: str,
        legacy_json: str,
        envelope: ExtractionEnvelope,
    ) -> tuple[str, str]:
        try:
            legacy_proposals = json.loads(legacy_json)
        except (TypeError, json.JSONDecodeError) as exc:
            raise PermanentExtractionError(
                "legacy proposal snapshot is malformed", error_code="invalid_snapshot"
            ) from exc
        if not isinstance(legacy_proposals, list):
            raise PermanentExtractionError(
                "legacy proposal snapshot is malformed", error_code="invalid_snapshot"
            )
        if len(legacy_proposals) > MAX_EXTRACTED_MEMORIES:
            raise PermanentExtractionError(
                "legacy proposal snapshot exceeds proposal limit",
                error_code="proposal_limit",
            )

        normalized: list[dict[str, Any]] = []
        spans = _eligible_envelope_spans(envelope)
        for proposal in legacy_proposals:
            if (
                not isinstance(proposal, dict)
                or set(proposal) != {"content", "category", "tags"}
                or not isinstance(proposal["content"], str)
                or not proposal["content"].strip()
                or not isinstance(proposal["category"], str)
                or not proposal["category"].strip()
                or not isinstance(proposal["tags"], str)
            ):
                raise PermanentExtractionError(
                    "legacy proposal snapshot contains invalid fields",
                    error_code="invalid_snapshot",
                )
            content = proposal["content"].strip()
            candidates = []
            if content in envelope.transcript:
                candidates.append(content)
            candidates.extend(
                span.text for span in spans if span.text not in candidates
            )
            item = None
            for excerpt in candidates:
                candidate = {
                    "category": proposal["category"].strip(),
                    "content": content,
                    "evidence_excerpt": excerpt,
                    "sensitivity": "normal",
                    "tags": proposal["tags"],
                }
                if not self._credential_shaped_snapshot_item(candidate):
                    item = candidate
                    break
            if item is not None:
                normalized.append(item)

        snapshot = {
            "extractor_identity": LEGACY_EXTRACTION_CLIENT_ID,
            "proposals": normalized,
            "version": 1,
        }
        canonical = json.dumps(
            snapshot,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        proposal_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        try:
            self._conn.execute("BEGIN IMMEDIATE")
            cursor = self._conn.execute(
                """
                UPDATE extract_queue
                SET proposal_json = ?, proposal_hash = ?
                WHERE id = ? AND status = 'processing' AND lease_owner = ?
                  AND lease_token = ? AND proposal_json = ? AND proposal_hash IS NULL
                """,
                (
                    canonical,
                    proposal_hash,
                    row_id,
                    self._worker_id,
                    lease_token,
                    legacy_json,
                ),
            )
            if cursor.rowcount != 1:
                raise RuntimeError(
                    "extraction lease was lost before legacy snapshot conversion"
                )
            self._conn.commit()
        except BaseException:
            if self._conn.in_transaction:
                self._conn.rollback()
            raise
        return canonical, proposal_hash

    @staticmethod
    def _decode_envelope(payload: str) -> ExtractionEnvelope:
        try:
            try:
                data = json.loads(payload)
            except json.JSONDecodeError as exc:
                stripped = payload.strip()
                if not stripped or stripped[0] in '{["':
                    raise
                raise PermanentExtractionError(
                    "legacy extraction payloads are quarantined",
                    error_code="legacy_extraction_quarantined",
                ) from exc
            if isinstance(data, str):
                raise PermanentExtractionError(
                    "legacy extraction payloads are quarantined",
                    error_code="legacy_extraction_quarantined",
                )
            if not isinstance(data, dict) or data.get("version") != 1:
                raise ValueError("unsupported extraction envelope version")
            provenance = data["provenance"]
            if not isinstance(provenance, dict):
                raise ValueError("provenance must be an object")
            metadata = data.get("metadata", {})
            if not isinstance(metadata, dict):
                raise ValueError("metadata must be an object")
            scope = validate_scope(str(data.get("scope", "private")))
            context = ClientContext.from_dict(provenance)
            if context.client_id == LEGACY_EXTRACTION_CLIENT_ID:
                raise ValueError("reserved extraction client id is not permitted")
            if not scope_authorized(scope, context.access_scopes) or scope == "secret":
                raise ValueError("extraction scope is unauthorized or secret")
            has_transcript = "transcript" in data
            has_turns = "turns" in data
            if has_transcript == has_turns:
                raise ValueError("envelope must contain transcript or turns")
            transcript, turns = normalize_transcript(
                data["transcript"] if has_transcript else data["turns"]
            )
            source = str(data["source"]).strip()
            if not transcript or not source:
                raise ValueError("transcript and source must be non-empty")
            return ExtractionEnvelope(
                transcript,
                source,
                scope,
                context,
                metadata,
                turns,
            )
        except PermanentExtractionError:
            raise
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise PermanentExtractionError(
                f"invalid extraction envelope: {exc}", error_code="invalid_envelope"
            ) from exc

    @staticmethod
    def _legacy_envelope(transcript: str) -> ExtractionEnvelope:
        return ExtractionEnvelope(
            transcript=transcript,
            source="legacy_extract_queue",
            scope="private",
            context=ClientContext(
                client_id=LEGACY_EXTRACTION_CLIENT_ID,
                surface="legacy",
                agent_id="legacy",
                session_id=LEGACY_EXTRACTION_CLIENT_ID,
                access_scopes=("private",),
            ),
            metadata={"legacy_queue_payload": True},
            legacy_payload=True,
        )

    def _make_snapshot(
        self,
        proposals: Sequence[ExtractedMemory],
        envelope: ExtractionEnvelope,
    ) -> tuple[str, str]:
        if len(proposals) > MAX_EXTRACTED_MEMORIES:
            raise PermanentExtractionError(
                "extractor returned too many memories", error_code="proposal_limit"
            )
        normalized: list[dict[str, Any]] = []
        credential_rejected = False
        grounding_rejected = False
        for proposal in proposals:
            if not isinstance(proposal, ExtractedMemory):
                raise PermanentExtractionError(
                    "extractor returned an invalid proposal",
                    error_code="invalid_proposal",
                )
            if not isinstance(proposal.content, str) or not proposal.content.strip():
                raise PermanentExtractionError(
                    "proposal content must be non-empty text",
                    error_code="invalid_proposal",
                )
            if proposal.scope is not None:
                try:
                    requested_scope = validate_scope(proposal.scope)
                except (TypeError, ValueError) as exc:
                    raise PermanentExtractionError(
                        "proposal scope is invalid",
                        error_code="proposal_scope_rejected",
                    ) from exc
                if requested_scope != envelope.scope:
                    raise PermanentExtractionError(
                        "automatic extraction cannot change envelope scope",
                        error_code="proposal_scope_rejected",
                    )
            if proposal.sensitivity not in {"normal", "sensitive"}:
                raise PermanentExtractionError(
                    "proposal sensitivity is not permitted",
                    error_code="proposal_sensitivity_rejected",
                )
            if not isinstance(proposal.category, str) or not proposal.category.strip():
                raise PermanentExtractionError(
                    "proposal category must be non-empty text",
                    error_code="invalid_proposal",
                )
            if not isinstance(proposal.tags, str):
                raise PermanentExtractionError(
                    "proposal tags must be text", error_code="invalid_proposal"
                )
            excerpt = proposal.evidence_excerpt
            if (
                not isinstance(excerpt, str)
                or not excerpt.strip()
                or len(excerpt) > MAX_EVIDENCE_CHARS
                or excerpt not in envelope.transcript
            ):
                grounding_rejected = True
                continue
            item = {
                "category": proposal.category.strip(),
                "content": proposal.content.strip(),
                "evidence_excerpt": excerpt,
                "sensitivity": proposal.sensitivity,
                "tags": proposal.tags,
            }
            span_id = proposal.metadata.get("evidence_span_id")
            if not isinstance(span_id, str) or not span_id:
                grounding_rejected = True
                continue
            matching_span = next(
                (
                    span
                    for span in _envelope_spans(envelope)
                    if span.span_id == span_id
                ),
                None,
            )
            if matching_span is not None and matching_span.role != "user":
                continue
            if matching_span is None or matching_span.text != excerpt:
                grounding_rejected = True
                continue
            item["evidence_span_id"] = span_id
            verification = self._evidence_verifier.verify(
                proposal,
                evidence_excerpt=excerpt,
                envelope=envelope,
            )
            if (
                not isinstance(verification, EvidenceVerification)
                or verification.status not in {"verified", "needs_review"}
            ):
                raise PermanentExtractionError(
                    "proposal claim support requires review",
                    error_code="proposal_support_unverified",
                )
            item["evidence_verification"] = {
                "status": verification.status,
                "verifier_id": verification.verifier_id,
            }
            if verification.status == "verified":
                typed = self._normalize_typed_fields(
                    proposal.state, item["content"], scope=envelope.scope
                )
                if typed is not None:
                    item["typed"] = typed
                elif proposal.state is not None:
                    item["typed_demotion"] = {
                        "reason": "invalid_or_incomplete_typed_fields",
                    }
            contract_demotion = proposal.metadata.get("typed_demotion")
            if contract_demotion is not None:
                if (
                    not isinstance(contract_demotion, Mapping)
                    or set(contract_demotion) != {"reason"}
                    or contract_demotion.get("reason") != "incomplete_typed_fields"
                    or proposal.state is not None
                ):
                    raise PermanentExtractionError(
                        "proposal typed demotion is invalid",
                        error_code="invalid_proposal",
                    )
                item["typed_demotion"] = dict(contract_demotion)
            if self._credential_shaped_snapshot_item(item):
                credential_rejected = True
                continue
            skip_reason = ephemeral_extraction_reason(item["content"])
            if skip_reason is not None:
                logger.info(
                    "extraction: dropped proposal (%s): %r",
                    skip_reason,
                    item["content"][:80],
                )
                continue
            normalized.append(item)
        if not normalized and credential_rejected:
            raise PermanentExtractionError(
                "credential-shaped proposal rejected",
                error_code="proposal_credential_rejected",
            )
        if not normalized and grounding_rejected:
            raise PermanentExtractionError(
                "proposal evidence is not an exact bounded transcript excerpt",
                error_code="proposal_grounding_rejected",
            )
        snapshot = {
            "extractor_identity": self._extractor.identity,
            "proposals": normalized,
            "version": 1,
        }
        try:
            proposal_json = json.dumps(
                snapshot,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        except (TypeError, ValueError, RecursionError) as exc:
            raise PermanentExtractionError(
                "proposal snapshot is not JSON", error_code="invalid_proposal"
            ) from exc
        return proposal_json, hashlib.sha256(proposal_json.encode("utf-8")).hexdigest()

    @staticmethod
    def _credential_shaped_snapshot_item(item: Mapping[str, Any]) -> bool:
        content = str(item.get("content", "")).strip() or "invalid proposal"
        excerpt = str(item.get("evidence_excerpt", "")) or None
        if extracted_proposal_credential_decision(content, excerpt) is not None:
            return True
        request = WriteRequest(
            idempotency_key="extraction-snapshot-screen",
            content=content,
            source_type="automatic_extraction",
            category=str(item.get("category", "general")),
            tags=str(item.get("tags", "")),
            evidence_excerpt=excerpt,
        )
        encoded = json.dumps(
            item,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        return default_credential_screen(request, (encoded,)) is not None

    def _prepare_snapshot(
        self,
        proposal_json: str,
        proposal_hash: str,
        envelope: ExtractionEnvelope,
        row_id: int,
        digest: str,
        observed_at: str,
    ) -> tuple[dict[str, Any], ...]:
        if hashlib.sha256(proposal_json.encode("utf-8")).hexdigest() != proposal_hash:
            raise PermanentExtractionError(
                "proposal snapshot hash does not match",
                error_code="snapshot_hash_mismatch",
            )
        try:
            snapshot = json.loads(proposal_json)
            canonical = json.dumps(
                snapshot,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        except (TypeError, ValueError, RecursionError) as exc:
            raise PermanentExtractionError(
                "proposal snapshot is malformed", error_code="invalid_snapshot"
            ) from exc
        if canonical != proposal_json or not isinstance(snapshot, dict):
            raise PermanentExtractionError(
                "proposal snapshot is not canonical", error_code="invalid_snapshot"
            )
        if snapshot.get("version") != 1:
            raise PermanentExtractionError(
                "proposal snapshot version is unsupported",
                error_code="invalid_snapshot",
            )
        identity = snapshot.get("extractor_identity")
        proposals = snapshot.get("proposals")
        if (
            not isinstance(identity, str)
            or not identity.strip()
            or not isinstance(proposals, list)
        ):
            raise PermanentExtractionError(
                "proposal snapshot is malformed", error_code="invalid_snapshot"
            )
        if len(proposals) > MAX_EXTRACTED_MEMORIES:
            raise PermanentExtractionError(
                "proposal snapshot exceeds proposal limit", error_code="proposal_limit"
            )
        prepared: list[dict[str, Any]] = []
        for index, proposal in enumerate(proposals):
            base_fields = {
                "category",
                "content",
                "evidence_excerpt",
                "sensitivity",
                "tags",
            }
            if (
                not isinstance(proposal, dict)
                or not base_fields.issubset(proposal)
                or set(proposal)
                - base_fields
                - {
                    "evidence_span_id",
                    "evidence_verification",
                    "typed",
                    "typed_demotion",
                }
            ):
                raise PermanentExtractionError(
                    "proposal snapshot has unsupported fields",
                    error_code="invalid_snapshot",
                )
            content = proposal["content"]
            category = proposal["category"]
            tags = proposal["tags"]
            excerpt = proposal["evidence_excerpt"]
            sensitivity = proposal["sensitivity"]
            typed = proposal.get("typed")
            typed_demotion = proposal.get("typed_demotion")
            evidence_span_id = proposal.get("evidence_span_id")
            verification = proposal.get("evidence_verification")
            if (
                not isinstance(content, str)
                or not content
                or not isinstance(category, str)
                or not category
                or not isinstance(tags, str)
                or not isinstance(excerpt, str)
                or not excerpt.strip()
                or sensitivity not in {"normal", "sensitive"}
            ):
                raise PermanentExtractionError(
                    "proposal snapshot contains invalid fields",
                    error_code="invalid_snapshot",
                )
            skip_reason = ephemeral_extraction_reason(content)
            if skip_reason is not None:
                logger.info(
                    "extraction: dropped snapshot proposal (%s): %r",
                    skip_reason,
                    content[:80],
                )
                continue
            if len(excerpt) > MAX_EVIDENCE_CHARS or excerpt not in envelope.transcript:
                raise PermanentExtractionError(
                    "proposal snapshot evidence is not grounded in the transcript",
                    error_code="proposal_grounding_rejected",
                )
            if not isinstance(evidence_span_id, str) or not evidence_span_id:
                raise PermanentExtractionError(
                    "proposal snapshot lacks an evidence span id",
                    error_code="proposal_grounding_rejected",
                )
            matching_span = next(
                (
                    span
                    for span in _envelope_spans(envelope)
                    if span.span_id == evidence_span_id
                ),
                None,
            )
            if matching_span is not None and matching_span.role != "user":
                continue
            if matching_span is None or matching_span.text != excerpt:
                raise PermanentExtractionError(
                    "proposal snapshot evidence span id is invalid",
                    error_code="proposal_grounding_rejected",
                )
            if (
                not isinstance(verification, dict)
                or set(verification) != {"status", "verifier_id"}
                or verification.get("status") not in {"verified", "needs_review"}
                or not isinstance(verification.get("verifier_id"), str)
                or not verification["verifier_id"].strip()
            ):
                raise PermanentExtractionError(
                    "proposal snapshot lacks verified claim support",
                    error_code="proposal_support_unverified",
                )
            if verification["status"] != "verified":
                typed = None
            if typed is not None and not self._is_normalized_typed_fields(
                typed, scope=envelope.scope
            ):
                raise PermanentExtractionError(
                    "proposal snapshot contains invalid typed fields",
                    error_code="invalid_snapshot",
                )
            if typed_demotion is not None and (
                not isinstance(typed_demotion, dict)
                or set(typed_demotion) != {"reason"}
                or typed_demotion.get("reason")
                not in {
                    "incomplete_typed_fields",
                    "invalid_or_incomplete_typed_fields",
                }
                or typed is not None
            ):
                raise PermanentExtractionError(
                    "proposal snapshot contains invalid typed demotion",
                    error_code="invalid_snapshot",
                )
            if verification["status"] != "verified":
                tags = (
                    f"{tags},evidence_unreviewed" if tags.strip() else "evidence_unreviewed"
                )
            metadata = {
                "extraction_queue_id": row_id,
                "extraction_payload_sha256": digest,
                "extractor_identity": identity,
                "extraction_source": envelope.source,
                "proposal_snapshot_sha256": proposal_hash,
                "evidence_verifier": verification["verifier_id"],
            }
            if verification["status"] != "verified":
                metadata["evidence_verification"] = {
                    "status": verification["status"],
                    "verifier_id": verification["verifier_id"],
                }
            if typed is not None:
                metadata.update(
                    {
                        "extracted_kind": typed["kind"],
                        "extracted_confidence": typed["confidence"],
                        "extracted_negation": typed["negation"],
                        # This normalized payload is consumed only by the
                        # authoritative write service.  It keeps typed
                        # extraction queryable in first-class fact columns,
                        # rather than leaving it as provenance-only metadata.
                        "extraction_typed": typed,
                    }
                )
            if typed_demotion is not None:
                metadata["typed_demotion"] = typed_demotion
            metadata["evidence_span_id"] = evidence_span_id
            try:
                request = WriteRequest(
                    idempotency_key=(f"extract:{digest}:{proposal_hash[:24]}:{index}"),
                    content=content,
                    source_type="automatic_extraction",
                    category=category,
                    tags=tags,
                    trust_score=AUTOMATIC_TRUST_SCORE,
                    source_authority=AUTOMATIC_SOURCE_AUTHORITY[matching_span.role],
                    observation_content=excerpt,
                    asserted_by=matching_span.role,
                    observed_at=observed_at,
                    scope=envelope.scope,
                    sensitivity=sensitivity,
                    evidence_excerpt=excerpt,
                    relation="derived_from",
                    metadata_json=json.dumps(metadata, sort_keys=True, allow_nan=False),
                )
            except (TypeError, ValueError, RecursionError) as exc:
                raise PermanentExtractionError(
                    "invalid extraction proposal", error_code="invalid_proposal"
                ) from exc
            decision = default_credential_screen(request)
            if decision is not None:
                continue
            params: dict[str, Any] = {
                "idempotency_key": request.idempotency_key,
                "content": request.content,
                "source_type": request.source_type,
                "category": request.category,
                "tags": request.tags,
                "trust_score": request.trust_score,
                "source_authority": request.source_authority,
                "observation_content": request.observation_content,
                "asserted_by": request.asserted_by,
                "observed_at": request.observed_at,
                "scope": request.scope,
                "sensitivity": request.sensitivity,
                "evidence_excerpt": request.evidence_excerpt,
                "relation": request.relation,
                "metadata": metadata,
            }
            if verification["status"] != "verified":
                params["correction_status"] = "unreviewed"
            if typed is not None and typed["kind"] == "state":
                params["state"] = {
                    "subject_key": typed["subject_key"],
                    "predicate_key": typed["predicate_key"],
                    "object_value": typed["object_value"],
                    "valid_from": typed["valid_from"],
                }
            prepared.append(params)
        return tuple(self._demote_conflicting_batch_state(prepared))

    @staticmethod
    def _demote_conflicting_batch_state(
        prepared: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Leave typed state in place so write_batch can open a real conflict."""

        return prepared

    def _normalize_typed_fields(
        self, value: Any, content: str, *, scope: str
    ) -> dict[str, Any] | None:
        """Return a safe typed payload, or abstain without dropping content."""

        if value is None or not isinstance(value, Mapping):
            return None
        if not value or set(value) - _TYPED_FIELDS:
            return None
        kind = value.get("kind")
        confidence = value.get("confidence")
        negation = value.get("negation", False)
        if (
            kind not in _TYPED_KINDS
            or isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not math.isfinite(float(confidence))
            or not MIN_TYPED_CONFIDENCE <= float(confidence) <= 1.0
            or not isinstance(negation, bool)
        ):
            return None
        if "object" in value and "value" in value:
            return None
        if "occurred_at" in value and "valid_from" in value:
            return None
        object_value = value.get("object", value.get("value"))
        if negation:
            if object_value is not None:
                return None
            object_value = None
        elif not isinstance(object_value, str) or not object_value.strip():
            return None
        else:
            object_value = object_value.strip()
        valid_from = value.get("valid_from", value.get("occurred_at"))
        if valid_from is not None and (
            not isinstance(valid_from, str) or not valid_from.strip()
        ):
            return None
        try:
            subject_key = resolve_stored_subject_key(
                self._conn, value.get("subject"), scope=scope
            )
            if subject_key is None:
                return None
            registry = canonical_slot_registry(self._conn, scope=scope)
            predicate_key = resolve_extracted_predicate_key(
                value.get("predicate"),
                known_predicates=tuple(registry["predicates"]),
            )
            # Reuse the slot type's timestamp validation rather than maintaining
            # a subtly different parser at the model boundary.
            StateCandidate(
                content=content,
                subject_key=subject_key,
                predicate_key=predicate_key,
                object_value=object_value,
                valid_from=valid_from,
            )
        except (TypeError, ValueError):
            return None
        return {
            "confidence": float(confidence),
            "kind": kind,
            "negation": negation,
            "object_value": object_value,
            "predicate_key": predicate_key,
            "subject_key": subject_key,
            "valid_from": valid_from.strip() if valid_from is not None else None,
        }

    def _is_normalized_typed_fields(self, value: Any, *, scope: str) -> bool:
        if not isinstance(value, dict) or set(value) != {
            "confidence",
            "kind",
            "negation",
            "object_value",
            "predicate_key",
            "subject_key",
            "valid_from",
        }:
            return False
        normalized = self._normalize_typed_fields(
            {
                "confidence": value["confidence"],
                "kind": value["kind"],
                "negation": value["negation"],
                "subject": value["subject_key"],
                "predicate": value["predicate_key"],
                "value": value["object_value"],
                "valid_from": value["valid_from"],
            },
            "snapshot validation",
            scope=scope,
        )
        return normalized == value

    @staticmethod
    def _safe_error_code(exc: BaseException | str) -> str:
        """Return an allowlisted operational code, never adapter/model text."""

        if isinstance(exc, str):
            candidates = (exc,)
        else:
            candidates = (
                getattr(exc, "error_code", None),
                getattr(exc, "code", None),
            )
        for candidate in candidates:
            if isinstance(candidate, str) and candidate in _SAFE_ERROR_CODES:
                return candidate
        return "extractor_failed"
