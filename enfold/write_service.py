"""Transactional, provenance-aware fact write envelope."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import sqlite3
from typing import Callable, Optional, Sequence
import urllib.error
import uuid

import numpy as np

from .cluster_merge import NearDuplicateCandidate, find_write_near_duplicates
from .hybrid_retrieval import StoredEmbeddingError
from .policy import MemoryPolicy, PolicyDecision, scope_authorized
from .provenance import ConnectionContext, WriteOutcome, WriteRequest
from .state_slots import (
    SlotDecision,
    StateCandidate,
    add_conflict_member,
    add_untyped_conflict_member,
    decide_state_write,
    normalize_predicate_key,
    normalize_subject_key,
    open_state_conflict,
    open_untyped_conflict,
    resolve_state_conflict,
)
from .temporal import (
    _content_tokens,
    _has_negation_mismatch,
    _has_opposing_state_words,
    _has_subjectish_overlap,
    _is_value_update,
    _norm_token_sequence,
    _subjectish_tokens,
)


_UNSET = object()
_UNTYPED_CONTRADICTION_PREFIX = 2
_UNTYPED_RESIDUE_NEGATION = frozenset(("never", "without"))
_STATE_FRAME_WORDS = frozenset(
    "is are was were uses use used using prefers prefer preferred "
    "runs run running".split()
)


@dataclass(frozen=True, slots=True)
class _UntypedContradiction:
    fact_id: int
    content: str
    subject_key: str
    predicate_key: str


def _canonical_shared_token(token: str, subjectish: set[str]) -> str | None:
    if token in subjectish:
        return token
    for suffix in ("ation", "ing", "ed", "ic", "ion", "s"):
        if token.endswith(suffix) and len(token) > len(suffix) + 2:
            stem = token[: -len(suffix)]
            if stem in subjectish:
                return stem
    return None


def guess_untyped_conflict_slot(left: str, right: str) -> tuple[str, str]:
    """Label a dispute from shared subject-like tokens, never from a lone word."""

    right_subjectish = _subjectish_tokens(right)
    shared: list[str] = []
    seen: set[str] = set()
    for token in _norm_token_sequence(left):
        canonical = _canonical_shared_token(token, right_subjectish)
        if canonical is None or canonical in seen:
            continue
        seen.add(canonical)
        shared.append(canonical)
    if len(shared) >= 2:
        subject = ".".join(shared[:-1])
        predicate = shared[-1]
    elif shared:
        subject = shared[0]
        predicate = "claim"
    else:
        subject = "untyped"
        predicate = "claim"
    if len(subject) > 128:
        subject = subject[:128]
    if len(predicate) > 64:
        predicate = predicate[:64]
    return normalize_subject_key(subject), normalize_predicate_key(predicate)


def _is_suffix_value_substitution(left: str, right: str) -> bool:
    """True when two claims share a prefix frame and replace the trailing value."""

    if not _has_subjectish_overlap(left, right):
        return False
    left_seq = _norm_token_sequence(left)
    right_seq = _norm_token_sequence(right)
    prefix = 0
    limit = min(len(left_seq), len(right_seq))
    while prefix < limit and left_seq[prefix] == right_seq[prefix]:
        prefix += 1
    if prefix < _UNTYPED_CONTRADICTION_PREFIX:
        return False
    left_tail = left_seq[prefix:]
    right_tail = right_seq[prefix:]
    if not left_tail or not right_tail:
        return False
    if set(left_tail) & set(right_tail):
        return False
    if set(left_tail) & _STATE_FRAME_WORDS or set(right_tail) & _STATE_FRAME_WORDS:
        return False
    left_values = all(any(char.isdigit() for char in token) for token in left_tail)
    right_values = all(any(char.isdigit() for char in token) for token in right_tail)
    shared_noncopular_frame = (
        set(left_seq[:prefix]) & _STATE_FRAME_WORDS
    ) - {"is", "are", "was", "were"}
    return (
        prefix >= 3
        or bool(shared_noncopular_frame)
        or (left_values and right_values)
    )


def _has_state_frame(text: str) -> bool:
    return bool(set(_norm_token_sequence(text)) & _STATE_FRAME_WORDS)


def _untyped_claim_residue(text: str) -> set[str]:
    """Subject-like tokens after polarity words, collapsed to shared stems."""

    pool = _subjectish_tokens(text) - _UNTYPED_RESIDUE_NEGATION
    residue: set[str] = set()
    for token in pool:
        collapsed = token
        for suffix in ("ation", "ing", "ed", "ic", "ion", "s"):
            if token.endswith(suffix) and len(token) > len(suffix) + 2:
                stem = token[: -len(suffix)]
                if stem in pool:
                    collapsed = stem
                    break
        residue.add(collapsed)
    return residue


def untyped_contents_contradict(left: str, right: str) -> bool:
    """Reuse the existing polarity detectors; value swaps need a state frame.

    Numbered event series such as "load memory 1" / "load memory 2" share a
    frame and differ by a value token, but they are not a single-valued slot.
    Those stay additive. A copula or use/prefer verb is required before a
    value substitution is treated as a contradiction.

    Polarity alone is not enough: the remaining claim tokens must match, so
    unrelated properties of the same subject stay additive.
    """

    if _has_opposing_state_words(left, right) or _has_negation_mismatch(left, right):
        left_residue = _untyped_claim_residue(left)
        right_residue = _untyped_claim_residue(right)
        return bool(left_residue) and left_residue == right_residue
    if not (_has_state_frame(left) and _has_state_frame(right)):
        return False
    return _is_value_update(left, right) or _is_suffix_value_substitution(left, right)


class IdempotencyConflict(ValueError):
    """An idempotency key was retried with a different request payload."""


class ClientIdentityConflict(ValueError):
    """A stable client id was reused for a different surface or agent."""


class SessionContextConflict(ValueError):
    """A client/session identity was reused with different immutable context."""


@dataclass(frozen=True, slots=True)
class FactWriteResult:
    """Narrow result contract implemented by the existing fact store."""

    fact_id: int
    outcome: str = "inserted"
    existing_fact_id: Optional[int] = None
    detail_json: str = "{}"

    def __post_init__(self) -> None:
        if self.fact_id <= 0:
            raise ValueError("fact_id must be positive")
        if self.existing_fact_id is not None and self.existing_fact_id <= 0:
            raise ValueError("existing_fact_id must be positive")
        decoded = json.loads(self.detail_json)
        if not isinstance(decoded, dict):
            raise ValueError("detail_json must be a JSON object")
        object.__setattr__(
            self,
            "detail_json",
            json.dumps(decoded, sort_keys=True, separators=(",", ":")),
        )


FactWriter = Callable[
    [sqlite3.Connection, WriteRequest, int],
    FactWriteResult,
]


@dataclass(frozen=True, slots=True)
class WriteBatchOutcome:
    """Ordered batch outcomes plus whether their transaction committed."""

    outcomes: tuple[WriteOutcome, ...]
    committed: bool


@dataclass(frozen=True, slots=True)
class NearDedupConfig:
    """Conservative controls for embedding-backed write-time consolidation.

    ``query_embedder`` remains intentionally separate from the durable
    embedding job queue: when it is absent, no embedding identity is supplied,
    or any candidate lacks a stored vector, writes retain the exact-dedup path
    instead of waiting on a job.
    """

    enabled: bool = True
    cosine_threshold: float = 0.97
    candidate_limit: int = 64
    embedding_identity: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("near dedup enabled must be a boolean")
        if not 0.0 <= self.cosine_threshold <= 1.0:
            raise ValueError("near dedup cosine threshold must be between 0 and 1")
        if self.candidate_limit <= 0:
            raise ValueError("near dedup candidate limit must be positive")
        if self.embedding_identity is not None and not self.embedding_identity.strip():
            raise ValueError("near dedup embedding identity must be non-empty")


@dataclass(frozen=True, slots=True)
class ExtractedTypedFields:
    """Normalized typed extraction fields safe to persist on a new fact.

    The public protocol deliberately exposes state slots only.  The extractor
    stores its independently normalized typed payload in provenance metadata,
    and this write-boundary value object is the sole place that permits it to
    reach queryable fact columns.
    """

    kind: str
    subject_key: str
    predicate_key: str
    object_value: str | None
    valid_from: str | None
    confidence: float


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _request_sha256(
    context: ConnectionContext,
    request: WriteRequest,
    state_candidate: Optional[StateCandidate] = None,
) -> str:
    # Session identity is provenance on the log row, not part of the replay
    # contract.  A reconnect mints a new session_id for the same client and
    # the same request body; that retry must hash as the original write.
    state_payload = (
        asdict(state_candidate) if state_candidate is not None else None
    )
    if isinstance(state_payload, dict) and state_payload.get("valid_to") is None:
        state_payload = {
            key: value for key, value in state_payload.items() if key != "valid_to"
        }
    payload = {
        "client_id": context.client_id,
        "request": asdict(request),
        "state_candidate": state_payload,
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _observation_sha256(context: ConnectionContext, request: WriteRequest) -> str:
    payload = {
        "content": request.observation_content or request.content,
        "source_type": request.source_type,
        "source_uri": request.source_uri,
        "asserted_by": request.asserted_by,
        "performed_by": request.performed_by,
        "observed_at": request.observed_at,
        "scope": request.scope,
        "sensitivity": request.sensitivity,
        "metadata_json": request.metadata_json,
        "project_root": context.project_root,
        "repository": context.repository,
        "branch": context.branch,
        "commit_sha": context.commit_sha,
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


_ENVIRONMENTAL_REJECTION_REASONS = frozenset(
    {
        "requested write scope is not server-authorized",
        "sensitive durable write is not authorized",
    }
)


def _factless_decision_is_durable(decision: PolicyDecision) -> bool:
    """Persist only rejections that are a function of the request body."""

    return (
        decision.outcome == "rejected"
        and decision.reason not in _ENVIRONMENTAL_REJECTION_REASONS
    )


def _ephemeral_factless_outcome(decision: PolicyDecision) -> WriteOutcome:
    """Return a factless decision without consuming the idempotency key."""

    return WriteOutcome(
        write_id=str(uuid.uuid4()),
        outcome=decision.outcome,
        fact_id=None,
        detail_json=json.dumps(
            {"policy_reason": decision.reason},
            sort_keys=True,
            separators=(",", ":"),
        ),
    )


class MemoryWriteService:
    """Perform one complete durable write using a caller-provided connection.

    ``fact_writer`` is the only integration point with the existing fact
    store.  It runs inside the same ``BEGIN IMMEDIATE`` transaction and must
    neither commit nor roll back.
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        fact_writer: FactWriter,
        policy: MemoryPolicy,
        embedding_enqueue: Callable[[int], int] | None = None,
        *,
        near_dedup: NearDedupConfig | None = None,
        query_embedder: Callable[[str], object] | None = None,
    ):
        self._conn = conn
        self._fact_writer = fact_writer
        self._policy = policy
        self._embedding_enqueue = embedding_enqueue
        self._near_dedup = near_dedup or NearDedupConfig()
        self._query_embedder = query_embedder
        if conn.in_transaction:
            raise RuntimeError("MemoryWriteService requires an idle connection")
        conn.execute("PRAGMA foreign_keys = ON")
        if int(conn.execute("PRAGMA foreign_keys").fetchone()[0]) != 1:
            raise RuntimeError("MemoryWriteService requires foreign key enforcement")

    def write(
        self,
        context: ConnectionContext,
        request: WriteRequest,
        *,
        state_candidate: Optional[StateCandidate] = None,
        _manage_transaction: bool = True,
        _near_dedup_query: object = _UNSET,
    ) -> WriteOutcome:
        if not isinstance(_manage_transaction, bool):
            raise TypeError("_manage_transaction must be a boolean")
        if _manage_transaction and self._conn.in_transaction:
            raise RuntimeError("MemoryWriteService requires an idle connection")
        if not _manage_transaction and not self._conn.in_transaction:
            raise RuntimeError(
                "caller-managed MemoryWriteService write requires a transaction"
            )
        context = self._policy.authorize_context(context)
        self._validate_state_candidate(request, state_candidate)
        extracted_typed = self._extracted_typed_fields(request, state_candidate)
        if _near_dedup_query is _UNSET:
            if not _manage_transaction:
                raise RuntimeError(
                    "caller-managed write requires a prepared near-dedup query"
                )
            near_dedup_query = self._prepare_near_dedup_query(
                context, request, state_candidate
            )
        else:
            near_dedup_query = _near_dedup_query

        request_hash = _request_sha256(context, request, state_candidate)
        recorded_at = _now()
        try:
            if _manage_transaction:
                self._conn.execute("BEGIN IMMEDIATE")
            self._register_client(context, recorded_at)
            self._register_session(context, recorded_at)
            prior = self._load_prior(context.client_id, request.idempotency_key)
            if prior is not None:
                if prior["request_sha256"] != request_hash:
                    raise IdempotencyConflict(
                        "idempotency key was already used for a different request"
                    )
                outcome = self._outcome_from_row(prior, replayed=True)
                if _manage_transaction:
                    self._conn.commit()
                return outcome
            sensitive_fields = ()
            if state_candidate is not None:
                sensitive_fields = tuple(
                    value
                    for value in (
                        state_candidate.subject_key,
                        state_candidate.predicate_key,
                        state_candidate.object_value,
                    )
                    if value
                )
            decision = self._policy.evaluate_write(
                request,
                client_id=context.client_id,
                sensitive_fields=sensitive_fields,
            )
            if decision is None and not scope_authorized(
                request.scope, context.access_scopes
            ):
                decision = PolicyDecision(
                    "rejected", "requested write scope is not server-authorized"
                )
            if (
                decision is None
                and request.sensitivity == "sensitive"
                and "sensitive" not in context.access_scopes
            ):
                decision = PolicyDecision(
                    "rejected", "sensitive durable write is not authorized"
                )
            if decision is None:
                decision = self._supersession_policy(request)
            # A factless policy decision must not inspect a secret or
            # unauthorized state slot.  The candidate remains part of the
            # idempotency hash above, but slot lookup is deferred until the
            # write itself is authorized.
            effective_candidate = state_candidate
            if (
                effective_candidate is not None
                and effective_candidate.valid_from is None
                and request.observed_at
                and request.source_type != "automatic_extraction"
            ):
                effective_candidate = replace(
                    effective_candidate,
                    valid_from=request.observed_at,
                )
            state_decision = (
                decide_state_write(self._conn, effective_candidate)
                if decision is None and state_candidate is not None
                else None
            )
            candidate_is_human = (
                request.source_type == "human_correction"
                or request.relation == "corrects"
                or request.correction_status
                in {"human_corrected", "human_confirmed"}
            )
            can_resolve_conflicts = self._policy.can_resolve_conflicts(
                context.client_id
            )
            if (
                decision is None
                and state_decision is not None
                and state_decision.action == "conflict"
                and candidate_is_human
                and can_resolve_conflicts
            ):
                placeholders = ",".join(
                    "?" for _ in state_decision.current_fact_ids
                )
                groups = {
                    row[0]
                    for row in self._conn.execute(
                        f"SELECT conflict_group FROM facts "
                        f"WHERE fact_id IN ({placeholders})",
                        state_decision.current_fact_ids,
                    ).fetchall()
                    if row[0] is not None
                }
                if len(groups) == 1:
                    state_decision = replace(
                        state_decision,
                        action="supersede",
                        target_fact_id=None,
                        reason="explicit human correction resolves open conflict",
                    )
            if (
                state_decision is not None
                and state_decision.action == "supersede"
                and state_decision.target_fact_id is None
                and not can_resolve_conflicts
            ):
                state_decision = replace(
                    state_decision,
                    action="conflict",
                    reason="client is not authorized to resolve open conflict",
                )
            if (
                decision is None
                and state_decision is not None
                and state_decision.action == "supersede"
                and state_decision.target_fact_id is None
                and not candidate_is_human
                and effective_candidate is not None
            ):
                for fact_id in state_decision.current_fact_ids:
                    decision = self._supersession_policy(
                        request,
                        target_id=fact_id,
                        candidate_authority=effective_candidate.source_authority,
                    )
                    if decision is not None:
                        break
            if (
                decision is None
                and state_decision is not None
                and state_decision.action == "supersede"
            ):
                decision = self._supersession_policy(
                    request,
                    target_id=state_decision.target_fact_id,
                    candidate_authority=effective_candidate.source_authority,
                )
            if decision is not None:
                if _factless_decision_is_durable(decision):
                    outcome = self._record_factless_decision(
                        context, request, request_hash, recorded_at, decision
                    )
                else:
                    outcome = _ephemeral_factless_outcome(decision)
                if _manage_transaction:
                    self._conn.commit()
                return outcome
            observation_id = self._record_observation(context, request, recorded_at)
            conflict_id: Optional[str] = None
            untyped_duplicate = (
                self._find_untyped_exact_duplicate(request)
                if state_candidate is None
                else None
            )
            untyped_contradiction = (
                self._find_untyped_contradiction(request)
                if state_candidate is None
                and extracted_typed is None
                and request.supersede_fact_id is None
                and untyped_duplicate is None
                else None
            )
            near_duplicate = (
                self._find_untyped_near_duplicate(request, near_dedup_query)
                if state_candidate is None
                and extracted_typed is None
                and request.supersede_fact_id is None
                and untyped_duplicate is None
                and untyped_contradiction is None
                else None
            )
            if near_duplicate is not None:
                converted = self._contradiction_from_near_duplicate(
                    request, near_duplicate
                )
                if converted is not None:
                    untyped_contradiction = converted
                    near_duplicate = None
            enqueue_fact_id: Optional[int] = None
            if (
                state_decision is not None
                and state_decision.action in {"dedup", "conflict"}
                and state_decision.target_fact_id is not None
            ):
                if state_decision.action == "conflict":
                    row = self._conn.execute(
                        "SELECT conflict_group FROM facts WHERE fact_id = ?",
                        (state_decision.target_fact_id,),
                    ).fetchone()
                    conflict_id = str(row[0]) if row and row[0] else None
                fact_result = FactWriteResult(
                    state_decision.target_fact_id,
                    outcome=state_decision.action,
                    existing_fact_id=state_decision.target_fact_id,
                )
            elif untyped_duplicate is not None:
                fact_result = FactWriteResult(
                    untyped_duplicate,
                    outcome="dedup",
                    existing_fact_id=untyped_duplicate,
                )
            else:
                if state_decision is not None:
                    conflict_id = self._prepare_state_mutation(
                        state_decision,
                        recorded_at,
                        close_valid_to=(
                            effective_candidate.valid_from
                            if effective_candidate is not None
                            else None
                        ),
                    )
                fact_result = self._fact_writer(self._conn, request, observation_id)
                if near_duplicate is not None:
                    fact_result, enqueue_fact_id = self._merge_near_duplicate(
                        fact_result, near_duplicate, request, recorded_at
                    )
                else:
                    enqueue_fact_id = fact_result.fact_id
                if untyped_contradiction is not None:
                    conflict_id = self._open_untyped_write_conflict(
                        untyped_contradiction,
                        fact_result.fact_id,
                        request,
                        recorded_at,
                    )
                    fact_result = FactWriteResult(
                        fact_result.fact_id,
                        outcome="conflict",
                        existing_fact_id=untyped_contradiction.fact_id,
                        detail_json=fact_result.detail_json,
                    )
                if effective_candidate is not None and state_decision is not None:
                    self._persist_state_candidate(
                        fact_result,
                        effective_candidate,
                        conflict_id,
                        confidence=(
                            extracted_typed.confidence
                            if extracted_typed is not None
                            else None
                        ),
                    )
                    if state_decision.action == "supersede":
                        if conflict_id is None:
                            self._finish_state_supersession(
                                state_decision.target_fact_id, fact_result.fact_id
                            )
                        else:
                            add_conflict_member(
                                self._conn, conflict_id, fact_result.fact_id
                            )
                            resolve_state_conflict(
                                self._conn,
                                conflict_id,
                                fact_result.fact_id,
                                resolved_by=(
                                    request.asserted_by
                                    or request.performed_by
                                    or context.agent_id
                                ),
                                reason=state_decision.reason,
                                resolved_at=recorded_at,
                                resolver_client_id=context.client_id,
                                resolver_session_id=context.session_id,
                                resolver_agent_id=context.agent_id,
                            )
                    elif state_decision.action == "conflict":
                        if conflict_id is None:
                            raise RuntimeError(
                                "state conflict group was not established"
                            )
                        add_conflict_member(
                            self._conn, conflict_id, fact_result.fact_id
                        )
                    fact_result = FactWriteResult(
                        fact_result.fact_id,
                        outcome=state_decision.action,
                        existing_fact_id=fact_result.existing_fact_id,
                        detail_json=fact_result.detail_json,
                    )
                elif extracted_typed is not None:
                    self._persist_nonstate_extracted_typed_fields(
                        fact_result, extracted_typed
                    )
            if enqueue_fact_id is not None and self._embedding_enqueue is not None:
                self._embedding_enqueue(enqueue_fact_id)
            self._authorize_fact_scope(fact_result.fact_id, request.scope)
            self._attach_provenance(
                fact_result.fact_id, observation_id, request, recorded_at
            )
            superseded = False
            if state_candidate is None:
                superseded = self._supersede_if_requested(
                    request.supersede_fact_id, fact_result.fact_id, recorded_at
                )

            detail = json.loads(fact_result.detail_json)
            if near_duplicate is not None:
                detail["near_duplicate"] = {
                    "candidate_fact_id": near_duplicate.fact_id,
                    "cosine": round(near_duplicate.cosine, 6),
                    "survivor_fact_id": fact_result.fact_id,
                }
            if superseded:
                detail["superseded_fact_id"] = request.supersede_fact_id
            if state_decision is not None:
                detail["state_action"] = state_decision.action
                detail["state_slot"] = {
                    "scope": state_decision.scope,
                    "subject_key": state_decision.subject_key,
                    "predicate_key": state_decision.predicate_key,
                }
                if state_decision.action == "supersede":
                    detail["superseded_fact_id"] = state_decision.target_fact_id
                if conflict_id is not None:
                    detail["conflict_id"] = conflict_id
            elif untyped_contradiction is not None and conflict_id is not None:
                detail["conflict_id"] = conflict_id
                detail["member_fact_ids"] = sorted(
                    {untyped_contradiction.fact_id, fact_result.fact_id}
                )
                detail["conflict_slot"] = {
                    "scope": request.scope,
                    "subject_key": untyped_contradiction.subject_key,
                    "predicate_key": untyped_contradiction.predicate_key,
                }
            detail_json = json.dumps(detail, sort_keys=True, separators=(",", ":"))
            write_id = str(uuid.uuid4())
            self._conn.execute(
                """
                INSERT INTO memory_write_log (
                    write_id, idempotency_key, client_id, session_id,
                    operation, outcome, fact_id, existing_fact_id,
                    observation_id, recorded_at, request_sha256, detail_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    write_id,
                    request.idempotency_key,
                    context.client_id,
                    context.session_id,
                    request.operation,
                    fact_result.outcome,
                    fact_result.fact_id,
                    fact_result.existing_fact_id,
                    observation_id,
                    recorded_at,
                    request_hash,
                    detail_json,
                ),
            )
            outcome = WriteOutcome(
                write_id=write_id,
                outcome=fact_result.outcome,
                fact_id=fact_result.fact_id,
                existing_fact_id=fact_result.existing_fact_id,
                observation_id=observation_id,
                detail_json=detail_json,
            )
            if _manage_transaction:
                self._conn.commit()
            return outcome
        except BaseException:
            if _manage_transaction and self._conn.in_transaction:
                self._conn.rollback()
            raise

    def write_batch(
        self,
        context: ConnectionContext,
        writes: Sequence[tuple[WriteRequest, Optional[StateCandidate]]],
        *,
        rollback_if: Callable[[WriteOutcome], bool] | None = None,
        before_commit: Callable[[], None] | None = None,
    ) -> WriteBatchOutcome:
        """Apply an ordered write batch in one owned transaction.

        ``rollback_if`` lets an authoritative caller turn a factless policy
        outcome into an all-or-nothing batch rejection without compensating
        deletes or leaking the rejected write into the durable write log.
        """

        if self._conn.in_transaction:
            raise RuntimeError("MemoryWriteService requires an idle connection")
        if isinstance(writes, (str, bytes)) or not isinstance(writes, Sequence):
            raise TypeError("writes must be a sequence")
        if not writes:
            raise ValueError("writes must not be empty")
        if rollback_if is not None and not callable(rollback_if):
            raise TypeError("rollback_if must be callable")
        if before_commit is not None and not callable(before_commit):
            raise TypeError("before_commit must be callable")
        context = self._policy.authorize_context(context)
        normalized: list[tuple[WriteRequest, Optional[StateCandidate], object]] = []
        for item in writes:
            if not isinstance(item, tuple) or len(item) != 2:
                raise TypeError("each batch item must be a request/candidate tuple")
            request, candidate = item
            if not isinstance(request, WriteRequest):
                raise TypeError("batch request must be a WriteRequest")
            if candidate is not None and not isinstance(candidate, StateCandidate):
                raise TypeError("batch candidate must be a StateCandidate or None")
            query = self._prepare_near_dedup_query(context, request, candidate)
            normalized.append((request, candidate, query))

        try:
            self._conn.execute("BEGIN IMMEDIATE")
            outcomes: list[WriteOutcome] = []
            for request, candidate, query in normalized:
                outcome = self.write(
                    context,
                    request,
                    state_candidate=candidate,
                    _manage_transaction=False,
                    _near_dedup_query=query,
                )
                outcomes.append(outcome)
                if rollback_if is not None and rollback_if(outcome):
                    self._conn.rollback()
                    return WriteBatchOutcome(tuple(outcomes), committed=False)
            if before_commit is not None:
                before_commit()
            self._conn.commit()
            return WriteBatchOutcome(tuple(outcomes), committed=True)
        except BaseException:
            if self._conn.in_transaction:
                self._conn.rollback()
            raise

    @staticmethod
    def _validate_state_candidate(
        request: WriteRequest, candidate: Optional[StateCandidate]
    ) -> None:
        if candidate is None:
            return
        if candidate.memory_kind != "state":
            raise ValueError("state_candidate must have memory_kind='state'")
        if candidate.content != request.content:
            raise ValueError("state candidate content must match the write request")
        if candidate.scope != request.scope:
            raise ValueError("state candidate scope must match the write request")
        if candidate.source_authority != request.source_authority:
            raise ValueError("state candidate authority must match the write request")
        if request.supersede_fact_id is not None:
            raise ValueError(
                "typed state writes derive supersession from the exact slot"
            )

    @staticmethod
    def _extracted_typed_fields(
        request: WriteRequest, state_candidate: Optional[StateCandidate]
    ) -> Optional[ExtractedTypedFields]:
        """Decode only processor-produced typed metadata at the write boundary."""

        if request.source_type != "automatic_extraction":
            return None
        metadata = json.loads(request.metadata_json)
        value = metadata.get("extraction_typed")
        if value is None:
            return None
        if not isinstance(value, dict) or set(value) != {
            "confidence",
            "kind",
            "negation",
            "object_value",
            "predicate_key",
            "subject_key",
            "valid_from",
        }:
            raise ValueError("automatic extraction typed metadata is invalid")
        kind = value["kind"]
        confidence = value["confidence"]
        object_value = value["object_value"]
        valid_from = value["valid_from"]
        if (
            kind not in {"state", "preference", "commitment", "event"}
            or isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not 0.0 <= float(confidence) <= 1.0
            or not isinstance(value["negation"], bool)
            or (
                object_value is not None
                and (not isinstance(object_value, str) or not object_value.strip())
            )
            or (
                valid_from is not None
                and (not isinstance(valid_from, str) or not valid_from.strip())
            )
        ):
            raise ValueError("automatic extraction typed metadata is invalid")
        try:
            # Reuse the state-slot value object for canonical keys and ISO
            # timestamp validation without assigning slot semantics here.
            normalized = StateCandidate(
                content=request.content,
                subject_key=value["subject_key"],
                predicate_key=value["predicate_key"],
                object_value=object_value,
                valid_from=valid_from,
                scope=request.scope,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("automatic extraction typed metadata is invalid") from exc
        fields = ExtractedTypedFields(
            kind=kind,
            subject_key=normalized.subject_key,
            predicate_key=normalized.predicate_key,
            object_value=normalized.object_value,
            valid_from=normalized.valid_from,
            confidence=float(confidence),
        )
        if fields.kind == "state":
            if state_candidate is None or (
                state_candidate.subject_key,
                state_candidate.predicate_key,
                state_candidate.object_value,
            ) != (
                fields.subject_key,
                fields.predicate_key,
                fields.object_value,
            ):
                raise ValueError(
                    "automatic extracted state must use its exact state slot"
                )
        elif state_candidate is not None:
            raise ValueError("only extracted state may use state-slot semantics")
        return fields

    def _prepare_state_mutation(
        self,
        decision: SlotDecision,
        now: str,
        *,
        close_valid_to: Optional[str] = None,
    ) -> Optional[str]:
        if decision.action == "supersede":
            if decision.target_fact_id is None:
                placeholders = ",".join("?" for _ in decision.current_fact_ids)
                groups = {
                    row[0]
                    for row in self._conn.execute(
                        f"SELECT conflict_group FROM facts "
                        f"WHERE fact_id IN ({placeholders})",
                        decision.current_fact_ids,
                    ).fetchall()
                    if row[0] is not None
                }
                if len(groups) != 1:
                    raise RuntimeError(
                        "conflict resolution must target one unresolved conflict"
                    )
                return str(next(iter(groups)))
            columns = {
                str(row[1]) for row in self._conn.execute("PRAGMA table_info(facts)")
            }
            assignments = ["invalid_at = ?"]
            values: list[object] = [now]
            if "expired_at" in columns:
                assignments.append("expired_at = ?")
                values.append(now)
            if close_valid_to is not None and "valid_to" in columns:
                assignments.append("valid_to = COALESCE(valid_to, ?)")
                values.append(close_valid_to)
            values.extend([decision.target_fact_id, decision.scope])
            cursor = self._conn.execute(
                f"""
                UPDATE facts SET {", ".join(assignments)}
                WHERE fact_id = ? AND scope = ?
                  AND invalid_at IS NULL AND superseded_by IS NULL
                  AND conflict_group IS NULL
                """,
                values,
            )
            if cursor.rowcount != 1:
                raise RuntimeError("state supersession target is no longer current")
            return None
        if decision.action != "conflict":
            return None

        placeholders = ",".join("?" for _ in decision.current_fact_ids)
        groups = {
            row[0]
            for row in self._conn.execute(
                f"SELECT conflict_group FROM facts WHERE fact_id IN ({placeholders})",
                decision.current_fact_ids,
            ).fetchall()
            if row[0] is not None
        }
        if groups:
            if len(groups) != 1:
                raise RuntimeError("state slot spans multiple unresolved conflicts")
            return str(next(iter(groups)))
        conflict = open_state_conflict(
            self._conn,
            decision.subject_key,
            decision.predicate_key,
            decision.current_fact_ids,
            scope=decision.scope,
            detected_at=now,
            detail_json=json.dumps({"reason": decision.reason}),
        )
        return conflict.conflict_id

    def _persist_state_candidate(
        self,
        result: FactWriteResult,
        candidate: StateCandidate,
        conflict_id: Optional[str],
        *,
        confidence: Optional[float] = None,
    ) -> None:
        if result.existing_fact_id is not None or result.outcome not in {
            "inserted",
            "add",
        }:
            raise RuntimeError(
                "typed state creation requires the fact writer to insert a new fact"
            )
        columns = {
            str(row[1]) for row in self._conn.execute("PRAGMA table_info(facts)")
        }
        assignments = [
            "memory_kind = 'state'",
            "subject_key = ?",
            "predicate_key = ?",
            "object_value = ?",
            "source_authority = ?",
            "valid_from = ?",
            "scope = ?",
            "conflict_group = ?",
            "confidence = ?",
        ]
        values: list[object] = [
            candidate.subject_key,
            candidate.predicate_key,
            candidate.object_value,
            candidate.source_authority,
            candidate.valid_from,
            candidate.scope,
            conflict_id,
            confidence,
        ]
        if "valid_to" in columns:
            assignments.append("valid_to = ?")
            values.append(candidate.valid_to)
            if candidate.valid_to is not None:
                assignments.append("invalid_at = ?")
                values.append(_now())
        values.append(result.fact_id)
        cursor = self._conn.execute(
            f"""
            UPDATE facts
            SET {", ".join(assignments)}
            WHERE fact_id = ? AND invalid_at IS NULL AND superseded_by IS NULL
            """,
            values,
        )
        if cursor.rowcount != 1:
            raise RuntimeError("fact writer returned an unusable state fact")

    def _persist_nonstate_extracted_typed_fields(
        self, result: FactWriteResult, fields: ExtractedTypedFields
    ) -> None:
        if fields.kind == "state":
            raise RuntimeError("state extraction must use state-slot persistence")
        if result.existing_fact_id is not None or result.outcome not in {
            "inserted",
            "add",
        }:
            raise RuntimeError(
                "typed extraction persistence requires a newly inserted fact"
            )
        columns = {row[1] for row in self._conn.execute("PRAGMA table_info(facts)")}
        required = {
            "memory_kind",
            "subject_key",
            "predicate_key",
            "object_value",
            "confidence",
            "valid_from",
        }
        if not required.issubset(columns):
            raise RuntimeError("facts schema cannot persist typed extraction fields")
        cursor = self._conn.execute(
            """
            UPDATE facts
            SET memory_kind = ?, subject_key = ?, predicate_key = ?,
                object_value = ?, confidence = ?, valid_from = ?
            WHERE fact_id = ? AND invalid_at IS NULL AND superseded_by IS NULL
            """,
            (
                fields.kind,
                fields.subject_key,
                fields.predicate_key,
                fields.object_value,
                fields.confidence,
                fields.valid_from,
                result.fact_id,
            ),
        )
        if cursor.rowcount != 1:
            raise RuntimeError("fact writer returned an unusable typed fact")

    def _finish_state_supersession(
        self, old_fact_id: Optional[int], new_fact_id: int
    ) -> None:
        if old_fact_id is None:
            raise RuntimeError("supersession decision has no target")
        cursor = self._conn.execute(
            """
            UPDATE facts SET superseded_by = ?
            WHERE fact_id = ? AND invalid_at IS NOT NULL
              AND superseded_by IS NULL
            """,
            (new_fact_id, old_fact_id),
        )
        if cursor.rowcount != 1:
            raise RuntimeError("could not finalize state supersession")

    def _record_factless_decision(
        self,
        context: ConnectionContext,
        request: WriteRequest,
        request_hash: str,
        recorded_at: str,
        decision: PolicyDecision,
    ) -> WriteOutcome:
        write_id = str(uuid.uuid4())
        detail_json = json.dumps(
            {"policy_reason": decision.reason},
            sort_keys=True,
            separators=(",", ":"),
        )
        self._conn.execute(
            """
            INSERT INTO memory_write_log (
                write_id, idempotency_key, client_id, session_id,
                operation, outcome, fact_id, existing_fact_id,
                observation_id, recorded_at, request_sha256, detail_json
            ) VALUES (?, ?, ?, ?, ?, ?, NULL, NULL, NULL, ?, ?, ?)
            """,
            (
                write_id,
                request.idempotency_key,
                context.client_id,
                context.session_id,
                request.operation,
                decision.outcome,
                recorded_at,
                request_hash,
                detail_json,
            ),
        )
        return WriteOutcome(
            write_id=write_id,
            outcome=decision.outcome,
            fact_id=None,
            detail_json=detail_json,
        )

    def _supersession_policy(
        self,
        request: WriteRequest,
        *,
        target_id: Optional[int] = None,
        candidate_authority: Optional[float] = None,
    ) -> Optional[PolicyDecision]:
        """Prevent automation from silently replacing protected truth."""

        target_id = target_id or request.supersede_fact_id
        if target_id is None:
            return None
        columns = {row[1] for row in self._conn.execute("PRAGMA table_info(facts)")}
        selected = ["fact_id"]
        selected.append(
            "COALESCE(source_authority, 0.5)"
            if "source_authority" in columns
            else "0.5"
        )
        selected.append(
            "correction_status" if "correction_status" in columns else "NULL"
        )
        selected.append("memory_kind" if "memory_kind" in columns else "NULL")
        selected.append("conflict_group" if "conflict_group" in columns else "NULL")
        row = self._conn.execute(
            f"SELECT {', '.join(selected)} FROM facts WHERE fact_id = ? AND scope = ?",
            (target_id, request.scope),
        ).fetchone()
        if row is None:
            return PolicyDecision("needs_review", "supersession target is unavailable")
        target_authority = float(row[1])
        correction_status = row[2]
        if request.supersede_fact_id is not None:
            if row[3] == "state":
                return PolicyDecision(
                    "needs_review", "typed state requires state-slot supersession"
                )
            if row[4] is not None:
                return PolicyDecision(
                    "needs_review",
                    "open-conflict members cannot be explicitly superseded",
                )
        candidate_is_human = (
            request.source_type == "human_correction"
            or request.relation == "corrects"
            or request.correction_status in {"human_corrected", "human_confirmed"}
        )
        if (
            correction_status in {"human_corrected", "human_confirmed"}
            and not candidate_is_human
        ):
            return PolicyDecision(
                "needs_review", "target is protected by human correction"
            )
        authority = (
            request.source_authority
            if candidate_authority is None
            else candidate_authority
        )
        if target_authority > authority:
            return PolicyDecision("needs_review", "target has higher source authority")
        return None

    def _find_untyped_exact_duplicate(self, request: WriteRequest) -> Optional[int]:
        """Find only an exact active duplicate inside the authorized write scope.

        This deliberately avoids semantic guesses at the write boundary and
        includes the scope predicate in the lookup, so neither result nor
        timing depends on facts the caller cannot access.
        """

        columns = {row[1] for row in self._conn.execute("PRAGMA table_info(facts)")}
        required = {"scope", "invalid_at", "superseded_by", "conflict_group"}
        if not required.issubset(columns):
            return None
        row = self._conn.execute(
            """
            SELECT fact_id FROM facts
            WHERE scope = ? AND content = ?
              AND invalid_at IS NULL AND superseded_by IS NULL
              AND conflict_group IS NULL
            ORDER BY fact_id LIMIT 1
            """,
            (request.scope, request.content),
        ).fetchone()
        return int(row[0]) if row is not None else None

    def _untyped_conflict_schema_ready(self) -> bool:
        tables = {
            str(row[0])
            for row in self._conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        if "fact_conflicts" not in tables or "fact_conflict_members" not in tables:
            return False
        columns = {str(row[1]) for row in self._conn.execute("PRAGMA table_info(facts)")}
        return {"scope", "invalid_at", "superseded_by", "conflict_group"}.issubset(
            columns
        )

    def _load_untyped_current_facts(self, scope: str) -> list[tuple[int, str]]:
        columns = {str(row[1]) for row in self._conn.execute("PRAGMA table_info(facts)")}
        kind_pred = ""
        if "memory_kind" in columns:
            kind_pred = "AND (memory_kind IS NULL OR memory_kind != 'state')"
        rows = self._conn.execute(
            f"""
            SELECT fact_id, content FROM facts
            WHERE scope = ?
              AND invalid_at IS NULL AND superseded_by IS NULL
              AND conflict_group IS NULL
              {kind_pred}
            ORDER BY fact_id
            """,
            (scope,),
        ).fetchall()
        return [(int(row[0]), str(row[1])) for row in rows]

    def _find_untyped_contradiction(
        self, request: WriteRequest
    ) -> Optional[_UntypedContradiction]:
        if not self._untyped_conflict_schema_ready():
            return None
        matches: list[_UntypedContradiction] = []
        for fact_id, content in self._load_untyped_current_facts(request.scope):
            if content == request.content:
                continue
            if not untyped_contents_contradict(request.content, content):
                continue
            try:
                subject_key, predicate_key = guess_untyped_conflict_slot(
                    request.content, content
                )
            except ValueError:
                continue
            matches.append(
                _UntypedContradiction(fact_id, content, subject_key, predicate_key)
            )
        if not matches:
            return None
        return max(
            matches,
            key=lambda item: (
                len(_content_tokens(request.content) & _content_tokens(item.content)),
                -item.fact_id,
            ),
        )

    def _contradiction_from_near_duplicate(
        self, request: WriteRequest, candidate: NearDuplicateCandidate
    ) -> Optional[_UntypedContradiction]:
        if not self._untyped_conflict_schema_ready():
            return None
        row = self._conn.execute(
            "SELECT content FROM facts WHERE fact_id = ?",
            (candidate.fact_id,),
        ).fetchone()
        if row is None:
            return None
        content = str(row[0])
        if not untyped_contents_contradict(request.content, content):
            return None
        try:
            subject_key, predicate_key = guess_untyped_conflict_slot(
                request.content, content
            )
        except ValueError:
            return None
        return _UntypedContradiction(
            candidate.fact_id, content, subject_key, predicate_key
        )

    def _open_untyped_write_conflict(
        self,
        match: _UntypedContradiction,
        new_fact_id: int,
        request: WriteRequest,
        recorded_at: str,
    ) -> str:
        conflict = open_untyped_conflict(
            self._conn,
            (match.fact_id,),
            scope=request.scope,
            subject_key=match.subject_key,
            predicate_key=match.predicate_key,
            detected_at=recorded_at,
            detail_json=json.dumps({"reason": "untyped contradiction"}),
        )
        columns = {str(row[1]) for row in self._conn.execute("PRAGMA table_info(facts)")}
        assignments = ["conflict_group = ?"]
        values: list[object] = [conflict.conflict_id]
        if "subject_key" in columns:
            assignments.append("subject_key = COALESCE(subject_key, ?)")
            values.append(match.subject_key)
        if "predicate_key" in columns:
            assignments.append("predicate_key = COALESCE(predicate_key, ?)")
            values.append(match.predicate_key)
        values.extend([new_fact_id, request.scope])
        cursor = self._conn.execute(
            f"""
            UPDATE facts SET {", ".join(assignments)}
            WHERE fact_id = ? AND scope = ?
              AND invalid_at IS NULL AND superseded_by IS NULL
              AND conflict_group IS NULL
            """,
            values,
        )
        if cursor.rowcount != 1:
            raise RuntimeError("untyped conflict member is no longer current")
        add_untyped_conflict_member(self._conn, conflict.conflict_id, new_fact_id)
        return conflict.conflict_id

    def _prepare_near_dedup_query(
        self,
        context: ConnectionContext,
        request: WriteRequest,
        state_candidate: Optional[StateCandidate],
    ) -> object:
        """Compute external query embeddings before transaction ownership."""

        policy_decision = self._policy.evaluate_write(
            request, client_id=context.client_id
        )
        if (
            state_candidate is not None
            or request.supersede_fact_id is not None
            or policy_decision is not None
            or not scope_authorized(request.scope, context.access_scopes)
            or (
                request.sensitivity == "sensitive"
                and "sensitive" not in context.access_scopes
            )
            or self._supersession_policy(request) is not None
            or not self._near_dedup.enabled
            or self._query_embedder is None
            or self._near_dedup.embedding_identity is None
            or self._load_prior(context.client_id, request.idempotency_key) is not None
            or self._find_untyped_exact_duplicate(request) is not None
            or request.correction_status == "unreviewed"
        ):
            return None
        try:
            return np.asarray(self._query_embedder(request.content), dtype=np.float32)
        except (
            TypeError,
            ValueError,
            sqlite3.DatabaseError,
            StoredEmbeddingError,
            urllib.error.URLError,
            TimeoutError,
            OSError,
        ):
            return None

    def _find_untyped_near_duplicate(
        self, request: WriteRequest, query_embedding: object
    ) -> Optional[NearDuplicateCandidate]:
        """Return the strongest safe FTS-bounded embedding match, if available."""
        if (
            query_embedding is None
            or not self._near_dedup.enabled
            or self._query_embedder is None
            or self._near_dedup.embedding_identity is None
        ):
            return None
        try:
            candidates = find_write_near_duplicates(
                self._conn,
                content=request.content,
                scope=request.scope,
                query_embedding=query_embedding,
                threshold=self._near_dedup.cosine_threshold,
                candidate_limit=self._near_dedup.candidate_limit,
                embedding_identity=self._near_dedup.embedding_identity,
            )
        except (TypeError, ValueError, sqlite3.DatabaseError):
            # The async embedding job may not have completed yet, or a legacy
            # store may not expose FTS/vector tables. Exact dedup remains safe.
            return None
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda item: (item.trust_score, item.created_at, item.fact_id),
        )

    def _merge_near_duplicate(
        self,
        inserted: FactWriteResult,
        candidate: NearDuplicateCandidate,
        request: WriteRequest,
        recorded_at: str,
    ) -> tuple[FactWriteResult, Optional[int]]:
        """Keep one fact active and retain the other in its history chain."""
        incoming_wins = not self._near_duplicate_candidate_is_protected(
            candidate.fact_id, request
        ) and (request.trust_score, recorded_at) > (
            candidate.trust_score,
            candidate.created_at,
        )
        if incoming_wins:
            self._supersede_near_duplicate(
                candidate.fact_id, inserted.fact_id, recorded_at
            )
            return (
                FactWriteResult(
                    inserted.fact_id,
                    outcome="near_dedup",
                    existing_fact_id=candidate.fact_id,
                    detail_json=inserted.detail_json,
                ),
                inserted.fact_id,
            )
        self._supersede_near_duplicate(inserted.fact_id, candidate.fact_id, recorded_at)
        return (
            FactWriteResult(
                candidate.fact_id,
                outcome="near_dedup",
                existing_fact_id=candidate.fact_id,
                detail_json=inserted.detail_json,
            ),
            None,
        )

    def _near_duplicate_candidate_is_protected(
        self, fact_id: int, request: WriteRequest
    ) -> bool:
        """Keep corrections and stronger sources out of implicit supersession.

        Near-dedup is a lossy similarity convenience, never an authority
        resolver.  Deliberate corrections and higher-authority claims require
        an explicit resolution path even when a new paraphrase scores highly.
        """

        columns = {row[1] for row in self._conn.execute("PRAGMA table_info(facts)")}
        selected = ["fact_id"]
        selected.append(
            "correction_status" if "correction_status" in columns else "NULL"
        )
        selected.append(
            "source_authority" if "source_authority" in columns else "NULL"
        )
        selected.append("memory_kind" if "memory_kind" in columns else "NULL")
        row = self._conn.execute(
            f"SELECT {', '.join(selected)} FROM facts "
            "WHERE fact_id = ? AND invalid_at IS NULL AND superseded_by IS NULL",
            (fact_id,),
        ).fetchone()
        if row is None:
            raise RuntimeError("near-duplicate candidate is no longer active")
        if row[3] == "state" or row[1] in {"human_corrected", "human_confirmed"}:
            return True
        try:
            candidate_authority = float(row[2]) if row[2] is not None else 0.0
        except (TypeError, ValueError):
            # A malformed legacy value must never weaken an implicit write.
            return True
        return candidate_authority > request.source_authority

    def _supersede_near_duplicate(
        self, loser_id: int, survivor_id: int, recorded_at: str
    ) -> None:
        columns = {row[1] for row in self._conn.execute("PRAGMA table_info(facts)")}
        expired_assignment = ", expired_at = ?" if "expired_at" in columns else ""
        values = (
            (recorded_at, recorded_at, survivor_id, loser_id)
            if expired_assignment
            else (recorded_at, survivor_id, loser_id)
        )
        cursor = self._conn.execute(
            f"""
            UPDATE facts SET invalid_at = ?{expired_assignment}, superseded_by = ?
            WHERE fact_id = ? AND invalid_at IS NULL AND superseded_by IS NULL
            """,
            values,
        )
        if cursor.rowcount != 1:
            raise RuntimeError("near-duplicate candidate is no longer active")

    def _register_client(self, context: ConnectionContext, now: str) -> None:
        self._conn.execute(
            """
            INSERT INTO memory_clients (
                client_id, surface, display_name, created_at
            ) VALUES (?, ?, ?, ?)
            ON CONFLICT(client_id) DO NOTHING
            """,
            (
                context.client_id,
                context.surface,
                context.display_name,
                now,
            ),
        )
        row = self._conn.execute(
            "SELECT surface, disabled_at FROM memory_clients WHERE client_id = ?",
            (context.client_id,),
        ).fetchone()
        if row is None:
            raise RuntimeError("failed to register memory client")
        if row[0] != context.surface:
            raise ClientIdentityConflict(
                "client_id is already registered to a different surface"
            )
        if row[1] is not None:
            raise PermissionError("memory client is disabled")

    def _register_session(self, context: ConnectionContext, now: str) -> None:
        capabilities_json = json.dumps(context.capabilities, separators=(",", ":"))
        access_scopes_json = json.dumps(context.access_scopes, separators=(",", ":"))
        self._conn.execute(
            """
            INSERT INTO memory_sessions (
                session_id, client_id, agent_id, parent_agent_id, project_root,
                repository, branch, commit_sha, capabilities_json,
                access_scopes_json, started_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(client_id, session_id) DO NOTHING
            """,
            (
                context.session_id,
                context.client_id,
                context.agent_id,
                context.parent_agent_id,
                context.project_root,
                context.repository,
                context.branch,
                context.commit_sha,
                capabilities_json,
                access_scopes_json,
                context.started_at or now,
            ),
        )
        # Branch and commit can advance during one long-running coding
        # session. Keep their latest values here; every observation below
        # records the exact values present for its write.
        self._conn.execute(
            """
            UPDATE memory_sessions SET branch = ?, commit_sha = ?
            WHERE client_id = ? AND session_id = ?
            """,
            (context.branch, context.commit_sha, context.client_id, context.session_id),
        )
        row = self._conn.execute(
            """
            SELECT agent_id, parent_agent_id, project_root, repository,
                   capabilities_json, access_scopes_json
            FROM memory_sessions
            WHERE client_id = ? AND session_id = ?
            """,
            (context.client_id, context.session_id),
        ).fetchone()
        expected = (
            context.agent_id,
            context.parent_agent_id,
            context.project_root,
            context.repository,
            capabilities_json,
            access_scopes_json,
        )
        if row is None or tuple(row) != expected:
            raise SessionContextConflict(
                "client_id/session_id was reused with different connection context"
            )

    def _record_observation(
        self,
        context: ConnectionContext,
        request: WriteRequest,
        now: str,
    ) -> int:
        content_hash = _observation_sha256(context, request)
        self._conn.execute(
            """
            INSERT INTO observations (
                client_id, session_id, source_type, source_uri,
                project_root, repository, branch, commit_sha, content,
                content_sha256, asserted_by, performed_by, observed_at,
                recorded_at, scope, sensitivity, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(client_id, content_sha256, session_id, source_type)
            DO NOTHING
            """,
            (
                context.client_id,
                context.session_id,
                request.source_type,
                request.source_uri,
                context.project_root,
                context.repository,
                context.branch,
                context.commit_sha,
                request.observation_content or request.content,
                content_hash,
                request.asserted_by,
                request.performed_by,
                request.observed_at,
                now,
                request.scope,
                request.sensitivity,
                request.metadata_json,
            ),
        )
        row = self._conn.execute(
            """
            SELECT observation_id FROM observations
            WHERE client_id = ? AND content_sha256 = ?
              AND session_id = ? AND source_type = ?
            """,
            (
                context.client_id,
                content_hash,
                context.session_id,
                request.source_type,
            ),
        ).fetchone()
        if row is None:
            raise RuntimeError("failed to record observation")
        return int(row[0])

    def _attach_provenance(
        self,
        fact_id: int,
        observation_id: int,
        request: WriteRequest,
        now: str,
    ) -> None:
        self._conn.execute(
            """
            INSERT INTO fact_provenance (
                fact_id, observation_id, relation, evidence_excerpt, created_at
            ) VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(fact_id, observation_id, relation) DO NOTHING
            """,
            (
                fact_id,
                observation_id,
                request.relation,
                request.evidence_excerpt,
                now,
            ),
        )

    def _supersede_if_requested(
        self,
        old_fact_id: Optional[int],
        new_fact_id: int,
        now: str,
    ) -> bool:
        if old_fact_id is None:
            return False
        if old_fact_id == new_fact_id:
            raise ValueError("a fact cannot supersede itself")
        columns = {row[1] for row in self._conn.execute("PRAGMA table_info(facts)")}
        required = {"invalid_at", "superseded_by"}
        if not required.issubset(columns):
            raise RuntimeError(
                "structural supersession requires temporal facts columns"
            )
        request_scope = "private"
        if "scope" in columns:
            replacement = self._conn.execute(
                "SELECT scope FROM facts WHERE fact_id = ?", (new_fact_id,)
            ).fetchone()
            if replacement is None:
                raise RuntimeError("replacement fact is unavailable")
            request_scope = str(replacement[0])
            target = self._conn.execute(
                "SELECT scope FROM facts WHERE fact_id = ? AND scope = ?",
                (old_fact_id, request_scope),
            ).fetchone()
            if target is None:
                raise ValueError("superseded fact is unavailable or no longer current")
        memory_expr = "memory_kind" if "memory_kind" in columns else "NULL"
        conflict_expr = "conflict_group" if "conflict_group" in columns else "NULL"
        protected = self._conn.execute(
            f"SELECT {memory_expr}, {conflict_expr} FROM facts WHERE fact_id = ?",
            (old_fact_id,),
        ).fetchone()
        if protected is None:
            raise ValueError("superseded fact is unavailable or no longer current")
        if protected[0] == "state" or protected[1] is not None:
            raise ValueError(
                "typed or conflicted facts require their dedicated resolution path"
            )
        expired_assignment = ", expired_at = ?" if "expired_at" in columns else ""
        values = (
            (now, now, new_fact_id, old_fact_id, request_scope)
            if expired_assignment
            else (now, new_fact_id, old_fact_id, request_scope)
        )
        cursor = self._conn.execute(
            f"""
            UPDATE facts
            SET invalid_at = ?{expired_assignment}, superseded_by = ?
            WHERE fact_id = ? AND scope = ?
              AND invalid_at IS NULL AND superseded_by IS NULL
            """,
            values,
        )
        if cursor.rowcount != 1:
            raise ValueError("superseded fact is unavailable or no longer current")
        return True

    def _authorize_fact_scope(self, fact_id: int, requested_scope: str) -> None:
        columns = {row[1] for row in self._conn.execute("PRAGMA table_info(facts)")}
        if "scope" not in columns:
            if requested_scope != "private":
                raise RuntimeError("legacy facts schema cannot persist requested scope")
            return
        row = self._conn.execute(
            "SELECT scope FROM facts WHERE fact_id = ?", (fact_id,)
        ).fetchone()
        if row is None:
            raise RuntimeError("fact writer returned a missing fact")
        if row[0] != requested_scope:
            raise PermissionError("fact writer persisted a different memory scope")

    def _load_prior(self, client_id: str, idempotency_key: str):
        old_factory = self._conn.row_factory
        try:
            self._conn.row_factory = sqlite3.Row
            return self._conn.execute(
                """
                SELECT * FROM memory_write_log
                WHERE client_id = ? AND idempotency_key = ?
                """,
                (client_id, idempotency_key),
            ).fetchone()
        finally:
            self._conn.row_factory = old_factory

    @staticmethod
    def _outcome_from_row(row: sqlite3.Row, replayed: bool) -> WriteOutcome:
        return WriteOutcome(
            write_id=row["write_id"],
            outcome=row["outcome"],
            fact_id=row["fact_id"],
            existing_fact_id=row["existing_fact_id"],
            observation_id=row["observation_id"],
            replayed=replayed,
            detail_json=row["detail_json"],
        )
