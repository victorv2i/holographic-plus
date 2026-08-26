"""Scoped, provenance-aware Enfold request service.

This module is the storage router behind a transport adapter.  It owns no
socket and opens no database: callers pass a connection that has already been
explicitly migrated to Enfold schema v1.  The protocol context is trusted only
as an identity assertion from the daemon; server-side :class:`MemoryPolicy`
grants still narrow every read and write.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
import sqlite3
from typing import Any, Callable, Mapping, Sequence

from .context import TRUNCATION_MARKER, pack_context
from .core_store import build_visibility_predicate, insert_fact
from .extraction_enqueue import ExtractionEnqueuer, MAX_EXTRACTION_PAYLOAD_BYTES
from .extraction_spans import TranscriptInput, normalize_transcript
from .embedding_jobs import EmbeddingOutbox
from .hybrid_retrieval import RetrieverFactory, deterministic_retriever_factory
from .policy import (
    MemoryPolicy,
    UnknownMemoryClient,
    default_credential_screen,
    is_run_scope,
    scope_authorized,
    validate_scope,
)
from .protocol import (
    IMMUTABLE_CONTEXT_FIELDS,
    ClientContext,
    ProtocolValidationError,
    Request,
    RequestHandlingError,
    SUPPORTED_SCHEMA_VERSION,
    optional_as_of_timestamp,
)
from .provenance import ConnectionContext, WriteRequest
from .projections import changes, entities, entity_dossier, timeline
from .schema import require_compatible_schema
from .state_slots import (
    ConflictReceipt,
    StateCandidate,
    list_conflict_receipts,
    list_state_conflicts,
    resolve_state_conflict,
)
from .temporal import row_matches_as_of
from .write_service import (
    FactWriteResult,
    IdempotencyConflict,
    MemoryWriteService,
    NearDedupConfig,
)


class ServiceRequestError(RequestHandlingError):
    """Safe request failure for a transport adapter to serialize."""

    def __init__(self, code: str, message: str):
        super().__init__(code, message)


@dataclass(frozen=True, slots=True)
class AtomicWriteBatchResult:
    """Internal batch responses with an explicit commit decision."""

    responses: tuple[dict[str, Any], ...]
    committed: bool


_FACT_FIELDS = (
    "fact_id",
    "content",
    "category",
    "tags",
    "trust_score",
    "retrieval_count",
    "helpful_count",
    "created_at",
    "updated_at",
    "valid_from",
    "valid_to",
    "invalid_at",
    "expired_at",
    "superseded_by",
    "memory_kind",
    "subject_key",
    "predicate_key",
    "object_value",
    "object_entity_id",
    "confidence",
    "source_authority",
    "scope",
    "sensitivity",
    "correction_status",
    "schema_version",
    "conflict_group",
)
_MIN_CONTEXT_TOKEN_BUDGET = 16
_MAX_CONTEXT_TOKEN_BUDGET = 4096
# Write text is capped well below the 1 MiB protocol frame so the service can
# return the stored fact and its provenance without exceeding the transport.
MAX_WRITE_TEXT_BYTES = 256 * 1024


@dataclass(frozen=True, slots=True)
class OutputBounds:
    """Service-layer trust defaults and serialized response limits."""

    default_min_trust: float = 0.3
    search_max_results: int = 20
    context_max_results: int = 12
    max_fact_chars: int = 2_000
    search_max_total_chars: int = 12_000
    context_max_total_chars: int = 16_000
    evidence_max_total_bytes: int = 512 * 1024
    history_max_total_bytes: int = 512 * 1024
    context_mmr_lambda: float = 0.7
    conflicts_max_total_bytes: int = 512 * 1024

    def __post_init__(self) -> None:
        integer_bounds = (
            self.search_max_results,
            self.context_max_results,
            self.max_fact_chars,
            self.search_max_total_chars,
            self.context_max_total_chars,
            self.evidence_max_total_bytes,
            self.history_max_total_bytes,
            self.conflicts_max_total_bytes,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in integer_bounds
        ):
            raise ValueError("output bounds must be integers")
        if (
            isinstance(self.default_min_trust, bool)
            or not isinstance(self.default_min_trust, (int, float))
            or not math.isfinite(self.default_min_trust)
        ):
            raise ValueError("default_min_trust must be a finite number")
        if not 0.0 <= self.default_min_trust <= 1.0:
            raise ValueError("default_min_trust must be between 0 and 1")
        if self.search_max_results < 1 or self.context_max_results < 1:
            raise ValueError("result caps must be positive")
        if self.max_fact_chars < len(TRUNCATION_MARKER):
            raise ValueError("max_fact_chars is too small for the truncation marker")
        if self.search_max_total_chars < 512 or self.context_max_total_chars < 512:
            raise ValueError("total character caps must be at least 512")
        if (
            self.evidence_max_total_bytes < 512
            or self.history_max_total_bytes < 512
            or self.conflicts_max_total_bytes < 512
        ):
            raise ValueError("total byte caps must be at least 512")
        if (
            isinstance(self.context_mmr_lambda, bool)
            or not isinstance(self.context_mmr_lambda, (int, float))
            or not math.isfinite(self.context_mmr_lambda)
            or not 0.0 <= self.context_mmr_lambda <= 1.0
        ):
            raise ValueError("context_mmr_lambda must be between 0 and 1")


DEFAULT_OUTPUT_BOUNDS = OutputBounds()
_QUERY_STOPWORDS = frozenset(
    "a an the is are was were be of to in on at for and or with by as it this "
    "that from what does do did how why who where when use uses using".split()
)


def _query_tokens(text: str) -> set[str]:
    cleaned = "".join(char.lower() if char.isalnum() else " " for char in text)
    return {token for token in cleaned.split() if token}


def _check_keys(
    params: Mapping[str, Any],
    required: set[str],
    optional: set[str] = frozenset(),
) -> None:
    missing = sorted(required - params.keys())
    unknown = sorted(params.keys() - required - optional)
    if missing:
        raise ServiceRequestError("invalid_params", f"missing parameters: {missing}")
    if unknown:
        raise ServiceRequestError("invalid_params", f"unknown parameters: {unknown}")


def _reject_nested_identity(value: Any, *, path: str = "params") -> None:
    if isinstance(value, Mapping):
        forbidden = sorted(IMMUTABLE_CONTEXT_FIELDS & value.keys())
        if forbidden:
            raise ServiceRequestError(
                "invalid_params",
                f"{path} cannot contain connection identity fields: {forbidden}",
            )
        for key, item in value.items():
            _reject_nested_identity(item, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_nested_identity(item, path=f"{path}[{index}]")


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ServiceRequestError(
            "invalid_params", f"{name} must be a non-empty string"
        )
    return value.strip()


def _optional_text(value: Any, name: str) -> str | None:
    if value is None:
        return None
    return _text(value, name)


def _number(value: Any, name: str, default: float) -> float:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ServiceRequestError("invalid_params", f"{name} must be a number")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ServiceRequestError(
            "invalid_params", f"{name} must be a number"
        ) from exc
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ServiceRequestError("invalid_params", f"{name} must be between 0 and 1")
    return result


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ServiceRequestError(
            "invalid_params", f"{name} must be a positive integer"
        )
    return value


def _limit(value: Any, *, default: int, maximum: int = 200) -> int:
    if value is None:
        return default
    result = _positive_int(value, "limit")
    if result > maximum:
        raise ServiceRequestError("invalid_params", f"limit must not exceed {maximum}")
    return result


def _token_budget(value: Any) -> int:
    if value is None:
        return 256
    if isinstance(value, bool) or not isinstance(value, int):
        raise ServiceRequestError("invalid_params", "token_budget must be an integer")
    if not _MIN_CONTEXT_TOKEN_BUDGET <= value <= _MAX_CONTEXT_TOKEN_BUDGET:
        raise ServiceRequestError(
            "invalid_params",
            "token_budget must be between "
            f"{_MIN_CONTEXT_TOKEN_BUDGET} and {_MAX_CONTEXT_TOKEN_BUDGET}",
        )
    return value


def _serialized_chars(value: Mapping[str, Any]) -> int:
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":")))


def _serialized_bytes(value: Mapping[str, Any]) -> int:
    return len(
        json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    )


def _truncate_fact_content(fact: dict[str, Any], maximum: int) -> bool:
    content = fact.get("content")
    if not isinstance(content, str) or len(content) <= maximum:
        return False
    fact["content"] = (
        content[: maximum - len(TRUNCATION_MARKER)].rstrip() + TRUNCATION_MARKER
    )
    fact["content_truncated"] = True
    return True


def _truncate_text_field(item: dict[str, Any], field: str, maximum: int) -> bool:
    content = item.get(field)
    if not isinstance(content, str) or len(content) <= maximum:
        return False
    item[field] = (
        content[: maximum - len(TRUNCATION_MARKER)].rstrip() + TRUNCATION_MARKER
    )
    item[f"{field}_truncated"] = True
    return True


def _write_text(value: Any, name: str) -> str:
    result = _text(value, name)
    if len(result.encode("utf-8")) > MAX_WRITE_TEXT_BYTES:
        raise ServiceRequestError(
            "invalid_params",
            f"{name} must not exceed {MAX_WRITE_TEXT_BYTES} UTF-8 bytes",
        )
    return result


def _optional_write_text(value: Any, name: str) -> str | None:
    if value is None:
        return None
    return _write_text(value, name)


def _boolean(value: Any, name: str, default: bool) -> bool:
    if value is None:
        return default
    if not isinstance(value, bool):
        raise ServiceRequestError("invalid_params", f"{name} must be a boolean")
    return value


def _json_object(value: Any, name: str) -> str:
    if value is None:
        return "{}"
    if not isinstance(value, dict):
        raise ServiceRequestError("invalid_params", f"{name} must be an object")
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError, RecursionError) as exc:
        raise ServiceRequestError(
            "invalid_params", f"{name} must contain JSON values"
        ) from exc


def _extraction_payload_bytes(
    context: ConnectionContext,
    transcript: TranscriptInput,
    source: str,
    scope: str,
    metadata: Mapping[str, Any],
) -> int:
    transcript_text, turns = normalize_transcript(transcript)
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
        "metadata": dict(metadata),
    }
    if turns is None:
        envelope["transcript"] = transcript_text
    else:
        envelope["turns"] = list(turns)
    payload = json.dumps(
        envelope,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return len(payload.encode("utf-8"))


def _protocol_context(context: ClientContext) -> ConnectionContext:
    """Copy only immutable, handshake-established context into provenance."""

    return ConnectionContext(
        client_id=context.client_id,
        surface=context.surface,
        agent_id=context.agent_id,
        session_id=context.session_id,
        parent_agent_id=context.parent_agent_id,
        project_root=context.project_root,
        repository=context.repository,
        branch=context.branch,
        commit_sha=context.commit_sha,
        access_scopes=context.access_scopes,
    )


def _fact_writer(
    conn: sqlite3.Connection, request: WriteRequest, observation_id: int
) -> FactWriteResult:
    del observation_id
    fact_id = insert_fact(
        conn,
        request.content,
        category=request.category,
        tags=request.tags,
        trust_score=request.trust_score,
        source_authority=request.source_authority,
        scope=request.scope,
        sensitivity=request.sensitivity,
    )
    if request.correction_status is not None:
        conn.execute(
            "UPDATE facts SET correction_status = ? WHERE fact_id = ?",
            (request.correction_status, fact_id),
        )
    return FactWriteResult(fact_id)


class EnfoldService:
    """Route typed protocol requests against one migrated SQLite connection."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        policy: MemoryPolicy,
        *,
        retriever_factory: RetrieverFactory | None = None,
        embedding_outbox: EmbeddingOutbox | None = None,
        extraction_enqueuer: ExtractionEnqueuer | None = None,
        extraction_processing_mode: str = "deferred",
        output_bounds: OutputBounds = DEFAULT_OUTPUT_BOUNDS,
        embedding_identity: str | None = None,
        query_embedder: Callable[[str], object] | None = None,
        near_dedup_enabled: bool = False,
    ):
        if conn.in_transaction:
            raise RuntimeError("EnfoldService requires an idle connection")
        version = require_compatible_schema(conn, for_writer=True)
        if version != SUPPORTED_SCHEMA_VERSION:
            raise RuntimeError(
                f"EnfoldService requires schema v{SUPPORTED_SCHEMA_VERSION}; found v{version}"
            )
        # Core retrieval returns mapping-shaped rows.  sqlite3.Row remains
        # index-compatible with the write/schema layers using this connection.
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        if int(conn.execute("PRAGMA foreign_keys").fetchone()[0]) != 1:
            raise RuntimeError("EnfoldService requires foreign key enforcement")
        self._conn = conn
        self._policy = policy
        self._output_bounds = output_bounds
        self._embedding_outbox = embedding_outbox
        self._writes = MemoryWriteService(
            conn,
            _fact_writer,
            policy,
            embedding_enqueue=(
                embedding_outbox.enqueue_in_transaction
                if embedding_outbox is not None
                else None
            ),
            near_dedup=NearDedupConfig(
                enabled=near_dedup_enabled,
                embedding_identity=embedding_identity,
            ),
            query_embedder=query_embedder,
        )
        self._retriever_factory = retriever_factory or deterministic_retriever_factory()
        self._extraction_enqueuer = extraction_enqueuer
        if extraction_processing_mode not in {
            "deferred",
            "disabled",
            "daemon-supervised",
        }:
            raise ValueError("extraction_processing_mode is invalid")
        self._extraction_processing_mode = extraction_processing_mode
        available = {
            str(row[1]) for row in conn.execute("PRAGMA table_info(facts)")
        }
        self._fact_field_names = tuple(
            name for name in _FACT_FIELDS if name in available
        )

    def retrieval_metadata_for(self, conn: sqlite3.Connection) -> dict[str, Any]:
        """Inspect retrieval capabilities on a caller-owned connection.

        Health must pass a read-only snapshot connection, not the writer
        connection used by ``MemoryWriteService``.
        """

        retriever = self._retriever_factory(conn, ("private",))
        return dict(retriever.metadata)

    @property
    def retrieval_metadata(self) -> dict[str, Any]:
        """Non-sensitive retrieval capabilities for health/inspection output."""

        return self.retrieval_metadata_for(self._conn)

    def __call__(self, context: ClientContext, request: Request) -> dict[str, Any]:
        return self.handle(context, request)

    def handle(self, context: ClientContext, request: Request) -> dict[str, Any]:
        if request.schema_version != SUPPORTED_SCHEMA_VERSION:
            raise ServiceRequestError(
                "incompatible_schema",
                f"request schema {request.schema_version}; service schema {SUPPORTED_SCHEMA_VERSION}",
            )
        _reject_nested_identity(request.params)
        effective = self._authorize(context)
        routes = {
            "memory.write": self._write,
            "memory.search": self._search,
            "memory.context": self._context,
            "memory.evidence": self._evidence,
            "memory.history": self._history,
            "memory.changes": self._changes,
            "memory.timeline": self._timeline,
            "memory.entities": self._entities,
            "memory.entity": self._entity,
            "memory.conflicts": self._conflicts,
            "memory.resolve_conflict": self._resolve_conflict,
            "memory.extraction.enqueue": self._enqueue_extraction,
            "memory.promote": self._promote,
        }
        route = routes.get(request.method)
        if route is None:
            raise ServiceRequestError(
                "unsupported_method", f"unsupported service method: {request.method}"
            )
        return route(effective, request.params)

    def handle_write_batch(
        self,
        context: ClientContext,
        requests: Sequence[Request],
        *,
        before_commit: Callable[[], None] | None = None,
    ) -> AtomicWriteBatchResult:
        """Apply ordered memory writes atomically through the normal policy path."""

        if isinstance(requests, (str, bytes)) or not isinstance(requests, Sequence):
            raise TypeError("requests must be a sequence")
        if not requests:
            return AtomicWriteBatchResult((), committed=True)
        for request in requests:
            if not isinstance(request, Request):
                raise TypeError("batch entries must be Request objects")
            if request.schema_version != SUPPORTED_SCHEMA_VERSION:
                raise ServiceRequestError(
                    "incompatible_schema",
                    f"request schema {request.schema_version}; service schema {SUPPORTED_SCHEMA_VERSION}",
                )
            if request.method != "memory.write":
                raise ServiceRequestError(
                    "unsupported_method", "write batch accepts only memory.write"
                )
            _reject_nested_identity(request.params)
        effective = self._authorize(context)
        try:
            prepared = tuple(
                self._prepare_write(effective, request.params) for request in requests
            )
            batch = self._writes.write_batch(
                effective,
                prepared,
                rollback_if=lambda outcome: (
                    outcome.outcome in {"rejected", "needs_review"}
                ),
                before_commit=before_commit,
            )
        except ServiceRequestError:
            raise
        except IdempotencyConflict as exc:
            raise ServiceRequestError("idempotency_conflict", str(exc)) from exc
        except (PermissionError, ValueError) as exc:
            raise ServiceRequestError("invalid_params", str(exc)) from exc
        results = tuple(self._write_result(outcome) for outcome in batch.outcomes)
        if not batch.committed and not any(
            result["outcome"] in {"rejected", "needs_review"} for result in results
        ):
            raise RuntimeError("write batch rolled back without a policy outcome")
        return AtomicWriteBatchResult(results, committed=batch.committed)

    def _authorize(self, context: ClientContext) -> ConnectionContext:
        try:
            return self._policy.authorize_context(_protocol_context(context))
        except UnknownMemoryClient as exc:
            raise ServiceRequestError(
                "access_denied", "memory client is not authorized"
            ) from exc
        except PermissionError as exc:
            raise ServiceRequestError(
                "access_denied", "no requested memory scope is authorized"
            ) from exc
        except ValueError as exc:
            raise ServiceRequestError("invalid_context", str(exc)) from exc

    def _write(
        self, context: ConnectionContext, params: Mapping[str, Any]
    ) -> dict[str, Any]:
        try:
            write, candidate = self._prepare_write(context, params)
            outcome = self._writes.write(context, write, state_candidate=candidate)
        except ServiceRequestError:
            raise
        except IdempotencyConflict as exc:
            raise ServiceRequestError("idempotency_conflict", str(exc)) from exc
        except (PermissionError, ValueError) as exc:
            raise ServiceRequestError("invalid_params", str(exc)) from exc
        return self._write_result(outcome)

    def _prepare_write(
        self, context: ConnectionContext, params: Mapping[str, Any]
    ) -> tuple[WriteRequest, StateCandidate | None]:
        required = {"idempotency_key", "content", "source_type"}
        optional = {
            "category",
            "tags",
            "trust_score",
            "source_authority",
            "source_uri",
            "observation_content",
            "asserted_by",
            "observed_at",
            "scope",
            "sensitivity",
            "correction_status",
            "evidence_excerpt",
            "relation",
            "metadata",
            "supersede_fact_id",
            "state",
        }
        _check_keys(params, required, optional)
        metadata = _json_object(params.get("metadata"), "metadata")
        write = WriteRequest(
            idempotency_key=_text(params["idempotency_key"], "idempotency_key"),
            content=_write_text(params["content"], "content"),
            source_type=_text(params["source_type"], "source_type"),
            category=_text(params.get("category", "general"), "category"),
            tags=(
                params.get("tags", "")
                if isinstance(params.get("tags", ""), str)
                else self._invalid("tags must be a string")
            ),
            trust_score=_number(params.get("trust_score"), "trust_score", 0.5),
            source_authority=_number(
                params.get("source_authority"), "source_authority", 0.5
            ),
            source_uri=_optional_text(params.get("source_uri"), "source_uri"),
            observation_content=_optional_write_text(
                params.get("observation_content"), "observation_content"
            ),
            asserted_by=_optional_text(params.get("asserted_by"), "asserted_by"),
            # The performing agent is connection provenance, never caller input.
            performed_by=context.agent_id,
            observed_at=_optional_text(params.get("observed_at"), "observed_at"),
            scope=_text(params.get("scope", "private"), "scope"),
            sensitivity=_text(params.get("sensitivity", "normal"), "sensitivity"),
            correction_status=_optional_text(
                params.get("correction_status"), "correction_status"
            ),
            evidence_excerpt=_optional_text(
                params.get("evidence_excerpt"), "evidence_excerpt"
            ),
            relation=_text(params.get("relation", "supports"), "relation"),
            metadata_json=metadata,
            supersede_fact_id=(
                None
                if params.get("supersede_fact_id") is None
                else _positive_int(params["supersede_fact_id"], "supersede_fact_id")
            ),
        )
        candidate = self._state_candidate(write, params.get("state"))
        return write, candidate

    @staticmethod
    def _write_result(outcome: Any) -> dict[str, Any]:
        result = asdict(outcome)
        result["detail"] = json.loads(result.pop("detail_json"))
        return result

    def _promote(
        self, context: ConnectionContext, params: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Copy a current run-scoped fact onto a durable granted scope."""

        _check_keys(params, {"fact_id", "idempotency_key"}, {"target_scope"})
        fact_id = _positive_int(params["fact_id"], "fact_id")
        try:
            target_scope = validate_scope(
                _text(params.get("target_scope", "private"), "target_scope")
            )
        except ValueError as exc:
            raise ServiceRequestError("invalid_params", str(exc)) from exc
        if is_run_scope(target_scope):
            raise ServiceRequestError(
                "invalid_params", "promote target must be a durable granted scope"
            )
        if target_scope not in context.access_scopes:
            raise ServiceRequestError(
                "access_denied", "requested memory scope is not authorized"
            )
        source = self._load_fact(fact_id)
        if source is None or not self._fact_is_visible(context, source):
            raise ServiceRequestError("access_denied", "fact is not visible")
        if not is_run_scope(source["scope"]):
            raise ServiceRequestError(
                "invalid_params", "only run-scoped facts can be promoted"
            )
        if source["invalid_at"] is not None or source["superseded_by"] is not None:
            raise ServiceRequestError("invalid_params", "source fact is not current")
        return self._write(
            context,
            {
                "idempotency_key": _text(params["idempotency_key"], "idempotency_key"),
                "content": source["content"],
                "source_type": "promotion",
                "category": source["category"] or "general",
                "tags": source["tags"] or "",
                "trust_score": source["trust_score"],
                "source_authority": source["source_authority"],
                "source_uri": f"enfold:fact:{fact_id}",
                "scope": target_scope,
                "sensitivity": source["sensitivity"] or "normal",
                "relation": "derived_from",
                "metadata": {
                    "promoted_from_fact_id": fact_id,
                    "promoted_from_scope": source["scope"],
                },
            },
        )

    def _fact_fields(self) -> tuple[str, ...]:
        cached = getattr(self, "_fact_field_names", None)
        if cached is not None:
            return cached
        available = {
            str(row[1]) for row in self._conn.execute("PRAGMA table_info(facts)")
        }
        return tuple(name for name in _FACT_FIELDS if name in available)

    def _load_fact(self, fact_id: int) -> dict[str, Any] | None:
        fields = self._fact_fields()
        columns = ", ".join(fields)
        row = self._conn.execute(
            f"SELECT {columns} FROM facts WHERE fact_id = ?",
            (fact_id,),
        ).fetchone()
        return dict(zip(fields, row)) if row is not None else None

    @staticmethod
    def _fact_is_visible(
        context: ConnectionContext, source: Mapping[str, Any]
    ) -> bool:
        if not scope_authorized(str(source["scope"]), context.access_scopes):
            return False
        sensitivity = source.get("sensitivity") or "normal"
        return not (
            sensitivity in {"sensitive", "secret"}
            and sensitivity not in context.access_scopes
        )

    def _authorized_read_scopes(
        self, context: ConnectionContext, fact_id: int
    ) -> tuple[str, ...] | None:
        source = self._load_fact(fact_id)
        if source is None or not self._fact_is_visible(context, source):
            return None
        return tuple(dict.fromkeys((*context.access_scopes, str(source["scope"]))))

    @staticmethod
    def _invalid(message: str) -> Any:
        raise ServiceRequestError("invalid_params", message)

    def _state_candidate(
        self, write: WriteRequest, value: Any
    ) -> StateCandidate | None:
        if value is None:
            return None
        if not isinstance(value, dict):
            raise ServiceRequestError("invalid_params", "state must be an object")
        _check_keys(
            value,
            {"subject_key", "predicate_key"},
            {"object_value", "valid_from", "valid_to"},
        )
        return StateCandidate(
            content=write.content,
            subject_key=_text(value["subject_key"], "state.subject_key"),
            predicate_key=_text(value["predicate_key"], "state.predicate_key"),
            object_value=_optional_text(
                value.get("object_value"), "state.object_value"
            ),
            source_authority=write.source_authority,
            valid_from=_optional_text(value.get("valid_from"), "state.valid_from"),
            valid_to=_optional_text(value.get("valid_to"), "state.valid_to"),
            scope=write.scope,
        )

    def _search(
        self, context: ConnectionContext, params: Mapping[str, Any]
    ) -> dict[str, Any]:
        _check_keys(
            params,
            {"query"},
            {
                "category",
                "min_trust",
                "limit",
                "scope",
                "include_unreviewed",
                "as_of_valid",
                "as_of_tx",
            },
        )
        query = _text(params["query"], "query")
        category = _optional_text(params.get("category"), "category")
        bounds = self._output_bounds
        min_trust = _number(
            params.get("min_trust"), "min_trust", bounds.default_min_trust
        )
        include_unreviewed = _boolean(
            params.get("include_unreviewed"), "include_unreviewed", False
        )
        requested_limit = _limit(params.get("limit"), default=20)
        limit = min(requested_limit, bounds.search_max_results)
        scopes = self._requested_scopes(context, params.get("scope"))
        try:
            as_of_valid = optional_as_of_timestamp(
                params.get("as_of_valid"), "as_of_valid"
            )
            as_of_tx = optional_as_of_timestamp(params.get("as_of_tx"), "as_of_tx")
        except ProtocolValidationError as exc:
            raise ServiceRequestError("invalid_params", str(exc)) from exc
        retriever = self._retriever_factory(self._conn, scopes)
        if as_of_valid is not None or as_of_tx is not None:
            rows = self._search_as_of(
                query,
                scopes=scopes,
                category=category,
                min_trust=min_trust,
                limit=limit + 1,
                include_unreviewed=include_unreviewed,
                as_of_valid=as_of_valid,
                as_of_tx=as_of_tx,
            )
        else:
            rows = retriever.search(
                query,
                category=category,
                min_trust=min_trust,
                limit=limit + 1,
                include_unreviewed=include_unreviewed,
            )
        result_cap_truncated = len(rows) > limit
        rows = rows[:limit]
        facts = []
        content_truncated = False
        for row in rows:
            fact = self._safe_fact(row)
            fact["attribution"] = self._authorized_attribution(
                int(fact["fact_id"]), scopes
            )
            content_truncated |= _truncate_fact_content(fact, bounds.max_fact_chars)
            facts.append(fact)
        response = {
            "facts": facts,
            "retrieval": dict(retriever.metadata),
            "output_truncated": (
                content_truncated or requested_limit > limit or result_cap_truncated
            ),
            "open_conflicts": [
                self._receipt_as_dict(item)
                for item in self._relevant_conflict_receipts(scopes, query)
            ],
        }
        while facts and _serialized_chars(response) > bounds.search_max_total_chars:
            facts.pop()
            response["output_truncated"] = True
        if _serialized_chars(response) > bounds.search_max_total_chars:
            response["retrieval"] = {"output_truncated": True}
        if _serialized_chars(response) > bounds.search_max_total_chars:
            response["retrieval"] = {}
        return response

    def _search_as_of(
        self,
        query: str,
        *,
        scopes: Sequence[str],
        category: str | None,
        min_trust: float,
        limit: int,
        include_unreviewed: bool,
        as_of_valid: str | None,
        as_of_tx: str | None,
    ) -> list[dict[str, Any]]:
        selected = list(self._fact_fields())
        available = set(selected)
        columns = ", ".join(f"f.{name}" for name in selected)
        visibility_sql, visibility_params = build_visibility_predicate(
            scopes,
            scope_column="f.scope",
            sensitivity_column="f.sensitivity",
        )
        tokens = [token for token in query.replace(",", " ").split() if token]
        if not tokens:
            return []
        match = " AND ".join(tokens)
        predicates = [
            "facts_fts MATCH ?",
            visibility_sql,
            "f.trust_score >= ?",
        ]
        params: list[Any] = [match, *visibility_params, min_trust]
        if as_of_tx is None:
            predicates.append("f.conflict_group IS NULL")
        if category is not None:
            predicates.append("f.category = ?")
            params.append(category)
        if not include_unreviewed and "correction_status" in available:
            predicates.append("f.correction_status IS NOT ?")
            params.append("unreviewed")
        rows = self._conn.execute(
            f"""
            SELECT {columns}
            FROM facts f
            JOIN facts_fts ON facts_fts.rowid = f.fact_id
            WHERE {" AND ".join(predicates)}
            ORDER BY f.trust_score DESC, f.fact_id
            """,
            params,
        ).fetchall()
        matched: list[dict[str, Any]] = []
        for row in rows:
            fact = dict(zip(selected, row))
            if row_matches_as_of(
                fact, as_of_valid=as_of_valid, as_of_tx=as_of_tx
            ):
                matched.append(fact)
            if len(matched) >= limit:
                break
        return matched

    def _context(
        self, context: ConnectionContext, params: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Return a bounded, cited context pack over authorized current facts.

        This is a read-only projection over the ordinary retriever.  The
        retriever applies scope/current/conflict predicates before ranking; the
        pure packer repeats lifecycle checks defensively before formatting.
        """

        _check_keys(params, {"query", "token_budget"}, {"scope", "min_trust"})
        query = _text(params["query"], "query")
        token_budget = _token_budget(params["token_budget"])
        bounds = self._output_bounds
        min_trust = _number(
            params.get("min_trust"), "min_trust", bounds.default_min_trust
        )
        scopes = self._requested_scopes(context, params.get("scope"))
        retriever = self._retriever_factory(self._conn, scopes)
        rows = retriever.search(
            query,
            min_trust=min_trust,
            limit=bounds.context_max_results * 4 + 1,
        )
        candidate_cap = bounds.context_max_results * 4
        result_cap_truncated = len(rows) > candidate_cap
        rows = rows[:candidate_cap]
        candidates: list[dict[str, Any]] = []
        for row in rows:
            fact = self._safe_fact(row)
            if "_mmr_embedding" in row:
                fact["_mmr_embedding"] = row["_mmr_embedding"]
            fact["attribution"] = self._authorized_attribution(
                int(fact["fact_id"]), scopes
            )
            candidates.append(fact)
        receipts = self._relevant_conflict_receipts(scopes, query)
        receipt_maps = [self._receipt_as_dict(item) for item in receipts]
        output_truncated = result_cap_truncated
        while True:
            packed = pack_context(
                candidates,
                token_budget=token_budget,
                max_fact_chars=bounds.max_fact_chars,
                max_facts=bounds.context_max_results,
                mmr_lambda=bounds.context_mmr_lambda,
                conflict_receipts=receipt_maps,
            ).as_dict()
            output_truncated |= any(
                bool(fact.get("context_truncated")) for fact in packed["facts"]
            )
            packed["retrieval"] = dict(retriever.metadata)
            packed["output_truncated"] = output_truncated
            packed["open_conflicts"] = receipt_maps
            packed["facts"] = [
                fact
                for fact in packed["facts"]
                if fact.get("exclusion_reason") != "open_conflict"
            ]
            if _serialized_chars(packed) <= bounds.context_max_total_chars:
                return packed
            output_truncated = True
            if candidates:
                candidates.pop()
                continue
            packed["retrieval"] = {"output_truncated": True}
            if _serialized_chars(packed) <= bounds.context_max_total_chars:
                return packed
            packed["retrieval"] = {}
            return packed

    def _enqueue_extraction(
        self, context: ConnectionContext, params: Mapping[str, Any]
    ) -> dict[str, Any]:
        _check_keys(params, {"transcript", "source"}, {"scope", "metadata"})
        if self._extraction_enqueuer is None:
            raise ServiceRequestError(
                "extraction_unavailable",
                "durable extraction enqueue is not configured; automatic LLM extraction remains deferred",
            )
        scope = _text(params.get("scope", "private"), "scope")
        try:
            scope = validate_scope(scope)
        except ValueError as exc:
            raise ServiceRequestError("invalid_params", str(exc)) from exc
        if not scope_authorized(scope, context.access_scopes):
            raise ServiceRequestError(
                "access_denied", "extraction scope is not authorized"
            )
        if scope == "secret":
            return {
                "outcome": "rejected",
                "reason": "secret durable extraction payloads are disabled",
                "queue_id": None,
            }
        try:
            transcript_text, turns = normalize_transcript(params["transcript"])
        except (TypeError, ValueError) as exc:
            raise ServiceRequestError("invalid_params", str(exc)) from exc
        transcript: TranscriptInput = transcript_text if turns is None else turns
        source = _text(params["source"], "source")
        metadata_json = _json_object(params.get("metadata"), "metadata")
        metadata = json.loads(metadata_json)
        if (
            _extraction_payload_bytes(context, transcript, source, scope, metadata)
            > MAX_EXTRACTION_PAYLOAD_BYTES
        ):
            raise ServiceRequestError(
                "invalid_params",
                "canonical extraction payload must not exceed "
                f"{MAX_EXTRACTION_PAYLOAD_BYTES} UTF-8 bytes",
            )
        screen_request = WriteRequest(
            idempotency_key="extraction-screen",
            content=transcript_text,
            source_type="conversation_transcript",
            scope=scope,
            metadata_json=metadata_json,
        )
        decision = default_credential_screen(screen_request)
        if decision is not None:
            return {"outcome": "rejected", "reason": decision.reason, "queue_id": None}
        result = self._extraction_enqueuer.enqueue_after_commit(
            context,
            transcript,
            source=source,
            scope=scope,
            metadata=metadata,
        )
        return {
            "outcome": "queued",
            "queue_id": result.queue_id,
            "payload_sha256": result.payload_sha256,
            "replayed": result.replayed,
            "automatic_llm_extraction": self._extraction_processing_mode,
        }

    def _evidence(
        self, context: ConnectionContext, params: Mapping[str, Any]
    ) -> dict[str, Any]:
        _check_keys(params, {"fact_id"}, {"limit"})
        fact_id = _positive_int(params["fact_id"], "fact_id")
        limit = _limit(params.get("limit"), default=100)
        fact = self._historical_fact(fact_id, context.access_scopes)
        if fact is None:
            scopes = self._authorized_read_scopes(context, fact_id)
            if scopes is None:
                raise ServiceRequestError("not_found", "fact was not found")
            fact = self._historical_fact(fact_id, scopes)
            if fact is None:
                raise ServiceRequestError("not_found", "fact was not found")
        else:
            scopes = tuple(
                dict.fromkeys((*context.access_scopes, str(fact["scope"])))
            )
        observation_visibility_sql, observation_visibility_params = (
            build_visibility_predicate(
                scopes,
                scope_column="o.scope",
                sensitivity_column="o.sensitivity",
            )
        )
        cursor = self._conn.execute(
            f"""
            SELECT o.observation_id, o.client_id, o.session_id, o.source_type,
                   o.source_uri, o.project_root, o.repository, o.branch,
                   o.commit_sha, o.content, o.asserted_by, o.performed_by,
                   o.observed_at, o.recorded_at, o.scope, o.sensitivity,
                   o.redacted_at, o.metadata_json, p.relation,
                   p.evidence_excerpt, p.created_at
            FROM fact_provenance p
            JOIN observations o ON o.observation_id = p.observation_id
            WHERE p.fact_id = ? AND {observation_visibility_sql}
            ORDER BY p.created_at, o.observation_id
            LIMIT ?
            """,
            (fact_id, *observation_visibility_params, limit + 1),
        )
        keys = (
            "observation_id",
            "client_id",
            "session_id",
            "source_type",
            "source_uri",
            "project_root",
            "repository",
            "branch",
            "commit_sha",
            "content",
            "asserted_by",
            "performed_by",
            "observed_at",
            "recorded_at",
            "scope",
            "sensitivity",
            "redacted_at",
            "metadata",
            "relation",
            "evidence_excerpt",
            "provenance_created_at",
        )
        evidence = []
        content_truncated = _truncate_fact_content(
            fact, self._output_bounds.max_fact_chars
        )
        response = {
            "fact": fact,
            "evidence": evidence,
            "output_truncated": content_truncated,
        }
        used_bytes = _serialized_bytes(response)
        if used_bytes > self._output_bounds.evidence_max_total_bytes:
            cursor.close()
            response["fact"] = {"fact_id": fact_id, "output_truncated": True}
            response["output_truncated"] = True
            return response
        try:
            for index, row in enumerate(cursor):
                if index >= limit:
                    response["output_truncated"] = True
                    break
                item = dict(zip(keys, row))
                item["metadata"] = json.loads(item["metadata"])
                item_truncated = _truncate_text_field(
                    item, "content", self._output_bounds.max_fact_chars
                )
                item_truncated |= _truncate_text_field(
                    item, "evidence_excerpt", self._output_bounds.max_fact_chars
                )
                item_bytes = _serialized_bytes(item) + (1 if evidence else 0)
                if (
                    used_bytes + item_bytes
                    > self._output_bounds.evidence_max_total_bytes
                ):
                    response["output_truncated"] = True
                    break
                evidence.append(item)
                used_bytes += item_bytes
                response["output_truncated"] |= item_truncated
        finally:
            cursor.close()
        return response

    def _history(
        self, context: ConnectionContext, params: Mapping[str, Any]
    ) -> dict[str, Any]:
        optional = {"fact_id", "subject_key", "predicate_key", "scope", "limit"}
        _check_keys(params, set(), optional)
        by_id = "fact_id" in params
        by_slot = (
            "subject_key" in params or "predicate_key" in params or "scope" in params
        )
        if by_id == by_slot:
            raise ServiceRequestError(
                "invalid_params",
                "history requires either fact_id or subject_key and predicate_key",
            )
        limit = _limit(params.get("limit"), default=100)
        if by_id:
            fact_id = _positive_int(params["fact_id"], "fact_id")
            anchor = self._historical_fact(fact_id, context.access_scopes)
            if anchor is None:
                scopes = self._authorized_read_scopes(context, fact_id)
                if scopes is None:
                    raise ServiceRequestError("not_found", "fact was not found")
                anchor = self._historical_fact(fact_id, scopes)
                if anchor is None:
                    raise ServiceRequestError("not_found", "fact was not found")
            else:
                scopes = tuple(
                    dict.fromkeys((*context.access_scopes, str(anchor["scope"])))
                )
            if anchor.get("subject_key") and anchor.get("predicate_key"):
                scopes = tuple(
                    dict.fromkeys(
                        (
                            str(anchor["scope"]),
                            *(
                                scope
                                for scope in context.access_scopes
                                if scope in {"sensitive", "secret"}
                            ),
                        )
                    )
                )
                subject = str(anchor["subject_key"])
                predicate = str(anchor["predicate_key"])
                rows = self._slot_history(scopes, subject, predicate, limit + 1)
            else:
                rows = self._fact_history(scopes, fact_id, limit + 1)
        else:
            if "subject_key" not in params or "predicate_key" not in params:
                raise ServiceRequestError(
                    "invalid_params", "subject_key and predicate_key are both required"
                )
            scopes = self._requested_scopes(context, params.get("scope"))
            rows = self._slot_history(
                scopes,
                _text(params["subject_key"], "subject_key"),
                _text(params["predicate_key"], "predicate_key"),
                limit + 1,
            )
        facts: list[dict[str, Any]] = []
        response = {
            "facts": facts,
            "output_truncated": False,
        }
        used_bytes = _serialized_bytes(response)
        try:
            for index, row in enumerate(rows):
                if index >= limit:
                    response["output_truncated"] = True
                    break
                fact = dict(row)
                fact_truncated = _truncate_fact_content(
                    fact, self._output_bounds.max_fact_chars
                )
                fact_bytes = _serialized_bytes(fact) + (1 if facts else 0)
                if (
                    used_bytes + fact_bytes
                    > self._output_bounds.history_max_total_bytes
                ):
                    response["output_truncated"] = True
                    break
                facts.append(fact)
                used_bytes += fact_bytes
                response["output_truncated"] |= fact_truncated
        finally:
            rows.close()
        return response

    @staticmethod
    def _receipt_as_dict(receipt: ConflictReceipt) -> dict[str, Any]:
        return {
            "conflict_id": receipt.conflict_id,
            "scope": receipt.scope,
            "subject_key": receipt.subject_key,
            "predicate_key": receipt.predicate_key,
            "member_fact_ids": list(receipt.member_fact_ids),
            "summary": receipt.summary,
        }

    def _relevant_conflict_receipts(
        self, scopes: Sequence[str], query: str
    ) -> tuple[ConflictReceipt, ...]:
        tokens = _query_tokens(query) - _QUERY_STOPWORDS
        if not tokens:
            tokens = _query_tokens(query)
        if not tokens:
            return ()
        receipts: list[ConflictReceipt] = []
        seen: set[str] = set()
        for scope in scopes:
            for receipt in list_conflict_receipts(
                self._conn,
                scope,
                visibility_scopes=tuple(scopes),
            ):
                if receipt.conflict_id in seen:
                    continue
                seen.add(receipt.conflict_id)
                receipts.append(receipt)
        matched: list[ConflictReceipt] = []
        for receipt in receipts:
            haystack = _query_tokens(
                f"{receipt.subject_key} {receipt.predicate_key} {receipt.summary}"
            )
            for fact_id in receipt.member_fact_ids:
                row = self._conn.execute(
                    "SELECT content FROM facts WHERE fact_id = ?",
                    (fact_id,),
                ).fetchone()
                if row is not None:
                    haystack.update(_query_tokens(str(row[0])))
            overlap = tokens & haystack
            distinctive = {token for token in tokens if len(token) >= 6}
            if len(overlap) >= 2 or distinctive & haystack:
                matched.append(receipt)
        return tuple(matched)

    def _conflicts(
        self, context: ConnectionContext, params: Mapping[str, Any]
    ) -> dict[str, Any]:
        _check_keys(params, set(), {"scope", "unresolved_only", "limit"})
        scopes = self._requested_scopes(context, params.get("scope"))
        unresolved = _boolean(params.get("unresolved_only"), "unresolved_only", True)
        limit = _limit(params.get("limit"), default=100)
        member_limit = min(limit, self._output_bounds.search_max_results)
        records = []
        for scope in scopes:
            remaining = limit + 1 - len(records)
            if remaining <= 0:
                break
            scoped = list_state_conflicts(
                self._conn,
                scope,
                unresolved_only=unresolved,
                limit=min(remaining, 200),
                member_limit=member_limit,
                visibility_scopes=scopes,
            )
            records.extend(scoped[:remaining])
            if remaining > len(scoped) and len(scoped) == 200:
                records.extend(
                    list_state_conflicts(
                        self._conn,
                        scope,
                        unresolved_only=unresolved,
                        limit=1,
                        offset=200,
                        member_limit=member_limit,
                        visibility_scopes=scopes,
                    )
                )
        output_truncated = len(records) > limit or any(
            record.members_truncated for record in records[:limit]
        )
        records = records[:limit]
        member_ids = tuple(
            dict.fromkeys(
                fact_id for record in records for fact_id in record.member_fact_ids
            )
        )
        members_by_id: dict[int, dict[str, Any]] = {}
        if member_ids:
            placeholders = ",".join("?" for _ in member_ids)
            scope_placeholders = ",".join("?" for _ in scopes)
            fields = self._fact_fields()
            columns = ", ".join(fields)
            rows = self._conn.execute(
                f"SELECT {columns} FROM facts "
                f"WHERE fact_id IN ({placeholders}) "
                f"AND scope IN ({scope_placeholders})",
                (*member_ids, *scopes),
            ).fetchall()
            members_by_id = {
                int(row[0]): dict(zip(fields, row)) for row in rows
            }
        conflicts: list[dict[str, Any]] = []
        response = {"conflicts": conflicts, "output_truncated": output_truncated}
        for record in records:
            item = asdict(record)
            item["member_fact_ids"] = list(item["member_fact_ids"])
            item["members"] = []
            conflicts.append(item)
            if (
                _serialized_bytes(response)
                > self._output_bounds.conflicts_max_total_bytes
            ):
                conflicts.pop()
                response["output_truncated"] = True
                break
            retained_ids = []
            for fact_id in item["member_fact_ids"]:
                member = members_by_id.get(fact_id)
                if member is None:
                    continue
                member_truncated = _truncate_fact_content(
                    member, self._output_bounds.max_fact_chars
                )
                retained_ids.append(fact_id)
                item["members"].append(member)
                if (
                    _serialized_bytes(response)
                    > self._output_bounds.conflicts_max_total_bytes
                ):
                    retained_ids.pop()
                    item["members"].pop()
                    item["members_truncated"] = True
                    response["output_truncated"] = True
                    break
                response["output_truncated"] |= member_truncated
            item["member_fact_ids"] = retained_ids
        return response

    def _changes(
        self, context: ConnectionContext, params: Mapping[str, Any]
    ) -> dict[str, Any]:
        _check_keys(params, {"since", "until"}, {"scope", "limit"})
        scopes = self._requested_scopes(context, params.get("scope"))
        try:
            return changes(
                self._conn,
                _text(params["since"], "since"),
                _text(params["until"], "until"),
                scopes,
                limit=_limit(params.get("limit"), default=100),
            )
        except ValueError as exc:
            raise ServiceRequestError("invalid_params", str(exc)) from exc

    def _timeline(
        self, context: ConnectionContext, params: Mapping[str, Any]
    ) -> dict[str, Any]:
        _check_keys(params, {"subject_or_query"}, {"scope", "limit"})
        scopes = self._requested_scopes(context, params.get("scope"))
        try:
            return timeline(
                self._conn,
                _text(params["subject_or_query"], "subject_or_query"),
                scopes,
                limit=_limit(params.get("limit"), default=100),
            )
        except ValueError as exc:
            raise ServiceRequestError("invalid_params", str(exc)) from exc

    def _entities(
        self, context: ConnectionContext, params: Mapping[str, Any]
    ) -> dict[str, Any]:
        _check_keys(params, set(), {"scope", "min_facts", "limit"})
        scopes = self._requested_scopes(context, params.get("scope"))
        min_facts = _positive_int(params.get("min_facts", 1), "min_facts")
        try:
            return entities(
                self._conn,
                scopes,
                min_facts,
                limit=_limit(params.get("limit"), default=100),
            )
        except ValueError as exc:
            raise ServiceRequestError("invalid_params", str(exc)) from exc

    def _entity(
        self, context: ConnectionContext, params: Mapping[str, Any]
    ) -> dict[str, Any]:
        _check_keys(params, {"name"}, {"scope", "limit"})
        scopes = self._requested_scopes(context, params.get("scope"))
        try:
            return entity_dossier(
                self._conn,
                _text(params["name"], "name"),
                scopes,
                limit=_limit(params.get("limit"), default=100),
            )
        except ValueError as exc:
            raise ServiceRequestError("invalid_params", str(exc)) from exc

    def _resolve_conflict(
        self, context: ConnectionContext, params: Mapping[str, Any]
    ) -> dict[str, Any]:
        _check_keys(params, {"conflict_id", "resolution_fact_id", "reason"})
        if not self._policy.can_resolve_conflicts(context.client_id):
            raise ServiceRequestError(
                "access_denied", "memory client is not authorized to resolve conflicts"
            )
        conflict_id = _text(params["conflict_id"], "conflict_id")
        resolution_fact_id = _positive_int(
            params["resolution_fact_id"], "resolution_fact_id"
        )
        reason = _text(params["reason"], "reason")
        placeholders = ",".join("?" for _ in context.access_scopes)
        resolution_visibility_sql, resolution_visibility_params = (
            build_visibility_predicate(
                context.access_scopes,
                scope_column="f.scope",
                sensitivity_column="f.sensitivity",
            )
        )
        resolved_at = datetime.now(timezone.utc).isoformat(timespec="microseconds")
        try:
            self._conn.execute("BEGIN IMMEDIATE")
            visible = self._conn.execute(
                f"""
                SELECT c.scope FROM fact_conflicts c
                JOIN facts f
                  ON f.fact_id = ? AND f.scope = c.scope
                WHERE c.conflict_id = ? AND c.scope IN ({placeholders})
                  AND c.resolved_at IS NULL AND {resolution_visibility_sql}
                """,
                (
                    resolution_fact_id,
                    conflict_id,
                    *context.access_scopes,
                    *resolution_visibility_params,
                ),
            ).fetchone()
            if visible is None:
                raise ServiceRequestError("not_found", "conflict was not found")
            self._writes._register_client(context, resolved_at)
            self._writes._register_session(context, resolved_at)
            resolution = resolve_state_conflict(
                self._conn,
                conflict_id,
                resolution_fact_id,
                resolved_by=context.agent_id,
                reason=reason,
                resolved_at=resolved_at,
                resolver_client_id=context.client_id,
                resolver_session_id=context.session_id,
                resolver_agent_id=context.agent_id,
            )
            if self._embedding_outbox is not None:
                self._embedding_outbox.enqueue_in_transaction(resolution_fact_id)
            self._conn.commit()
        except ServiceRequestError:
            if self._conn.in_transaction:
                self._conn.rollback()
            raise
        except (ValueError, RuntimeError) as exc:
            if self._conn.in_transaction:
                self._conn.rollback()
            raise ServiceRequestError("invalid_resolution", str(exc)) from exc
        except BaseException:
            if self._conn.in_transaction:
                self._conn.rollback()
            raise
        result = asdict(resolution)
        result["superseded_fact_ids"] = list(result["superseded_fact_ids"])
        result["scope"] = str(visible[0])
        return {"resolution": result}

    @staticmethod
    def _safe_fact(row: Mapping[str, Any]) -> dict[str, Any]:
        score_fields = (
            "score",
            "fts_score",
            "jaccard_score",
            "dense_score",
            "trust_score_component",
            "memory_kind_score",
            "recency_score",
        )
        return {key: row[key] for key in (*_FACT_FIELDS, *score_fields) if key in row}

    def _authorized_attribution(
        self, fact_id: int, scopes: Sequence[str]
    ) -> dict[str, Any] | None:
        """Return latest visible provenance plus a visible-only evidence count."""

        visibility_sql, visibility_params = build_visibility_predicate(
            scopes,
            scope_column="o.scope",
            sensitivity_column="o.sensitivity",
        )
        row = self._conn.execute(
            f"""
            SELECT o.performed_by, s.agent_id, o.session_id, o.source_type,
                   o.repository, o.branch, o.commit_sha,
                   COUNT(*) OVER () AS evidence_count
            FROM fact_provenance p
            JOIN observations o ON o.observation_id = p.observation_id
            JOIN memory_sessions s
              ON s.client_id = o.client_id AND s.session_id = o.session_id
            WHERE p.fact_id = ? AND {visibility_sql}
            ORDER BY o.recorded_at DESC, o.observation_id DESC
            LIMIT 1
            """,
            (fact_id, *visibility_params),
        ).fetchone()
        if row is None:
            return None
        keys = (
            "performed_by",
            "agent_id",
            "session_id",
            "source_type",
            "repository",
            "branch",
            "commit_sha",
            "evidence_count",
        )
        return dict(zip(keys, row))

    def _historical_fact(
        self, fact_id: int, scopes: Sequence[str]
    ) -> dict[str, Any] | None:
        fields = self._fact_fields()
        columns = ", ".join(fields)
        visibility_sql, visibility_params = build_visibility_predicate(scopes)
        row = self._conn.execute(
            f"SELECT {columns} FROM facts WHERE fact_id = ? AND {visibility_sql}",
            (fact_id, *visibility_params),
        ).fetchone()
        return dict(zip(fields, row)) if row is not None else None

    def _slot_history(
        self, scopes: Sequence[str], subject: str, predicate: str, limit: int
    ) -> sqlite3.Cursor:
        columns = ", ".join(self._fact_fields())
        visibility_sql, visibility_params = build_visibility_predicate(scopes)
        return self._conn.execute(
            f"""
            SELECT {columns} FROM facts
            WHERE {visibility_sql}
              AND subject_key = ? AND predicate_key = ?
            ORDER BY COALESCE(valid_from, created_at), fact_id
            LIMIT ?
            """,
            (*visibility_params, subject, predicate, limit),
        )

    def _fact_history(
        self, scopes: Sequence[str], fact_id: int, limit: int
    ) -> sqlite3.Cursor:
        columns = ", ".join(f"f.{name}" for name in self._fact_fields())
        anchor_sql, anchor_params = build_visibility_predicate(scopes)
        previous_sql, previous_params = build_visibility_predicate(
            scopes,
            scope_column="previous.scope",
            sensitivity_column="previous.sensitivity",
        )
        following_sql, following_params = build_visibility_predicate(
            scopes,
            scope_column="following.scope",
            sensitivity_column="following.sensitivity",
        )
        result_sql, result_params = build_visibility_predicate(
            scopes, scope_column="f.scope", sensitivity_column="f.sensitivity"
        )
        return self._conn.execute(
            f"""
            WITH RECURSIVE chain(fact_id, superseded_by) AS (
                SELECT fact_id, superseded_by FROM facts
                WHERE fact_id = ? AND {anchor_sql}
                UNION
                SELECT previous.fact_id, previous.superseded_by
                FROM facts previous
                JOIN chain current ON previous.superseded_by = current.fact_id
                WHERE {previous_sql}
                UNION
                SELECT following.fact_id, following.superseded_by
                FROM facts following
                JOIN chain current ON following.fact_id = current.superseded_by
                WHERE {following_sql}
                LIMIT ?
            )
            SELECT {columns}
            FROM chain
            JOIN facts f ON f.fact_id = chain.fact_id
            WHERE {result_sql}
            ORDER BY COALESCE(f.created_at, ''), f.fact_id
            LIMIT ?
            """,
            (
                fact_id,
                *anchor_params,
                *previous_params,
                *following_params,
                limit,
                *result_params,
                limit,
            ),
        )

    @staticmethod
    def _requested_scopes(
        context: ConnectionContext, requested: Any
    ) -> tuple[str, ...]:
        if requested is None:
            return context.access_scopes
        try:
            scope = validate_scope(_text(requested, "scope"))
        except ValueError as exc:
            raise ServiceRequestError("invalid_params", str(exc)) from exc
        if not scope_authorized(scope, context.access_scopes):
            raise ServiceRequestError(
                "access_denied", "requested memory scope is not authorized"
            )
        return (scope,)
