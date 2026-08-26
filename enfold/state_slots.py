"""Deterministic typed-state decisions and durable conflict records.

State slots are explicit structured identity, never inferred here from prose.
The module has no LLM or similarity-search dependency and performs no commits;
callers compose its mutations into their write transaction.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import re
import sqlite3
from typing import Literal, Optional
import uuid

from .core_store import build_visibility_predicate


DecisionAction = Literal["add", "dedup", "supersede", "conflict"]


_SUBJECT_KEY = re.compile(r"^[a-z0-9][a-z0-9._:/-]{0,127}$")
_PREDICATE_KEY = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
_CANONICAL_SUBJECT_KINDS = {
    "agent": "software agent as agent:<stable-name>",
    "bot": "named bot as bot:<stable-name>",
    "course": "catalog course as course:<letters><digits>; omit section suffixes",
    "course_section": "specific course section as course_section:<stable-name>",
    "cron": "scheduled job as cron:<stable-name>",
    "environment": "runtime environment as environment:<stable-name>",
    "person": "named person as person:<stable-name>",
    "project": "named project as project:<stable-name>",
    "service": "software service as service:<stable-name>",
    "trip": "specific trip as trip:<stable-name>",
}
_CANONICAL_PREDICATES = {
    "clothing_store": "preferred store for clothing",
    "data_analysis_course": "preferred course for a data analysis requirement",
    "decision_council": "preferred council or panel for decisions",
    "employer": "current employer",
    "cru_schedule_mode": "preferred CRU schedule delivery mode",
    "job_status": "current job status",
    "keyboard": "preferred keyboard type",
    "live_version": "currently deployed or live version",
    "location": "current location",
    "model": "current or preferred model",
    "model_routing": "preferred routing of work among models",
    "port": "network port",
    "quiz_time_accommodation": "quiz time accommodation or multiplier",
    "response_style": "preferred response, answer, or reply style",
    "result_format": "preferred format for final results",
    "typography": "typography style",
    "threadwell_use": "constraint on when to use or buy from Threadwell",
}
_STATE_SLOT_INDEX_SQL = """
    CREATE UNIQUE INDEX uq_facts_current_state_slot
    ON facts(scope, subject_key, predicate_key)
    WHERE memory_kind = 'state'
      AND subject_key IS NOT NULL AND predicate_key IS NOT NULL
      AND invalid_at IS NULL AND superseded_by IS NULL
      AND conflict_group IS NULL
"""
_MAX_CONFLICT_LIMIT = 200


class StateSlotInvariantError(RuntimeError):
    """The persisted state violates exact-slot invariants."""


def _required(value: str, name: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError(f"{name} must not be empty")
    return value


def normalize_subject_key(value: str) -> str:
    """Canonicalize a model-produced slot subject to Enfold key style."""

    if not isinstance(value, str):
        raise ValueError("subject_key must be text")
    normalized = re.sub(r"\s+", "_", value.strip().casefold())
    if not _SUBJECT_KEY.fullmatch(normalized):
        raise ValueError("subject_key is not a canonical slot key")
    return normalized


def normalize_predicate_key(value: str) -> str:
    """Canonicalize a model-produced predicate to lowercase snake case."""

    if not isinstance(value, str):
        raise ValueError("predicate_key must be text")
    normalized = re.sub(r"[\s-]+", "_", value.strip().casefold())
    if not _PREDICATE_KEY.fullmatch(normalized):
        raise ValueError("predicate_key is not a canonical slot key")
    return normalized


def canonical_slot_registry(
    conn: sqlite3.Connection | None = None, *, scope: str | None = None
) -> dict[str, dict[str, str]]:
    """Return the provider-neutral canonical vocabulary for typed extraction."""

    predicates = dict(_CANONICAL_PREDICATES)
    if conn is not None:
        if not isinstance(scope, str) or not scope.strip():
            raise ValueError("scope is required for a store-backed slot registry")
        rows = conn.execute(
            "SELECT DISTINCT predicate_key FROM facts "
            "WHERE scope = ? AND predicate_key IS NOT NULL "
            "AND invalid_at IS NULL AND superseded_by IS NULL "
            "AND conflict_group IS NULL ORDER BY predicate_key",
            (scope.strip(),),
        ).fetchall()
        for row in rows:
            try:
                predicate = normalize_predicate_key(str(row[0]))
            except ValueError:
                continue
            predicates.setdefault(predicate, "existing canonical store predicate")
    return {
        "subject_kinds": dict(_CANONICAL_SUBJECT_KINDS),
        "predicates": predicates,
    }


def _plural_key(value: str) -> str:
    prefix, separator, final = value.rpartition("_")
    if final.endswith(("s", "x", "z", "ch", "sh")):
        plural = final + "es"
    elif len(final) > 1 and final.endswith("y") and final[-2] not in "aeiou":
        plural = final[:-1] + "ies"
    else:
        plural = final + "s"
    return f"{prefix}{separator}{plural}" if separator else plural


def resolve_extracted_subject_key(value: str) -> str:
    """Resolve only syntactic subject near-misses from extractor output."""

    if not isinstance(value, str):
        raise ValueError("subject_key must be text")
    normalized = normalize_subject_key(re.sub(r"\s*:\s*", ":", value))
    if ":" not in normalized:
        return normalized
    kind, payload = normalized.split(":", 1)
    kinds = set(_CANONICAL_SUBJECT_KINDS)
    kind_matches = [candidate for candidate in kinds if _plural_key(candidate) == kind]
    if kind not in kinds and len(kind_matches) == 1:
        kind = kind_matches[0]
    payload = payload.strip("_-")
    if kind == "course":
        match = re.fullmatch(
            r"([a-z]{2,4})[_-]*(\d{4})(?:[_-]*[a-z][a-z0-9]{1,7})?",
            payload,
        )
        if match is not None:
            payload = "".join(match.groups()[:2])
    return normalize_subject_key(f"{kind}:{payload}")


def resolve_extracted_predicate_key(
    value: str, *, known_predicates: tuple[str, ...] = ()
) -> str:
    """Resolve separators and unambiguous final-token plurals only."""

    normalized = normalize_predicate_key(value)
    known = {
        normalize_predicate_key(candidate)
        for candidate in (*_CANONICAL_PREDICATES, *known_predicates)
    }
    if normalized in known:
        return normalized
    matches = [candidate for candidate in known if _plural_key(candidate) == normalized]
    return matches[0] if len(matches) == 1 else normalized


def _entity_aliases(value: object) -> frozenset[str]:
    if not isinstance(value, str) or not value.strip():
        return frozenset()
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError:
        decoded = None
    if isinstance(decoded, list) and all(isinstance(item, str) for item in decoded):
        aliases = decoded
    else:
        aliases = re.split(r"[,;|\n]", value)
    return frozenset(alias.strip().casefold() for alias in aliases if alias.strip())


def resolve_stored_subject_key(
    conn: sqlite3.Connection, value: str, *, scope: str
) -> str | None:
    """Resolve a generic user subject only from unique trusted store evidence."""

    subject = resolve_extracted_subject_key(value)
    if subject not in {"user", "person:user", "person:the_user", "person:the-user"}:
        return subject
    rows = conn.execute(
        """
        SELECT DISTINCT f.subject_key, e.name, e.entity_type, e.aliases
        FROM facts AS f
        JOIN fact_entities AS fe ON fe.fact_id = f.fact_id
        JOIN entities AS e ON e.entity_id = fe.entity_id
        JOIN fact_provenance AS fp ON fp.fact_id = f.fact_id
        JOIN observations AS o ON o.observation_id = fp.observation_id
        WHERE f.scope = ? AND f.memory_kind = 'state'
          AND f.subject_key IS NOT NULL
          AND f.invalid_at IS NULL AND f.superseded_by IS NULL
          AND f.conflict_group IS NULL
          AND (f.correction_status = 'human_confirmed'
               OR (o.asserted_by = 'user'
                   AND o.source_type != 'automatic_extraction'))
        ORDER BY f.subject_key, e.entity_id
        """,
        (scope,),
    ).fetchall()
    candidates: set[str] = set()
    for stored_subject, name, entity_type, aliases in rows:
        if str(entity_type).casefold() != "person":
            continue
        if not {"user", "the user"}.intersection(_entity_aliases(aliases)):
            continue
        try:
            candidate = resolve_extracted_subject_key(str(stored_subject))
            entity_subject = resolve_extracted_subject_key(f"person:{name}")
        except ValueError:
            continue
        if candidate == entity_subject and candidate.startswith("person:"):
            candidates.add(candidate)
    return next(iter(candidates)) if len(candidates) == 1 else None


def _json_object(value: str, name: str = "detail_json") -> str:
    try:
        decoded = json.loads(value)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} must be a JSON object") from exc
    if not isinstance(decoded, dict):
        raise ValueError(f"{name} must be a JSON object")
    return json.dumps(decoded, sort_keys=True, separators=(",", ":"))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _parse_timestamp(value: Optional[str]) -> Optional[datetime]:
    if value is None:
        return None
    candidate = value.strip()
    if not candidate:
        return None
    if candidate.endswith("Z"):
        candidate = candidate[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError as exc:
        raise ValueError(f"invalid ISO timestamp: {value}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _normalized_schema_sql(sql: str) -> str:
    normalized = " ".join(sql.strip().rstrip(";").lower().split())
    return normalized.replace(" if not exists", "")


def _state_slot_index_has_expected_shape(conn: sqlite3.Connection) -> bool:
    row = next(
        (
            item
            for item in conn.execute('PRAGMA index_list("facts")')
            if str(item[1]) == "uq_facts_current_state_slot"
        ),
        None,
    )
    if row is None or not bool(row[2]) or str(row[3]) != "c" or not bool(row[4]):
        return False
    columns = tuple(
        str(item[2])
        for item in conn.execute(
            'PRAGMA index_info("uq_facts_current_state_slot")'
        )
    )
    sql_row = conn.execute(
        "SELECT tbl_name, sql FROM sqlite_master "
        "WHERE type = 'index' AND name = ?",
        ("uq_facts_current_state_slot",),
    ).fetchone()
    return bool(
        columns == ("scope", "subject_key", "predicate_key")
        and sql_row is not None
        and str(sql_row[0]) == "facts"
        and sql_row[1] is not None
        and _normalized_schema_sql(str(sql_row[1]))
        == _normalized_schema_sql(_STATE_SLOT_INDEX_SQL)
    )


def _canonicalize_existing_state_slots(
    conn: sqlite3.Connection, columns: set[str]
) -> None:
    timestamp_columns = [
        name for name in ("valid_from", "created_at") if name in columns
    ]
    selected = [
        "fact_id", "subject_key", "predicate_key", "scope", "memory_kind",
        "invalid_at", "superseded_by", "conflict_group", *timestamp_columns,
    ]
    rows = conn.execute(
        f"SELECT {', '.join(selected)} FROM facts "
        "WHERE subject_key IS NOT NULL AND predicate_key IS NOT NULL"
    ).fetchall()
    canonical: list[tuple[sqlite3.Row | tuple[object, ...], str, str, bool]] = []
    for row in rows:
        try:
            subject_key = normalize_subject_key(str(row[1]))
            predicate_key = normalize_predicate_key(str(row[2]))
        except ValueError as exc:
            raise StateSlotInvariantError(
                f"fact {int(row[0])} has invalid persisted state-slot keys"
            ) from exc
        canonical.append(
            (
                row,
                subject_key,
                predicate_key,
                (subject_key, predicate_key) != row[1:3],
            )
            )

    for row, subject_key, predicate_key, changed in canonical:
        if changed:
            conn.execute(
                "UPDATE facts SET subject_key = ?, predicate_key = ? "
                "WHERE fact_id = ?",
                (subject_key, predicate_key, int(row[0])),
            )


def _open_conflicts_for_duplicate_current_slots(conn: sqlite3.Connection) -> None:
    groups = conn.execute(
        """
        SELECT scope, subject_key, predicate_key
        FROM facts
        WHERE memory_kind = 'state'
          AND subject_key IS NOT NULL AND predicate_key IS NOT NULL
          AND invalid_at IS NULL AND superseded_by IS NULL
          AND conflict_group IS NULL
        GROUP BY scope, subject_key, predicate_key
        HAVING COUNT(*) > 1
        """
    ).fetchall()
    for scope, subject_key, predicate_key in groups:
        ids = tuple(
            int(row[0])
            for row in conn.execute(
                """
                SELECT fact_id FROM facts
                WHERE memory_kind = 'state'
                  AND scope = ? AND subject_key = ? AND predicate_key = ?
                  AND invalid_at IS NULL AND superseded_by IS NULL
                  AND conflict_group IS NULL
                ORDER BY fact_id
                """,
                (scope, subject_key, predicate_key),
            )
        )
        open_state_conflict(
            conn,
            str(subject_key),
            str(predicate_key),
            ids,
            scope=str(scope),
            detail_json='{"reason":"schema repair duplicate current slot"}',
        )


@dataclass(frozen=True, slots=True)
class StateCandidate:
    content: str
    subject_key: str
    predicate_key: str
    object_value: Optional[str] = None
    source_authority: float = 0.5
    valid_from: Optional[str] = None
    memory_kind: str = "state"
    scope: str = "private"
    valid_to: Optional[str] = None

    def __post_init__(self) -> None:
        for name in (
            "content", "scope", "subject_key", "predicate_key", "memory_kind"
        ):
            object.__setattr__(self, name, _required(getattr(self, name), name))
        object.__setattr__(
            self, "subject_key", normalize_subject_key(self.subject_key)
        )
        object.__setattr__(
            self, "predicate_key", normalize_predicate_key(self.predicate_key)
        )
        if not 0.0 <= self.source_authority <= 1.0:
            raise ValueError("source_authority must be between 0 and 1")
        start = _parse_timestamp(self.valid_from)
        end = _parse_timestamp(self.valid_to)
        if start is not None and end is not None and end <= start:
            raise ValueError("valid_to must be after valid_from")


@dataclass(frozen=True, slots=True)
class CurrentStateFact:
    fact_id: int
    content: str
    subject_key: str
    predicate_key: str
    scope: str
    object_value: Optional[str]
    source_authority: float
    valid_from: Optional[str]
    conflict_group: Optional[str]
    valid_to: Optional[str] = None


@dataclass(frozen=True, slots=True)
class SlotDecision:
    action: DecisionAction
    scope: str
    subject_key: str
    predicate_key: str
    current_fact_ids: tuple[int, ...] = ()
    target_fact_id: Optional[int] = None
    reason: str = ""


@dataclass(frozen=True, slots=True)
class ConflictRecord:
    conflict_id: str
    scope: str
    subject_key: str
    predicate_key: str
    member_fact_ids: tuple[int, ...]
    detected_at: str
    members_truncated: bool = False


@dataclass(frozen=True, slots=True)
class ConflictResolution:
    conflict_id: str
    resolution_fact_id: int
    superseded_fact_ids: tuple[int, ...]
    resolved_by: str
    reason: str
    resolved_at: str


@dataclass(frozen=True, slots=True)
class ReviewRecord:
    review_id: str
    scope: str
    reason: str
    content: Optional[str]
    subject_key: Optional[str]
    predicate_key: Optional[str]
    recorded_at: str
    detail_json: str


@dataclass(frozen=True, slots=True)
class ConflictReceipt:
    conflict_id: str
    scope: str
    subject_key: str
    predicate_key: str
    member_fact_ids: tuple[int, ...]
    summary: str


def format_conflict_receipt_summary(
    conflict_id: str,
    subject_key: str,
    predicate_key: str,
    member_count: int,
) -> str:
    """Return the one-line recall receipt for an open conflict."""

    return (
        f"[conflict:{conflict_id} slot:{subject_key}.{predicate_key} "
        f"members:{member_count} - do not treat either as current]"
    )


_TYPED_COLUMNS = (
    ("memory_kind", "TEXT"),
    ("subject_key", "TEXT"),
    ("predicate_key", "TEXT"),
    ("object_value", "TEXT"),
    ("object_entity_id", "INTEGER"),
    ("confidence", "REAL"),
    ("source_authority", "REAL"),
    ("scope", "TEXT NOT NULL DEFAULT 'private'"),
    ("sensitivity", "TEXT"),
    ("correction_status", "TEXT"),
    ("schema_version", "INTEGER"),
    ("conflict_group", "TEXT"),
)


def ensure_state_slot_schema(conn: sqlite3.Connection) -> bool:
    """Add typed-fact columns, conflict tables, and the strict-slot index.

    Returns ``True`` when the partial uniqueness invariant is installed.
    It is skipped when temporal columns are absent. Remaining active
    duplicate slots are opened as conflicts so the unique index can install.
    No commit is performed.
    """

    if conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='facts'"
    ).fetchone() is None:
        raise RuntimeError("facts table must exist before state-slot schema")

    columns = {row[1] for row in conn.execute("PRAGMA table_info(facts)")}
    for name, column_type in (
        *_TYPED_COLUMNS,
        ("valid_to", "TIMESTAMP"),
        ("expired_at", "TIMESTAMP"),
    ):
        if name not in columns:
            conn.execute(f"ALTER TABLE facts ADD COLUMN {name} {column_type}")
            columns.add(name)

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS fact_conflicts (
            conflict_id TEXT PRIMARY KEY,
            scope TEXT NOT NULL DEFAULT 'private',
            subject_key TEXT NOT NULL,
            predicate_key TEXT NOT NULL,
            detected_at TEXT NOT NULL,
            resolved_at TEXT,
            resolution_fact_id INTEGER,
            resolved_by TEXT,
            resolution_reason TEXT,
            detail_json TEXT NOT NULL DEFAULT '{}',
            FOREIGN KEY (resolution_fact_id) REFERENCES facts(fact_id)
        )
        """
    )
    conflict_columns = {
        row[1] for row in conn.execute("PRAGMA table_info(fact_conflicts)")
    }
    for name in ("resolved_by", "resolution_reason"):
        if name not in conflict_columns:
            conn.execute(f"ALTER TABLE fact_conflicts ADD COLUMN {name} TEXT")
    if "scope" not in conflict_columns:
        conn.execute(
            "ALTER TABLE fact_conflicts ADD COLUMN scope TEXT NOT NULL DEFAULT 'private'"
        )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS fact_conflict_members (
            conflict_id TEXT NOT NULL,
            fact_id INTEGER NOT NULL,
            PRIMARY KEY (conflict_id, fact_id),
            FOREIGN KEY (conflict_id) REFERENCES fact_conflicts(conflict_id),
            FOREIGN KEY (fact_id) REFERENCES facts(fact_id)
        )
        """
    )
    conn.execute(
        """CREATE INDEX IF NOT EXISTS idx_fact_conflict_members_fact
           ON fact_conflict_members(fact_id)"""
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS memory_review_queue (
            review_id TEXT PRIMARY KEY,
            scope TEXT NOT NULL DEFAULT 'private',
            reason TEXT NOT NULL,
            content TEXT,
            subject_key TEXT,
            predicate_key TEXT,
            detail_json TEXT NOT NULL DEFAULT '{}',
            recorded_at TEXT NOT NULL,
            resolved_at TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS fact_conflict_resolutions (
            conflict_id TEXT PRIMARY KEY,
            resolution_fact_id INTEGER NOT NULL,
            resolver_client_id TEXT NOT NULL,
            resolver_session_id TEXT NOT NULL,
            resolver_agent_id TEXT NOT NULL,
            reason TEXT NOT NULL,
            resolved_at TEXT NOT NULL,
            FOREIGN KEY (conflict_id) REFERENCES fact_conflicts(conflict_id),
            FOREIGN KEY (resolution_fact_id) REFERENCES facts(fact_id),
            FOREIGN KEY (resolver_client_id, resolver_session_id)
                REFERENCES memory_sessions(client_id, session_id)
        )
        """
    )

    if not {"invalid_at", "superseded_by"}.issubset(columns):
        return False
    if conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'index' "
        "AND name = 'uq_facts_current_state_slot'"
    ).fetchone() is not None:
        conn.execute("DROP INDEX uq_facts_current_state_slot")
    _canonicalize_existing_state_slots(conn, columns)
    _open_conflicts_for_duplicate_current_slots(conn)
    conn.execute(_STATE_SLOT_INDEX_SQL)
    return _state_slot_index_has_expected_shape(conn)


def _intervals_overlap(
    left_from: Optional[str],
    left_to: Optional[str],
    right_from: Optional[str],
    right_to: Optional[str],
) -> bool:
    """True unless one ``[start, end)`` interval ends at or before the other starts."""

    left_start = _parse_timestamp(left_from)
    left_end = _parse_timestamp(left_to)
    right_start = _parse_timestamp(right_from)
    right_end = _parse_timestamp(right_to)
    if left_end is not None and right_start is not None and left_end <= right_start:
        return False
    if right_end is not None and left_start is not None and right_end <= left_start:
        return False
    return True


def _fact_columns(conn: sqlite3.Connection) -> set[str]:
    return {str(row[1]) for row in conn.execute("PRAGMA table_info(facts)")}


def current_state_facts(
    conn: sqlite3.Connection,
    subject_key: str,
    predicate_key: str,
    scope: str = "private",
) -> tuple[CurrentStateFact, ...]:
    """Return active facts for exactly one structured state slot."""

    subject_key = _required(subject_key, "subject_key")
    predicate_key = _required(predicate_key, "predicate_key")
    scope = _required(scope, "scope")
    columns = _fact_columns(conn)
    valid_to_select = "valid_to" if "valid_to" in columns else "NULL"
    open_interval = "AND valid_to IS NULL" if "valid_to" in columns else ""
    rows = conn.execute(
        f"""
        SELECT fact_id, content, subject_key, predicate_key, scope, object_value,
               COALESCE(source_authority, 0.5), valid_from, conflict_group,
               {valid_to_select}
        FROM facts
        WHERE memory_kind = 'state'
          AND scope = ? AND subject_key = ? AND predicate_key = ?
          AND invalid_at IS NULL AND superseded_by IS NULL
          {open_interval}
        ORDER BY fact_id
        """,
        (scope, subject_key, predicate_key),
    ).fetchall()
    return tuple(
        CurrentStateFact(
            fact_id=int(row[0]),
            content=row[1],
            subject_key=row[2],
            predicate_key=row[3],
            scope=row[4],
            object_value=row[5],
            source_authority=float(row[6]),
            valid_from=row[7],
            conflict_group=row[8],
            valid_to=row[9],
        )
        for row in rows
    )


def _believed_state_facts(
    conn: sqlite3.Connection,
    subject_key: str,
    predicate_key: str,
    scope: str,
) -> tuple[CurrentStateFact, ...]:
    """Return same-slot facts the store still believes, including closed intervals."""

    columns = _fact_columns(conn)
    valid_to_select = "valid_to" if "valid_to" in columns else "NULL"
    if "expired_at" in columns and "valid_to" in columns:
        believed = (
            "AND expired_at IS NULL "
            "AND (invalid_at IS NULL OR valid_to IS NOT NULL)"
        )
    elif "expired_at" in columns:
        believed = "AND expired_at IS NULL"
    else:
        believed = "AND invalid_at IS NULL AND superseded_by IS NULL"
    rows = conn.execute(
        f"""
        SELECT fact_id, content, subject_key, predicate_key, scope, object_value,
               COALESCE(source_authority, 0.5), valid_from, conflict_group,
               {valid_to_select}
        FROM facts
        WHERE memory_kind = 'state'
          AND scope = ? AND subject_key = ? AND predicate_key = ?
          {believed}
        ORDER BY fact_id
        """,
        (scope, subject_key, predicate_key),
    ).fetchall()
    return tuple(
        CurrentStateFact(
            fact_id=int(row[0]),
            content=row[1],
            subject_key=row[2],
            predicate_key=row[3],
            scope=row[4],
            object_value=row[5],
            source_authority=float(row[6]),
            valid_from=row[7],
            conflict_group=row[8],
            valid_to=row[9],
        )
        for row in rows
    )


def read_current_state(
    conn: sqlite3.Connection,
    subject_key: str,
    predicate_key: str,
    scope: str = "private",
) -> Optional[CurrentStateFact]:
    """Return an unambiguous current value, abstaining on conflicts."""

    facts = current_state_facts(conn, subject_key, predicate_key, scope)
    if not facts or any(fact.conflict_group for fact in facts):
        return None
    if len(facts) != 1:
        raise StateSlotInvariantError(
            "strict state slot has multiple non-conflicted current facts"
        )
    return facts[0]


def list_state_conflicts(
    conn: sqlite3.Connection,
    scope: str = "private",
    *,
    unresolved_only: bool = True,
    limit: int = _MAX_CONFLICT_LIMIT,
    offset: int = 0,
    member_limit: Optional[int] = None,
    visibility_scopes: tuple[str, ...] | None = None,
) -> tuple[ConflictRecord, ...]:
    """List durable conflicts, optionally bounding members per conflict."""

    scope = _required(scope, "scope")
    if (
        isinstance(limit, bool)
        or not isinstance(limit, int)
        or not 1 <= limit <= _MAX_CONFLICT_LIMIT
    ):
        raise ValueError(f"limit must be between 1 and {_MAX_CONFLICT_LIMIT}")
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        raise ValueError("offset must be a non-negative integer")
    if member_limit is not None and (
        isinstance(member_limit, bool)
        or not isinstance(member_limit, int)
        or member_limit < 1
    ):
        raise ValueError("member_limit must be a positive integer or None")
    predicate = "AND c.resolved_at IS NULL" if unresolved_only else ""
    if visibility_scopes is None:
        visibility_sql, visibility_params = "1", ()
    else:
        visibility_sql, visibility_params = build_visibility_predicate(
            visibility_scopes,
            scope_column="f.scope",
            sensitivity_column="f.sensitivity",
        )
    rows = conn.execute(
        f"""
        WITH selected_conflicts AS (
            SELECT c.conflict_id, c.scope, c.subject_key, c.predicate_key,
                   c.detected_at
            FROM fact_conflicts c
            WHERE c.scope = ? {predicate}
              AND EXISTS (
                  SELECT 1
                  FROM fact_conflict_members m
                  JOIN facts f ON f.fact_id = m.fact_id
                  WHERE m.conflict_id = c.conflict_id AND {visibility_sql}
              )
            ORDER BY c.detected_at, c.conflict_id
            LIMIT ? OFFSET ?
        ), ranked_members AS (
            SELECT m.conflict_id, m.fact_id,
                   ROW_NUMBER() OVER (
                       PARTITION BY m.conflict_id ORDER BY m.fact_id
                   ) AS member_ordinal,
                   COUNT(*) OVER (
                       PARTITION BY m.conflict_id
                   ) AS member_count
            FROM fact_conflict_members m
            JOIN selected_conflicts c ON c.conflict_id = m.conflict_id
            JOIN facts f ON f.fact_id = m.fact_id
            WHERE {visibility_sql}
        )
        SELECT c.conflict_id, c.scope, c.subject_key, c.predicate_key,
               c.detected_at, m.fact_id, m.member_count
        FROM selected_conflicts c
        JOIN ranked_members m
          ON m.conflict_id = c.conflict_id
         AND (? IS NULL OR m.member_ordinal <= ?)
        ORDER BY c.detected_at, c.conflict_id, m.fact_id
        """,
        (
            scope,
            *visibility_params,
            limit,
            offset,
            *visibility_params,
            member_limit,
            member_limit,
        ),
    ).fetchall()
    grouped: dict[str, tuple[str, str, str, str, list[int], bool]] = {}
    for row in rows:
        conflict_id = str(row[0])
        entry = grouped.get(conflict_id)
        if entry is None:
            entry = (
                str(row[1]),
                str(row[2]),
                str(row[3]),
                str(row[4]),
                [],
                member_limit is not None and int(row[6]) > member_limit,
            )
            grouped[conflict_id] = entry
        entry[4].append(int(row[5]))
    return tuple(
        ConflictRecord(
            conflict_id,
            values[0],
            values[1],
            values[2],
            tuple(values[4]),
            values[3],
            values[5],
        )
        for conflict_id, values in grouped.items()
    )


def list_conflict_receipts(
    conn: sqlite3.Connection,
    scope: str = "private",
    *,
    limit: int = _MAX_CONFLICT_LIMIT,
    offset: int = 0,
    visibility_scopes: tuple[str, ...] | None = None,
) -> tuple[ConflictReceipt, ...]:
    """Return compact receipts for unresolved conflicts on the default read path."""

    records = list_state_conflicts(
        conn,
        scope,
        unresolved_only=True,
        limit=limit,
        offset=offset,
        visibility_scopes=visibility_scopes,
    )
    return tuple(
        ConflictReceipt(
            record.conflict_id,
            record.scope,
            record.subject_key,
            record.predicate_key,
            record.member_fact_ids,
            format_conflict_receipt_summary(
                record.conflict_id,
                record.subject_key,
                record.predicate_key,
                len(record.member_fact_ids),
            ),
        )
        for record in records
    )


def record_needs_review(
    conn: sqlite3.Connection,
    *,
    reason: str,
    scope: str = "private",
    content: Optional[str] = None,
    subject_key: Optional[str] = None,
    predicate_key: Optional[str] = None,
    detail_json: str = "{}",
    recorded_at: Optional[str] = None,
) -> ReviewRecord:
    """Persist a policy needs_review row so default recall can list it."""

    scope = _required(scope, "scope")
    reason = _required(reason, "reason")
    detail_json = _json_object(detail_json)
    recorded_at = recorded_at or _utc_now()
    _parse_timestamp(recorded_at)
    if subject_key is not None:
        subject_key = normalize_subject_key(subject_key)
    if predicate_key is not None:
        predicate_key = normalize_predicate_key(predicate_key)
    review_id = str(uuid.uuid4())
    conn.execute(
        """
        INSERT INTO memory_review_queue (
            review_id, scope, reason, content, subject_key, predicate_key,
            detail_json, recorded_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            review_id,
            scope,
            reason,
            content,
            subject_key,
            predicate_key,
            detail_json,
            recorded_at,
        ),
    )
    return ReviewRecord(
        review_id,
        scope,
        reason,
        content,
        subject_key,
        predicate_key,
        recorded_at,
        detail_json,
    )


def list_needs_review(
    conn: sqlite3.Connection,
    scope: str = "private",
    *,
    unresolved_only: bool = True,
    limit: int = _MAX_CONFLICT_LIMIT,
    offset: int = 0,
) -> tuple[ReviewRecord, ...]:
    """List persisted needs_review rows with the same pagination as conflicts."""

    scope = _required(scope, "scope")
    if (
        isinstance(limit, bool)
        or not isinstance(limit, int)
        or not 1 <= limit <= _MAX_CONFLICT_LIMIT
    ):
        raise ValueError(f"limit must be between 1 and {_MAX_CONFLICT_LIMIT}")
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        raise ValueError("offset must be a non-negative integer")
    predicate = "AND resolved_at IS NULL" if unresolved_only else ""
    rows = conn.execute(
        f"""
        SELECT review_id, scope, reason, content, subject_key, predicate_key,
               recorded_at, detail_json
        FROM memory_review_queue
        WHERE scope = ? {predicate}
        ORDER BY recorded_at, review_id
        LIMIT ? OFFSET ?
        """,
        (scope, limit, offset),
    ).fetchall()
    return tuple(
        ReviewRecord(
            str(row[0]),
            str(row[1]),
            str(row[2]),
            row[3],
            row[4],
            row[5],
            str(row[6]),
            str(row[7]),
        )
        for row in rows
    )


def decide_state_write(
    conn: sqlite3.Connection, candidate: StateCandidate
) -> SlotDecision:
    """Choose an exact-slot action without mutating the database.

    Authority wins only when the candidate is not demonstrably older. Equal
    authority requires a strictly newer timestamp. Conflicting authority and
    freshness signals remain visible instead of being guessed away.
    Non-state kinds always add and never supersede slot members.
    """

    if candidate.memory_kind != "state":
        return SlotDecision(
            "add",
            candidate.scope,
            candidate.subject_key,
            candidate.predicate_key,
            reason="non-state memories coexist",
        )

    current = current_state_facts(
        conn, candidate.subject_key, candidate.predicate_key, candidate.scope
    )
    believed = _believed_state_facts(
        conn, candidate.subject_key, candidate.predicate_key, candidate.scope
    )
    overlapping = tuple(
        fact
        for fact in believed
        if _intervals_overlap(
            fact.valid_from, fact.valid_to, candidate.valid_from, candidate.valid_to
        )
    )
    if not current and not overlapping:
        return SlotDecision(
            "add",
            candidate.scope,
            candidate.subject_key,
            candidate.predicate_key,
            reason="slot has no current overlapping value",
        )

    peers = current if current else overlapping
    exact = next(
        (
            fact
            for fact in peers
            if fact.content == candidate.content
            or (
                candidate.object_value is not None
                and fact.object_value == candidate.object_value
            )
        ),
        None,
    )
    # An unresolved conflict is the truth of the slot.  A matching candidate
    # may add evidence to one member, but must never make an ordinary write
    # look like an unambiguous deduplication.
    if len(current) > 1 or any(fact.conflict_group for fact in current):
        candidate_time = _parse_timestamp(candidate.valid_from)
        current_with_times = tuple(
            (fact, _parse_timestamp(fact.valid_from)) for fact in current
        )
        dated_candidate_wins = candidate_time is not None and all(
            (
                candidate.source_authority > fact.source_authority
                and (existing_time is None or candidate_time >= existing_time)
            )
            or (
                candidate.source_authority == fact.source_authority
                and (existing_time is None or candidate_time > existing_time)
            )
            for fact, existing_time in current_with_times
        )
        if dated_candidate_wins:
            return SlotDecision(
                "supersede",
                candidate.scope,
                candidate.subject_key,
                candidate.predicate_key,
                tuple(fact.fact_id for fact in current),
                reason="dated claim supersedes every open conflict member",
            )
        if candidate.source_authority > max(
            fact.source_authority for fact in current
        ):
            return SlotDecision(
                "supersede",
                candidate.scope,
                candidate.subject_key,
                candidate.predicate_key,
                tuple(fact.fact_id for fact in current),
                reason="higher authority resolves open conflict",
            )
        return SlotDecision(
            "conflict",
            candidate.scope,
            candidate.subject_key,
            candidate.predicate_key,
            tuple(fact.fact_id for fact in current),
            target_fact_id=exact.fact_id if exact is not None else None,
            reason="slot already has an unresolved conflict",
        )

    if exact is not None:
        return SlotDecision(
            "dedup",
            candidate.scope,
            candidate.subject_key,
            candidate.predicate_key,
            tuple(fact.fact_id for fact in peers),
            target_fact_id=exact.fact_id,
            reason="identical current content or structured value",
        )

    if not overlapping:
        return SlotDecision(
            "add",
            candidate.scope,
            candidate.subject_key,
            candidate.predicate_key,
            tuple(fact.fact_id for fact in current),
            reason="valid intervals do not overlap",
        )

    existing = overlapping[0] if overlapping else current[0]
    candidate_time = _parse_timestamp(candidate.valid_from)
    existing_time = _parse_timestamp(existing.valid_from)
    candidate_is_not_older = existing_time is None or (
        candidate_time is not None and candidate_time >= existing_time
    )
    authority_higher = candidate.source_authority > existing.source_authority
    authority_equal = candidate.source_authority == existing.source_authority
    freshness_higher = candidate_time is not None and (
        existing_time is None or candidate_time > existing_time
    )

    if authority_higher and candidate_is_not_older:
        return SlotDecision(
            "supersede",
            candidate.scope,
            candidate.subject_key,
            candidate.predicate_key,
            (existing.fact_id,),
            target_fact_id=existing.fact_id,
            reason="higher authority and not older",
        )
    if authority_equal and freshness_higher:
        return SlotDecision(
            "supersede",
            candidate.scope,
            candidate.subject_key,
            candidate.predicate_key,
            (existing.fact_id,),
            target_fact_id=existing.fact_id,
            reason="same authority with newer validity time",
        )
    return SlotDecision(
        "conflict",
        candidate.scope,
        candidate.subject_key,
        candidate.predicate_key,
        (existing.fact_id,),
        reason="authority and freshness do not establish a clear winner",
    )


def open_state_conflict(
    conn: sqlite3.Connection,
    subject_key: str,
    predicate_key: str,
    existing_fact_ids: tuple[int, ...],
    *,
    scope: str = "private",
    detected_at: Optional[str] = None,
    detail_json: str = "{}",
) -> ConflictRecord:
    """Open a conflict and move existing members outside the unique projection.

    The caller can then insert the candidate with the returned ``conflict_id``
    as its ``conflict_group`` and call :func:`add_conflict_member`, all within
    the same transaction.
    """

    subject_key = _required(subject_key, "subject_key")
    predicate_key = _required(predicate_key, "predicate_key")
    scope = _required(scope, "scope")
    if not existing_fact_ids:
        raise ValueError("a conflict requires at least one existing fact")
    if len(set(existing_fact_ids)) != len(existing_fact_ids):
        raise ValueError("conflict member ids must be unique")
    detail_json = _json_object(detail_json)
    detected_at = detected_at or _utc_now()
    _parse_timestamp(detected_at)

    placeholders = ",".join("?" for _ in existing_fact_ids)
    rows = conn.execute(
        f"""
        SELECT fact_id FROM facts
        WHERE fact_id IN ({placeholders})
          AND memory_kind = 'state'
          AND scope = ? AND subject_key = ? AND predicate_key = ?
          AND invalid_at IS NULL AND superseded_by IS NULL
          AND conflict_group IS NULL
        """,
        (*existing_fact_ids, scope, subject_key, predicate_key),
    ).fetchall()
    found = {int(row[0]) for row in rows}
    if found != set(existing_fact_ids):
        raise StateSlotInvariantError(
            "every conflict member must be an active state fact in the exact slot"
        )

    conflict_id = str(uuid.uuid4())
    conn.execute(
        """
        INSERT INTO fact_conflicts (
            conflict_id, scope, subject_key, predicate_key, detected_at, detail_json
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        (conflict_id, scope, subject_key, predicate_key, detected_at, detail_json),
    )
    conn.execute(
        f"UPDATE facts SET conflict_group = ? WHERE fact_id IN ({placeholders})",
        (conflict_id, *existing_fact_ids),
    )
    conn.executemany(
        "INSERT INTO fact_conflict_members(conflict_id, fact_id) VALUES (?, ?)",
        ((conflict_id, fact_id) for fact_id in existing_fact_ids),
    )
    return ConflictRecord(
        conflict_id,
        scope,
        subject_key,
        predicate_key,
        tuple(existing_fact_ids),
        detected_at,
    )


def open_untyped_conflict(
    conn: sqlite3.Connection,
    existing_fact_ids: tuple[int, ...],
    *,
    scope: str,
    subject_key: str,
    predicate_key: str,
    detected_at: Optional[str] = None,
    detail_json: str = "{}",
) -> ConflictRecord:
    """Open a conflict for current untyped facts that disagree in the same scope.

    Slot keys are caller-supplied labels for the receipt, not inferred here.
    Members stay non-state so the typed unique projection is unchanged.
    """

    subject_key = normalize_subject_key(_required(subject_key, "subject_key"))
    predicate_key = normalize_predicate_key(_required(predicate_key, "predicate_key"))
    scope = _required(scope, "scope")
    if not existing_fact_ids:
        raise ValueError("a conflict requires at least one existing fact")
    if len(set(existing_fact_ids)) != len(existing_fact_ids):
        raise ValueError("conflict member ids must be unique")
    detail_json = _json_object(detail_json)
    detected_at = detected_at or _utc_now()
    _parse_timestamp(detected_at)

    columns = _fact_columns(conn)
    kind_pred = ""
    if "memory_kind" in columns:
        kind_pred = "AND (memory_kind IS NULL OR memory_kind != 'state')"
    placeholders = ",".join("?" for _ in existing_fact_ids)
    rows = conn.execute(
        f"""
        SELECT fact_id FROM facts
        WHERE fact_id IN ({placeholders})
          AND scope = ?
          AND invalid_at IS NULL AND superseded_by IS NULL
          AND conflict_group IS NULL
          {kind_pred}
        """,
        (*existing_fact_ids, scope),
    ).fetchall()
    found = {int(row[0]) for row in rows}
    if found != set(existing_fact_ids):
        raise StateSlotInvariantError(
            "every untyped conflict member must be an active non-state fact in scope"
        )

    conflict_id = str(uuid.uuid4())
    conn.execute(
        """
        INSERT INTO fact_conflicts (
            conflict_id, scope, subject_key, predicate_key, detected_at, detail_json
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        (conflict_id, scope, subject_key, predicate_key, detected_at, detail_json),
    )
    assignments = ["conflict_group = ?"]
    values: list[object] = [conflict_id]
    if "subject_key" in columns:
        assignments.append("subject_key = COALESCE(subject_key, ?)")
        values.append(subject_key)
    if "predicate_key" in columns:
        assignments.append("predicate_key = COALESCE(predicate_key, ?)")
        values.append(predicate_key)
    conn.execute(
        f"UPDATE facts SET {', '.join(assignments)} WHERE fact_id IN ({placeholders})",
        (*values, *existing_fact_ids),
    )
    conn.executemany(
        "INSERT INTO fact_conflict_members(conflict_id, fact_id) VALUES (?, ?)",
        ((conflict_id, fact_id) for fact_id in existing_fact_ids),
    )
    return ConflictRecord(
        conflict_id,
        scope,
        subject_key,
        predicate_key,
        tuple(existing_fact_ids),
        detected_at,
    )


def add_untyped_conflict_member(
    conn: sqlite3.Connection, conflict_id: str, fact_id: int
) -> None:
    """Attach a newly inserted untyped fact to an open conflict."""

    row = conn.execute(
        """
        SELECT c.scope FROM fact_conflicts c
        WHERE c.conflict_id = ? AND c.resolved_at IS NULL
        """,
        (conflict_id,),
    ).fetchone()
    if row is None:
        raise ValueError("conflict does not exist or is already resolved")
    columns = _fact_columns(conn)
    kind_select = "memory_kind" if "memory_kind" in columns else "NULL"
    fact = conn.execute(
        f"""
        SELECT scope, conflict_group, {kind_select}, invalid_at, superseded_by
        FROM facts WHERE fact_id = ?
        """,
        (fact_id,),
    ).fetchone()
    if fact is None or fact[0] != row[0] or str(fact[1]) != conflict_id:
        raise StateSlotInvariantError(
            "new untyped conflict member must be current in the same scope and group"
        )
    if fact[2] == "state":
        raise StateSlotInvariantError("typed state must use the state conflict path")
    if fact[3] is not None or fact[4] is not None:
        raise StateSlotInvariantError("new conflict member must be current")
    conn.execute(
        "INSERT INTO fact_conflict_members(conflict_id, fact_id) VALUES (?, ?)",
        (conflict_id, fact_id),
    )


def add_conflict_member(
    conn: sqlite3.Connection, conflict_id: str, fact_id: int
) -> None:
    """Attach a newly inserted, exact-slot fact to an open conflict."""

    row = conn.execute(
        """
        SELECT c.scope, c.subject_key, c.predicate_key
        FROM fact_conflicts c
        WHERE c.conflict_id = ? AND c.resolved_at IS NULL
        """,
        (conflict_id,),
    ).fetchone()
    if row is None:
        raise ValueError("conflict does not exist or is already resolved")
    fact = conn.execute(
        """
        SELECT scope, subject_key, predicate_key, conflict_group, memory_kind,
               invalid_at, superseded_by
        FROM facts WHERE fact_id = ?
        """,
        (fact_id,),
    ).fetchone()
    if fact is None or fact[0:5] != (
        row[0], row[1], row[2], conflict_id, "state"
    ):
        raise StateSlotInvariantError(
            "new conflict member must be a state fact in the exact slot and group"
        )
    if fact[5] is not None or fact[6] is not None:
        raise StateSlotInvariantError("new conflict member must be current")
    conn.execute(
        "INSERT INTO fact_conflict_members(conflict_id, fact_id) VALUES (?, ?)",
        (conflict_id, fact_id),
    )


def resolve_state_conflict(
    conn: sqlite3.Connection,
    conflict_id: str,
    resolution_fact_id: int,
    *,
    resolved_by: str,
    reason: str,
    resolved_at: Optional[str] = None,
    resolver_client_id: Optional[str] = None,
    resolver_session_id: Optional[str] = None,
    resolver_agent_id: Optional[str] = None,
) -> ConflictResolution:
    """Resolve a conflict to one member and retain an audited history."""

    resolved_by = _required(resolved_by, "resolved_by")
    reason = _required(reason, "reason")
    resolved_at = resolved_at or _utc_now()
    _parse_timestamp(resolved_at)
    conflict = conn.execute(
        "SELECT resolved_at FROM fact_conflicts WHERE conflict_id = ?",
        (conflict_id,),
    ).fetchone()
    if conflict is None:
        raise ValueError("conflict does not exist")
    if conflict[0] is not None:
        raise ValueError("conflict is already resolved")
    members = tuple(
        int(row[0])
        for row in conn.execute(
            """SELECT fact_id FROM fact_conflict_members
               WHERE conflict_id = ? ORDER BY fact_id""",
            (conflict_id,),
        ).fetchall()
    )
    if resolution_fact_id not in members:
        raise ValueError("resolution fact must be a conflict member")
    losers = tuple(fact_id for fact_id in members if fact_id != resolution_fact_id)
    if losers:
        placeholders = ",".join("?" for _ in losers)
        cursor = conn.execute(
            f"""
            UPDATE facts
            SET invalid_at = ?, expired_at = ?, superseded_by = ?,
                valid_to = (
                    SELECT valid_from FROM facts WHERE fact_id = ?
                )
            WHERE fact_id IN ({placeholders})
              AND conflict_group = ? AND invalid_at IS NULL
            """,
            (
                resolved_at,
                resolved_at,
                resolution_fact_id,
                resolution_fact_id,
                *losers,
                conflict_id,
            ),
        )
        if cursor.rowcount != len(losers):
            raise StateSlotInvariantError("not all losing conflict members are current")
    cursor = conn.execute(
        """
        UPDATE facts SET conflict_group = NULL
        WHERE fact_id = ? AND conflict_group = ?
          AND invalid_at IS NULL AND superseded_by IS NULL
        """,
        (resolution_fact_id, conflict_id),
    )
    if cursor.rowcount != 1:
        raise StateSlotInvariantError("resolution fact is not a current member")
    conn.execute(
        """
        UPDATE fact_conflicts
        SET resolved_at = ?, resolution_fact_id = ?, resolved_by = ?,
            resolution_reason = ?
        WHERE conflict_id = ? AND resolved_at IS NULL
        """,
        (resolved_at, resolution_fact_id, resolved_by, reason, conflict_id),
    )
    if any(
        value is not None
        for value in (resolver_client_id, resolver_session_id, resolver_agent_id)
    ):
        resolver_client_id = _required(resolver_client_id or "", "resolver_client_id")
        resolver_session_id = _required(resolver_session_id or "", "resolver_session_id")
        resolver_agent_id = _required(resolver_agent_id or "", "resolver_agent_id")
        conn.execute(
            """
            INSERT INTO fact_conflict_resolutions (
                conflict_id, resolution_fact_id, resolver_client_id,
                resolver_session_id, resolver_agent_id, reason, resolved_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                conflict_id,
                resolution_fact_id,
                resolver_client_id,
                resolver_session_id,
                resolver_agent_id,
                reason,
                resolved_at,
            ),
        )
    return ConflictResolution(
        conflict_id,
        resolution_fact_id,
        losers,
        resolved_by,
        reason,
        resolved_at,
    )
