"""Bounded time and entity projections over the Enfold fact store."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
import json
import sqlite3
from typing import Any

from .core_store import active_facts, build_visibility_predicate


DEFAULT_LIMIT = 100
MAX_LIMIT = 200
MAX_OUTPUT_CHARS = 60_000
MAX_QUERY_CHARS = 16_000
MAX_ENTITY_CHARS = 2_000
PROJECTION_SCAN_LIMIT = 10_000
_ITEMS_CHAR_BUDGET = MAX_OUTPUT_CHARS - 256
_FACT_FIELDS = (
    "fact_id", "content", "category", "tags", "trust_score", "created_at",
    "updated_at", "valid_from", "invalid_at", "superseded_by", "memory_kind",
    "subject_key", "predicate_key", "object_value", "confidence",
    "source_authority", "scope", "sensitivity", "correction_status",
    "schema_version", "conflict_group",
)


def _scopes(scope: str | Sequence[str]) -> tuple[str, ...]:
    values = (scope,) if isinstance(scope, str) else tuple(scope)
    cleaned = tuple(dict.fromkeys(value.strip() for value in values if value.strip()))
    if not cleaned:
        raise ValueError("scope must not be empty")
    return cleaned


def _limit(value: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 1 <= value <= MAX_LIMIT
    ):
        raise ValueError(f"limit must be between 1 and {MAX_LIMIT}")
    return value


def _timestamp(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty ISO-8601 timestamp")
    candidate = value.strip()
    try:
        parsed = datetime.fromisoformat(candidate.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{name} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat(timespec="microseconds")


def _fact(row: sqlite3.Row | dict[str, Any]) -> dict[str, Any]:
    return {key: row[key] for key in _FACT_FIELDS if key in row.keys()}


def _json_chars(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":")))


def _bounded(items: Iterable[dict[str, Any]], limit: int) -> tuple[list[dict[str, Any]], bool]:
    result: list[dict[str, Any]] = []
    used = 2
    truncated = False
    for item in items:
        if len(result) >= limit:
            truncated = True
            break
        item_size = _json_chars(item) + (1 if result else 0)
        if used + item_size > _ITEMS_CHAR_BUDGET:
            truncated = True
            break
        result.append(item)
        used += item_size
    return result, truncated


def _event_rows(
    conn: sqlite3.Connection,
    scopes: tuple[str, ...],
    *,
    since: str | None = None,
    until: str | None = None,
) -> tuple[list[dict[str, Any]], bool]:
    visibility_sql, visibility_params = build_visibility_predicate(
        scopes, scope_column="f.scope", sensitivity_column="f.sensitivity"
    )
    window_sql = ""
    window_params: tuple[str, ...] = ()
    if since is not None and until is not None:
        window_sql = (
            "AND julianday(e.changed_at) >= julianday(?) "
            "AND julianday(e.changed_at) < julianday(?)"
        )
        window_params = (since, until)
    selected = ", ".join(f"f.{name}" for name in _FACT_FIELDS)
    rows = conn.execute(
        f"""
        WITH fact_events AS (
            SELECT 'created' AS kind, f.created_at AS changed_at, f.fact_id
            FROM facts f
            WHERE {visibility_sql} AND f.conflict_group IS NULL
            UNION ALL
            SELECT 'superseded', f.invalid_at, f.fact_id
            FROM facts f
            WHERE {visibility_sql} AND f.invalid_at IS NOT NULL
              AND f.superseded_by IS NOT NULL
            UNION ALL
            SELECT 'resolved', c.resolved_at, c.resolution_fact_id
            FROM fact_conflicts c
            JOIN facts f ON f.fact_id = c.resolution_fact_id
            WHERE {visibility_sql} AND c.resolved_at IS NOT NULL
            UNION ALL
            SELECT 'conflicted', c.detected_at, MIN(f.fact_id)
            FROM fact_conflicts c
            JOIN fact_conflict_members m ON m.conflict_id = c.conflict_id
            JOIN facts f ON f.fact_id = m.fact_id
            WHERE {visibility_sql} AND c.resolved_at IS NULL
            GROUP BY c.conflict_id, c.detected_at
        ), newest_events AS (
            SELECT e.kind, e.changed_at, {selected}
            FROM fact_events e
            JOIN facts f ON f.fact_id = e.fact_id
            WHERE e.changed_at IS NOT NULL {window_sql}
            ORDER BY julianday(e.changed_at) DESC, e.changed_at DESC,
                     CASE e.kind
                         WHEN 'created' THEN 0
                         WHEN 'superseded' THEN 1
                         WHEN 'conflicted' THEN 2
                         ELSE 3
                     END DESC,
                     f.fact_id DESC
            LIMIT ?
        )
        SELECT * FROM newest_events
        ORDER BY julianday(changed_at), changed_at,
                 CASE kind
                     WHEN 'created' THEN 0
                     WHEN 'superseded' THEN 1
                     WHEN 'conflicted' THEN 2
                     ELSE 3
                 END,
                 fact_id
        """,
        (
            *visibility_params,
            *visibility_params,
            *visibility_params,
            *visibility_params,
            *window_params,
            PROJECTION_SCAN_LIMIT,
        ),
    ).fetchall()
    events = [
        {
            "kind": str(row["kind"]),
            "changed_at": str(row["changed_at"]),
            "fact": _fact(row),
        }
        for row in rows
    ]
    return events, len(rows) == PROJECTION_SCAN_LIMIT


def changes(
    conn: sqlite3.Connection,
    since: str,
    until: str,
    scope: str | Sequence[str],
    *,
    limit: int = DEFAULT_LIMIT,
) -> dict[str, Any]:
    """Return settled fact changes in the half-open interval ``[since, until)``."""

    since_value = _timestamp(since, "since")
    until_value = _timestamp(until, "until")
    if since_value >= until_value:
        raise ValueError("since must be earlier than until")
    scanned, scan_truncated = _event_rows(
        conn, _scopes(scope), since=since_value, until=until_value
    )
    events, output_truncated = _bounded(scanned, _limit(limit))
    return {"changes": events, "truncated": scan_truncated or output_truncated}


def _entity_names(fact: dict[str, Any]) -> dict[str, str]:
    names: dict[str, str] = {}
    subject = fact.get("subject_key")
    if isinstance(subject, str) and subject.strip():
        names[subject.strip().casefold()] = subject.strip()
    tags = fact.get("tags")
    if isinstance(tags, str):
        for tag in tags.split(","):
            if tag.strip():
                names.setdefault(tag.strip().casefold(), tag.strip())
    return names


def _matches_entity(fact: dict[str, Any], name: str) -> bool:
    return name.casefold() in _entity_names(fact)


def _matches_entity_fields(subject: Any, tags: Any, name: str) -> bool:
    return _matches_entity({"subject_key": subject, "tags": tags}, name)


def _entity_events(
    conn: sqlite3.Connection,
    entity: str,
    scopes: tuple[str, ...],
    limit: int,
) -> tuple[list[dict[str, Any]], bool]:
    scanned, scan_truncated = _event_rows(conn, scopes)
    matched = [
        event
        for event in scanned
        if _matches_entity(event["fact"], entity)
    ]
    selected = matched[-limit:]
    events, chars_truncated = _bounded(selected, limit)
    return events, scan_truncated or len(matched) > limit or chars_truncated


def _entity_conflicts(
    conn: sqlite3.Connection,
    entity: str,
    scopes: tuple[str, ...],
    limit: int,
) -> tuple[list[dict[str, Any]], bool]:
    scope_values = ", ".join("(?, ?)" for _ in scopes)
    scope_params = tuple(
        value for ordinal, scope in enumerate(scopes) for value in (scope, ordinal)
    )
    selected = ", ".join(
        f"f.{name} AS member_{name}" for name in _FACT_FIELDS
    )
    entity_visibility_sql, entity_visibility_params = build_visibility_predicate(
        scopes,
        scope_column="entity_fact.scope",
        sensitivity_column="entity_fact.sensitivity",
    )
    member_visibility_sql, member_visibility_params = build_visibility_predicate(
        scopes,
        scope_column="f.scope",
        sensitivity_column="f.sensitivity",
    )
    conn.create_function(
        "_enfold_matches_entity", 3, _matches_entity_fields, deterministic=True
    )
    cursor = conn.execute(
        f"""
        WITH authorized_scopes(scope, ordinal) AS (VALUES {scope_values}),
        matching_conflicts AS (
            SELECT c.*, authorized.ordinal
            FROM authorized_scopes authorized
            JOIN fact_conflicts c ON c.scope = authorized.scope
            WHERE c.resolved_at IS NULL
              AND EXISTS (
                  SELECT 1
                  FROM fact_conflict_members visible_member
                  JOIN facts entity_fact
                    ON entity_fact.fact_id = visible_member.fact_id
                   AND entity_fact.scope = c.scope
                  WHERE visible_member.conflict_id = c.conflict_id
                    AND {entity_visibility_sql}
              )
              AND (
                  _enfold_matches_entity(c.subject_key, NULL, ?)
                  OR EXISTS (
                      SELECT 1
                      FROM fact_conflict_members entity_member
                      JOIN facts entity_fact
                        ON entity_fact.fact_id = entity_member.fact_id
                       AND entity_fact.scope = c.scope
                      WHERE entity_member.conflict_id = c.conflict_id
                        AND {entity_visibility_sql}
                        AND _enfold_matches_entity(
                            entity_fact.subject_key, entity_fact.tags, ?
                        )
                  )
              )
            ORDER BY authorized.ordinal, c.detected_at, c.conflict_id
            LIMIT ?
        ), ranked_members AS (
            SELECT m.conflict_id, m.fact_id,
                   ROW_NUMBER() OVER (
                       PARTITION BY m.conflict_id ORDER BY m.fact_id
                   ) AS member_ordinal,
                   COUNT(*) OVER (
                       PARTITION BY m.conflict_id
                   ) AS member_count
            FROM fact_conflict_members m
            JOIN matching_conflicts c ON c.conflict_id = m.conflict_id
            JOIN facts f ON f.fact_id = m.fact_id
            WHERE {member_visibility_sql}
        )
        SELECT c.conflict_id, c.scope, c.subject_key, c.predicate_key,
               c.detected_at, m.member_count, {selected}
        FROM matching_conflicts c
        JOIN ranked_members m
          ON m.conflict_id = c.conflict_id AND m.member_ordinal <= ?
        JOIN facts f ON f.fact_id = m.fact_id AND f.scope = c.scope
        WHERE c.resolved_at IS NULL
        ORDER BY c.ordinal, c.detected_at, c.conflict_id, m.fact_id
        """,
        (
            *scope_params,
            *entity_visibility_params,
            entity,
            *entity_visibility_params,
            entity,
            limit + 1,
            *member_visibility_params,
            limit,
        ),
    )
    conflicts: list[dict[str, Any]] = []
    conflict_id: str | None = None
    conflict: dict[str, Any] | None = None
    members_truncated = False

    def finish_conflict() -> None:
        nonlocal members_truncated
        if conflict is None:
            return
        if conflict["members_truncated"]:
            members_truncated = True
        conflicts.append(conflict)

    try:
        for row in cursor:
            row_conflict_id = str(row["conflict_id"])
            if row_conflict_id != conflict_id:
                finish_conflict()
                if len(conflicts) > limit:
                    break
                conflict_id = row_conflict_id
                conflict = {
                    "conflict_id": row_conflict_id,
                    "scope": str(row["scope"]),
                    "subject_key": str(row["subject_key"]),
                    "predicate_key": str(row["predicate_key"]),
                    "detected_at": str(row["detected_at"]),
                    "members": [],
                    "members_truncated": int(row["member_count"]) > limit,
                }
            if conflict is not None:
                conflict["members"].append({
                    name: row[f"member_{name}"] for name in _FACT_FIELDS
                })
        else:
            finish_conflict()
    finally:
        cursor.close()
    return conflicts[:limit], len(conflicts) > limit or members_truncated


def timeline(
    conn: sqlite3.Connection,
    subject_or_query: str,
    scope: str | Sequence[str],
    *,
    limit: int = DEFAULT_LIMIT,
) -> dict[str, Any]:
    """Return matching events from the newest ``PROJECTION_SCAN_LIMIT`` events."""

    if not isinstance(subject_or_query, str) or not subject_or_query.strip():
        raise ValueError("subject_or_query must not be empty")
    if len(subject_or_query) > MAX_QUERY_CHARS:
        raise ValueError(
            f"subject_or_query must not exceed {MAX_QUERY_CHARS} characters"
        )
    cap = _limit(limit)
    query = subject_or_query.strip().casefold()
    scanned, scan_truncated = _event_rows(conn, _scopes(scope))
    matched = []
    for event in scanned:
        fact = event["fact"]
        text = " ".join(
            str(fact.get(field) or "")
            for field in ("content", "subject_key", "predicate_key", "object_value", "tags")
        ).casefold()
        if query in text:
            matched.append(event)
    selected = matched[-cap:]
    events, chars_truncated = _bounded(selected, cap)
    return {
        "events": events,
        "truncated": scan_truncated or len(matched) > cap or chars_truncated,
    }


def entities(
    conn: sqlite3.Connection,
    scope: str | Sequence[str],
    min_facts: int = 1,
    *,
    limit: int = DEFAULT_LIMIT,
) -> dict[str, Any]:
    """Rank entities across the newest ``PROJECTION_SCAN_LIMIT`` current facts."""

    if (
        isinstance(min_facts, bool)
        or not isinstance(min_facts, int)
        or min_facts < 1
    ):
        raise ValueError("min_facts must be a positive integer")
    counts: dict[str, set[int]] = defaultdict(set)
    display: dict[str, str] = {}
    sources: dict[str, set[str]] = defaultdict(set)
    scanned = active_facts(
        conn, allowed_scopes=_scopes(scope), limit=PROJECTION_SCAN_LIMIT
    )
    for fact in scanned:
        fact_id = int(fact["fact_id"])
        subject = fact.get("subject_key")
        if isinstance(subject, str) and subject.strip():
            key = subject.strip().casefold()
            display.setdefault(key, subject.strip())
            counts[key].add(fact_id)
            sources[key].add("subject")
        tags = fact.get("tags")
        if isinstance(tags, str):
            for raw in tags.split(","):
                if not raw.strip():
                    continue
                key = raw.strip().casefold()
                display.setdefault(key, raw.strip())
                counts[key].add(fact_id)
                sources[key].add("tag")
    ranked = [
        {
            "name": display[key],
            "fact_count": len(fact_ids),
            "derived_from": sorted(sources[key]),
        }
        for key, fact_ids in counts.items()
        if len(fact_ids) >= min_facts
    ]
    ranked.sort(
        key=lambda item: (
            -item["fact_count"],
            item["name"].casefold(),
            item["name"],
        )
    )
    result, output_truncated = _bounded(ranked, _limit(limit))
    return {
        "entities": result,
        "truncated": len(scanned) == PROJECTION_SCAN_LIMIT or output_truncated,
    }


def entity_dossier(
    conn: sqlite3.Connection,
    name: str,
    scope: str | Sequence[str],
    *,
    limit: int = DEFAULT_LIMIT,
) -> dict[str, Any]:
    """Return an entity dossier, scanning at most ``PROJECTION_SCAN_LIMIT`` facts/events."""

    if not isinstance(name, str) or not name.strip():
        raise ValueError("name must not be empty")
    if len(name) > MAX_ENTITY_CHARS:
        raise ValueError(f"name must not exceed {MAX_ENTITY_CHARS} characters")
    entity = name.strip()
    scopes = _scopes(scope)
    cap = _limit(limit)
    scanned_current = active_facts(
        conn, allowed_scopes=scopes, limit=PROJECTION_SCAN_LIMIT
    )
    all_current = [
        _fact(fact)
        for fact in scanned_current
        if _matches_entity(fact, entity)
    ]
    current = all_current[:cap]
    recent, recent_truncated = _entity_events(conn, entity, scopes, cap)
    conflicts, conflicts_truncated = _entity_conflicts(conn, entity, scopes, cap)
    result: dict[str, Any] = {
        "entity": entity,
        "current_facts": current,
        "recent_changes": recent,
        "open_conflicts": conflicts,
        "truncated": (
            len(all_current) > cap
            or len(scanned_current) == PROJECTION_SCAN_LIMIT
            or conflicts_truncated
            or recent_truncated
        ),
    }
    sections = ("recent_changes", "current_facts", "open_conflicts")
    while _json_chars(result) > MAX_OUTPUT_CHARS:
        for section in sections:
            if result[section]:
                result[section].pop()
                result["truncated"] = True
                break
        else:
            break
    return result
