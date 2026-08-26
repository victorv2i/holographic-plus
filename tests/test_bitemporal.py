"""Bitemporal valid-time and transaction-time for facts.

Migration safety is tested first: an old-shape store with superseded rows
and conflicts must answer every pre-migration query identically after the
additive columns are applied.
"""

from __future__ import annotations

import sqlite3

from datetime import datetime, timezone

import pytest

from enfold.core_store import insert_fact
from enfold.policy import MemoryPolicy
from enfold.protocol import ClientContext, Request, optional_as_of_timestamp
from enfold.schema import SchemaError, migrate, require_compatible_schema
from enfold.service import EnfoldService
from enfold.state_slots import (
    StateCandidate,
    add_conflict_member,
    current_state_facts,
    decide_state_write,
    ensure_state_slot_schema,
    open_state_conflict,
    resolve_state_conflict,
)
from enfold.temporal import supersede


_LEGACY_FACT_COLUMNS = (
    "fact_id",
    "content",
    "category",
    "tags",
    "trust_score",
    "valid_from",
    "invalid_at",
    "superseded_by",
    "memory_kind",
    "subject_key",
    "predicate_key",
    "object_value",
    "source_authority",
    "scope",
    "conflict_group",
    "created_at",
    "updated_at",
)


def _legacy_snapshot(conn: sqlite3.Connection) -> dict[str, tuple]:
    columns = ", ".join(_LEGACY_FACT_COLUMNS)
    all_facts = tuple(
        tuple(row)
        for row in conn.execute(
            f"SELECT {columns} FROM facts ORDER BY fact_id"
        )
    )
    current = tuple(
        tuple(row)
        for row in conn.execute(
            f"""
            SELECT {columns} FROM facts
            WHERE invalid_at IS NULL AND superseded_by IS NULL
            ORDER BY fact_id
            """
        )
    )
    conflicts = tuple(
        tuple(row)
        for row in conn.execute(
            """
            SELECT conflict_id, scope, subject_key, predicate_key
            FROM fact_conflicts
            ORDER BY conflict_id
            """
        )
    )
    members = tuple(
        tuple(row)
        for row in conn.execute(
            """
            SELECT conflict_id, fact_id FROM fact_conflict_members
            ORDER BY conflict_id, fact_id
            """
        )
    )
    return {
        "all_facts": all_facts,
        "current": current,
        "conflicts": conflicts,
        "members": members,
    }


def _strip_bitemporal_columns(conn: sqlite3.Connection) -> None:
    """Force the pre-bitemporal v1 facts shape."""

    conn.execute("DROP INDEX IF EXISTS uq_facts_current_state_slot")
    columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(facts)")}
    for name in ("valid_to", "expired_at"):
        if name in columns:
            conn.execute(f'ALTER TABLE facts DROP COLUMN "{name}"')
    conn.execute(
        """
        CREATE UNIQUE INDEX uq_facts_current_state_slot
        ON facts(scope, subject_key, predicate_key)
        WHERE memory_kind = 'state'
          AND subject_key IS NOT NULL AND predicate_key IS NOT NULL
          AND invalid_at IS NULL AND superseded_by IS NULL
          AND conflict_group IS NULL
        """
    )
    conn.commit()


def _old_shape_store() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    migrate(conn)
    _strip_bitemporal_columns(conn)

    first = insert_fact(
        conn,
        "Acme CEO is Alice",
        memory_kind="state",
        subject_key="acme",
        predicate_key="ceo",
        object_value="alice",
        source_authority=0.8,
        valid_from="2022-01-01T00:00:00Z",
    )
    conn.execute(
        "UPDATE facts SET invalid_at = '2026-08-01T12:00:00+00:00' WHERE fact_id = ?",
        (first,),
    )
    second = insert_fact(
        conn,
        "Acme CEO is Bob",
        memory_kind="state",
        subject_key="acme",
        predicate_key="ceo",
        object_value="bob",
        source_authority=0.8,
        valid_from="2024-01-01T00:00:00Z",
    )
    conn.execute(
        "UPDATE facts SET superseded_by = ? WHERE fact_id = ?",
        (second, first),
    )

    undated = insert_fact(
        conn,
        "Widget port is 3100",
        memory_kind="state",
        subject_key="widget",
        predicate_key="port",
        object_value="3100",
        source_authority=0.5,
    )
    conn.execute(
        "UPDATE facts SET invalid_at = '2026-08-02T09:00:00+00:00' WHERE fact_id = ?",
        (undated,),
    )
    successor = insert_fact(
        conn,
        "Widget port is 3200",
        memory_kind="state",
        subject_key="widget",
        predicate_key="port",
        object_value="3200",
        source_authority=0.5,
    )
    conn.execute(
        "UPDATE facts SET superseded_by = ? WHERE fact_id = ?",
        (successor, undated),
    )

    left = insert_fact(
        conn,
        "Prefers Terra",
        memory_kind="state",
        subject_key="user:avery",
        predicate_key="model",
        object_value="terra",
        source_authority=0.6,
        valid_from="2026-07-01T00:00:00Z",
    )
    conflict = open_state_conflict(
        conn,
        "user:avery",
        "model",
        (left,),
        detected_at="2026-07-11T12:00:00Z",
        detail_json='{"reason":"authority-freshness-disagreement"}',
    )
    right = insert_fact(
        conn,
        "Prefers Helios",
        memory_kind="state",
        subject_key="user:avery",
        predicate_key="model",
        object_value="helios",
        source_authority=0.6,
        valid_from="2026-07-02T00:00:00Z",
    )
    conn.execute(
        "UPDATE facts SET conflict_group = ? WHERE fact_id = ?",
        (conflict.conflict_id, right),
    )
    conn.execute(
        "INSERT INTO fact_conflict_members(conflict_id, fact_id) VALUES (?, ?)",
        (conflict.conflict_id, right),
    )
    insert_fact(conn, "Untyped note about the office plants")
    conn.commit()
    return conn


def test_migration_preserves_pre_migration_query_bytes():
    conn = _old_shape_store()
    before = _legacy_snapshot(conn)
    assert before["current"]
    assert any(row[7] is not None for row in before["all_facts"])
    assert before["conflicts"]

    assert migrate(conn) == 1
    after = _legacy_snapshot(conn)

    assert after == before
    columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(facts)")}
    assert {"valid_to", "expired_at"} <= columns

    rows = conn.execute(
        """
        SELECT fact_id, superseded_by, invalid_at, expired_at, valid_from, valid_to
        FROM facts ORDER BY fact_id
        """
    ).fetchall()
    by_id = {int(row["fact_id"]): row for row in rows}
    alice = next(
        row for row in rows if row["valid_from"] == "2022-01-01T00:00:00Z"
    )
    bob = next(
        row for row in rows if row["valid_from"] == "2024-01-01T00:00:00Z"
    )
    undated = next(row for row in rows if row["valid_from"] is None and row["superseded_by"])
    assert alice["expired_at"] == alice["invalid_at"] == "2026-08-01T12:00:00+00:00"
    assert alice["valid_to"] == bob["valid_from"] == "2024-01-01T00:00:00Z"
    assert undated["expired_at"] == undated["invalid_at"]
    assert undated["valid_to"] is None
    assert bob["expired_at"] is None
    assert bob["valid_to"] is None
    assert by_id[alice["fact_id"]]["superseded_by"] == bob["fact_id"]
    require_compatible_schema(conn, for_writer=True)
    conn.close()


def test_unpatched_bitemporal_store_cannot_open_a_writer(tmp_path):
    conn = sqlite3.connect(tmp_path / "crash-mid-migration.db")
    conn.execute("PRAGMA foreign_keys=ON")
    migrate(conn)
    _strip_bitemporal_columns(conn)

    assert require_compatible_schema(conn) == 1
    with pytest.raises(SchemaError, match="migrate"):
        EnfoldService(conn, MemoryPolicy({"terminal-install": ("private",)}))

    assert migrate(conn) == 1
    service, context = _service(conn)
    written = _write_state(
        service,
        context,
        "bounded-boston",
        "Alice lived in Boston",
        subject_key="alice",
        predicate_key="city",
        object_value="boston",
        valid_from="2018-01-01T00:00:00Z",
        valid_to="2020-01-01T00:00:00Z",
    )
    assert written["outcome"] == "add"
    row = conn.execute(
        "SELECT valid_from, valid_to, expired_at FROM facts WHERE fact_id = ?",
        (written["fact_id"],),
    ).fetchone()
    assert tuple(row) == (
        "2018-01-01T00:00:00Z",
        "2020-01-01T00:00:00Z",
        None,
    )
    conn.close()


def test_old_binary_legacy_supersession_on_migrated_store(tmp_path):
    conn = sqlite3.connect(tmp_path / "old-binary.db")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    migrate(conn)
    alice = insert_fact(
        conn,
        "Acme CEO is Alice",
        memory_kind="state",
        subject_key="acme",
        predicate_key="ceo",
        object_value="alice",
        source_authority=0.8,
        valid_from="2020-01-01T00:00:00Z",
        trust_score=0.9,
    )
    conn.execute(
        "UPDATE facts SET invalid_at = '2026-08-01T12:00:00+00:00' WHERE fact_id = ?",
        (alice,),
    )
    bob = insert_fact(
        conn,
        "Acme CEO is Bob",
        memory_kind="state",
        subject_key="acme",
        predicate_key="ceo",
        object_value="bob",
        source_authority=0.8,
        valid_from="2022-01-01T00:00:00Z",
        trust_score=0.9,
    )
    conn.execute(
        "UPDATE facts SET superseded_by = ? WHERE fact_id = ?",
        (bob, alice),
    )
    conn.execute(
        "UPDATE facts SET created_at = '2020-01-01T00:00:00Z' WHERE fact_id = ?",
        (alice,),
    )
    conn.execute(
        "UPDATE facts SET created_at = '2022-01-01T00:00:00Z' WHERE fact_id = ?",
        (bob,),
    )
    conn.commit()

    assert require_compatible_schema(conn) == 1
    with pytest.raises(SchemaError, match="migrate"):
        require_compatible_schema(conn, for_writer=True)
    with pytest.raises(SchemaError, match="migrate"):
        _service(conn)

    assert migrate(conn) == 1
    assert require_compatible_schema(conn, for_writer=True) == 1
    service, context = _service(conn)
    later = service.handle(
        context,
        Request(
            "old-binary-as-of",
            "memory.search",
            {
                "query": "Acme CEO",
                "as_of_tx": "2026-09-01T00:00:00Z",
            },
        ),
    )
    later_ids = [fact["fact_id"] for fact in later["facts"]]
    assert later_ids == [bob]

    written = _write_state(
        service,
        context,
        "widget-port",
        "Widget port was 3100",
        subject_key="widget",
        predicate_key="port",
        object_value="3100",
        valid_from="2019-01-01T00:00:00Z",
        valid_to="2020-01-01T00:00:00Z",
    )
    assert written["outcome"] == "add"
    stored = conn.execute(
        "SELECT valid_to FROM facts WHERE fact_id = ?",
        (written["fact_id"],),
    ).fetchone()
    assert stored["valid_to"] == "2020-01-01T00:00:00Z"

    repaired = conn.execute(
        "SELECT expired_at, valid_to FROM facts WHERE fact_id = ?",
        (alice,),
    ).fetchone()
    assert repaired["expired_at"] == "2026-08-01T12:00:00+00:00"
    assert repaired["valid_to"] == "2022-01-01T00:00:00Z"
    conn.close()


def test_as_of_tx_reconstructs_belief_before_conflict_resolution(tmp_path):
    conn = sqlite3.connect(tmp_path / "conflict-as-of.db")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    migrate(conn)
    alice = insert_fact(
        conn,
        "Acme CEO is Alice",
        memory_kind="state",
        subject_key="acme",
        predicate_key="ceo",
        object_value="alice",
        source_authority=0.8,
        valid_from="2020-01-01T00:00:00Z",
        trust_score=0.9,
    )
    conn.execute(
        "UPDATE facts SET created_at = '2020-01-01T00:00:00Z' WHERE fact_id = ?",
        (alice,),
    )
    conflict = open_state_conflict(
        conn, "acme", "ceo", (alice,), detected_at="2023-01-01T00:00:00Z"
    )
    bob = insert_fact(
        conn,
        "Acme CEO is Bob",
        memory_kind="state",
        subject_key="acme",
        predicate_key="ceo",
        object_value="bob",
        source_authority=0.8,
        valid_from="2022-01-01T00:00:00Z",
        trust_score=0.9,
    )
    conn.execute(
        "UPDATE facts SET created_at = '2022-01-01T00:00:00Z' WHERE fact_id = ?",
        (bob,),
    )
    conn.execute(
        "UPDATE facts SET conflict_group = ? WHERE fact_id = ?",
        (conflict.conflict_id, bob),
    )
    add_conflict_member(conn, conflict.conflict_id, bob)
    resolve_state_conflict(
        conn,
        conflict.conflict_id,
        bob,
        resolved_by="hermes-install",
        reason="keep Bob",
        resolved_at="2024-01-01T00:00:00Z",
    )
    conn.commit()

    service, context = _service(conn)
    before = service.handle(
        context,
        Request(
            "as-of-2021",
            "memory.search",
            {
                "query": "Acme CEO",
                "as_of_tx": "2021-01-01T00:00:00Z",
            },
        ),
    )
    before_ids = [fact["fact_id"] for fact in before["facts"]]
    assert before_ids == [alice]

    after = service.handle(
        context,
        Request(
            "as-of-2025",
            "memory.search",
            {
                "query": "Acme CEO",
                "as_of_tx": "2025-01-01T00:00:00Z",
            },
        ),
    )
    after_ids = [fact["fact_id"] for fact in after["facts"]]
    assert after_ids == [bob]

    current = service.handle(
        context,
        Request("search-now", "memory.search", {"query": "Acme CEO"}),
    )
    current_ids = [fact["fact_id"] for fact in current["facts"]]
    assert current_ids == [bob]
    conn.close()


def _slot_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    migrate(conn)
    assert ensure_state_slot_schema(conn) is True
    return conn


def _insert_slot(
    conn: sqlite3.Connection,
    content: str,
    *,
    object_value: str,
    valid_from: str | None = None,
    valid_to: str | None = None,
    authority: float = 0.8,
    subject: str = "alice",
    predicate: str = "city",
):
    fact_id = insert_fact(
        conn,
        content,
        memory_kind="state",
        subject_key=subject,
        predicate_key=predicate,
        object_value=object_value,
        source_authority=authority,
        valid_from=valid_from,
    )
    if valid_to is not None:
        conn.execute(
            "UPDATE facts SET valid_to = ? WHERE fact_id = ?",
            (valid_to, fact_id),
        )
    return fact_id


def test_non_overlapping_intervals_both_stay_believed():
    conn = _slot_conn()
    boston = _insert_slot(
        conn,
        "Alice lived in Boston",
        object_value="boston",
        valid_from="2018-01-01T00:00:00Z",
        valid_to="2021-01-01T00:00:00Z",
    )
    decision = decide_state_write(
        conn,
        StateCandidate(
            "Alice lives in Denver",
            "alice",
            "city",
            "denver",
            source_authority=0.8,
            valid_from="2022-01-01T00:00:00Z",
        ),
    )
    assert decision.action == "add"
    current = current_state_facts(conn, "alice", "city")
    assert [fact.fact_id for fact in current] == []
    believed = conn.execute(
        "SELECT fact_id FROM facts WHERE fact_id = ? AND expired_at IS NULL",
        (boston,),
    ).fetchone()
    assert believed is not None
    conn.close()


def test_overlap_closes_old_interval_at_new_valid_from():
    conn = _slot_conn()
    boston = _insert_slot(
        conn,
        "Alice lived in Boston",
        object_value="boston",
        valid_from="2018-01-01T00:00:00Z",
    )
    decision = decide_state_write(
        conn,
        StateCandidate(
            "Alice lives in Denver",
            "alice",
            "city",
            "denver",
            source_authority=0.8,
            valid_from="2022-01-01T00:00:00Z",
        ),
    )
    assert decision.action == "supersede"
    assert decision.target_fact_id == boston
    conn.close()


def test_same_day_overlap_still_conflicts_without_a_clear_winner():
    conn = _slot_conn()
    _insert_slot(
        conn,
        "Alice lived in Boston",
        object_value="boston",
        valid_from="2022-01-01T00:00:00Z",
        authority=0.8,
    )
    decision = decide_state_write(
        conn,
        StateCandidate(
            "Alice lived in Denver",
            "alice",
            "city",
            "denver",
            source_authority=0.4,
            valid_from="2022-01-01T00:00:00Z",
        ),
    )
    assert decision.action == "conflict"
    conn.close()


def test_supersede_stamps_expired_at_and_optional_valid_to():
    conn = _slot_conn()
    old_id = insert_fact(conn, "port is 3100")
    new_id = insert_fact(conn, "port is 3200", valid_from="2024-03-01T00:00:00Z")
    before = datetime.now(timezone.utc)
    assert supersede(conn, old_id, new_id) is True
    row = conn.execute(
        "SELECT invalid_at, expired_at, valid_to, superseded_by FROM facts "
        "WHERE fact_id = ?",
        (old_id,),
    ).fetchone()
    assert row["superseded_by"] == new_id
    assert row["invalid_at"] is not None
    assert row["expired_at"] == row["invalid_at"]
    assert row["valid_to"] == "2024-03-01T00:00:00Z"
    stamped = datetime.fromisoformat(row["expired_at"].replace("Z", "+00:00"))
    if stamped.tzinfo is None:
        stamped = stamped.replace(tzinfo=timezone.utc)
    assert stamped >= before.replace(microsecond=0)
    conn.close()


def _service(conn: sqlite3.Connection) -> tuple[EnfoldService, ClientContext]:
    service = EnfoldService(conn, MemoryPolicy({"terminal-install": ("private",)}))
    context = ClientContext(
        client_id="terminal-install",
        surface="terminal",
        agent_id="terminal",
        session_id="bitemporal-session",
        access_scopes=("private",),
    )
    return service, context


def _write_state(service, context, key, content, **state):
    return service.handle(
        context,
        Request(
            f"req-{key}",
            "memory.write",
            {
                "idempotency_key": key,
                "content": content,
                "source_type": "agent_report",
                "source_authority": 0.9,
                "state": state,
            },
        ),
    )


def test_structural_supersede_does_not_block_a_second_service(tmp_path):
    path = tmp_path / "structural-supersede.db"
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys=ON")
    migrate(conn)
    service, context = _service(conn)
    original = service.handle(
        context,
        Request(
            "req-structural-original",
            "memory.write",
            {
                "idempotency_key": "structural-original",
                "content": "The deployment uses model v1",
                "source_type": "agent_report",
            },
        ),
    )

    service.handle(
        context,
        Request(
            "req-structural-replacement",
            "memory.write",
            {
                "idempotency_key": "structural-replacement",
                "content": "The deployment uses model v2",
                "source_type": "agent_report",
                "supersede_fact_id": original["fact_id"],
            },
        ),
    )
    conn.close()

    reopened = sqlite3.connect(path)
    EnfoldService(reopened, MemoryPolicy({"terminal-install": ("private",)}))
    reopened.close()


def test_optional_as_of_timestamp_rejects_empty_and_non_iso():
    assert optional_as_of_timestamp(None, "as_of_valid") is None
    assert (
        optional_as_of_timestamp("2023-04-15T00:00:00Z", "as_of_valid")
        == "2023-04-15T00:00:00Z"
    )
    try:
        optional_as_of_timestamp("not-a-date", "as_of_valid")
    except ValueError as exc:
        assert "as_of_valid" in str(exc)
    else:
        raise AssertionError("expected invalid as_of timestamp to fail")


def test_service_as_of_and_retroactive_correction(tmp_path):
    conn = sqlite3.connect(tmp_path / "bitemporal-service.db")
    conn.execute("PRAGMA foreign_keys=ON")
    migrate(conn)
    service, context = _service(conn)

    recorded = _write_state(
        service,
        context,
        "ceo-alice",
        "Acme CEO was Alice from March through June 2023",
        subject_key="acme",
        predicate_key="ceo",
        object_value="alice",
        valid_from="2023-03-01T00:00:00Z",
        valid_to="2023-06-01T00:00:00Z",
    )
    current = _write_state(
        service,
        context,
        "ceo-bob",
        "Acme CEO is Bob",
        subject_key="acme",
        predicate_key="ceo",
        object_value="bob",
        valid_from="2023-06-01T00:00:00Z",
    )
    assert recorded["outcome"] == "add"
    assert current["outcome"] == "add"

    default_search = service.handle(
        context,
        Request("search-now", "memory.search", {"query": "Acme CEO"}),
    )
    default_ids = [fact["fact_id"] for fact in default_search["facts"]]
    assert current["fact_id"] in default_ids
    assert recorded["fact_id"] not in default_ids

    april = service.handle(
        context,
        Request(
            "search-april",
            "memory.search",
            {
                "query": "Acme CEO",
                "as_of_valid": "2023-04-15T00:00:00Z",
            },
        ),
    )
    april_ids = [fact["fact_id"] for fact in april["facts"]]
    assert recorded["fact_id"] in april_ids
    assert current["fact_id"] not in april_ids

    february_belief = service.handle(
        context,
        Request(
            "search-feb-tx",
            "memory.search",
            {
                "query": "Acme CEO",
                "as_of_tx": "2023-02-01T00:00:00Z",
            },
        ),
    )
    assert february_belief["facts"] == []

    alice = conn.execute(
        "SELECT valid_from, valid_to, expired_at, invalid_at FROM facts "
        "WHERE fact_id = ?",
        (recorded["fact_id"],),
    ).fetchone()
    assert tuple(alice)[:2] == ("2023-03-01T00:00:00Z", "2023-06-01T00:00:00Z")
    assert alice[2] is None
    assert alice[3] is not None
    conn.close()


def test_service_overlap_closes_world_time_not_clock_now(tmp_path):
    conn = sqlite3.connect(tmp_path / "bitemporal-overlap.db")
    conn.execute("PRAGMA foreign_keys=ON")
    migrate(conn)
    service, context = _service(conn)
    first = _write_state(
        service,
        context,
        "boston",
        "Alice lived in Boston",
        subject_key="alice",
        predicate_key="city",
        object_value="boston",
        valid_from="2018-01-01T00:00:00Z",
    )
    second = _write_state(
        service,
        context,
        "denver",
        "Alice lives in Denver",
        subject_key="alice",
        predicate_key="city",
        object_value="denver",
        valid_from="2022-01-01T00:00:00Z",
    )
    assert second["outcome"] == "supersede"
    old = conn.execute(
        "SELECT valid_to, expired_at, invalid_at, superseded_by FROM facts "
        "WHERE fact_id = ?",
        (first["fact_id"],),
    ).fetchone()
    assert old["valid_to"] == "2022-01-01T00:00:00Z"
    assert old["superseded_by"] == second["fact_id"]
    assert old["invalid_at"] is not None
    assert old["expired_at"] == old["invalid_at"]
    assert old["valid_to"] != old["invalid_at"]
    conn.close()
