import sqlite3

from enfold.core_store import insert_fact
from enfold.schema import migrate
from enfold.state_slots import (
    StateCandidate,
    current_state_facts,
    decide_state_write,
    ensure_state_slot_schema,
    list_state_conflicts,
)


def test_state_candidates_canonicalize_before_slot_deduplication():
    conn = sqlite3.connect(":memory:")
    migrate(conn)
    first = StateCandidate(
        "User likes blue",
        "User",
        "favorite-color",
        object_value="blue",
    )
    fact_id = insert_fact(
        conn,
        first.content,
        memory_kind=first.memory_kind,
        subject_key=first.subject_key,
        predicate_key=first.predicate_key,
        object_value=first.object_value,
        scope=first.scope,
    )

    repeated = StateCandidate(
        "User's favorite color is blue",
        "user",
        "favorite_color",
        object_value="blue",
    )
    decision = decide_state_write(conn, repeated)

    assert (first.subject_key, first.predicate_key) == ("user", "favorite_color")
    assert decision.action == "dedup"
    assert decision.target_fact_id == fact_id
    assert len(current_state_facts(conn, "user", "favorite_color")) == 1


def test_migrate_canonicalizes_preexisting_state_slot_keys():
    conn = sqlite3.connect(":memory:")
    migrate(conn)
    fact_id = insert_fact(
        conn,
        "Avery's home zone is east",
        memory_kind="state",
        subject_key="Person:Avery",
        predicate_key="home-zone",
        object_value="east",
    )
    conn.commit()

    migrate(conn)

    decision = decide_state_write(
        conn,
        StateCandidate(
            "Avery lives in the east zone",
            "person:avery",
            "home_zone",
            object_value="east",
        ),
    )
    assert decision.action == "dedup"
    assert decision.target_fact_id == fact_id
    assert conn.execute(
        "SELECT subject_key, predicate_key FROM facts WHERE fact_id = ?",
        (fact_id,),
    ).fetchone() == ("person:avery", "home_zone")


def test_slot_key_repair_supersedes_older_canonical_collision():
    conn = sqlite3.connect(":memory:")
    migrate(conn)
    newer_id = insert_fact(
        conn,
        "Avery's current home zone is east",
        memory_kind="state",
        subject_key="Person:Avery",
        predicate_key="home-zone",
        object_value="east",
        valid_from="2026-07-20T00:00:00Z",
    )
    older_id = insert_fact(
        conn,
        "Avery's former home zone was west",
        memory_kind="state",
        subject_key="person:avery",
        predicate_key="home_zone",
        object_value="west",
        valid_from="2026-07-01T00:00:00Z",
    )
    conn.commit()

    assert ensure_state_slot_schema(conn) is True

    assert current_state_facts(conn, "person:avery", "home_zone")[0].fact_id == newer_id
    assert conn.execute(
        "SELECT invalid_at IS NOT NULL, superseded_by, subject_key, predicate_key "
        "FROM facts WHERE fact_id = ?",
        (older_id,),
    ).fetchone() == (1, newer_id, "person:avery", "home_zone")


def test_slot_key_repair_uses_created_at_when_valid_from_is_null():
    conn = sqlite3.connect(":memory:")
    migrate(conn)
    dated_id = insert_fact(
        conn,
        "Avery's former home zone was west",
        memory_kind="state",
        subject_key="Person:Avery",
        predicate_key="home-zone",
        object_value="west",
        valid_from="2026-01-01T00:00:00Z",
    )
    undated_id = insert_fact(
        conn,
        "Avery's current home zone is east",
        memory_kind="state",
        subject_key="person:avery",
        predicate_key="home_zone",
        object_value="east",
    )
    conn.execute(
        "UPDATE facts SET created_at = ? WHERE fact_id = ?",
        ("2026-07-20T00:00:00Z", undated_id),
    )
    conn.commit()

    assert ensure_state_slot_schema(conn) is True

    assert (
        current_state_facts(conn, "person:avery", "home_zone")[0].fact_id
        == undated_id
    )
    assert conn.execute(
        "SELECT invalid_at IS NOT NULL, superseded_by "
        "FROM facts WHERE fact_id = ?",
        (dated_id,),
    ).fetchone() == (1, undated_id)


def test_conflict_listing_caps_members_per_conflict_and_marks_truncation():
    conn = sqlite3.connect(":memory:")
    migrate(conn)
    conflict_members = {}
    for conflict_id, member_count in (
        ("conflict-large", 4),
        ("conflict-exact", 2),
    ):
        conn.execute(
            "INSERT INTO fact_conflicts("
            "conflict_id, subject_key, predicate_key, detected_at"
            ") VALUES (?, 'person:avery', 'status', ?)",
            (conflict_id, f"2026-07-20T00:00:0{len(conflict_members)}Z"),
        )
        fact_ids = []
        for number in range(member_count):
            fact_id = insert_fact(conn, f"{conflict_id} member {number}")
            conn.execute(
                "INSERT INTO fact_conflict_members(conflict_id, fact_id) "
                "VALUES (?, ?)",
                (conflict_id, fact_id),
            )
            fact_ids.append(fact_id)
        conflict_members[conflict_id] = tuple(fact_ids)

    unbounded = list_state_conflicts(conn)
    assert {
        record.conflict_id: record.member_fact_ids for record in unbounded
    } == conflict_members
    assert all(record.members_truncated is False for record in unbounded)

    capped = list_state_conflicts(conn, member_limit=2)
    assert {
        record.conflict_id: record.member_fact_ids for record in capped
    } == {
        conflict_id: fact_ids[:2]
        for conflict_id, fact_ids in conflict_members.items()
    }
    assert {
        record.conflict_id: record.members_truncated for record in capped
    } == {"conflict-large": True, "conflict-exact": False}
