from __future__ import annotations

from enfold.erasure import erase_fact
from enfold.projections import entity_dossier, timeline
from enfold.schema import migrate
from enfold.state_slots import (
    list_conflict_receipts,
    list_needs_review,
    read_current_state,
    record_needs_review,
)

from tests.test_erasure import _store, _write


def test_erased_conflict_member_cannot_be_recovered_as_settled_truth():
    conn, service, context = _store()
    secret = "Alex private timezone is UTC-8"
    slot = {"subject_key": "person:alex", "predicate_key": "timezone"}
    first = _write(
        service,
        context,
        "life-1",
        "Alex is in UTC-5",
        source_authority=0.8,
        state={**slot, "object_value": "UTC-5", "valid_from": "2026-01-01T00:00:00Z"},
    )
    second = _write(
        service,
        context,
        "life-2",
        secret,
        source_authority=0.2,
        state={**slot, "object_value": "UTC-8", "valid_from": "2026-01-02T00:00:00Z"},
    )
    insight_id = conn.execute(
        "INSERT INTO facts(content, category, tags) VALUES (?, 'insight', ?)",
        ("Derived from the erased timezone", f"source_facts:{second['fact_id']}"),
    ).lastrowid
    conn.commit()

    erase_fact(conn, second["fact_id"], requested_by="avery", reason="privacy")

    assert secret not in conn.execute(
        "SELECT content FROM facts WHERE fact_id = ?", (second["fact_id"],)
    ).fetchone()[0]
    assert conn.execute(
        "SELECT invalid_at FROM facts WHERE fact_id = ?", (insight_id,)
    ).fetchone()[0] is not None
    assert read_current_state(conn, "person:alex", "timezone") is None
    dossier = entity_dossier(conn, "person:alex", "private")
    assert secret not in str(dossier)
    assert all(
        fact["fact_id"] != first["fact_id"] for fact in dossier["current_facts"]
    )
    events = timeline(conn, "person:alex", "private")["events"]
    assert secret not in str(events)
    assert all(event["kind"] != "created" or event["fact"]["fact_id"] != first["fact_id"] for event in events)


def test_needs_review_and_conflict_receipts_are_reachable_after_migrate():
    conn, _service, _context = _store()
    migrate(conn)
    record_needs_review(
        conn,
        reason="client is not authorized to assert human correction",
        content="briefing uses Terra",
        subject_key="cron:briefing",
        predicate_key="model",
    )
    reviews = list_needs_review(conn)
    assert len(reviews) == 1
    assert reviews[0].subject_key == "cron:briefing"
    assert list_conflict_receipts(conn) == ()
