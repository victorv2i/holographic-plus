from __future__ import annotations

import os
import sqlite3

from enfold.core_store import insert_fact
from enfold.erasure import erase_fact
from enfold.export import export_current
from enfold.ops import main
from enfold.schema import migrate


def _v1(tmp_path):
    database = tmp_path / "memory.db"
    conn = sqlite3.connect(database)
    conn.execute("PRAGMA foreign_keys=ON")
    migrate(conn)
    return database, conn


def _observe(conn, fact_id, *, source_uri, excerpt):
    conn.execute(
        "INSERT OR IGNORE INTO memory_clients(client_id, surface, created_at) "
        "VALUES ('export-test', 'cli', '2026-08-24T00:00:00+00:00')"
    )
    conn.execute(
        "INSERT OR IGNORE INTO memory_sessions("
        "session_id, client_id, agent_id, capabilities_json, access_scopes_json"
        ") VALUES ('s1', 'export-test', 'owner', '[]', '[\"private\"]')"
    )
    cursor = conn.execute(
        "INSERT INTO observations("
        "client_id, session_id, source_type, source_uri, content, "
        "content_sha256, recorded_at, scope, sensitivity"
        ") VALUES ("
        "'export-test', 's1', 'user_statement', ?, ?, ?, "
        "'2026-08-24T00:00:00+00:00', 'private', 'normal')",
        (source_uri, excerpt, f"sha-{fact_id}"),
    )
    observation_id = int(cursor.lastrowid)
    conn.execute(
        "INSERT INTO fact_provenance("
        "fact_id, observation_id, relation, evidence_excerpt, created_at"
        ") VALUES (?, ?, 'supports', ?, '2026-08-24T00:00:00+00:00')",
        (fact_id, observation_id, excerpt),
    )


def test_export_current_writes_active_facts_and_parks_conflicts(tmp_path):
    database, conn = _v1(tmp_path)
    current_id = insert_fact(conn, "Ada prefers Terra for daily briefing.")
    _observe(
        conn,
        current_id,
        source_uri="file:///tmp/notes.md",
        excerpt="Ada prefers Terra for daily briefing.",
    )
    superseded = insert_fact(conn, "Ada preferred Codex last year.")
    conn.execute(
        "UPDATE facts SET superseded_by = ? WHERE fact_id = ?",
        (current_id, superseded),
    )
    left = insert_fact(conn, "Home lab uses SQLite.")
    right = insert_fact(conn, "Home lab uses Postgres.")
    conn.execute(
        "UPDATE facts SET conflict_group = 'lab-store' WHERE fact_id IN (?, ?)",
        (left, right),
    )
    conn.execute(
        "INSERT INTO fact_conflicts("
        "conflict_id, scope, subject_key, predicate_key, detected_at"
        ") VALUES ('lab-store', 'private', 'home-lab', 'store', "
        "'2026-08-24T00:00:00+00:00')"
    )
    conn.execute(
        "INSERT INTO fact_conflict_members(conflict_id, fact_id) VALUES "
        "('lab-store', ?), ('lab-store', ?)",
        (left, right),
    )
    unreviewed = insert_fact(conn, "Unverified capture about a weekend plan.")
    conn.execute(
        "UPDATE facts SET correction_status = 'unreviewed' WHERE fact_id = ?",
        (unreviewed,),
    )
    conn.commit()
    conn.close()

    destination = tmp_path / "export"
    report = export_current(database, destination)

    current = (destination / "current.md").read_text(encoding="utf-8")
    conflicts = (destination / "needs_review" / "conflicts.md").read_text(
        encoding="utf-8"
    )
    review = (destination / "needs_review" / "unreviewed.md").read_text(
        encoding="utf-8"
    )
    assert "Ada prefers Terra for daily briefing." in current
    assert "file:///tmp/notes.md" in current
    assert "Ada preferred Codex last year." not in current
    assert "Home lab uses SQLite." not in current
    assert "Unverified capture about a weekend plan." not in current
    assert "NEEDS REVIEW" in conflicts
    assert "Home lab uses SQLite." in conflicts
    assert "Home lab uses Postgres." in conflicts
    assert "Unverified capture about a weekend plan." in review
    assert report.current_facts == 1
    assert report.conflicted_facts == 2
    assert report.unreviewed_facts == 1


def test_export_current_omits_erased_content_and_is_not_an_import(tmp_path):
    database, conn = _v1(tmp_path)
    fact_id = insert_fact(conn, "The secret nickname is Moth.")
    _observe(
        conn,
        fact_id,
        source_uri="file:///tmp/diary.md",
        excerpt="The secret nickname is Moth.",
    )
    conn.commit()
    erase_fact(conn, fact_id, requested_by="owner", reason="privacy request")
    conn.commit()
    conn.close()

    destination = tmp_path / "export"
    export_current(database, destination)

    current = (destination / "current.md").read_text(encoding="utf-8")
    review_dir = destination / "needs_review"
    review_text = "".join(
        path.read_text(encoding="utf-8") for path in review_dir.glob("*.md")
    )
    assert "Moth" not in current
    assert "Moth" not in review_text
    assert "diary.md" not in current
    assert "The secret nickname is Moth." not in current
    assert "cannot be imported" in current.lower() or "not a store" in current.lower()
    with sqlite3.connect(database) as conn:
        assert conn.execute("SELECT content FROM facts").fetchone()[0].startswith(
            "[PRIVACY ERASED"
        )


def test_export_current_omits_erased_evidence_shared_by_current_fact(tmp_path):
    database, conn = _v1(tmp_path)
    secret = "PRIVATE-DIAGNOSIS-XYZ-99"
    erased_id = insert_fact(conn, "Avery completed a private clinic visit")
    kept_id = insert_fact(conn, "Avery listed a backup contact")
    _observe(
        conn,
        erased_id,
        source_uri="file:///tmp/clinic.md",
        excerpt=secret,
    )
    observation_id = conn.execute(
        "SELECT observation_id FROM fact_provenance WHERE fact_id = ?",
        (erased_id,),
    ).fetchone()[0]
    conn.execute(
        "INSERT INTO fact_provenance("
        "fact_id, observation_id, relation, evidence_excerpt, created_at"
        ") VALUES (?, ?, 'supports', ?, '2026-08-24T00:00:00+00:00')",
        (kept_id, observation_id, secret),
    )
    conn.commit()
    erase_fact(conn, erased_id, requested_by="owner", reason="privacy request")
    conn.commit()
    conn.close()

    destination = tmp_path / "export"
    export_current(database, destination)

    current = (destination / "current.md").read_text(encoding="utf-8")
    review_text = "".join(
        path.read_text(encoding="utf-8")
        for path in (destination / "needs_review").glob("*.md")
    )
    assert "Avery listed a backup contact" in current
    assert "Avery completed a private clinic visit" not in current
    assert secret not in current
    assert secret not in review_text


def test_export_current_creates_owner_only_output(tmp_path):
    database, conn = _v1(tmp_path)
    insert_fact(conn, "Visible current fact.")
    conn.commit()
    conn.close()

    destination = tmp_path / "export"
    previous_umask = os.umask(0o022)
    try:
        export_current(database, destination)
    finally:
        os.umask(previous_umask)

    current_path = destination / "current.md"
    review_dir = destination / "needs_review"
    assert destination.stat().st_mode & 0o777 == 0o700
    assert review_dir.stat().st_mode & 0o777 == 0o700
    assert current_path.stat().st_mode & 0o777 == 0o600
    assert (review_dir / "conflicts.md").stat().st_mode & 0o777 == 0o600
    assert (review_dir / "unreviewed.md").stat().st_mode & 0o777 == 0o600


def test_ops_export_current_is_read_only(tmp_path, capsys):
    database, conn = _v1(tmp_path)
    insert_fact(conn, "Visible current fact.")
    conn.commit()
    before = conn.execute("SELECT content FROM facts").fetchall()
    conn.close()
    destination = tmp_path / "out"

    assert main(["export", "--current", str(database), str(destination)]) == 0

    output = capsys.readouterr().out
    assert "current.md" in output or '"ok"' in output
    assert (destination / "current.md").is_file()
    assert "Visible current fact." in (
        destination / "current.md"
    ).read_text(encoding="utf-8")
    with sqlite3.connect(database) as conn:
        assert conn.execute("SELECT content FROM facts").fetchall() == before
