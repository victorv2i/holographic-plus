import json
import sqlite3
import hashlib
import os
from pathlib import Path

import numpy as np
import pytest

from enfold.embeddings import embedding_to_bytes
from enfold.ops import _connect, main
from enfold.core_store import insert_fact
from enfold.schema import migrate
from enfold.rehearsal import create_legacy_fixture
from enfold.server import DatabaseOwnership


def _database(path):
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE facts(fact_id INTEGER PRIMARY KEY, content TEXT)")
    conn.execute("INSERT INTO facts(content) VALUES ('shared memory')")
    conn.commit()
    conn.close()


def test_schema_status_is_read_only(tmp_path, capsys):
    database = tmp_path / "legacy.db"
    _database(database)

    assert main(["schema-status", str(database)]) == 0

    output = json.loads(capsys.readouterr().out)
    assert output["schema_version"] == 0
    with sqlite3.connect(database) as conn:
        assert conn.execute(
            "SELECT name FROM sqlite_master WHERE name = 'schema_migrations'"
        ).fetchone() is None


def test_read_only_connect_percent_encodes_sqlite_uri_path(tmp_path):
    database = tmp_path / "live?tenant=1.sqlite"
    _database(database)

    with _connect(database, read_only=True) as conn:
        assert conn.execute("SELECT content FROM facts").fetchone() == (
            "shared memory",
        )
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            conn.execute("INSERT INTO facts(content) VALUES ('must fail')")


def test_migrate_is_explicit_and_reports_versions(tmp_path, capsys):
    database = tmp_path / "memory.db"
    _database(database)

    assert main(["migrate", str(database)]) == 0

    output = json.loads(capsys.readouterr().out)
    assert output["schema_version_before"] == 0
    assert output["schema_version_after"] == 1


def test_migrate_under_hermes_requires_maintenance_override(tmp_path, capsys):
    database = tmp_path / ".hermes" / "memory.db"
    database.parent.mkdir()
    _database(database)

    assert main(["migrate", str(database)]) == 2

    error = capsys.readouterr().err
    assert "maintenance window" in error
    assert "--allow-live" in error
    with sqlite3.connect(database) as conn:
        assert conn.execute(
            "SELECT name FROM sqlite_master WHERE name = 'schema_migrations'"
        ).fetchone() is None

    assert main(["migrate", str(database), "--allow-live"]) == 0


def test_backup_may_read_hermes_source_with_explicit_destination(tmp_path, capsys):
    source = tmp_path / ".hermes" / "memory.db"
    source.parent.mkdir()
    destination = tmp_path / "safe" / "memory-backup.db"
    _database(source)

    assert main(["backup", str(source), str(destination)]) == 0

    output = json.loads(capsys.readouterr().out)
    assert output["ok"] is True
    assert destination.is_file()


def test_verify_reports_integrity_and_row_counts(tmp_path, capsys):
    database = tmp_path / "memory.db"
    _database(database)

    assert main(["verify", str(database)]) == 0

    output = json.loads(capsys.readouterr().out)
    assert output["ok"] is True
    assert output["report"]["integrity_check"] == ["ok"]
    assert output["report"]["row_counts"]["facts"] == 1


def test_live_verify_is_read_only_unless_explicit_fts_maintenance(tmp_path, capsys):
    database = tmp_path / ".hermes" / "memory.db"
    database.parent.mkdir()
    _database(database)

    assert main(["verify", str(database)]) == 0
    capsys.readouterr()

    assert main(["verify", str(database), "--check-fts"]) == 2
    assert "maintenance window" in capsys.readouterr().err

    assert main(
        ["verify", str(database), "--check-fts", "--allow-live"]
    ) == 0


def test_restore_under_hermes_requires_maintenance_override(tmp_path, capsys):
    source = tmp_path / "source.db"
    backup = tmp_path / "backup.db"
    destination = tmp_path / ".hermes" / "restored.db"
    destination.parent.mkdir()
    _database(source)
    assert main(["backup", str(source), str(backup)]) == 0
    capsys.readouterr()

    assert main(["restore", str(backup), str(destination)]) == 2

    error = capsys.readouterr().err
    assert "maintenance window" in error
    assert "--allow-live" in error
    assert not destination.exists()

    assert main(
        ["restore", str(backup), str(destination), "--allow-live"]
    ) == 0
    with sqlite3.connect(destination) as conn:
        assert conn.execute("SELECT content FROM facts").fetchone() == (
            "shared memory",
        )


def test_erase_fact_is_explicit_audited_maintenance(tmp_path, capsys):
    database = tmp_path / "memory.db"
    _database(database)
    assert main(["migrate", str(database)]) == 0
    capsys.readouterr()

    assert main([
        "erase-fact", str(database), "1",
        "--requested-by", "avery",
        "--reason", "privacy request",
    ]) == 0

    output = json.loads(capsys.readouterr().out)
    assert output["ok"] is True
    assert output["report"]["fact_id"] == 1
    with sqlite3.connect(database) as conn:
        assert conn.execute("SELECT content FROM facts").fetchone()[0] == (
            "[PRIVACY ERASED fact:1]"
        )
        assert conn.execute(
            "SELECT requested_by, reason FROM privacy_erasure_log"
        ).fetchone() == ("avery", "privacy request")


def test_rehearse_command_uses_only_explicit_offline_snapshot(tmp_path, capsys):
    snapshot = create_legacy_fixture(tmp_path / "snapshot.sqlite")
    workdir = tmp_path / "rehearsal"

    assert main(["rehearse", str(snapshot), str(workdir)]) == 0

    output = json.loads(capsys.readouterr().out)
    assert output["ok"] is True
    assert output["report"]["migrated_schema_version"] == 1
    assert output["report"]["restored_schema_version"] == 0
    assert output["report"]["source_unchanged"] is True


def test_rebuild_vector_index_command_is_explicit_and_idempotent(tmp_path, capsys):
    database = tmp_path / "memory.db"
    _database(database)
    assert main(["migrate", str(database)]) == 0
    capsys.readouterr()
    with sqlite3.connect(database) as conn:
        conn.execute(
            "INSERT INTO fact_embeddings(fact_id, embedding, dim, embedding_identity) "
            "VALUES (1, ?, 2, 'fixture')",
            (embedding_to_bytes(np.asarray((1.0, 0.0), dtype=np.float32)),),
        )
        conn.commit()

    command = [
        "rebuild-vector-index", str(database),
        "--embedding-identity", "fixture", "--dimensions", "2",
    ]
    assert main(command) == 0
    first = json.loads(capsys.readouterr().out)
    assert main(command) == 0
    second = json.loads(capsys.readouterr().out)

    assert first["report"]["indexed_count"] == 1
    assert second["report"] == first["report"]


def test_extraction_dead_status_is_read_only_and_redacts_unsafe_errors(
    tmp_path, capsys
):
    database = tmp_path / "memory.db"
    _database(database)
    assert main(["migrate", str(database)]) == 0
    capsys.readouterr()
    with sqlite3.connect(database) as conn:
        conn.executemany(
            "INSERT INTO extract_queue(payload, payload_hash, status, attempts, "
            "last_error) VALUES (?, ?, 'dead', ?, ?)",
            [
                ("private transcript one", "a" * 64, 3, "adapter_exit"),
                ("private transcript two", "b" * 64, 1, "unsafe raw detail"),
            ],
        )
        conn.commit()

    assert main(["extraction-dead-status", str(database)]) == 0

    output = json.loads(capsys.readouterr().out)
    assert output["dead"] == 2
    assert output["read_only"] is True
    assert [row["error_code"] for row in output["rows"]] == [
        "adapter_exit",
        "redacted",
    ]
    assert "private transcript" not in json.dumps(output)
    with sqlite3.connect(database) as conn:
        assert conn.execute(
            "SELECT count(*) FROM extract_queue WHERE status = 'dead'"
        ).fetchone()[0] == 2


def test_revive_extraction_dead_requires_exact_ids_and_error(tmp_path, capsys):
    database = tmp_path / "memory.db"
    _database(database)
    assert main(["migrate", str(database)]) == 0
    capsys.readouterr()
    with sqlite3.connect(database) as conn:
        conn.executemany(
            "INSERT INTO extract_queue(payload, payload_hash, status, attempts, "
            "last_error, lease_owner, lease_until, lease_token) "
            "VALUES (?, ?, 'dead', 3, ?, 'old-worker', 123, 'old-token')",
            [
                ("one", "a" * 64, "adapter_exit"),
                ("two", "b" * 64, "invalid_proposal"),
            ],
        )
        conn.commit()
        ids = [
            int(row[0])
            for row in conn.execute("SELECT id FROM extract_queue ORDER BY id")
        ]

    wrong = [
        "revive-extraction-dead", str(database),
        "--id", str(ids[0]), "--id", str(ids[1]),
        "--expected-error", "adapter_exit",
    ]
    assert main(wrong) == 2
    assert "expected error" in capsys.readouterr().err
    with sqlite3.connect(database) as conn:
        assert conn.execute(
            "SELECT count(*) FROM extract_queue WHERE status = 'dead'"
        ).fetchone()[0] == 2

    command = [
        "revive-extraction-dead", str(database),
        "--id", str(ids[0]), "--expected-error", "adapter_exit",
    ]
    assert main(command) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["revived"] == [ids[0]]
    with sqlite3.connect(database) as conn:
        assert tuple(conn.execute(
            "SELECT status, attempts, not_before, lease_owner, lease_until, "
            "lease_token, last_error FROM extract_queue WHERE id = ?",
            (ids[0],),
        ).fetchone()) == (
            "pending", 0, None, None, None, None, "adapter_exit"
        )


def test_acknowledge_extraction_dead_requires_exact_ids_and_error(tmp_path, capsys):
    database = tmp_path / "memory.db"
    _database(database)
    assert main(["migrate", str(database)]) == 0
    capsys.readouterr()
    with sqlite3.connect(database) as conn:
        conn.executemany(
            "INSERT INTO extract_queue(payload, payload_hash, status, attempts, "
            "last_error) VALUES (?, ?, 'dead', 3, ?)",
            [
                ("one", "a" * 64, "adapter_exit"),
                ("two", "b" * 64, "invalid_proposal"),
            ],
        )
        conn.commit()
        ids = [
            int(row[0])
            for row in conn.execute("SELECT id FROM extract_queue ORDER BY id")
        ]

    wrong = [
        "acknowledge-extraction-dead", str(database),
        "--id", str(ids[0]), "--id", str(ids[1]),
        "--expected-error", "adapter_exit",
    ]
    assert main(wrong) == 2
    assert "expected error" in capsys.readouterr().err

    command = [
        "acknowledge-extraction-dead", str(database),
        "--id", str(ids[0]), "--expected-error", "adapter_exit",
    ]
    assert main(command) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["acknowledged"] == [ids[0]]
    with sqlite3.connect(database) as conn:
        assert tuple(conn.execute(
            "SELECT status, attempts, last_error FROM extract_queue WHERE id = ?",
            (ids[0],),
        ).fetchone()) == ("acknowledged", 3, "adapter_exit")
        assert conn.execute(
            "SELECT status FROM extract_queue WHERE id = ?", (ids[1],)
        ).fetchone()[0] == "dead"


def test_revive_acknowledged_extraction_requires_explicit_source_status(
    tmp_path, capsys
):
    database = tmp_path / "memory.db"
    _database(database)
    assert main(["migrate", str(database)]) == 0
    capsys.readouterr()
    with sqlite3.connect(database) as conn:
        cursor = conn.execute(
            "INSERT INTO extract_queue(payload, payload_hash, status, attempts, "
            "last_error) VALUES ('one', ?, 'acknowledged', 3, 'adapter_exit')",
            ("a" * 64,),
        )
        row_id = int(cursor.lastrowid)
        conn.commit()

    command = [
        "revive-extraction-dead", str(database),
        "--id", str(row_id), "--expected-error", "adapter_exit",
    ]
    assert main(command) == 2
    assert "not dead" in capsys.readouterr().err

    assert main([*command, "--from-status", "acknowledged"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["revived"] == [row_id]
    assert output["from_status"] == "acknowledged"
    with sqlite3.connect(database) as conn:
        assert tuple(conn.execute(
            "SELECT status, attempts, last_error FROM extract_queue WHERE id = ?",
            (row_id,),
        ).fetchone()) == ("pending", 0, "adapter_exit")


def test_mutating_maintenance_refuses_daemon_owned_database(tmp_path, capsys):
    database = tmp_path / "memory.db"
    _database(database)
    ownership = DatabaseOwnership(database)
    ownership.acquire()
    try:
        assert main(["migrate", str(database)]) == 2
        assert "daemon owns" in capsys.readouterr().err
    finally:
        ownership.release()

    assert main(["migrate", str(database)]) == 0


def test_browse_snapshot_is_policy_filtered_read_only_and_idempotent(tmp_path, capsys):
    database = tmp_path / "memory.db"
    conn = sqlite3.connect(database)
    migrate(conn)
    visible = insert_fact(conn, "Visible browse fact", scope="private")
    insert_fact(conn, "Out of scope fact", scope="work")
    insert_fact(conn, "Sensitive browse fact", scope="private", sensitivity="sensitive")
    superseded = insert_fact(conn, "Superseded browse fact", scope="private")
    conflicted = insert_fact(conn, "Conflicted browse fact", scope="private")
    conn.execute("UPDATE facts SET superseded_by = ? WHERE fact_id = ?", (visible, superseded))
    conn.execute("UPDATE facts SET conflict_group = 'disputed' WHERE fact_id = ?", (conflicted,))
    conn.commit()
    conn.close()
    config = tmp_path / "server.json"
    config.write_text(json.dumps({
        "database_path": str(database),
        "socket_path": str(tmp_path / "enfold.sock"),
        "grants": {"browser": ["private"]},
        "browse_scopes": ["private"],
        "retrieval": {"mode": "ci", "allow_nonproduction": True, "dimensions": 64},
    }), encoding="utf-8")
    config.chmod(0o600)
    destination = tmp_path / "browse" / "browse-snapshot.db"

    assert main(["browse-snapshot", str(config), "--destination", str(destination)]) == 0
    assert json.loads(capsys.readouterr().out)["ok"] is True
    first_digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    with sqlite3.connect(f"file:{destination}?mode=ro", uri=True) as browse:
        assert browse.execute("SELECT content FROM facts").fetchall() == [("Visible browse fact",)]
        assert browse.execute("SELECT rowid FROM facts_fts WHERE facts_fts MATCH 'visible'").fetchall() == [(visible,)]
    assert (destination.stat().st_mode & 0o222) == 0
    metadata = json.loads((destination.parent / "metadata.json").read_text())
    assert metadata["title"] == "Enfold Second Brain"
    assert metadata["scope_allowlist"] == ["private"]

    assert main(["browse-snapshot", str(config), "--destination", str(destination)]) == 0
    capsys.readouterr()
    assert hashlib.sha256(destination.read_bytes()).hexdigest() == first_digest


@pytest.mark.parametrize(
    ("database_name", "destination_name", "symlink", "message"),
    [
        ("memory.db", "memory.db", False, "collides with live database path"),
        ("memory.db", "memory.db-wal", False, "collides with live database path"),
        ("memory.db", "memory.db-shm", False, "collides with live database path"),
        ("memory.db", "memory.db.enfold.lock", False, "collides with live database path"),
        ("memory.db", "memory.db.mcp-write.lock", False, "collides with live database path"),
        ("metadata.json", "browse-snapshot.db", False, "collides with live database path"),
        ("browse-snapshot.db-wal", "browse-snapshot.db", False, "collides with live database path"),
        ("browse-snapshot.db-shm", "browse-snapshot.db", False, "collides with live database path"),
        ("memory.db", "database-link.db", True, "must not be a symlink"),
    ],
)
def test_browse_snapshot_refuses_live_database_path_collisions(
    tmp_path, capsys, database_name, destination_name, symlink, message
):
    database = tmp_path / database_name
    with sqlite3.connect(database) as conn:
        migrate(conn)
        insert_fact(conn, "Live fact", scope="private")
    config = tmp_path / "server.json"
    config.write_text(json.dumps({
        "database_path": str(database),
        "socket_path": str(tmp_path / "enfold.sock"),
        "grants": {"browser": ["private"]},
        "browse_scopes": ["private"],
        "retrieval": {"mode": "ci", "allow_nonproduction": True, "dimensions": 64},
    }), encoding="utf-8")
    config.chmod(0o600)
    destination = tmp_path / destination_name
    if symlink:
        destination.symlink_to(database)

    assert main([
        "browse-snapshot", str(config),
        "--destination", str(destination),
    ]) == 2

    assert message in capsys.readouterr().err
    with sqlite3.connect(database) as conn:
        assert conn.execute("SELECT content FROM facts").fetchall() == [
            ("Live fact",)
        ]


def test_browse_snapshot_refuses_unrelated_existing_sqlite(tmp_path, capsys):
    database = tmp_path / "memory.db"
    with sqlite3.connect(database) as conn:
        migrate(conn)
        insert_fact(conn, "Live fact", scope="private")
    config = tmp_path / "server.json"
    config.write_text(json.dumps({
        "database_path": str(database),
        "socket_path": str(tmp_path / "enfold.sock"),
        "grants": {"browser": ["private"]},
        "browse_scopes": ["private"],
        "retrieval": {"mode": "ci", "allow_nonproduction": True, "dimensions": 64},
    }), encoding="utf-8")
    config.chmod(0o600)
    destination = tmp_path / "unrelated.db"
    _database(destination)

    assert main([
        "browse-snapshot", str(config), "--destination", str(destination)
    ]) == 2

    assert "is not an Enfold browse snapshot" in capsys.readouterr().err
    with sqlite3.connect(destination) as conn:
        assert conn.execute("SELECT content FROM facts").fetchall() == [
            ("shared memory",)
        ]


def test_browse_snapshot_refuses_unrelated_existing_metadata(tmp_path, capsys):
    database = tmp_path / "memory.db"
    with sqlite3.connect(database) as conn:
        migrate(conn)
        insert_fact(conn, "Live fact", scope="private")
    config = tmp_path / "server.json"
    config.write_text(json.dumps({
        "database_path": str(database),
        "socket_path": str(tmp_path / "enfold.sock"),
        "grants": {"browser": ["private"]},
        "browse_scopes": ["private"],
        "retrieval": {"mode": "ci", "allow_nonproduction": True, "dimensions": 64},
    }), encoding="utf-8")
    config.chmod(0o600)
    destination = tmp_path / "browse" / "browse-snapshot.db"
    destination.parent.mkdir()
    metadata = destination.parent / "metadata.json"
    unrelated = {"title": "Unrelated Datasette configuration"}
    metadata.write_text(json.dumps(unrelated), encoding="utf-8")

    assert main([
        "browse-snapshot", str(config), "--destination", str(destination)
    ]) == 2

    assert "is not Enfold browse snapshot metadata" in capsys.readouterr().err
    assert json.loads(metadata.read_text(encoding="utf-8")) == unrelated
    assert not destination.exists()


def test_browse_snapshot_refuses_metadata_filename_destination(tmp_path, capsys):
    database = tmp_path / "memory.db"
    with sqlite3.connect(database) as conn:
        migrate(conn)
        insert_fact(conn, "Live fact", scope="private")
    config = tmp_path / "server.json"
    config.write_text(json.dumps({
        "database_path": str(database),
        "socket_path": str(tmp_path / "enfold.sock"),
        "grants": {"browser": ["private"]},
        "browse_scopes": ["private"],
        "retrieval": {"mode": "ci", "allow_nonproduction": True, "dimensions": 64},
    }), encoding="utf-8")
    config.chmod(0o600)
    destination = tmp_path / "browse" / "metadata.json"

    assert main([
        "browse-snapshot", str(config), "--destination", str(destination)
    ]) == 2

    assert "must differ from its metadata path" in capsys.readouterr().err
    assert not destination.exists()


def test_browse_snapshot_rechecks_destination_before_replace(
    tmp_path, capsys, monkeypatch
):
    database = tmp_path / "memory.db"
    with sqlite3.connect(database) as conn:
        migrate(conn)
        insert_fact(conn, "Live fact", scope="private")
    config = tmp_path / "server.json"
    config.write_text(json.dumps({
        "database_path": str(database),
        "socket_path": str(tmp_path / "enfold.sock"),
        "grants": {"browser": ["private"]},
        "browse_scopes": ["private"],
        "retrieval": {"mode": "ci", "allow_nonproduction": True, "dimensions": 64},
    }), encoding="utf-8")
    config.chmod(0o600)
    destination = tmp_path / "browse" / "browse-snapshot.db"
    real_chmod = os.chmod

    def collide(path, mode):
        real_chmod(path, mode)
        if Path(path).name.startswith(f".{destination.name}."):
            destination.write_text("concurrent owner", encoding="utf-8")

    monkeypatch.setattr("enfold.ops.os.chmod", collide)

    assert main([
        "browse-snapshot", str(config), "--destination", str(destination)
    ]) == 2

    assert "is not an Enfold browse snapshot" in capsys.readouterr().err
    assert destination.read_text(encoding="utf-8") == "concurrent owner"


def test_browse_snapshot_rechecks_metadata_before_replace(
    tmp_path, capsys, monkeypatch
):
    database = tmp_path / "memory.db"
    with sqlite3.connect(database) as conn:
        migrate(conn)
        insert_fact(conn, "Live fact", scope="private")
    config = tmp_path / "server.json"
    config.write_text(json.dumps({
        "database_path": str(database),
        "socket_path": str(tmp_path / "enfold.sock"),
        "grants": {"browser": ["private"]},
        "browse_scopes": ["private"],
        "retrieval": {"mode": "ci", "allow_nonproduction": True, "dimensions": 64},
    }), encoding="utf-8")
    config.chmod(0o600)
    destination = tmp_path / "browse" / "browse-snapshot.db"
    metadata = destination.with_name("metadata.json")
    unrelated = {"title": "concurrent owner"}
    real_dump = json.dump

    def collide(value, handle, *args, **kwargs):
        result = real_dump(value, handle, *args, **kwargs)
        metadata.write_text(json.dumps(unrelated), encoding="utf-8")
        return result

    monkeypatch.setattr("enfold.ops.json.dump", collide)

    assert main([
        "browse-snapshot", str(config), "--destination", str(destination)
    ]) == 2

    assert "is not Enfold browse snapshot metadata" in capsys.readouterr().err
    assert json.loads(metadata.read_text(encoding="utf-8")) == unrelated
