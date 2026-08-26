"""Explicit Enfold schema, backup, verification, and restore operations.

This module is intentionally separate from provider initialization.  It never
chooses a database path implicitly and requires an explicit maintenance-window
override before migrating or restoring anything below a ``.hermes`` directory.

Run with ``python -m enfold.ops --help``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Sequence

from .backup import (
    BackupError,
    backup_database,
    maintenance_database_lock,
    restore_database,
    sqlite_file_uri,
    verify_database,
)
from .core_store import DEFAULT_BUSY_TIMEOUT_MS
from .erasure import ErasureError, erase_fact
from .export import ExportError, export_current
from .extraction_enqueue import (
    CaptureDisabled,
    CaptureEnableError,
    ExtractionEnqueuer,
    ExtractionQueueUnavailable,
    capture_status,
    enable_capture,
)
from .provenance import ConnectionContext
from .rehearsal import RehearsalError, rehearse_snapshot
from .schema import (
    SUPPORTED_SCHEMA_VERSION,
    SchemaError,
    migrate,
    require_compatible_schema,
)
from .server import load_config
from .sqlite_vec_index import SQLiteVecError, rebuild_sqlite_vec_index


LIVE_PATH_MESSAGE = (
    "refusing to modify a database under .hermes; stop all client, MCP, and "
    "Enfold writers during a maintenance window, then pass "
    "--allow-live explicitly"
)
_SAFE_ERROR_CODE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_BROWSE_APPLICATION_ID = 0x454E4644


def _resolved(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _is_under_hermes(path: str | Path) -> bool:
    return ".hermes" in _resolved(path).parts


def _require_live_override(path: str | Path, *, allow_live: bool) -> None:
    if _is_under_hermes(path) and not allow_live:
        raise BackupError(LIVE_PATH_MESSAGE)


def _connect(path: str | Path, *, read_only: bool) -> sqlite3.Connection:
    resolved = _resolved(path)
    if not resolved.is_file():
        raise BackupError(f"database does not exist: {resolved}")
    mode = "ro" if read_only else "rw"
    conn = sqlite3.connect(sqlite_file_uri(resolved, mode=mode), uri=True)
    try:
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute(f"PRAGMA busy_timeout={int(DEFAULT_BUSY_TIMEOUT_MS)}")
        if not read_only:
            journal_mode = str(conn.execute("PRAGMA journal_mode=WAL").fetchone()[0])
            if journal_mode.lower() != "wal":
                raise BackupError("database cannot enable WAL journal mode")
        return conn
    except BaseException:
        conn.close()
        raise


def _print_json(value: object) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def _schema_status(args: argparse.Namespace) -> None:
    with _connect(args.database, read_only=True) as conn:
        version = require_compatible_schema(conn)
    _print_json(
        {
            "database": str(_resolved(args.database)),
            "schema_version": version,
            "supported_schema_version": SUPPORTED_SCHEMA_VERSION,
            "compatible": True,
        }
    )


def _migrate(args: argparse.Namespace) -> None:
    _require_live_override(args.database, allow_live=args.allow_live)
    with maintenance_database_lock(args.database):
        with _connect(args.database, read_only=False) as conn:
            before = require_compatible_schema(conn)
            after = migrate(conn, target_version=args.target)
    _print_json(
        {
            "database": str(_resolved(args.database)),
            "schema_version_before": before,
            "schema_version_after": after,
        }
    )


def _backup(args: argparse.Namespace) -> None:
    report = backup_database(
        args.source,
        args.destination,
        overwrite=args.overwrite,
        secondary_directory=args.secondary_directory,
        age_recipient_path=args.age_recipient_path,
    )
    _print_json(
        {
            "source": str(_resolved(args.source)),
            "destination": str(_resolved(args.destination)),
            "report": asdict(report),
            "ok": report.ok,
        }
    )


def _verify(args: argparse.Namespace) -> None:
    # FTS5's integrity command uses INSERT syntax even though it rolls back to
    # a savepoint. Keep ordinary verification genuinely read-only; the fuller
    # FTS check is an explicit maintenance operation.
    if args.check_fts:
        _require_live_override(args.database, allow_live=args.allow_live)
    if args.check_fts:
        with maintenance_database_lock(args.database):
            with _connect(args.database, read_only=False) as conn:
                report = verify_database(conn, check_fts=True)
    else:
        with _connect(args.database, read_only=True) as conn:
            report = verify_database(conn, check_fts=False)
    _print_json(
        {
            "database": str(_resolved(args.database)),
            "report": asdict(report),
            "ok": report.ok,
        }
    )
    if not report.ok:
        raise BackupError("database verification failed")


def _restore(args: argparse.Namespace) -> None:
    _require_live_override(args.destination, allow_live=args.allow_live)
    with maintenance_database_lock(args.destination):
        report = restore_database(
            args.backup, args.destination, overwrite=args.overwrite
        )
    _print_json(
        {
            "backup": str(_resolved(args.backup)),
            "destination": str(_resolved(args.destination)),
            "report": asdict(report),
            "ok": report.ok,
        }
    )


def _erase_fact(args: argparse.Namespace) -> None:
    _require_live_override(args.database, allow_live=args.allow_live)
    with maintenance_database_lock(args.database):
        with _connect(args.database, read_only=False) as conn:
            require_compatible_schema(conn, for_writer=True)
            report = erase_fact(
                conn,
                args.fact_id,
                requested_by=args.requested_by,
                reason=args.reason,
            )
    _print_json(
        {
            "database": str(_resolved(args.database)),
            "report": asdict(report),
            "ok": True,
        }
    )


def _rehearse(args: argparse.Namespace) -> None:
    report = rehearse_snapshot(args.snapshot, args.workdir)
    _print_json({"report": asdict(report), "ok": True})


def _browse_metadata(scopes: tuple[str, ...]) -> dict[str, object]:
    return {
        "title": "Enfold Second Brain",
        "databases": {
            "browse-snapshot": {
                "tables": {"facts": {"fts_table": "facts_fts"}}
            }
        },
        "scope_allowlist": list(scopes),
        "filters": {"lifecycle": "settled_active", "sensitivity": "normal_only"},
    }


def _atomic_json(
    path: Path, value: object, *, before_replace: Callable[[], None] | None = None
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        if before_replace is not None:
            before_replace()
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _browse_output_paths(destination: Path) -> set[Path]:
    return {
        _resolved(destination),
        _resolved(f"{destination}-wal"),
        _resolved(f"{destination}-shm"),
        _resolved(destination.with_name("metadata.json")),
    }


def _database_write_paths(database: Path) -> set[Path]:
    return {
        _resolved(database),
        _resolved(f"{database}-wal"),
        _resolved(f"{database}-shm"),
        _resolved(f"{database}.enfold.lock"),
        _resolved(f"{database}.mcp-write.lock"),
    }


def _require_browse_metadata(metadata_path: Path) -> None:
    if not metadata_path.exists() and not metadata_path.is_symlink():
        return
    try:
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeError) as exc:
        raise BackupError(
            f"browse snapshot metadata already exists and is not Enfold browse "
            f"snapshot metadata: {metadata_path}"
        ) from exc
    scopes = existing.get("scope_allowlist") if isinstance(existing, dict) else None
    if (
        not isinstance(scopes, list)
        or not all(isinstance(scope, str) for scope in scopes)
        or existing != _browse_metadata(tuple(scopes))
    ):
        raise BackupError(
            f"browse snapshot metadata already exists and is not Enfold browse "
            f"snapshot metadata: {metadata_path}"
        )


def _require_safe_browse_destination(database: Path, destination: Path) -> None:
    metadata_path = destination.with_name("metadata.json")
    if destination == metadata_path:
        raise BackupError(
            "browse snapshot destination must differ from its metadata path"
        )
    if destination.is_symlink():
        raise BackupError("browse snapshot destination must not be a symlink")
    if metadata_path.is_symlink():
        raise BackupError("browse snapshot metadata must not be a symlink")
    collisions = _browse_output_paths(destination) & _database_write_paths(database)
    if collisions:
        collision = min(collisions, key=str)
        raise BackupError(
            f"browse snapshot output collides with live database path: {collision}"
        )
    _require_browse_metadata(metadata_path)
    if not destination.exists():
        return
    try:
        with sqlite3.connect(
            sqlite_file_uri(destination, mode="ro"), uri=True
        ) as existing:
            application_id = int(existing.execute("PRAGMA application_id").fetchone()[0])
    except sqlite3.DatabaseError as exc:
        raise BackupError(
            f"browse snapshot destination already exists and is not an Enfold "
            f"browse snapshot: {destination}"
        ) from exc
    if application_id != _BROWSE_APPLICATION_ID:
        raise BackupError(
            f"browse snapshot destination already exists and is not an Enfold "
            f"browse snapshot: {destination}"
        )


def _browse_snapshot(args: argparse.Namespace) -> None:
    """Copy approved current facts from a read-only live snapshot into SQLite."""

    config = load_config(args.config, allow_live=True)
    requested_destination = Path(
        args.destination or "~/.local/state/enfold/browse/browse-snapshot.db"
    ).expanduser()
    if requested_destination.is_symlink():
        raise BackupError("browse snapshot destination must not be a symlink")
    destination = _resolved(requested_destination.parent) / requested_destination.name
    metadata_path = destination.with_name("metadata.json")
    _require_safe_browse_destination(_resolved(config.database_path), destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    source = _connect(config.database_path, read_only=True)
    temporary_source = tempfile.NamedTemporaryFile(
        prefix="enfold-browse-source-", suffix=".db", delete=False
    )
    temporary_source.close()
    temporary_destination = tempfile.NamedTemporaryFile(
        prefix=f".{destination.name}.", suffix=".db", dir=destination.parent, delete=False
    )
    temporary_destination.close()
    try:
        with sqlite3.connect(temporary_source.name) as snapshot:
            source.backup(snapshot)
        source.close()
        source = None
        with sqlite3.connect(temporary_source.name) as snapshot, sqlite3.connect(
            temporary_destination.name
        ) as browse:
            require_compatible_schema(snapshot)
            browse.execute(f"PRAGMA application_id = {_BROWSE_APPLICATION_ID}")
            browse.executescript(
                """
                CREATE TABLE facts (
                    fact_id INTEGER PRIMARY KEY,
                    content TEXT NOT NULL,
                    category TEXT NOT NULL,
                    tags TEXT NOT NULL,
                    trust_score REAL,
                    memory_kind TEXT,
                    subject_key TEXT,
                    predicate_key TEXT,
                    object_value TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    scope TEXT NOT NULL
                );
                CREATE VIRTUAL TABLE facts_fts USING fts5(content);
                """
            )
            placeholders = ",".join("?" for _ in config.browse_scopes)
            rows = snapshot.execute(
                f"""
                SELECT fact_id, content, COALESCE(category, 'general'), COALESCE(tags, ''),
                       trust_score, memory_kind, subject_key, predicate_key,
                       object_value, created_at, updated_at, scope
                FROM facts
                WHERE invalid_at IS NULL AND superseded_by IS NULL
                  AND conflict_group IS NULL
                  AND COALESCE(sensitivity, 'normal') = 'normal'
                  AND scope IN ({placeholders})
                ORDER BY fact_id
                """,
                config.browse_scopes,
            )
            facts = list(rows)
            browse.executemany(
                "INSERT INTO facts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", facts
            )
            browse.executemany(
                "INSERT INTO facts_fts(rowid, content) VALUES (?, ?)",
                ((row[0], row[1]) for row in facts),
            )
        os.chmod(temporary_destination.name, 0o400)
        _require_safe_browse_destination(
            _resolved(config.database_path), destination
        )
        os.replace(temporary_destination.name, destination)
        _atomic_json(
            metadata_path,
            _browse_metadata(config.browse_scopes),
            before_replace=lambda: _require_safe_browse_destination(
                _resolved(config.database_path), destination
            ),
        )
    finally:
        if source is not None:
            source.close()
        Path(temporary_source.name).unlink(missing_ok=True)
        Path(temporary_destination.name).unlink(missing_ok=True)
    _print_json(
        {
            "database": str(destination),
            "metadata": str(metadata_path),
            "scope_allowlist": list(config.browse_scopes),
            "ok": True,
        }
    )


def _rebuild_vector_index(args: argparse.Namespace) -> None:
    _require_live_override(args.database, allow_live=args.allow_live)
    with maintenance_database_lock(args.database):
        with _connect(args.database, read_only=False) as conn:
            require_compatible_schema(conn, for_writer=True)
            report = rebuild_sqlite_vec_index(
                conn, args.embedding_identity, args.dimensions
            )
    _print_json({
        "database": str(_resolved(args.database)),
        "report": asdict(report),
        "ok": True,
    })


def _public_error_code(value: object) -> str:
    return value if isinstance(value, str) and _SAFE_ERROR_CODE.fullmatch(value) else "redacted"


def _extraction_dead_status(args: argparse.Namespace) -> None:
    """Inspect dead extraction rows without exposing transcript/model content."""

    with _connect(args.database, read_only=True) as conn:
        require_compatible_schema(conn)
        rows = conn.execute(
            "SELECT id, created_at, attempts, last_error, payload_hash, "
            "proposal_hash IS NOT NULL FROM extract_queue "
            "WHERE status = 'dead' ORDER BY id"
        ).fetchall()
    safe_rows = [
        {
            "id": int(row[0]),
            "created_at": str(row[1]),
            "attempts": int(row[2]),
            "error_code": _public_error_code(row[3]),
            "payload_sha256": str(row[4]),
            "has_proposal_snapshot": bool(row[5]),
        }
        for row in rows
    ]
    _print_json(
        {
            "database": str(_resolved(args.database)),
            "dead": len(safe_rows),
            "rows": safe_rows,
            "read_only": True,
        }
    )


def _revive_extraction_dead(args: argparse.Namespace) -> None:
    """Revive explicitly selected dead rows after a stopped-writer review."""

    _require_live_override(args.database, allow_live=args.allow_live)
    source_status = args.from_status
    ids = tuple(dict.fromkeys(args.id))
    if not ids:
        raise BackupError("at least one dead extraction row id is required")
    expected_error = args.expected_error
    if not _SAFE_ERROR_CODE.fullmatch(expected_error):
        raise BackupError("expected extraction error must be a safe error code")
    placeholders = ",".join("?" for _ in ids)
    with maintenance_database_lock(args.database):
        with _connect(args.database, read_only=False) as conn:
            require_compatible_schema(conn, for_writer=True)
            rows = conn.execute(
                f"SELECT id, status, last_error FROM extract_queue "
                f"WHERE id IN ({placeholders}) ORDER BY id",
                ids,
            ).fetchall()
            found = {int(row[0]) for row in rows}
            missing = sorted(set(ids) - found)
            if missing:
                raise BackupError(f"extraction rows were not found: {missing}")
            invalid = [
                int(row[0])
                for row in rows
                if row[1] != source_status or row[2] != expected_error
            ]
            if invalid:
                raise BackupError(
                    f"extraction rows are not {source_status} with the expected error: "
                    f"{invalid}"
                )
            conn.execute("BEGIN IMMEDIATE")
            cursor = conn.execute(
                f"UPDATE extract_queue SET status = 'pending', attempts = 0, "
                "not_before = NULL, lease_owner = NULL, lease_until = NULL, "
                f"lease_token = NULL WHERE id IN ({placeholders}) "
                "AND status = ? AND last_error = ?",
                (*ids, source_status, expected_error),
            )
            if cursor.rowcount != len(ids):
                conn.rollback()
                raise BackupError("dead extraction rows changed during revival")
            conn.commit()
    _print_json(
        {
            "database": str(_resolved(args.database)),
            "revived": list(ids),
            "expected_error": expected_error,
            "from_status": source_status,
            "ok": True,
        }
    )


def _acknowledge_extraction_dead(args: argparse.Namespace) -> None:
    """Acknowledge explicitly reviewed dead rows without discarding evidence."""

    _require_live_override(args.database, allow_live=args.allow_live)
    ids = tuple(dict.fromkeys(args.id))
    if not ids:
        raise BackupError("at least one dead extraction row id is required")
    expected_error = args.expected_error
    if not _SAFE_ERROR_CODE.fullmatch(expected_error):
        raise BackupError("expected extraction error must be a safe error code")
    placeholders = ",".join("?" for _ in ids)
    with maintenance_database_lock(args.database):
        with _connect(args.database, read_only=False) as conn:
            require_compatible_schema(conn, for_writer=True)
            rows = conn.execute(
                f"SELECT id, status, last_error FROM extract_queue "
                f"WHERE id IN ({placeholders}) ORDER BY id",
                ids,
            ).fetchall()
            found = {int(row[0]) for row in rows}
            missing = sorted(set(ids) - found)
            if missing:
                raise BackupError(f"extraction rows were not found: {missing}")
            invalid = [
                int(row[0])
                for row in rows
                if row[1] != "dead" or row[2] != expected_error
            ]
            if invalid:
                raise BackupError(
                    "extraction rows are not dead with the expected error: "
                    f"{invalid}"
                )
            conn.execute("BEGIN IMMEDIATE")
            cursor = conn.execute(
                f"UPDATE extract_queue SET status = 'acknowledged' "
                f"WHERE id IN ({placeholders}) "
                "AND status = 'dead' AND last_error = ?",
                (*ids, expected_error),
            )
            if cursor.rowcount != len(ids):
                conn.rollback()
                raise BackupError(
                    "dead extraction rows changed during acknowledgement"
                )
            conn.commit()
    _print_json(
        {
            "database": str(_resolved(args.database)),
            "acknowledged": list(ids),
            "expected_error": expected_error,
            "ok": True,
        }
    )


def _capture_status(args: argparse.Namespace) -> None:
    with _connect(args.database, read_only=True) as conn:
        require_compatible_schema(conn)
        status = capture_status(conn)
    _print_json(
        {
            "database": str(_resolved(args.database)),
            **status,
        }
    )


def _capture_enable(args: argparse.Namespace) -> None:
    _require_live_override(args.database, allow_live=args.allow_live)
    with _connect(args.database, read_only=False) as conn:
        require_compatible_schema(conn, for_writer=True)
        status = enable_capture(
            conn,
            allow_unreviewed=args.allow_unreviewed,
            verifier_ready=args.verifier_ready,
        )
    _print_json(
        {
            "database": str(_resolved(args.database)),
            **status,
            "ok": True,
        }
    )


def _capture_session(args: argparse.Namespace) -> None:
    _require_live_override(args.database, allow_live=args.allow_live)
    transcript_path = _resolved(args.transcript)
    if not transcript_path.is_file():
        raise BackupError(f"transcript does not exist: {transcript_path}")
    transcript = transcript_path.read_text(encoding="utf-8")
    context = ConnectionContext(
        client_id=args.client_id,
        surface=args.surface,
        agent_id=args.agent_id,
        session_id=args.session_id,
        access_scopes=(args.scope,),
    )
    with _connect(args.database, read_only=False) as conn:
        require_compatible_schema(conn, for_writer=True)
        result = ExtractionEnqueuer(conn).enqueue_session_capture(
            context, transcript, scope=args.scope
        )
    _print_json(
        {
            "database": str(_resolved(args.database)),
            "queue_id": result.queue_id,
            "payload_sha256": result.payload_sha256,
            "replayed": result.replayed,
            "ok": True,
        }
    )


def _export_current(args: argparse.Namespace) -> None:
    if not args.current:
        raise BackupError("export requires --current")
    report = export_current(args.database, args.destination)
    _print_json(
        {
            "database": str(_resolved(args.database)),
            "current": str(report.current_path),
            "needs_review": str(report.needs_review_dir),
            "current_facts": report.current_facts,
            "conflicted_facts": report.conflicted_facts,
            "unreviewed_facts": report.unreviewed_facts,
            "omitted_erased": report.omitted_erased,
            "ok": True,
        }
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    status = commands.add_parser(
        "schema-status", help="inspect schema compatibility without modifying it"
    )
    status.add_argument("database", help="explicit SQLite database path")
    status.set_defaults(handler=_schema_status)

    migration = commands.add_parser(
        "migrate", help="explicitly apply registered schema migrations"
    )
    migration.add_argument("database", help="explicit SQLite database path")
    migration.add_argument(
        "--target", type=int, default=SUPPORTED_SCHEMA_VERSION, help="target version"
    )
    migration.add_argument(
        "--allow-live",
        action="store_true",
        help="allow a .hermes path after all writers are stopped for maintenance",
    )
    migration.set_defaults(handler=_migrate)

    backup = commands.add_parser(
        "backup", help="create a verified backup using SQLite's backup API"
    )
    backup.add_argument("source", help="explicit source SQLite database path")
    backup.add_argument("destination", help="explicit destination backup path")
    backup.add_argument("--overwrite", action="store_true")
    backup.add_argument(
        "--secondary-directory",
        help="best-effort secondary destination directory",
    )
    backup.add_argument(
        "--age-recipient-path",
        help="age recipients or identity file used with age -R",
    )
    backup.set_defaults(handler=_backup)

    verify = commands.add_parser(
        "verify", help="run integrity, foreign-key, FTS, and row-count checks"
    )
    verify.add_argument("database", help="explicit SQLite database path")
    verify.add_argument(
        "--check-fts",
        action="store_true",
        help="run the FTS5 write-syntax integrity command inside a rollback",
    )
    verify.add_argument(
        "--allow-live",
        action="store_true",
        help="allow --check-fts under .hermes during a maintenance window",
    )
    verify.set_defaults(handler=_verify)

    restore = commands.add_parser(
        "restore", help="restore a verified backup using SQLite's backup API"
    )
    restore.add_argument("backup", help="explicit source backup path")
    restore.add_argument("destination", help="explicit restore destination path")
    restore.add_argument("--overwrite", action="store_true")
    restore.add_argument(
        "--allow-live",
        action="store_true",
        help="allow a .hermes path after all writers are stopped for maintenance",
    )
    restore.set_defaults(handler=_restore)

    erasure = commands.add_parser(
        "erase-fact",
        help="privacy/legal erasure of a fact and all known content copies",
    )
    erasure.add_argument("database", help="explicit schema-v1 SQLite database path")
    erasure.add_argument("fact_id", type=int)
    erasure.add_argument("--requested-by", required=True)
    erasure.add_argument("--reason", required=True)
    erasure.add_argument(
        "--allow-live",
        action="store_true",
        help="allow a .hermes path after every writer is stopped",
    )
    erasure.set_defaults(handler=_erase_fact)

    rehearsal = commands.add_parser(
        "rehearse",
        help="backup/migrate/smoke/restore an explicit offline snapshot copy",
    )
    rehearsal.add_argument("snapshot", help="offline snapshot outside .hermes")
    rehearsal.add_argument("workdir", help="empty artifact directory outside .hermes")
    rehearsal.set_defaults(handler=_rehearse)

    browse = commands.add_parser(
        "browse-snapshot",
        help="build a policy-filtered SQLite snapshot for a local Datasette browser",
    )
    browse.add_argument("config", help="explicit Enfold server JSON configuration")
    browse.add_argument(
        "--destination",
        help="snapshot path (default: ~/.local/state/enfold/browse/browse-snapshot.db)",
    )
    browse.set_defaults(handler=_browse_snapshot)

    vector_rebuild = commands.add_parser(
        "rebuild-vector-index",
        help="atomically rebuild sqlite-vec from canonical fact embeddings",
    )
    vector_rebuild.add_argument("database", help="explicit schema-v1 SQLite database path")
    vector_rebuild.add_argument("--embedding-identity", required=True)
    vector_rebuild.add_argument("--dimensions", required=True, type=int)
    vector_rebuild.add_argument(
        "--allow-live", action="store_true",
        help="allow a .hermes path after every writer is stopped",
    )
    vector_rebuild.set_defaults(handler=_rebuild_vector_index)

    dead_status = commands.add_parser(
        "extraction-dead-status",
        help="inspect dead extraction rows without exposing queued content",
    )
    dead_status.add_argument(
        "database", help="explicit schema-v1 SQLite database path"
    )
    dead_status.set_defaults(handler=_extraction_dead_status)

    dead_revive = commands.add_parser(
        "revive-extraction-dead",
        help="revive explicitly reviewed extraction dead letters",
    )
    dead_revive.add_argument(
        "database", help="explicit schema-v1 SQLite database path"
    )
    dead_revive.add_argument(
        "--id", action="append", type=int, required=True,
        help="dead extraction row id; repeat for multiple reviewed rows",
    )
    dead_revive.add_argument(
        "--expected-error", required=True,
        help="require every selected row to have this exact safe error code",
    )
    dead_revive.add_argument(
        "--from-status",
        choices=("dead", "acknowledged"),
        default="dead",
        help="explicit source status; acknowledged rows require an opt-in",
    )
    dead_revive.add_argument(
        "--allow-live", action="store_true",
        help="allow a .hermes path after every writer is stopped",
    )
    dead_revive.set_defaults(handler=_revive_extraction_dead)

    dead_acknowledge = commands.add_parser(
        "acknowledge-extraction-dead",
        help="acknowledge explicitly reviewed extraction dead letters",
    )
    dead_acknowledge.add_argument(
        "database", help="explicit schema-v1 SQLite database path"
    )
    dead_acknowledge.add_argument(
        "--id", action="append", type=int, required=True,
        help="dead extraction row id; repeat for multiple reviewed rows",
    )
    dead_acknowledge.add_argument(
        "--expected-error", required=True,
        help="require every selected row to have this exact safe error code",
    )
    dead_acknowledge.add_argument(
        "--allow-live", action="store_true",
        help="allow a .hermes path after every writer is stopped",
    )
    dead_acknowledge.set_defaults(handler=_acknowledge_extraction_dead)

    capture = commands.add_parser(
        "capture",
        help="opt in to session capture that enqueues without memory_write",
    )
    capture_commands = capture.add_subparsers(dest="capture_command", required=True)
    capture_status = capture_commands.add_parser(
        "status", help="report whether session capture is enabled"
    )
    capture_status.add_argument(
        "database", help="explicit schema-v1 SQLite database path"
    )
    capture_status.set_defaults(handler=_capture_status)
    capture_enable = capture_commands.add_parser(
        "enable",
        help="opt in to session capture; refused unless a verifier or unreviewed ack",
    )
    capture_enable.add_argument(
        "database", help="explicit schema-v1 SQLite database path"
    )
    capture_enable.add_argument(
        "--allow-unreviewed",
        action="store_true",
        help="enqueue anyway; captured rows stay excluded from default recall",
    )
    capture_enable.add_argument(
        "--verifier-ready",
        action="store_true",
        help="operator attests a local evidence verifier is configured",
    )
    capture_enable.add_argument(
        "--allow-live",
        action="store_true",
        help="allow a .hermes path after every writer is stopped",
    )
    capture_enable.set_defaults(handler=_capture_enable)
    capture_session = capture_commands.add_parser(
        "session",
        help="enqueue one session transcript without writing a fact",
    )
    capture_session.add_argument(
        "database", help="explicit schema-v1 SQLite database path"
    )
    capture_session.add_argument("--transcript", required=True)
    capture_session.add_argument("--client-id", required=True)
    capture_session.add_argument("--session-id", required=True)
    capture_session.add_argument("--agent-id", required=True)
    capture_session.add_argument("--surface", required=True)
    capture_session.add_argument("--scope", default="private")
    capture_session.add_argument(
        "--allow-live",
        action="store_true",
        help="allow a .hermes path after every writer is stopped",
    )
    capture_session.set_defaults(handler=_capture_session)

    export_cmd = commands.add_parser(
        "export",
        help="write a read-only Markdown projection of current facts",
    )
    export_cmd.add_argument(
        "--current",
        action="store_true",
        help="export non-superseded, non-conflicted facts plus source references",
    )
    export_cmd.add_argument("database", help="explicit schema-v1 SQLite database path")
    export_cmd.add_argument(
        "destination", help="directory for current.md and needs_review/"
    )
    export_cmd.set_defaults(handler=_export_current)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.handler(args)
    except (
        BackupError,
        CaptureDisabled,
        CaptureEnableError,
        ErasureError,
        ExportError,
        ExtractionQueueUnavailable,
        RehearsalError,
        SchemaError,
        SQLiteVecError,
        sqlite3.DatabaseError,
        OSError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    raise SystemExit(main())
