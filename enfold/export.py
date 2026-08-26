"""Read-only Markdown projection of current Enfold facts.

This is not a second store and has no import path.  Erased facts are omitted
so an export taken after erasure cannot resurrect the original content.
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .backup import sqlite_file_uri
from .core_store import DEFAULT_BUSY_TIMEOUT_MS
from .schema import SchemaError, require_compatible_schema


class ExportError(RuntimeError):
    """A current-facts export could not be produced safely."""


@dataclass(frozen=True, slots=True)
class ExportReport:
    current_path: Path
    needs_review_dir: Path
    current_facts: int
    conflicted_facts: int
    unreviewed_facts: int
    omitted_erased: int


_ERASED_PREFIX = "[PRIVACY ERASED"
_CURRENT_PREAMBLE = (
    "# Enfold current facts\n"
    "\n"
    "This is a read-only projection of non-superseded, non-conflicted facts.\n"
    "It is not a second store and cannot be imported back into Enfold.\n"
    "\n"
)
_CONFLICT_PREAMBLE = (
    "# NEEDS REVIEW\n"
    "\n"
    "These facts are parked in an unresolved conflict. They are not current truth.\n"
    "\n"
)
_UNREVIEWED_PREAMBLE = (
    "# NEEDS REVIEW\n"
    "\n"
    "These facts were captured without verified evidence support.\n"
    "They are excluded from default recall and prompt context.\n"
    "\n"
)


def _resolved(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _private_directory(path: Path) -> None:
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(path, 0o700)


def _write_private_text(path: Path, text: str) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(fd)
    os.unlink(temporary)
    handle = os.open(temporary, flags, 0o600)
    try:
        payload = text.encode("utf-8")
        written = 0
        while written < len(payload):
            written += os.write(handle, payload[written:])
        os.close(handle)
        handle = -1
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    finally:
        if handle >= 0:
            os.close(handle)


def _connect_read_only(path: str | Path) -> sqlite3.Connection:
    resolved = _resolved(path)
    if not resolved.is_file():
        raise ExportError(f"database does not exist: {resolved}")
    conn = sqlite3.connect(sqlite_file_uri(resolved, mode="ro"), uri=True)
    try:
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute(f"PRAGMA busy_timeout={int(DEFAULT_BUSY_TIMEOUT_MS)}")
        return conn
    except BaseException:
        conn.close()
        raise


def _is_erased(content: object) -> bool:
    return isinstance(content, str) and content.startswith(_ERASED_PREFIX)


def _is_erasure_clone(content_sha256: object) -> bool:
    return isinstance(content_sha256, str) and content_sha256.startswith("clone:")


def _sources(conn: sqlite3.Connection, fact_id: int) -> list[tuple[str, str, str]]:
    rows = conn.execute(
        """
        SELECT COALESCE(o.source_type, ''), COALESCE(o.source_uri, ''),
               COALESCE(p.evidence_excerpt, ''), COALESCE(o.content_sha256, '')
        FROM fact_provenance p
        JOIN observations o ON o.observation_id = p.observation_id
        WHERE p.fact_id = ?
        ORDER BY p.created_at, o.observation_id
        """,
        (fact_id,),
    ).fetchall()
    sources = []
    for source_type, source_uri, excerpt, content_sha256 in rows:
        if _is_erased(excerpt) or (
            isinstance(excerpt, str) and excerpt.startswith("[PRIVACY ERASED")
        ):
            continue
        if _is_erasure_clone(content_sha256):
            continue
        sources.append((str(source_type), str(source_uri), str(excerpt)))
    return sources


def _render_fact(
    fact_id: int,
    content: str,
    category: str,
    scope: str,
    sources: Sequence[tuple[str, str, str]],
    *,
    extra: str | None = None,
) -> str:
    lines = [
        f"## Fact {fact_id}",
        "",
        content,
        "",
        f"- category: {category}",
        f"- scope: {scope}",
    ]
    if extra:
        lines.append(extra)
    if sources:
        lines.append("- sources:")
        for source_type, source_uri, excerpt in sources:
            uri = source_uri or "(no source uri)"
            lines.append(f"  - {source_type} {uri}")
            if excerpt:
                lines.append(f"    excerpt: {excerpt}")
    else:
        lines.append("- sources: (none recorded)")
    lines.append("")
    return "\n".join(lines)


def export_current(database: str | Path, destination: str | Path) -> ExportReport:
    """Dump current facts and park conflicts/unreviewed under needs_review/."""

    dest = _resolved(destination)
    db_path = _resolved(database)
    if dest == db_path:
        raise ExportError("export destination must not be the database")
    _private_directory(dest)
    review_dir = dest / "needs_review"
    _private_directory(review_dir)

    current_facts = 0
    conflicted_facts = 0
    unreviewed_facts = 0
    omitted_erased = 0
    current_blocks: list[str] = []
    conflict_blocks: list[str] = []
    unreviewed_blocks: list[str] = []

    conn = _connect_read_only(database)
    try:
        require_compatible_schema(conn)
        rows = conn.execute(
            """
            SELECT fact_id, content, COALESCE(category, 'general'),
                   COALESCE(scope, 'private'), conflict_group, correction_status
            FROM facts
            WHERE invalid_at IS NULL AND superseded_by IS NULL
            ORDER BY fact_id
            """
        ).fetchall()
        for fact_id, content, category, scope, conflict_group, correction_status in rows:
            fact_id = int(fact_id)
            content = str(content)
            if _is_erased(content):
                omitted_erased += 1
                continue
            sources = _sources(conn, fact_id)
            if conflict_group is not None:
                conflicted_facts += 1
                conflict_blocks.append(
                    _render_fact(
                        fact_id,
                        content,
                        str(category),
                        str(scope),
                        sources,
                        extra=f"- conflict_group: {conflict_group}",
                    )
                )
                continue
            if correction_status == "unreviewed":
                unreviewed_facts += 1
                unreviewed_blocks.append(
                    _render_fact(fact_id, content, str(category), str(scope), sources)
                )
                continue
            current_facts += 1
            current_blocks.append(
                _render_fact(fact_id, content, str(category), str(scope), sources)
            )
    finally:
        conn.close()

    current_path = dest / "current.md"
    _write_private_text(current_path, _CURRENT_PREAMBLE + "".join(current_blocks))
    _write_private_text(
        review_dir / "conflicts.md",
        _CONFLICT_PREAMBLE + "".join(conflict_blocks),
    )
    _write_private_text(
        review_dir / "unreviewed.md",
        _UNREVIEWED_PREAMBLE + "".join(unreviewed_blocks),
    )
    return ExportReport(
        current_path=current_path,
        needs_review_dir=review_dir,
        current_facts=current_facts,
        conflicted_facts=conflicted_facts,
        unreviewed_facts=unreviewed_facts,
        omitted_erased=omitted_erased,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--current",
        action="store_true",
        help="export non-superseded, non-conflicted facts plus source references",
    )
    parser.add_argument("database", help="explicit schema-v1 SQLite database path")
    parser.add_argument("destination", help="directory for current.md and needs_review/")
    args = parser.parse_args(argv)
    if not args.current:
        print("error: export requires --current", file=sys.stderr)
        return 2
    try:
        report = export_current(args.database, args.destination)
    except (ExportError, SchemaError, sqlite3.DatabaseError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(
        f"wrote {report.current_path} ({report.current_facts} current, "
        f"{report.conflicted_facts} conflicted, "
        f"{report.unreviewed_facts} unreviewed)"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    raise SystemExit(main())
