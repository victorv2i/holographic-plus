"""Five-minute walkthrough of Enfold's truth model.

Two clients write competing typed state at equal authority. Recall returns
a conflict receipt, not a silent winner. A human authority resolves once.
History and evidence stay visible. Erasure is then shown against export.
The store is temporary and never the live Hermes path.

Invoke with ``enfold demo`` or ``python -m enfold.demo``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sqlite3
import tempfile
from typing import Any, Sequence

from .bootstrap import _private_directory, resolve_socket_path
from .client import ClientConfig, EnfoldClient
from .erasure import erase_fact
from .export import export_current
from .protocol import ClientContext
from .schema import migrate
from .server import ServerApplication, load_config


CLIENT_A = "demo-client-a"
CLIENT_B = "demo-client-b"
HUMAN = "demo-human"
SUBJECT_KEY = "env:staging"
PREDICATE_KEY = "port"
VALUE_A = "3100"
VALUE_B = "3200"
AUTHORITY = 0.5
QUERY = "staging port"
LIVE_HERMES = Path.home() / ".hermes" / "memory_store.db"


def _hermes_fingerprint() -> tuple[object, ...]:
    if not LIVE_HERMES.exists():
        return ()
    info = LIVE_HERMES.stat()
    return (info.st_ino, info.st_size, info.st_mtime_ns)


def _content(value: str) -> str:
    return f"The staging port is {value}."


def _prepare_store(root: Path) -> Path:
    database = root / "memory.db"
    conn = sqlite3.connect(database)
    migrate(conn)
    conn.close()
    database.chmod(0o600)
    config_path = root / "server.json"
    socket_path = resolve_socket_path(
        root / "enfold.sock",
        data_directory=root,
        allow_runtime_fallback=True,
    )
    if socket_path.parent != root:
        _private_directory(socket_path.parent, "socket directory")
    config_path.write_text(
        json.dumps(
            {
                "database_path": str(database),
                "socket_path": str(socket_path),
                "grants": {
                    CLIENT_A: ["private"],
                    CLIENT_B: ["private"],
                    HUMAN: ["private"],
                },
                "conflict_resolution_authorities": [HUMAN],
                "retrieval": {"mode": "local-lexical"},
                "client_timeout": 2.0,
                "shutdown_timeout": 2.0,
            }
        ),
        encoding="utf-8",
    )
    config_path.chmod(0o600)
    return config_path


def _client(socket_path: Path, client_id: str, agent_id: str) -> EnfoldClient:
    return EnfoldClient(
        ClientConfig(
            socket_path,
            ClientContext(
                client_id=client_id,
                surface="demo",
                agent_id=agent_id,
                session_id=f"demo-{client_id}",
                access_scopes=("private",),
            ),
        )
    )


def _write_port(client: EnfoldClient, *, client_id: str, value: str, key: str) -> dict[str, Any]:
    written = client.request(
        "memory.write",
        {
            "idempotency_key": key,
            "content": _content(value),
            "source_type": "operator",
            "scope": "private",
            "source_authority": AUTHORITY,
            "state": {
                "subject_key": SUBJECT_KEY,
                "predicate_key": PREDICATE_KEY,
                "object_value": value,
            },
        },
    )
    return {
        "client_id": client_id,
        "outcome": written.get("outcome"),
        "fact_id": written.get("fact_id"),
        "object_value": value,
        "source_authority": AUTHORITY,
        "detail": written.get("detail") or {},
        "content": _content(value),
    }


def _fact_contents(payload: dict[str, Any]) -> str:
    return " ".join(str(row.get("content") or "") for row in payload.get("facts") or [])


def _collect_evidence_clients(client: EnfoldClient, fact_ids: Sequence[int]) -> list[str]:
    named: list[str] = []
    seen: set[str] = set()
    for fact_id in fact_ids:
        if not isinstance(fact_id, int):
            continue
        evidence = client.request("memory.evidence", {"fact_id": fact_id})
        for row in evidence.get("evidence") or ():
            client_id = row.get("client_id")
            if isinstance(client_id, str) and client_id not in seen:
                seen.add(client_id)
                named.append(client_id)
    return named


def _read_export_text(destination: Path) -> str:
    parts: list[str] = []
    current = destination / "current.md"
    if current.is_file():
        parts.append(current.read_text(encoding="utf-8"))
    review = destination / "needs_review"
    if review.is_dir():
        for path in sorted(review.glob("*.md")):
            parts.append(path.read_text(encoding="utf-8"))
    return "".join(parts)


def _erase_and_export(database: Path, destination: Path, fact_id: int) -> dict[str, Any]:
    conn = sqlite3.connect(database)
    try:
        erase_fact(
            conn,
            fact_id,
            requested_by=HUMAN,
            reason="demo privacy erasure",
        )
    finally:
        conn.close()
    report = export_current(database, destination)
    export_text = _read_export_text(destination)
    return {
        "shown": True,
        "fact_id": fact_id,
        "omitted_erased": report.omitted_erased,
        "export_text": export_text,
        "path": "maintenance erase_fact then export_current",
    }


def _exercise(socket_path: Path) -> dict[str, Any]:
    client_a = _client(socket_path, CLIENT_A, "demo-a")
    client_b = _client(socket_path, CLIENT_B, "demo-b")
    human = _client(socket_path, HUMAN, "demo-human")
    report: dict[str, Any] = {
        "ok": False,
        "retrieval": "local-lexical",
        "network": "disabled",
        "live_hermes_path": str(LIVE_HERMES),
        "diagnosis": "demo did not finish",
    }
    first = _write_port(
        client_a, client_id=CLIENT_A, value=VALUE_A, key="enfold-demo-port-a"
    )
    report["client_a"] = first
    if first["outcome"] != "add" or not isinstance(first["fact_id"], int):
        report["diagnosis"] = (
            f"client A write did not add typed state; outcome={first['outcome']!r}"
        )
        return report

    second = _write_port(
        client_b, client_id=CLIENT_B, value=VALUE_B, key="enfold-demo-port-b"
    )
    report["client_b"] = second
    if second["outcome"] != "conflict" or not isinstance(second["fact_id"], int):
        report["diagnosis"] = (
            "client B write at equal authority did not open a conflict; "
            f"outcome={second['outcome']!r}"
        )
        return report

    conflict_id = (second.get("detail") or {}).get("conflict_id")
    recall = client_a.request("memory.search", {"query": QUERY})
    report["conflict_recall"] = {
        "facts": recall.get("facts") or [],
        "open_conflicts": recall.get("open_conflicts") or [],
    }
    receipts = report["conflict_recall"]["open_conflicts"]
    if not receipts:
        report["diagnosis"] = "recall returned no conflict receipt"
        return report
    receipt = receipts[0]
    if not conflict_id:
        conflict_id = receipt.get("conflict_id")
    report["conflict_id"] = conflict_id
    contents = _fact_contents(report["conflict_recall"])
    if VALUE_A in contents or VALUE_B in contents:
        report["diagnosis"] = (
            "recall returned a silent winner instead of a conflict receipt"
        )
        return report
    if receipt.get("conflict_id") != conflict_id:
        report["diagnosis"] = "recall receipt did not match the write conflict"
        return report

    resolved = human.request(
        "memory.resolve_conflict",
        {
            "conflict_id": conflict_id,
            "resolution_fact_id": second["fact_id"],
            "reason": "operator chose the current staging port",
        },
    )
    resolution = resolved.get("resolution") or {}
    report["resolution"] = resolution
    if resolution.get("resolution_fact_id") != second["fact_id"]:
        report["diagnosis"] = "human resolve did not keep 3200 as current"
        return report

    settled = client_a.request("memory.search", {"query": QUERY})
    report["settled_recall"] = {
        "facts": settled.get("facts") or [],
        "open_conflicts": settled.get("open_conflicts") or [],
    }
    settled_contents = _fact_contents(report["settled_recall"])
    if VALUE_B not in settled_contents or VALUE_A in settled_contents:
        report["diagnosis"] = (
            "after resolve, recall did not return only 3200 as current"
        )
        return report
    if report["settled_recall"]["open_conflicts"]:
        report["diagnosis"] = "after resolve, recall still reported an open conflict"
        return report

    history = client_a.request(
        "memory.history",
        {"subject_key": SUBJECT_KEY, "predicate_key": PREDICATE_KEY},
    )
    report["history"] = {"facts": history.get("facts") or []}
    older = next(
        (
            row
            for row in report["history"]["facts"]
            if row.get("fact_id") == first["fact_id"]
        ),
        None,
    )
    if older is None or older.get("superseded_by") != second["fact_id"]:
        report["diagnosis"] = "history did not keep 3100 with its superseded provenance"
        return report

    report["evidence_client_ids"] = _collect_evidence_clients(
        client_a, (first["fact_id"], second["fact_id"])
    )
    if CLIENT_A not in report["evidence_client_ids"] or CLIENT_B not in report[
        "evidence_client_ids"
    ]:
        report["diagnosis"] = "evidence did not name both writing clients"
        return report

    report["ok"] = True
    report["diagnosis"] = (
        "wrote competing typed state, returned a conflict receipt, resolved "
        "as the human authority, and kept history plus evidence"
    )
    return report


def _finish_erasure(report: dict[str, Any], database: Path, export_dir: Path) -> None:
    fact_id = (report.get("client_b") or {}).get("fact_id")
    if not report.get("ok") or not isinstance(fact_id, int):
        report.setdefault(
            "erasure",
            {
                "shown": False,
                "fact_id": fact_id,
                "omitted_erased": 0,
                "export_text": "",
                "path": "maintenance erase_fact then export_current",
                "error": "protocol walk did not finish",
            },
        )
        return
    try:
        report["erasure"] = _erase_and_export(database, export_dir, fact_id)
    except Exception as exc:
        report["ok"] = False
        report["erasure"] = {
            "shown": False,
            "fact_id": fact_id,
            "omitted_erased": 0,
            "export_text": "",
            "path": "maintenance erase_fact then export_current",
            "error": str(exc),
        }
        report["diagnosis"] = f"erasure or export could not be shown: {exc}"
        return
    if VALUE_B in report["erasure"]["export_text"]:
        report["ok"] = False
        report["diagnosis"] = "export still contained the erased 3200 value"
        return
    report["diagnosis"] = (
        "wrote competing typed state, returned a conflict receipt, resolved "
        "as the human authority, then showed history, evidence, and erasure"
    )


def run_demo() -> dict[str, Any]:
    """Prove the conflict receipt on an isolated local-lexical daemon."""

    before = _hermes_fingerprint()
    with tempfile.TemporaryDirectory(prefix="enfold-demo-") as tmp:
        root = Path(tmp)
        os.chmod(root, 0o700)
        config_path = _prepare_store(root)
        config = load_config(config_path)
        extra_socket_parent = (
            config.socket_path.parent if config.socket_path.parent != root else None
        )
        export_dir = root / "export"
        try:
            with ServerApplication(config) as application:
                application.daemon.start()
                report = _exercise(config.socket_path)
            _finish_erasure(report, config.database_path, export_dir)
        finally:
            if extra_socket_parent is not None:
                try:
                    config.socket_path.unlink(missing_ok=True)
                except OSError:
                    pass
                try:
                    extra_socket_parent.rmdir()
                except OSError:
                    pass
        report["touched_live_hermes"] = _hermes_fingerprint() != before
        report["store_path"] = str(root)
    report["store_discarded"] = not Path(report["store_path"]).exists()
    return report


def render_demo(report: dict[str, Any]) -> str:
    """Turn a live demo report into a short, honest walkthrough."""

    lines = [
        "Enfold demo",
        "Disposable store. Offline. Local-lexical retrieval.",
        "This is not a bag of notes: current state, conflict, and human resolve are first-class.",
        "",
    ]
    first = report.get("client_a") or {}
    second = report.get("client_b") or {}
    lines.append("1. Client A writes typed state")
    lines.append(f"   client: {first.get('client_id', CLIENT_A)}")
    lines.append(f"   claim:  {SUBJECT_KEY}.{PREDICATE_KEY} = {VALUE_A}")
    lines.append(
        f"   write:  outcome={first.get('outcome')} fact_id={first.get('fact_id')} "
        f"authority={first.get('source_authority')}"
    )
    lines.append("")
    lines.append("2. Client B writes the same slot at equal authority")
    lines.append(f"   client: {second.get('client_id', CLIENT_B)}")
    lines.append(f"   claim:  {SUBJECT_KEY}.{PREDICATE_KEY} = {VALUE_B}")
    lines.append(
        f"   write:  outcome={second.get('outcome')} fact_id={second.get('fact_id')} "
        f"authority={second.get('source_authority')}"
    )
    lines.append("")
    lines.append("3. Recall returns a conflict receipt, not 3100, not 3200")
    receipts = (report.get("conflict_recall") or {}).get("open_conflicts") or []
    facts = (report.get("conflict_recall") or {}).get("facts") or []
    if facts:
        lines.append(f"   current facts returned: {len(facts)}")
        for row in facts:
            lines.append(f"   - {row.get('content')}")
    else:
        lines.append("   current facts returned: none")
    if receipts:
        receipt = receipts[0]
        lines.append(f"   receipt: {receipt.get('summary')}")
        lines.append(f"   conflict_id: {receipt.get('conflict_id')}")
        lines.append(f"   members: {receipt.get('member_fact_ids')}")
        lines.append(
            "   This is the moment. A competitor would silently return one value, "
            "or both as if they were both current."
        )
    else:
        lines.append("   Could not show a conflict receipt on this build.")
    lines.append("")
    lines.append("4. Human authority resolves once")
    resolution = report.get("resolution") or {}
    if resolution:
        lines.append(f"   resolver: {HUMAN}")
        lines.append(
            f"   winner: fact_id={resolution.get('resolution_fact_id')} "
            f"(staging port {VALUE_B})"
        )
        lines.append(
            f"   superseded: {resolution.get('superseded_fact_ids')}"
        )
        lines.append("   reason: operator chose the current staging port")
    else:
        lines.append("   Could not resolve the conflict on this build.")
    lines.append("")
    lines.append("5. Recall now returns 3200. History keeps 3100.")
    settled = (report.get("settled_recall") or {}).get("facts") or []
    if settled:
        for row in settled:
            lines.append(f"   current: {row.get('content')}")
    else:
        lines.append("   Could not show a settled current value.")
    for row in (report.get("history") or {}).get("facts") or ():
        lines.append(
            f"   history: fact_id={row.get('fact_id')} "
            f"value={row.get('object_value')} "
            f"superseded_by={row.get('superseded_by')} "
            f"content={row.get('content')}"
        )
    named = report.get("evidence_client_ids") or []
    if named:
        lines.append(f"   evidence names: {', '.join(named)}")
    else:
        lines.append("   Could not name both clients from evidence.")
    lines.append("")
    lines.append("6. Erase the current fact. Export cannot recover it.")
    erasure = report.get("erasure") or {}
    if erasure.get("shown"):
        lines.append(f"   path: {erasure.get('path')}")
        lines.append(
            f"   erased fact_id={erasure.get('fact_id')}; "
            f"export omitted_erased={erasure.get('omitted_erased')}"
        )
        if VALUE_B in (erasure.get("export_text") or ""):
            lines.append("   Could not hide 3200 from export on this build.")
        else:
            lines.append("   export text does not contain 3200.")
    else:
        error = erasure.get("error") or "erasure step did not run"
        lines.append(f"   Could not show erasure: {error}")
    lines.append("")
    lines.append(f"Diagnosis: {report.get('diagnosis')}")
    if report.get("store_discarded"):
        lines.append("Disposable store discarded.")
    else:
        lines.append("Warning: disposable store was not discarded.")
    if report.get("touched_live_hermes"):
        lines.append(f"Warning: {LIVE_HERMES} changed during the demo.")
    else:
        lines.append(f"Live {LIVE_HERMES} was not used.")
    lines.append("")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    del argv
    report = run_demo()
    print(render_demo(report), end="")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
