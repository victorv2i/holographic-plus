"""End-to-end self-test for the local-lexical retrieval default.

Writes one fact, recalls it, and returns its evidence through the real
daemon service path. The store is temporary and isolated from any live
instance. Invoke with ``enfold doctor`` or ``python -m enfold.doctor``.
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
from .protocol import ClientContext
from .schema import migrate
from .server import ServerApplication, load_config


DOCTOR_CLIENT_ID = "enfold-doctor"
DOCTOR_CONTENT = "Doctor self-test marker: the test preference is dark mode."
DOCTOR_QUERY = "test preference is dark mode"


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
                "grants": {DOCTOR_CLIENT_ID: ["private"]},
                "retrieval": {"mode": "local-lexical"},
                "client_timeout": 2.0,
                "shutdown_timeout": 2.0,
            }
        ),
        encoding="utf-8",
    )
    config_path.chmod(0o600)
    return config_path


def _client(socket_path: Path) -> EnfoldClient:
    return EnfoldClient(
        ClientConfig(
            socket_path,
            ClientContext(
                client_id=DOCTOR_CLIENT_ID,
                surface="doctor",
                agent_id="enfold-doctor",
                session_id="doctor-self-test",
                access_scopes=("private",),
            ),
        )
    )


def _exercise(socket_path: Path) -> dict[str, Any]:
    client = _client(socket_path)
    report: dict[str, Any] = {
        "ok": False,
        "retrieval": "local-lexical",
        "network": "disabled",
        "write": "fail",
        "recall": "fail",
        "evidence": "fail",
        "fact_id": None,
        "evidence_count": 0,
        "output_truncated": False,
        "content": DOCTOR_CONTENT,
        "diagnosis": "self-test did not finish",
    }
    written = client.request(
        "memory.write",
        {
            "idempotency_key": "enfold-doctor-self-test",
            "content": DOCTOR_CONTENT,
            "source_type": "doctor",
            "scope": "private",
        },
    )
    fact_id = written.get("fact_id")
    if written.get("outcome") != "inserted" or not isinstance(fact_id, int):
        report["diagnosis"] = "write did not insert a fact through the daemon"
        return report
    report["write"] = "pass"
    report["fact_id"] = fact_id

    found = client.request("memory.search", {"query": DOCTOR_QUERY})
    facts = found.get("facts") or []
    if [row.get("fact_id") for row in facts] != [fact_id]:
        report["diagnosis"] = (
            f"wrote fact_id={fact_id} but local-lexical search did not recall it"
        )
        return report
    report["recall"] = "pass"

    evidence = client.request("memory.evidence", {"fact_id": fact_id})
    rows = evidence.get("evidence") or []
    truncated = bool(evidence.get("output_truncated"))
    report["evidence_count"] = len(rows)
    report["output_truncated"] = truncated
    if not rows or evidence.get("fact", {}).get("fact_id") != fact_id:
        report["diagnosis"] = (
            f"recalled fact_id={fact_id} but evidence returned no provenance"
        )
        return report
    if truncated:
        report["diagnosis"] = (
            "evidence was truncated; self-test cannot trust the receipt"
        )
        return report
    report["evidence"] = "pass"
    report["ok"] = True
    report["diagnosis"] = (
        "wrote, recalled, and returned evidence through the "
        "local-lexical daemon path"
    )
    return report


def run_self_test() -> dict[str, Any]:
    """Prove write, recall, and evidence on an isolated local-lexical daemon."""

    with tempfile.TemporaryDirectory(prefix="enfold-doctor-") as tmp:
        root = Path(tmp)
        os.chmod(root, 0o700)
        config = load_config(_prepare_store(root))
        extra_socket_parent = (
            config.socket_path.parent if config.socket_path.parent != root else None
        )
        try:
            with ServerApplication(config) as application:
                application.daemon.start()
                return _exercise(config.socket_path)
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


def main(argv: Sequence[str] | None = None) -> int:
    del argv
    report = run_self_test()
    print(json.dumps(report, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
