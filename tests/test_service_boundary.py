from __future__ import annotations

from pathlib import Path

import pytest

from enfold.client import ClientConfig, EnfoldClient, EnfoldRemoteError
from enfold.policy import run_scope_for_session
from enfold.protocol import ClientContext
from enfold.server import ServerApplication, load_config

from tests.test_server import _config


def _client(socket_path: Path, session_id: str, client_id: str = "client-a-install") -> EnfoldClient:
    return EnfoldClient(
        ClientConfig(
            socket_path,
            ClientContext(
                client_id=client_id,
                surface="client-a",
                agent_id="client-a",
                session_id=session_id,
                access_scopes=("private", "work"),
            ),
        )
    )


def test_run_scope_is_bound_to_the_creating_session(tmp_path):
    config = load_config(_config(tmp_path))
    writer_session = "writer-session"
    other_session = "other-session"
    with ServerApplication(config) as application:
        application.daemon.start()
        writer = _client(config.socket_path, writer_session)
        other = _client(config.socket_path, other_session, client_id="client-b-install")
        written = writer.request(
            "memory.write",
            {
                "idempotency_key": "run-bind-1",
                "content": "Worker scratch that must stay in this run",
                "source_type": "integration_test",
                "scope": run_scope_for_session(writer_session),
            },
        )
        fact_id = written["fact_id"]

        own = writer.request("memory.evidence", {"fact_id": fact_id})
        assert own["fact"]["fact_id"] == fact_id

        with pytest.raises(EnfoldRemoteError) as evidence:
            other.request("memory.evidence", {"fact_id": fact_id})
        assert evidence.value.code in {"access_denied", "not_found"}

        with pytest.raises(EnfoldRemoteError) as history:
            other.request("memory.history", {"fact_id": fact_id})
        assert history.value.code in {"access_denied", "not_found"}

        with pytest.raises(EnfoldRemoteError) as promote:
            other.request(
                "memory.promote",
                {
                    "fact_id": fact_id,
                    "idempotency_key": "steal-run-1",
                    "target_scope": "private",
                },
            )
        assert promote.value.code == "access_denied"

        with pytest.raises(EnfoldRemoteError) as search:
            other.request(
                "memory.search",
                {
                    "query": "Worker scratch that must stay in this run",
                    "scope": run_scope_for_session(writer_session),
                },
            )
        assert search.value.code == "access_denied"

        with pytest.raises(EnfoldRemoteError) as write_foreign:
            other.request(
                "memory.write",
                {
                    "idempotency_key": "run-bind-foreign",
                    "content": "Must not write into another session run",
                    "source_type": "integration_test",
                    "scope": run_scope_for_session(writer_session),
                },
            )
        assert write_foreign.value.code == "access_denied"
