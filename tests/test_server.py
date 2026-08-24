from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import socket
import sqlite3
import stat
import sys
import threading
import time

import numpy as np
import pytest

from enfold.client import (
    ClientConfig,
    EnfoldClient,
    EnfoldHandshakeError,
    EnfoldRemoteError,
)
from enfold.embeddings import embedding_to_bytes
from enfold.extraction_processor import EvidenceVerification, ExtractedMemory
from enfold.extraction_spans import transcript_spans
from enfold.extractor_artifact import bundled_ollama_components
from enfold.host_extractor import HostExtractorConfig
from enfold.ollama_artifact import ArtifactAttestation, ArtifactAttestationError
from enfold.protocol import (
    ClientContext,
    Handshake,
    Request,
    Response,
    decode_frame,
    encode_frame,
)
from enfold.schema import migrate
from enfold.server import (
    DatabaseOwnershipError,
    ServerApplication,
    ServerConfigError,
    inspect_config,
    load_config,
    main,
)


_ARTIFACT_DIGEST = "sha256:" + "a" * 64
_EXTRACTION_MODEL_DIGEST = "sha256:" + "b" * 64


class FakeArtifactAttestor:
    def __init__(self, *, failure: Exception | None = None):
        self.failure = failure
        self.calls: list[tuple[str, str]] = []

    def attest(self, *, model: str, expected_digest: str) -> ArtifactAttestation:
        self.calls.append((model, expected_digest))
        if self.failure is not None:
            raise self.failure
        return ArtifactAttestation()


def _database(path: Path) -> Path:
    conn = sqlite3.connect(path)
    migrate(conn)
    conn.close()
    return path


def _config(tmp_path: Path, **changes) -> Path:
    data = {
        "database_path": str(_database(tmp_path / "memory.db")),
        "socket_path": str(tmp_path / "enfold.sock"),
        "grants": {
            "client-a-install": ["private", "work"],
            "client-b-install": ["private"],
            "hermes-install": ["private", "work"],
        },
        "retrieval": {
            "mode": "ci",
            "allow_nonproduction": True,
            "dimensions": 64,
        },
        "client_timeout": 0.5,
        "shutdown_timeout": 1.0,
    }
    data.update(changes)
    path = tmp_path / "server.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    path.chmod(0o600)
    return path


def _client(socket_path: Path) -> EnfoldClient:
    context = ClientContext(
        client_id="client-a-install",
        surface="client-a",
        agent_id="client-a",
        session_id="server-integration",
        access_scopes=("private", "work"),
    )
    return EnfoldClient(ClientConfig(socket_path, context))


def test_application_composes_service_health_and_write_in_temp_directory(tmp_path):
    config = load_config(_config(tmp_path))
    with ServerApplication(config) as application:
        application.daemon.start()
        client = _client(config.socket_path)
        health = client.request("health")
        assert health["status"] == "ok"
        assert health["schema_version"] == 1
        assert health["storage"] == "sqlite"
        assert health["retrieval"]["filter_before_dense_ranking"] is True
        assert health["retrieval"]["embedder_production_ready"] is False
        assert health["automatic_llm_extraction"] == {"status": "disabled"}
        result = client.request(
            "memory.write",
            {
                "idempotency_key": "server-test-1",
                "content": "Client A exercised the packaged Enfold daemon",
                "source_type": "integration_test",
                "scope": "private",
            },
        )
        assert result["outcome"] == "inserted"
        found = client.request(
            "memory.search", {"query": "packaged Enfold daemon"}
        )
        assert [row["fact_id"] for row in found["facts"]] == [result["fact_id"]]
    assert not config.socket_path.exists()


def test_protocol_health_rejects_ungranted_client_context(tmp_path):
    config = load_config(_config(tmp_path))
    context = ClientContext(
        client_id="ungranted-health-client",
        surface="operator-health",
        agent_id="operator-health",
        session_id="ungranted-health",
        access_scopes=("secret",),
    )
    client = EnfoldClient(ClientConfig(config.socket_path, context))
    with ServerApplication(config) as application:
        application.daemon.start()

        with pytest.raises(EnfoldRemoteError) as denied:
            client.request("health")

    assert denied.value.code == "access_denied"
    assert denied.value.message == "memory client is not authorized"


def test_protocol_health_degrades_after_process_local_vector_fallback(tmp_path):
    config = load_config(_config(tmp_path))
    with ServerApplication(config) as application:
        application.vector_fallback_telemetry.record("sqlite_vec_query_error")

        health = application.daemon._health(
            ClientContext(
                client_id="client-a-install",
                surface="client-a",
                agent_id="client-a",
                session_id="vector-fallback-health",
                access_scopes=("private",),
            )
        )

        assert health["status"] == "degraded"
        assert health["retrieval"]["vector_fallback_active"] is True
        assert health["retrieval"]["vector_fallback_count"] == 1
        assert (
            health["retrieval"]["vector_last_fallback_reason"]
            == "sqlite_vec_query_error"
        )


def test_database_sidecar_refuses_second_server_and_allows_clean_reacquire(tmp_path):
    config = load_config(_config(tmp_path))
    first = ServerApplication(config)
    lock_path = config.database_path.with_name(config.database_path.name + ".enfold.lock")
    try:
        assert lock_path.is_file()
        assert lock_path.stat().st_mode & 0o777 == 0o600
        with pytest.raises(DatabaseOwnershipError, match="another Enfold server"):
            ServerApplication(config)
    finally:
        first.close()

    # The stable sidecar inode persists, but releasing flock permits the next
    # server to acquire sole ownership without a split-lock unlink race.
    with ServerApplication(config):
        pass


def test_database_sidecar_symlink_is_refused(tmp_path):
    config = load_config(_config(tmp_path))
    target = tmp_path / "elsewhere"
    target.write_text("", encoding="utf-8")
    lock_path = config.database_path.with_name(config.database_path.name + ".enfold.lock")
    lock_path.symlink_to(target)
    with pytest.raises(DatabaseOwnershipError, match="cannot open database lock"):
        ServerApplication(config)


def test_check_reports_version_schema_and_grants_without_binding(tmp_path, capsys):
    config_path = _config(tmp_path)
    assert main(["--config", str(config_path), "check"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "ready"
    assert report["schema_version"] == 1
    assert report["database"] == "compatible"
    assert report["socket"] == "absent"
    assert report["grant_count"] == 3
    assert report["retrieval"]["embedder_production_ready"] is False


def test_health_command_queries_the_live_protocol(tmp_path, capsys):
    config_path = _config(tmp_path)
    config = load_config(config_path)
    with ServerApplication(config) as application:
        application.daemon.start()

        assert main([
            "--config", str(config_path), "health",
            "--client-id", "client-a-install",
            "--surface", "operator-health",
            "--agent-id", "operator-health",
            "--access-scope", "private",
        ]) == 0

    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "ok"
    assert report["service_version"] == "0.8.0"
    assert report["schema_version"] == 1


def test_health_readiness_accepts_configured_healthcheck_public_grant(
    tmp_path, capsys
):
    config_path = _config(
        tmp_path,
        grants={"healthcheck-install": ["public"]},
    )
    config = load_config(config_path)
    with ServerApplication(config) as application:
        application.daemon.start()

        assert main([
            "--config", str(config_path), "health", "--readiness",
        ]) == 0

    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "ok"
    assert report["service_version"] == "0.8.0"
    assert report["schema_version"] == 1


def test_health_command_reports_transport_failure_without_traceback(tmp_path, capsys):
    config_path = _config(tmp_path)

    assert main([
        "--config", str(config_path), "health",
        "--client-id", "client-a-install",
        "--access-scope", "private",
    ]) == 1

    report = json.loads(capsys.readouterr().out)
    assert report == {
        "error": "enfold_client_error",
        "ok": False,
        "problems": ["daemon_unavailable_or_protocol_error"],
        "status": "unavailable",
    }


def test_health_readiness_accepts_authenticated_degraded_response(tmp_path, capsys):
    config_path = _config(tmp_path)
    config = load_config(config_path)
    command = [
        "--config", str(config_path), "health",
        "--client-id", "client-a-install",
        "--access-scope", "private",
    ]
    with ServerApplication(config) as application:
        application.vector_fallback_telemetry.record("sqlite_vec_query_error")
        application.daemon.start()

        assert main(command) == 1
        assert json.loads(capsys.readouterr().out)["status"] == "degraded"
        assert main([*command, "--readiness"]) == 0

    assert json.loads(capsys.readouterr().out)["status"] == "degraded"


def test_status_reports_stale_socket_unhealthy_and_exits_nonzero(tmp_path, capsys):
    config_path = _config(tmp_path, cleanup_stale_socket=True)
    config = load_config(config_path)
    stale = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    stale.bind(os.fspath(config.socket_path))
    stale.close()

    assert main(["--config", str(config_path), "status"]) == 2
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "blocked"
    assert report["socket"] == "stale-or-unreachable"
    assert (
        report["activation_blocker"]
        == "configured socket is stale or unreachable"
    )


def test_server_with_explicit_cleanup_recovers_stale_socket(tmp_path):
    config = load_config(_config(tmp_path, cleanup_stale_socket=True))
    stale = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    stale.bind(os.fspath(config.socket_path))
    stale.close()

    with ServerApplication(config) as application:
        application.daemon.start()
        assert _client(config.socket_path).request("health")["status"] == "ok"

    assert not config.socket_path.exists()


def test_privileged_memory_actions_require_named_granted_clients(tmp_path):
    config = load_config(
        _config(
            tmp_path,
            correction_authorities=["hermes-install"],
            conflict_resolution_authorities=["hermes-install"],
        )
    )
    assert config.correction_authorities == ("hermes-install",)
    assert config.conflict_resolution_authorities == ("hermes-install",)

    with pytest.raises(ServerConfigError, match="clients without grants"):
        load_config(
            _config(
                tmp_path,
                correction_authorities=["self-claimed-client"],
            )
        )


def test_server_client_credentials_prevent_same_uid_grant_impersonation(tmp_path):
    token = "secure-client-token-with-at-least-32-characters"
    digest = "sha256:" + hashlib.sha256(token.encode()).hexdigest()
    client_b_digest = "sha256:" + hashlib.sha256(b"client-b-distinct-token").hexdigest()
    hermes_digest = "sha256:" + hashlib.sha256(b"hermes-distinct-token").hexdigest()
    config = load_config(
        _config(
            tmp_path,
            client_credentials={
                "client-a-install": digest,
                "client-b-install": client_b_digest,
                "hermes-install": hermes_digest,
            },
        )
    )
    assert config.client_credentials["client-a-install"] == digest

    with ServerApplication(config) as application:
        application.daemon.start()
        context = ClientContext(
            client_id="client-a-install",
            surface="forged-surface",
            agent_id="forged-agent",
            session_id="forged-session",
            access_scopes=("private", "work"),
        )
        with pytest.raises(EnfoldHandshakeError) as denied:
            EnfoldClient(
                ClientConfig(config.socket_path, context, credential="wrong-token")
            ).request("health")
        assert denied.value.code == "invalid_client_credentials"

        health = EnfoldClient(
            ClientConfig(config.socket_path, context, credential=token)
        ).request("health")
        assert health["identity_authentication"] == "client-credential"

    with pytest.raises(ServerConfigError, match="unique per client"):
        load_config(
            _config(
                tmp_path,
                client_credentials={
                    "client-a-install": digest,
                    "client-b-install": digest,
                    "hermes-install": hermes_digest,
                },
            )
        )


def test_missing_database_is_not_created_or_migrated(tmp_path):
    missing = tmp_path / "missing.db"
    config_path = _config(tmp_path, database_path=str(missing))
    config = load_config(config_path)
    with pytest.raises(ServerConfigError, match="never creates"):
        ServerApplication(config)
    assert not missing.exists()


def test_unmigrated_database_is_rejected_without_schema_changes(tmp_path):
    database = tmp_path / "empty.db"
    sqlite3.connect(database).close()
    config = load_config(_config(tmp_path, database_path=str(database)))
    with pytest.raises(ServerConfigError, match="must already be schema v1; found v0"):
        inspect_config(config)
    conn = sqlite3.connect(database)
    assert conn.execute("SELECT COUNT(*) FROM sqlite_master").fetchone()[0] == 0
    conn.close()


def test_live_paths_require_explicit_allow_live():
    path = Path.home() / ".hermes" / "enfold-server.json"
    with pytest.raises(ServerConfigError, match="--allow-live"):
        load_config(path)


def test_live_paths_through_symlinked_ancestor_require_allow_live(tmp_path):
    alias = tmp_path / "apparently-safe"
    alias.symlink_to(Path.home() / ".hermes", target_is_directory=True)

    with pytest.raises(ServerConfigError, match="--allow-live"):
        load_config(alias / "enfold-server.json")


def test_config_and_socket_parent_permissions_fail_closed(tmp_path):
    config_path = _config(tmp_path)
    config_path.chmod(0o622)
    with pytest.raises(ServerConfigError, match="group/world writable"):
        load_config(config_path)

    config_path.chmod(0o600)
    config = load_config(config_path)
    tmp_path.chmod(0o777)
    try:
        with pytest.raises(ServerConfigError, match="socket parent"):
            ServerApplication(config)
    finally:
        tmp_path.chmod(0o700)


@pytest.mark.parametrize(
    ("unsafe_target", "message"),
    [
        ("database", "database must not be group/world writable"),
        ("parent", "database parent must not be group/world writable"),
    ],
)
def test_database_permissions_fail_before_writer_lock(
    tmp_path, unsafe_target, message
):
    database_parent = tmp_path / "database"
    database_parent.mkdir(mode=0o700)
    database = _database(database_parent / "memory.db")
    config = load_config(_config(tmp_path, database_path=str(database)))
    target = database if unsafe_target == "database" else database_parent
    original_mode = stat.S_IMODE(target.stat().st_mode)
    target.chmod(original_mode | 0o022)
    lock_path = database.with_name(database.name + ".enfold.lock")
    try:
        with pytest.raises(ServerConfigError, match=message):
            ServerApplication(config)
        assert not lock_path.exists()
    finally:
        target.chmod(original_mode)


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        ('"client_timeout": NaN', "non-finite JSON number"),
        ('"client_timeout": Infinity', "non-finite JSON number"),
        ('"client_timeout": 1e309', "client_timeout must be a positive number"),
    ],
)
def test_config_rejects_non_finite_numbers(tmp_path, replacement, message):
    path = _config(tmp_path)
    text = path.read_text(encoding="utf-8").replace(
        '"client_timeout": 0.5', replacement
    )
    path.write_text(text, encoding="utf-8")

    with pytest.raises(ServerConfigError, match=message):
        load_config(path)


def test_config_rejects_duplicate_json_keys(tmp_path):
    path = _config(tmp_path)
    text = path.read_text(encoding="utf-8").replace(
        '"client_timeout": 0.5',
        '"client_timeout": 0.5, "client_timeout": 0.6',
    )
    path.write_text(text, encoding="utf-8")

    with pytest.raises(ServerConfigError, match="duplicate JSON object key"):
        load_config(path)


def test_config_normalizes_grant_ids_and_rejects_normalized_collisions(tmp_path):
    config = load_config(_config(tmp_path, grants={" client-a ": ["private"]}))
    assert config.grants == {"client-a": ("private",)}

    with pytest.raises(ServerConfigError, match="unique after whitespace"):
        load_config(
            _config(
                tmp_path,
                grants={" client-a ": ["private"], "client-a": ["work"]},
            )
        )


def test_config_reads_and_validates_one_open_descriptor(tmp_path, monkeypatch):
    path = _config(tmp_path)
    real_fdopen = os.fdopen
    real_read_text = Path.read_text
    replaced = False

    def replace_path():
        nonlocal replaced
        if replaced:
            return
        replaced = True
        path.unlink()
        path.write_text("{not valid JSON", encoding="utf-8")
        path.chmod(0o600)

    def racing_fdopen(descriptor, *args, **kwargs):
        replace_path()
        return real_fdopen(descriptor, *args, **kwargs)

    def racing_read_text(candidate, *args, **kwargs):
        if candidate == path:
            replace_path()
        return real_read_text(candidate, *args, **kwargs)

    monkeypatch.setattr(os, "fdopen", racing_fdopen)
    monkeypatch.setattr(Path, "read_text", racing_read_text)

    config = load_config(path)
    assert replaced is True
    assert "client-a-install" in config.grants


@pytest.mark.parametrize(
    ("digits", "message"),
    [
        (4000, "client_timeout must be a positive number"),
        (5000, "cannot read config JSON"),
    ],
)
def test_huge_json_integers_use_config_error_path(
    tmp_path, capsys, digits, message
):
    path = _config(tmp_path)
    text = path.read_text(encoding="utf-8").replace(
        '"client_timeout": 0.5', f'"client_timeout": {"9" * digits}'
    )
    path.write_text(text, encoding="utf-8")

    with pytest.raises(ServerConfigError, match=message):
        load_config(path)
    assert main(["--config", str(path), "check"]) == 2
    assert message in capsys.readouterr().err


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"unknown": True}, "unknown config fields"),
        ({"grants": {}}, "non-empty object"),
        ({"grants": {"client-a": ["made-up"]}}, "unsupported memory scope"),
        ({"cleanup_stale_socket": "yes"}, "must be a boolean"),
        ({"retrieval": {"mode": "ci"}}, "allow_nonproduction=true"),
        ({"retrieval": {"mode": "mystery"}}, "must be 'ci' or 'stored'"),
    ],
)
def test_strict_config_validation(tmp_path, change, message):
    with pytest.raises(ServerConfigError, match=message):
        load_config(_config(tmp_path, **change))


def test_near_dedup_off_switch_is_strictly_parsed(tmp_path):
    path = _config(tmp_path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["retrieval"]["near_dedup_enabled"] = False
    path.write_text(json.dumps(raw), encoding="utf-8")

    assert load_config(path).retrieval.near_dedup_enabled is False

    raw["retrieval"]["near_dedup_enabled"] = "no"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ServerConfigError, match="near_dedup_enabled must be a boolean"):
        load_config(path)


def test_retrieval_selection_is_required_and_ci_needs_explicit_opt_in(tmp_path):
    path = _config(tmp_path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    del raw["retrieval"]
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ServerConfigError, match="missing config fields.*retrieval"):
        load_config(path)


def test_stored_retrieval_readiness_checks_identity_without_model_call(tmp_path):
    database = _database(tmp_path / "memory.db")
    conn = sqlite3.connect(database)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS fact_embeddings(
            fact_id INTEGER NOT NULL,
            embedding BLOB NOT NULL,
            dim INTEGER NOT NULL,
            embedding_identity TEXT NOT NULL,
            PRIMARY KEY(fact_id, embedding_identity)
        )
        """
    )
    conn.execute(
        "INSERT INTO fact_embeddings(fact_id, embedding, dim, embedding_identity) "
        "VALUES (?, ?, ?, ?)",
        (
            1,
            embedding_to_bytes(np.asarray((1.0, 0.0), dtype=np.float32)),
            2,
            f"ollama:fixture:document:none:{_ARTIFACT_DIGEST}",
        ),
    )
    conn.execute(
        "INSERT INTO facts(fact_id, content, scope) VALUES (1, 'fixture fact', 'private')"
    )
    conn.commit()
    conn.close()
    retrieval = {
        "mode": "stored",
        "provider": "ollama",
        "model": "fixture",
        "dimensions": 2,
        "query_identity": f"ollama:fixture:query:none:{_ARTIFACT_DIGEST}",
        "document_identity": f"ollama:fixture:document:none:{_ARTIFACT_DIGEST}",
        "embedding_version": _ARTIFACT_DIGEST,
        "model_fingerprint": _ARTIFACT_DIGEST,
        "prefix_policy": "none",
        "processor": {"mode": "daemon-supervised", "poll_seconds": 0.01},
    }
    config = load_config(_config(tmp_path, retrieval=retrieval))

    attestor = FakeArtifactAttestor()
    report = inspect_config(config, artifact_attestor=attestor)

    assert report["status"] == "ready"
    assert report["activation_blocker"] is None
    assert report["retrieval"]["embedder_production_ready"] is True
    assert report["retrieval"]["document_embedding_identity"] == (
        f"ollama:fixture:document:none:{_ARTIFACT_DIGEST}"
    )
    assert report["retrieval"]["missing_embedding_behavior"] == "fail-closed"
    assert report["artifact_attestation"] == {
        "provider": "ollama", "status": "verified"
    }
    assert attestor.calls == [("fixture", _ARTIFACT_DIGEST)]

    with ServerApplication(
        config, artifact_attestor=FakeArtifactAttestor()
    ) as application:
        application.daemon.start()
        health = _client(config.socket_path).request("health")
        assert health["artifact_attestation"] == {
            "provider": "ollama", "status": "verified"
        }

    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(str(config.socket_path))
    listener.listen(1)
    try:
        live = inspect_config(
            config, probe_socket=True, artifact_attestor=FakeArtifactAttestor()
        )
    finally:
        listener.close()
        config.socket_path.unlink()
    assert live["status"] == "blocked"
    assert live["embedding_worker"]["state"] == "live-health-unverified"
    assert "protocol health" in live["activation_blocker"]

    missing = dict(retrieval)
    missing["query_identity"] = f"ollama:missing:query:none:{_ARTIFACT_DIGEST}"
    missing["document_identity"] = (
        f"ollama:missing:document:none:{_ARTIFACT_DIGEST}"
    )
    with pytest.raises(ServerConfigError, match="canonically derived"):
        load_config(_config(tmp_path, retrieval=missing))


def test_stored_retrieval_requires_an_immutable_artifact_digest(tmp_path):
    retrieval = {
        "mode": "stored",
        "provider": "ollama",
        "model": "fixture",
        "dimensions": 2,
        "query_identity": "ollama:fixture:query:none:v1",
        "document_identity": "ollama:fixture:document:none:v1",
        "embedding_version": "v1",
        "model_fingerprint": "v1",
        "prefix_policy": "none",
        "processor": {"mode": "daemon-supervised"},
    }
    with pytest.raises(ServerConfigError, match="64 lowercase hexadecimal"):
        load_config(_config(tmp_path, retrieval=retrieval))

    missing = dict(retrieval)
    missing.pop("model_fingerprint")
    with pytest.raises(ServerConfigError, match="model_fingerprint"):
        load_config(_config(tmp_path, retrieval=missing))


def test_stored_startup_fails_before_backfill_when_attestation_fails(tmp_path):
    database = _database(tmp_path / "memory.db")
    with sqlite3.connect(database) as conn:
        conn.execute(
            "INSERT INTO facts(content, scope) VALUES ('must not be backfilled', 'private')"
        )
        conn.commit()
    retrieval = {
        "mode": "stored",
        "provider": "ollama",
        "model": "fixture",
        "dimensions": 2,
        "query_identity": f"ollama:fixture:query:none:{_ARTIFACT_DIGEST}",
        "document_identity": f"ollama:fixture:document:none:{_ARTIFACT_DIGEST}",
        "embedding_version": _ARTIFACT_DIGEST,
        "model_fingerprint": _ARTIFACT_DIGEST,
        "prefix_policy": "none",
        "processor": {"mode": "daemon-supervised"},
    }
    config = load_config(_config(tmp_path, retrieval=retrieval))
    attestor = FakeArtifactAttestor(
        failure=ArtifactAttestationError("fixture attestation failure")
    )

    with pytest.raises(ServerConfigError, match="artifact attestation failed"):
        ServerApplication(config, artifact_attestor=attestor)

    with sqlite3.connect(database) as conn:
        assert conn.execute("SELECT COUNT(*) FROM embedding_jobs").fetchone() == (0,)
    assert not (tmp_path / "memory.db.enfold.lock").exists()
    assert attestor.calls == [("fixture", _ARTIFACT_DIGEST)]


@pytest.mark.parametrize(
    ("field", "value"),
    [("poll_seconds", 0), ("drain_limit", 1.5), ("lease_seconds", True),
     ("max_attempts", -1), ("heartbeat_stale_seconds", float("inf"))],
)
def test_stored_processor_numeric_configuration_fails_early(tmp_path, field, value):
    retrieval = {
        "mode": "stored", "provider": "ollama", "model": "fixture",
        "dimensions": 2,
        "query_identity": f"ollama:fixture:query:none:{_ARTIFACT_DIGEST}",
        "document_identity": f"ollama:fixture:document:none:{_ARTIFACT_DIGEST}",
        "embedding_version": _ARTIFACT_DIGEST,
        "model_fingerprint": _ARTIFACT_DIGEST,
        "prefix_policy": "none",
        "processor": {"mode": "daemon-supervised", field: value},
    }
    message = (
        "non-finite JSON number"
        if field == "heartbeat_stale_seconds"
        else f"processor.{field}"
    )
    with pytest.raises(ServerConfigError, match=message):
        load_config(_config(tmp_path, retrieval=retrieval))


def test_stored_fastembed_is_blocked_until_worker_process_isolation(tmp_path):
    retrieval = {
        "mode": "stored", "provider": "fastembed", "model": "fixture",
        "dimensions": 2,
        "query_identity": "fastembed:fixture:query:none:v1",
        "document_identity": "fastembed:fixture:document:none:v1",
        "embedding_version": "v1", "model_fingerprint": "v1",
        "prefix_policy": "none",
        "processor": {"mode": "daemon-supervised"},
    }
    with pytest.raises(ServerConfigError, match="killable process isolation"):
        load_config(_config(tmp_path, retrieval=retrieval))


def test_close_finishes_cleanup_when_a_worker_stop_fails(tmp_path):
    application = ServerApplication(load_config(_config(tmp_path)))

    class FailingWorker:
        def __init__(self):
            self.calls = 0

        def stop(self, _timeout):
            self.calls += 1
            raise RuntimeError("join timeout")

    class RecordingWorker:
        def __init__(self):
            self.calls = 0

        def stop(self, _timeout):
            self.calls += 1

    extraction_worker = FailingWorker()
    embedding_worker = RecordingWorker()
    application.extraction_worker = extraction_worker
    application.embedding_worker = embedding_worker
    with pytest.raises(ExceptionGroup, match="server cleanup failed") as raised:
        application.close()
    assert "join timeout" in str(raised.value.exceptions[0])
    assert application._closed is True
    assert application.ownership._fd is None
    assert extraction_worker.calls == 1
    assert embedding_worker.calls == 1
    with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
        application.connection.execute("SELECT 1")

    application.close()
    assert extraction_worker.calls == 1
    assert embedding_worker.calls == 1


@pytest.mark.parametrize("live_thread_owner", ["daemon", "worker"])
def test_close_preserves_shared_state_while_a_thread_is_alive(
    tmp_path, live_thread_owner
):
    application = ServerApplication(
        load_config(_config(tmp_path, shutdown_timeout=0.01))
    )
    release = threading.Event()
    thread = threading.Thread(target=release.wait, name="blocked-enfold-thread")
    thread.start()

    class BlockingWorker:
        def __init__(self, worker_thread):
            self._thread = worker_thread

        def stop(self, timeout):
            self._thread.join(timeout)
            if self._thread.is_alive():
                raise RuntimeError("worker did not stop cleanly")

    if live_thread_owner == "daemon":
        application.daemon._client_threads.add(thread)
    else:
        application.extraction_worker = BlockingWorker(thread)

    try:
        with pytest.raises(BaseExceptionGroup, match="server cleanup failed"):
            application.close()
        assert application._closed is False
        assert application._close_degraded is True
        assert application.ownership._fd is not None
        assert application.connection.execute("SELECT 1").fetchone()[0] == 1
        with pytest.raises(DatabaseOwnershipError, match="another Enfold server"):
            ServerApplication(application.config)
    finally:
        release.set()
        thread.join(1.0)
        application.close()

    assert application._closed is True
    assert application._close_degraded is False
    assert application.ownership._fd is None
    with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
        application.connection.execute("SELECT 1")


def _extraction_config(**changes):
    host = {
        "type": "subprocess",
        "argv": [
            sys.executable,
            "-m", "enfold.ollama_extractor_child",
            "--model", "fixture-model:latest",
            "--model-identity", "ollama:fixture-model",
            "--prompt-identity", "durable-memory-v3",
        ],
        "model_identity": "ollama:fixture-model",
        "prompt_identity": "durable-memory-v3",
        "timeout_seconds": 0.2,
        "terminate_grace_seconds": 0.1,
    }
    host_config = HostExtractorConfig(
        argv=tuple(host["argv"]),
        model_identity=host["model_identity"],
        prompt_identity=host["prompt_identity"],
        timeout_seconds=host["timeout_seconds"],
        terminate_grace_seconds=host["terminate_grace_seconds"],
    )
    recipe = host_config.inference_recipe(
        model_artifact_digest=_EXTRACTION_MODEL_DIGEST,
        **bundled_ollama_components(sys.executable),
    )
    config = {
        "mode": "daemon-supervised",
        "host": host,
        "artifact": {
            "provider": "ollama",
            "model": "fixture-model:latest",
            "model_digest": _EXTRACTION_MODEL_DIGEST,
            "recipe_digest": recipe.digest,
        },
        "poll_seconds": 0.01,
        "lease_seconds": 1.0,
        "heartbeat_seconds": 0.1,
        "heartbeat_stale_seconds": 0.5,
        "pending_stale_seconds": 30.0,
    }
    config.update(changes)
    return config


class FakeExtractionAdapter:
    identity = "fake:fixture-model:fixture-prompt-v1"

    def __init__(self):
        self.calls = 0

    def extract(self, envelope):
        self.calls += 1
        return [
            ExtractedMemory(
                "Avery uses supervised shared memory.",
                evidence_excerpt="Avery uses supervised shared memory.",
                metadata={
                    "evidence_span_id": transcript_spans(envelope.transcript)[0].span_id,
                },
            )
        ]


class VerifiedTestEvidence:
    identity = "test-evidence-v1"

    def verify(self, _proposal, *, evidence_excerpt, envelope):
        assert evidence_excerpt in envelope.transcript
        return EvidenceVerification("verified", self.identity)


def test_supervised_extraction_uses_dedicated_connection_and_reports_health(tmp_path):
    adapter = FakeExtractionAdapter()
    factory_calls = []

    def factory(config):
        factory_calls.append(config)
        return adapter

    config = load_config(_config(tmp_path, extraction=_extraction_config()))
    with ServerApplication(
        config,
        extraction_extractor_factory=factory,
        extraction_artifact_attestor=FakeArtifactAttestor(),
        extraction_evidence_verifier=VerifiedTestEvidence(),
    ) as application:
        assert application.extraction_connection is not application.connection
        assert factory_calls == [config.extraction.host]
        application.extraction_worker.start()
        application.daemon.start()
        client = _client(config.socket_path)
        queued = client.request(
            "memory.extraction.enqueue",
            {
                "transcript": "Avery uses supervised shared memory.",
                "source": "integration_test",
            },
        )
        assert queued["outcome"] == "queued"
        assert queued["automatic_llm_extraction"] == "daemon-supervised"
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            health = client.request("health")
            extraction_health = health["automatic_llm_extraction"]
            queue_health = extraction_health.get("queue", {})
            if (
                extraction_health["status"] == "ready"
                and adapter.calls
                and queue_health.get("pending") == 0
                and queue_health.get("processing") == 0
            ):
                break
            time.sleep(0.01)
        assert health["status"] == "ok"
        assert health["extraction_artifact_attestation"] == {
            "provider": "ollama",
            "status": "verified",
            "recipe_version": 1,
        }
        extraction = health["automatic_llm_extraction"]
        assert extraction["status"] == "ready"
        assert extraction["worker"]["last_error"] is None
        assert extraction["queue"] == {
            "pending": 0,
            "processing": 0,
            "dead": 0,
            "acknowledged": 0,
            "oldest_active_age_seconds": None,
            "pending_stale": False,
        }
        assert adapter.calls == 1


def test_supervised_extraction_without_evidence_verifier_is_degraded_and_quarantines(
    tmp_path,
):
    config = load_config(_config(tmp_path, extraction=_extraction_config()))
    with ServerApplication(
        config,
        extraction_extractor_factory=lambda _config: FakeExtractionAdapter(),
        extraction_artifact_attestor=FakeArtifactAttestor(),
    ) as application:
        application.extraction_worker.start()
        application.daemon.start()
        client = _client(config.socket_path)
        queued = client.request(
            "memory.extraction.enqueue",
            {
                "transcript": "Avery uses supervised shared memory.",
                "source": "integration_test",
            },
        )
        assert queued["outcome"] == "queued"
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            health = client.request("health")
            queue = health["automatic_llm_extraction"].get("queue", {})
            if queue.get("dead") == 1:
                break
            time.sleep(0.01)

        extraction = health["automatic_llm_extraction"]
        assert health["status"] == "degraded"
        assert extraction["status"] == "degraded"
        assert extraction["evidence_verifier"] == {
            "configured": False,
            "verifier_id": "unconfigured",
        }
        assert extraction["queue"]["dead"] == 1
        assert application.connection.execute("SELECT count(*) FROM facts").fetchone()[0] == 0


def test_oversized_extraction_enqueue_is_invalid_params(tmp_path):
    config = load_config(_config(tmp_path, extraction=_extraction_config()))
    with ServerApplication(
        config,
        extraction_extractor_factory=lambda _config: FakeExtractionAdapter(),
        extraction_artifact_attestor=FakeArtifactAttestor(),
        extraction_evidence_verifier=VerifiedTestEvidence(),
    ) as application:
        enqueuer = application.service._extraction_enqueuer

        def enqueue_directly(context, request):
            return enqueuer.enqueue_after_commit(
                context,
                request.params["transcript"],
                source=request.params["source"],
            )

        application.daemon._handler = enqueue_directly

        class RecordingClient:
            def __init__(self):
                self.frames = []

            def sendall(self, frame):
                self.frames.append(frame)

        context = ClientContext(
            client_id="client-a-install",
            surface="client-a",
            agent_id="client-a",
            session_id="oversized-extraction",
            access_scopes=("private",),
        )
        client = RecordingClient()
        keep_open, connection = application.daemon._handle_frame(
            client, encode_frame(Handshake(context)), None
        )
        assert keep_open is True

        application.daemon._handle_frame(
            client,
            encode_frame(Request(
                "oversized",
                "memory.extraction.enqueue",
                {"transcript": "x" * 13000, "source": "integration_test"},
            )),
            connection,
        )

        response = decode_frame(client.frames[-1])
        assert isinstance(response, Response)
        assert response.error.code == "invalid_params"
        assert "size limit" in response.error.message


def test_periodic_attestation_keeps_previous_state_during_recheck(tmp_path):
    entered = threading.Event()
    release = threading.Event()

    class BlockingAttestor(FakeArtifactAttestor):
        def attest(self, *, model, expected_digest):
            if self.calls:
                entered.set()
                release.wait(1.0)
            return super().attest(model=model, expected_digest=expected_digest)

    attestor = BlockingAttestor()
    config = load_config(_config(tmp_path, extraction=_extraction_config()))
    with ServerApplication(
        config,
        extraction_extractor_factory=lambda _config: FakeExtractionAdapter(),
        extraction_artifact_attestor=attestor,
    ) as application:
        thread = threading.Thread(
            target=application.extraction_worker._prerequisites_ready
        )
        thread.start()
        assert entered.wait(1.0)
        try:
            assert application.extraction_artifact_attestation is not None
        finally:
            release.set()
            thread.join(1.0)
        assert not thread.is_alive()


def test_health_degrades_when_required_attestation_is_unverified(
    tmp_path, monkeypatch
):
    config = load_config(_config(tmp_path, extraction=_extraction_config()))
    with ServerApplication(
        config,
        extraction_extractor_factory=lambda _config: FakeExtractionAdapter(),
        extraction_artifact_attestor=FakeArtifactAttestor(),
        extraction_evidence_verifier=VerifiedTestEvidence(),
    ) as application:
        monkeypatch.setattr(
            application.extraction_worker,
            "health",
            lambda **_kwargs: {
                "running": True,
                "stopping": False,
                "heartbeat_age_seconds": 0.0,
                "heartbeat_stale": False,
                "last_success_age_seconds": None,
                "last_error": None,
            },
        )
        application.extraction_artifact_attestation = None
        health = application.daemon._health(
            ClientContext(
                client_id="client-a-install",
                surface="client-a",
                agent_id="client-a",
                session_id="unverified-attestation",
                access_scopes=("private",),
            )
        )

        assert health["automatic_llm_extraction"]["status"] == "ready"
        assert health["extraction_artifact_attestation"]["status"] == "unverified"
        assert health["status"] == "degraded"


def test_health_reports_acknowledged_extraction_without_degrading(
    tmp_path, monkeypatch
):
    config = load_config(_config(tmp_path, extraction=_extraction_config()))
    with ServerApplication(
        config,
        extraction_extractor_factory=lambda _config: FakeExtractionAdapter(),
        extraction_artifact_attestor=FakeArtifactAttestor(),
        extraction_evidence_verifier=VerifiedTestEvidence(),
    ) as application:
        monkeypatch.setattr(
            application.extraction_worker,
            "health",
            lambda **_kwargs: {
                "running": True,
                "stopping": False,
                "heartbeat_age_seconds": 0.0,
                "heartbeat_stale": False,
                "last_success_age_seconds": None,
                "last_error": None,
            },
        )
        application.connection.execute(
            "INSERT INTO extract_queue(payload, payload_hash, status, attempts, "
            "last_error) VALUES ('reviewed', ?, 'acknowledged', 3, 'adapter_exit')",
            ("a" * 64,),
        )
        application.connection.commit()

        health = application.daemon._health(
            ClientContext(
                client_id="client-a-install",
                surface="client-a",
                agent_id="client-a",
                session_id="acknowledged-extraction",
                access_scopes=("private",),
            )
        )

        assert health["status"] == "ok"
        extraction = health["automatic_llm_extraction"]
        assert extraction["status"] == "ready"
        assert extraction["queue"]["acknowledged"] == 1
        assert extraction["queue"]["dead"] == 0


def test_failed_periodic_attestation_replaces_startup_verified_health(tmp_path):
    attestor = FakeArtifactAttestor()
    config = load_config(_config(tmp_path, extraction=_extraction_config()))
    with ServerApplication(
        config,
        extraction_extractor_factory=lambda _config: FakeExtractionAdapter(),
        extraction_artifact_attestor=attestor,
    ) as application:
        attestor.failure = ArtifactAttestationError("mutable tag drifted")

        assert application.extraction_worker._prerequisites_ready() is False
        health = application.daemon._health(
            ClientContext(
                client_id="client-a-install",
                surface="client-a",
                agent_id="client-a",
                session_id="attestation-drift",
                access_scopes=("private",),
            )
        )

        assert health["extraction_artifact_attestation"] == {
            "provider": "ollama",
            "status": "verified",
            "recipe_version": 1,
        }
        assert health["status"] == "degraded"
        assert health["automatic_llm_extraction"]["worker"]["last_error"] == (
            "worker_prerequisite_failed"
        )
        assert len(attestor.calls) == 2


@pytest.mark.parametrize(
    ("extraction", "message"),
    [
        ({"mode": "daemon-supervised"}, "requires host"),
        ({"mode": "disabled", "poll_seconds": 1}, "accepts only mode"),
        (
            _extraction_config(host={
                "type": "subprocess", "argv": ["relative-command"],
                "model_identity": "fixture", "prompt_identity": "v1",
            }),
            r"argv\[0\] must be absolute",
        ),
        (
            _extraction_config(heartbeat_stale_seconds=0.3),
            "must exceed the host timeout",
        ),
    ],
)
def test_extraction_configuration_fails_closed(tmp_path, extraction, message):
    with pytest.raises(ServerConfigError, match=message):
        load_config(_config(tmp_path, extraction=extraction))


def test_check_reports_configured_extraction_without_starting_adapter(tmp_path):
    config = load_config(_config(tmp_path, extraction=_extraction_config()))
    report = inspect_config(
        config, extraction_artifact_attestor=FakeArtifactAttestor()
    )
    assert report["automatic_llm_extraction"] == {
        "status": "configured-ready-to-start"
    }
    assert report["extraction_artifact_attestation"] == {
        "provider": "ollama",
        "status": "verified",
        "recipe_version": 1,
    }


def test_supervised_extraction_requires_immutable_artifact_pins(tmp_path):
    extraction = _extraction_config()
    extraction.pop("artifact")

    with pytest.raises(ServerConfigError, match="immutable artifact pins"):
        load_config(_config(tmp_path, extraction=extraction))


def test_extraction_model_tag_must_match_immutable_digest_before_database_open(
    tmp_path, monkeypatch
):
    config = load_config(_config(tmp_path, extraction=_extraction_config()))
    opened = False

    def must_not_open(*_args, **_kwargs):
        nonlocal opened
        opened = True
        raise AssertionError("database opened before extraction attestation")

    monkeypatch.setattr("enfold.server._open_existing_v1", must_not_open)
    attestor = FakeArtifactAttestor(
        failure=ArtifactAttestationError("mutable tag drifted")
    )

    with pytest.raises(
        ServerConfigError, match="automatic extraction artifact attestation failed"
    ):
        ServerApplication(config, extraction_artifact_attestor=attestor)

    assert opened is False
    assert attestor.calls == [("fixture-model:latest", _EXTRACTION_MODEL_DIGEST)]


def test_extraction_recipe_change_fails_attestation_without_exposing_digest(tmp_path):
    extraction = _extraction_config()
    extraction["artifact"]["recipe_digest"] = "sha256:" + "0" * 64
    config = load_config(_config(tmp_path, extraction=extraction))

    with pytest.raises(
        ServerConfigError, match="automatic extraction artifact attestation failed"
    ):
        inspect_config(
            config, extraction_artifact_attestor=FakeArtifactAttestor()
        )
