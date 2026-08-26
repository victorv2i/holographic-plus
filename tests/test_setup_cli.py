from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import time

import pytest

from enfold.bootstrap import bootstrap, main
from enfold.client import ClientConfig, EnfoldClient, EnfoldHandshakeError
from enfold.protocol import ClientContext


def _stop_pid(pid_path: Path) -> None:
    if not pid_path.is_file():
        return
    try:
        pid = int(pid_path.read_text(encoding="utf-8").strip())
    except ValueError:
        return
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        return
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except OSError:
            return
        time.sleep(0.02)


def test_setup_cursor_is_a_reversible_credentialed_transaction(
    tmp_path: Path, capsys
) -> None:
    config_dir = tmp_path / "config"
    data_dir = tmp_path / "data"
    pid_path = data_dir / "enfold.pid"

    try:
        result = main([
            "setup",
            "--client",
            "cursor",
            "--config-dir",
            str(config_dir),
            "--data-dir",
            str(data_dir),
        ])
        captured = capsys.readouterr()
        assert result == 0
        payload = json.loads(captured.out.splitlines()[-1])
        token = payload["client_credential"]
        client_id = payload["client_id"]
        assert client_id.startswith("cursor-install-")
        assert token.startswith("enf_")
        assert "Remember that" in payload["agent_sentence"]
        snippet = payload["snippet"]
        assert "ENFOLD_CLIENT_CREDENTIAL" in snippet
        assert token in snippet
        assert "Remember that" not in snippet
        raw = json.loads((config_dir / "server.json").read_text(encoding="utf-8"))
        assert token not in (config_dir / "server.json").read_text(encoding="utf-8")
        assert client_id in raw["grants"]
        assert client_id in raw["client_credentials"]
        manifest = json.loads(
            (config_dir / "install-manifest.json").read_text(encoding="utf-8")
        )
        assert any(item["path"].endswith("server.json") for item in manifest["paths"])

        denied = EnfoldClient(
            ClientConfig(
                Path(raw["socket_path"]),
                ClientContext(
                    client_id=client_id,
                    surface="cursor",
                    agent_id="cursor",
                    session_id="setup-forged",
                    access_scopes=("private",),
                ),
                credential="forged-token",
            )
        )
        with pytest.raises(EnfoldHandshakeError) as forged:
            denied.request("health")
        assert forged.value.code == "invalid_client_credentials"

        client = EnfoldClient(
            ClientConfig(
                Path(raw["socket_path"]),
                ClientContext(
                    client_id=client_id,
                    surface="cursor",
                    agent_id="cursor",
                    session_id="setup-recall",
                    access_scopes=("private",),
                ),
                credential=token,
            )
        )
        found = client.request(
            "memory.search",
            {"query": "Enfold setup smoke remembered the local install"},
        )
        assert found["facts"]
        assert found["facts"][0]["fact_id"] == payload["smoke"]["fact_id"]
        assert raw["retrieval"] == {"mode": "local-lexical"}
    finally:
        _stop_pid(pid_path)


def _overlong_socket_path(root: Path) -> Path:
    parent = root
    while len(os.fsencode(os.fspath(parent / "enfold.sock"))) <= 107:
        parent = parent / ("pad-" + "x" * 40)
    parent.mkdir(parents=True)
    os.chmod(parent, 0o700)
    for cursor in parent.parents:
        if cursor == root or root in cursor.parents:
            try:
                os.chmod(cursor, 0o700)
            except OSError:
                break
        else:
            break
    return parent / "enfold.sock"


def test_setup_does_not_move_existing_store_when_socket_path_is_too_long(
    tmp_path: Path, capsys
) -> None:
    config_dir = tmp_path / "config"
    data_dir = tmp_path / "data"
    report = bootstrap(config_dir=config_dir, data_dir=data_dir)
    long_socket = _overlong_socket_path(tmp_path / "sock")
    raw = json.loads(report.config_path.read_text(encoding="utf-8"))
    raw["socket_path"] = str(long_socket)
    report.config_path.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")
    os.chmod(report.config_path, 0o600)
    original = report.config_path.read_bytes()
    started = time.monotonic()

    result = main([
        "setup",
        "--client",
        "generic",
        "--config-dir",
        str(config_dir),
        "--data-dir",
        str(data_dir),
    ])
    elapsed = time.monotonic() - started
    captured = capsys.readouterr()

    assert result == 2
    assert "AF_UNIX" in captured.err
    assert str(len(os.fsencode(os.fspath(long_socket)))) in captured.err
    assert "--socket-path" in captured.err
    assert report.config_path.read_bytes() == original
    assert report.database_path.is_file()
    assert elapsed < 2.0


def test_setup_accepts_explicit_short_socket_path_and_refuses_an_overlong_one(
    tmp_path: Path, capsys
) -> None:
    config_dir = tmp_path / "config"
    data_dir = tmp_path / "data"
    socket_path = tmp_path / "run" / "enfold.sock"
    pid_path = data_dir / "enfold.pid"
    try:
        result = main([
            "setup",
            "--client",
            "generic",
            "--config-dir",
            str(config_dir),
            "--data-dir",
            str(data_dir),
            "--socket-path",
            str(socket_path),
        ])
        captured = capsys.readouterr()
        assert result == 0, captured.err
        payload = json.loads(captured.out.splitlines()[-1])
        assert payload["socket_path"] == str(socket_path)
        raw = json.loads((config_dir / "server.json").read_text(encoding="utf-8"))
        assert raw["socket_path"] == str(socket_path)
        assert raw["database_path"] == str(data_dir / "memory.db")
    finally:
        _stop_pid(pid_path)

    long_socket = _overlong_socket_path(tmp_path / "too-long")
    refused = main([
        "setup",
        "--client",
        "generic",
        "--config-dir",
        str(tmp_path / "config-long"),
        "--data-dir",
        str(tmp_path / "data-long"),
        "--socket-path",
        str(long_socket),
    ])
    err = capsys.readouterr().err
    assert refused == 2
    assert "AF_UNIX" in err
    assert "--socket-path" in err
    assert not (tmp_path / "config-long" / "server.json").exists()
    assert not (tmp_path / "data-long" / "memory.db").exists()


def test_setup_under_long_xdg_then_self_test_and_doctor_succeed(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    from enfold import mcp_launcher

    runtime = tmp_path / "run"
    runtime.mkdir(mode=0o700)
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(runtime))
    long_root = tmp_path / "xdg"
    while len(os.fsencode(os.fspath(long_root / "data" / "enfold.sock"))) <= 107:
        long_root = long_root / ("pad-" + "x" * 40)
    long_root.mkdir(parents=True)
    os.chmod(long_root, 0o700)
    config_dir = long_root / "config"
    data_dir = long_root / "data"
    pid_path = data_dir / "enfold.pid"
    try:
        assert main([
            "setup",
            "--client",
            "generic",
            "--config-dir",
            str(config_dir),
            "--data-dir",
            str(data_dir),
        ]) == 0
        setup_out = capsys.readouterr()
        payload = json.loads(setup_out.out.splitlines()[-1])
        assert len(os.fsencode(payload["socket_path"])) <= 107
        assert payload["database_path"] == str(data_dir / "memory.db")
        assert Path(payload["socket_path"]).parent == runtime / "enfold"
        assert Path(payload["socket_path"]).name.endswith(".sock")
        result = mcp_launcher.product_main([
            "--self-test",
            "--config-dir",
            str(config_dir),
            "--data-dir",
            str(data_dir),
        ])
        self_test = capsys.readouterr()
        assert result == 0, self_test.err
        recalled = json.loads(self_test.out)
        assert recalled["ok"] is True
        assert recalled["fact_id"]
        assert main(["doctor"]) == 0
        doctor = json.loads(capsys.readouterr().out)
        assert doctor["ok"] is True
        assert doctor["write"] == "pass"
        assert doctor["recall"] == "pass"
        assert doctor["evidence"] == "pass"
    finally:
        _stop_pid(pid_path)


def test_setup_then_mcp_self_test_reuses_the_registered_surface(
    tmp_path: Path, capsys
) -> None:
    from enfold import mcp_launcher

    config_dir = tmp_path / "config"
    data_dir = tmp_path / "data"
    pid_path = data_dir / "enfold.pid"
    try:
        assert main([
            "setup",
            "--client",
            "generic",
            "--config-dir",
            str(config_dir),
            "--data-dir",
            str(data_dir),
        ]) == 0
        capsys.readouterr()
        result = mcp_launcher.product_main([
            "--self-test",
            "--config-dir",
            str(config_dir),
            "--data-dir",
            str(data_dir),
        ])
        captured = capsys.readouterr()
        assert result == 0, captured.err
        payload = json.loads(captured.out)
        assert payload["ok"] is True
        assert payload["fact_id"]
    finally:
        _stop_pid(pid_path)


def test_setup_explicit_socket_path_repairs_existing_overlong_socket_without_moving_store(
    tmp_path: Path, capsys
) -> None:
    config_dir = tmp_path / "config"
    data_dir = tmp_path / "data"
    pid_path = data_dir / "enfold.pid"
    report = bootstrap(config_dir=config_dir, data_dir=data_dir)
    long_socket = _overlong_socket_path(tmp_path / "sock")
    raw = json.loads(report.config_path.read_text(encoding="utf-8"))
    raw["socket_path"] = str(long_socket)
    report.config_path.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")
    os.chmod(report.config_path, 0o600)
    short_socket = tmp_path / "run" / "enfold.sock"
    try:
        result = main([
            "setup",
            "--client",
            "generic",
            "--config-dir",
            str(config_dir),
            "--data-dir",
            str(data_dir),
            "--socket-path",
            str(short_socket),
        ])
        captured = capsys.readouterr()
        assert result == 0, captured.err
        updated = json.loads(report.config_path.read_text(encoding="utf-8"))
        assert updated["socket_path"] == str(short_socket)
        assert updated["database_path"] == str(report.database_path)
        assert report.database_path.is_file()
    finally:
        _stop_pid(pid_path)


def test_setup_second_client_reloads_live_daemon_grants(
    tmp_path: Path, capsys
) -> None:
    config_dir = tmp_path / "config"
    data_dir = tmp_path / "data"
    pid_path = data_dir / "enfold.pid"
    try:
        first = main([
            "setup",
            "--client",
            "cursor",
            "--config-dir",
            str(config_dir),
            "--data-dir",
            str(data_dir),
        ])
        first_out = capsys.readouterr()
        assert first == 0, first_out.err
        first_payload = json.loads(first_out.out.splitlines()[-1])

        second = main([
            "setup",
            "--client",
            "generic",
            "--config-dir",
            str(config_dir),
            "--data-dir",
            str(data_dir),
        ])
        second_out = capsys.readouterr()
        assert second == 0, second_out.err
        second_payload = json.loads(second_out.out.splitlines()[-1])
        assert second_payload["client_id"] != first_payload["client_id"]
        raw = json.loads((config_dir / "server.json").read_text(encoding="utf-8"))
        assert first_payload["client_id"] in raw["grants"]
        assert second_payload["client_id"] in raw["grants"]

        client = EnfoldClient(
            ClientConfig(
                Path(raw["socket_path"]),
                ClientContext(
                    client_id=second_payload["client_id"],
                    surface="generic",
                    agent_id="generic",
                    session_id="second-client-recall",
                    access_scopes=("private",),
                ),
                credential=second_payload["client_credential"],
            )
        )
        found = client.request(
            "memory.search",
            {"query": "Enfold setup smoke remembered the local install"},
        )
        assert found["facts"]
        assert found["facts"][0]["fact_id"] == second_payload["smoke"]["fact_id"]
    finally:
        _stop_pid(pid_path)


def test_setup_rollback_restores_overwritten_snippet_and_manifest(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    from enfold import setup_cli

    config_dir = tmp_path / "config"
    data_dir = tmp_path / "data"
    pid_path = data_dir / "enfold.pid"
    snippet_path = config_dir / "clients" / "generic.snippet"
    manifest_path = config_dir / "install-manifest.json"
    try:
        assert main([
            "setup",
            "--client",
            "generic",
            "--config-dir",
            str(config_dir),
            "--data-dir",
            str(data_dir),
        ]) == 0
        capsys.readouterr()
        snippet_before = snippet_path.read_bytes()
        manifest_before = manifest_path.read_bytes()
        config_before = (config_dir / "server.json").read_bytes()

        def boom(*_args, **_kwargs):
            raise RuntimeError("injected smoke failure")

        monkeypatch.setattr(setup_cli, "run_protocol_smoke", boom)
        result = main([
            "setup",
            "--client",
            "generic",
            "--config-dir",
            str(config_dir),
            "--data-dir",
            str(data_dir),
        ])
        captured = capsys.readouterr()
        assert result == 2
        assert "injected smoke failure" in captured.err
        assert snippet_path.read_bytes() == snippet_before
        assert manifest_path.read_bytes() == manifest_before
        assert (config_dir / "server.json").read_bytes() == config_before
    finally:
        _stop_pid(pid_path)


def test_setup_rolls_back_when_smoke_test_fails(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    from enfold import setup_cli

    def boom(*_args, **_kwargs):
        raise RuntimeError("injected smoke failure")

    monkeypatch.setattr(setup_cli, "run_protocol_smoke", boom)
    result = main([
        "setup",
        "--client",
        "generic",
        "--config-dir",
        str(tmp_path / "config"),
        "--data-dir",
        str(tmp_path / "data"),
    ])
    captured = capsys.readouterr()
    assert result == 2
    assert "injected smoke failure" in captured.err
    assert not (tmp_path / "config" / "server.json").exists()
    assert not (tmp_path / "data" / "memory.db").exists()
    assert not (tmp_path / "config" / "install-manifest.json").exists()


def test_uninstall_dry_run_lists_then_purge_removes_created_files(
    tmp_path: Path, capsys
) -> None:
    config_dir = tmp_path / "config"
    data_dir = tmp_path / "data"
    pid_path = data_dir / "enfold.pid"
    try:
        assert main([
            "setup",
            "--client",
            "codex",
            "--config-dir",
            str(config_dir),
            "--data-dir",
            str(data_dir),
        ]) == 0
        capsys.readouterr()
        assert (config_dir / "server.json").is_file()
        assert (data_dir / "memory.db").is_file()

        assert main([
            "uninstall",
            "--config-dir",
            str(config_dir),
            "--data-dir",
            str(data_dir),
            "--dry-run",
        ]) == 0
        listed = json.loads(capsys.readouterr().out)
        assert listed["dry_run"] is True
        assert (config_dir / "server.json").is_file()
        assert any(
            item.endswith("memory.db") for item in listed["paths"]
        )

        assert main([
            "uninstall",
            "--config-dir",
            str(config_dir),
            "--data-dir",
            str(data_dir),
            "--purge-data",
        ]) == 0
        removed = json.loads(capsys.readouterr().out)
        assert removed["dry_run"] is False
        assert not (config_dir / "server.json").exists()
        assert not (data_dir / "memory.db").exists()
        assert not (config_dir / "install-manifest.json").exists()
    finally:
        _stop_pid(pid_path)
