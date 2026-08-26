from __future__ import annotations

from pathlib import Path
import json
import re
from types import SimpleNamespace

import pytest

from enfold import mcp_launcher


def _arguments(tmp_path: Path) -> list[str]:
    return [
        "--socket-path", str(tmp_path / "enfold.sock"),
        "--client-id", "client-a-install-1",
        "--surface", "client-a",
        "--agent-id", "client-a",
        "--access-scope", "private",
        "--access-scope", "work",
    ]


def test_launcher_generates_fresh_session_and_ignores_identity_environment(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        mcp_launcher,
        "discover_provenance",
        lambda *args, **kwargs: mcp_launcher.ProjectProvenance(
            "/workspace/project", "github.com/example/project", "main", "a" * 40
        ),
    )
    hostile = {
        "ENFOLD_CLIENT_ID": "attacker",
        "ENFOLD_SURFACE": "attacker",
        "ENFOLD_AGENT_ID": "attacker",
        "ENFOLD_SESSION_ID": "attacker",
        "ENFOLD_ACCESS_SCOPES": "secret",
        "ENFOLD_CLIENT_CREDENTIAL": "credential-from-supervisor",
    }

    first = mcp_launcher.parse_config(_arguments(tmp_path), environ=hostile)
    second = mcp_launcher.parse_config(_arguments(tmp_path), environ=hostile)

    assert first.context.client_id == "client-a-install-1"
    assert first.context.surface == first.context.agent_id == "client-a"
    assert first.context.access_scopes == ("private", "work")
    assert first.context.session_id != second.context.session_id
    assert re.fullmatch(r"client-a-[A-Za-z0-9_-]{32}", first.context.session_id)
    assert first.context.repository == "github.com/example/project"
    assert first.credential == "credential-from-supervisor"


def test_explicit_session_is_supported_for_a_trusted_supervisor(tmp_path, monkeypatch):
    monkeypatch.setattr(
        mcp_launcher,
        "discover_provenance",
        lambda *args, **kwargs: mcp_launcher.ProjectProvenance(),
    )
    config = mcp_launcher.parse_config(
        [*_arguments(tmp_path), "--session-id", "client-a-thread-7"], environ={}
    )
    assert config.context.session_id == "client-a-thread-7"


def test_discovery_uses_argv_without_shell_and_strips_remote_credentials(
    tmp_path, monkeypatch
):
    root = tmp_path / "project;touch-pwned"
    root.mkdir()
    responses = {
        ("rev-parse", "--show-toplevel"): (0, str(root)),
        ("config", "--get", "remote.origin.url"): (
            0,
            "https://token:"
            + "secret"
            + "@example.com/owner/project.git?credential=bad#fragment",
        ),
        ("symbolic-ref", "--quiet", "--short", "HEAD"): (0, "feature/safe"),
        ("rev-parse", "--verify", "HEAD"): (0, "A" * 40),
    }
    calls = []

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))
        returncode, stdout = responses[tuple(argv[3:])]
        return SimpleNamespace(returncode=returncode, stdout=stdout + "\n")

    monkeypatch.setattr(mcp_launcher.subprocess, "run", fake_run)
    result = mcp_launcher.discover_provenance(root, environ={"PATH": "/bin"})

    assert result == mcp_launcher.ProjectProvenance(
        str(root), "example.com/owner/project", "feature/safe", "a" * 40
    )
    assert len(calls) == 4
    assert all(call[1].get("shell") is None for call in calls)
    assert all(Path(call[0][0]).name == "git" for call in calls)
    assert all(call[0][1:3] == ["-C", str(root)] for call in calls)
    assert all(call[1]["timeout"] == 1.0 for call in calls)
    assert all("token" not in str(call[1]["env"]) for call in calls)


def test_non_repository_and_git_failures_keep_only_safe_cwd(tmp_path, monkeypatch):
    monkeypatch.setattr(
        mcp_launcher.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=128, stdout=""),
    )
    result = mcp_launcher.discover_provenance(tmp_path, environ={})
    assert result == mcp_launcher.ProjectProvenance(
        str(tmp_path.resolve()), None, None, None
    )


def test_malformed_git_outputs_are_omitted(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    responses = iter(
        [str(root), "git@example.com:owner/repo.git", "bad\x00branch", "not-a-sha"]
    )
    monkeypatch.setattr(
        mcp_launcher.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=next(responses)),
    )
    result = mcp_launcher.discover_provenance(root, environ={})
    assert result.repository == "example.com/owner/repo"
    assert result.branch is None
    assert result.commit_sha is None


def test_requires_static_registration_identity_scope_and_absolute_socket(tmp_path):
    with pytest.raises(SystemExit):
        mcp_launcher.parse_config(
            [
                "--socket-path", str(tmp_path / "enfold.sock"),
                "--client-id", "client-a",
                "--surface", "client-a",
                "--agent-id", "client-a",
            ],
            environ={},
        )
    with pytest.raises(SystemExit):
        mcp_launcher.parse_config(
            [
                "--socket-path", "relative.sock",
                "--client-id", "client-a",
                "--surface", "client-a",
                "--agent-id", "client-a",
                "--access-scope", "private",
            ],
            environ={},
        )


def test_main_starts_stdio_proxy_in_process(monkeypatch):
    config = object()
    server = SimpleNamespace(run=lambda **kwargs: calls.append(kwargs))
    calls = []
    monkeypatch.setattr(mcp_launcher, "parse_config", lambda argv: config)
    monkeypatch.setattr(
        mcp_launcher, "build_server", lambda received: server if received is config else None
    )

    assert mcp_launcher.main(["ignored-by-test-parser"]) == 0
    assert calls == [{"transport": "stdio"}]


def test_product_entry_self_test_creates_store_starts_daemon_and_recalls(
    tmp_path, capsys
):
    config_dir = tmp_path / "config"
    data_dir = tmp_path / "data"
    pid_path = data_dir / "enfold.pid"
    try:
        result = mcp_launcher.product_main(
            [
                "--self-test",
                "--config-dir",
                str(config_dir),
                "--data-dir",
                str(data_dir),
            ]
        )
        assert result == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["ok"] is True
        assert payload["fact_id"]
        assert (config_dir / "server.json").is_file()
        assert (data_dir / "memory.db").is_file()
        first = payload["fact_id"]

        again = mcp_launcher.product_main(
            [
                "--self-test",
                "--config-dir",
                str(config_dir),
                "--data-dir",
                str(data_dir),
            ]
        )
        assert again == 0
        reused = json.loads(capsys.readouterr().out)
        assert reused["ok"] is True
        assert reused["fact_id"] != first
        assert reused["daemon"] == "reused"
    finally:
        if pid_path.is_file():
            import os
            import signal
            import time

            pid = int(pid_path.read_text(encoding="utf-8").strip())
            try:
                os.kill(pid, signal.SIGTERM)
            except OSError:
                pass
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                try:
                    os.kill(pid, 0)
                except OSError:
                    break
                time.sleep(0.02)


def test_product_first_run_with_client_id_uses_minted_token_without_env(
    tmp_path, capsys, monkeypatch
):
    monkeypatch.delenv("ENFOLD_CLIENT_CREDENTIAL", raising=False)
    config_dir = tmp_path / "config"
    data_dir = tmp_path / "data"
    pid_path = data_dir / "enfold.pid"
    try:
        result = mcp_launcher.product_main(
            [
                "--self-test",
                "--config-dir",
                str(config_dir),
                "--data-dir",
                str(data_dir),
                "--client-id",
                "fresh-install",
            ]
        )
        captured = capsys.readouterr()
        assert result == 0, captured.err
        payload = json.loads(captured.out)
        assert payload["ok"] is True
        raw = json.loads((config_dir / "server.json").read_text(encoding="utf-8"))
        assert "fresh-install" in raw["grants"]
    finally:
        if pid_path.is_file():
            import os
            import signal
            import time

            pid = int(pid_path.read_text(encoding="utf-8").strip())
            try:
                os.kill(pid, signal.SIGTERM)
            except OSError:
                pass
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                try:
                    os.kill(pid, 0)
                except OSError:
                    break
                time.sleep(0.02)


def test_product_launcher_does_not_self_serve_another_clients_stored_credential(
    tmp_path, capsys, monkeypatch
):
    from enfold.bootstrap import (
        bootstrap,
        credential_digest,
        new_client_token,
        replace_config,
        write_client_credential,
    )

    config_dir = tmp_path / "config"
    data_dir = tmp_path / "data"
    pid_path = data_dir / "enfold.pid"
    bootstrap(config_dir=config_dir, data_dir=data_dir, owner_client_id="low")
    privileged_token = new_client_token()
    raw = json.loads((config_dir / "server.json").read_text(encoding="utf-8"))
    raw["grants"]["privileged"] = ["private"]
    raw["client_credentials"]["privileged"] = credential_digest(privileged_token)
    replace_config(config_dir / "server.json", raw)
    write_client_credential(config_dir, "privileged", privileged_token)
    monkeypatch.delenv("ENFOLD_CLIENT_CREDENTIAL", raising=False)

    try:
        result = mcp_launcher.product_main(
            [
                "--self-test",
                "--config-dir",
                str(config_dir),
                "--data-dir",
                str(data_dir),
                "--client-id",
                "privileged",
            ]
        )
        captured = capsys.readouterr()
        assert result == 2
        assert privileged_token not in captured.out
        assert privileged_token not in captured.err
        assert "credential" in captured.err
        assert "ok" not in captured.out

        monkeypatch.setenv("ENFOLD_CLIENT_CREDENTIAL", privileged_token)
        held = mcp_launcher.product_main(
            [
                "--self-test",
                "--config-dir",
                str(config_dir),
                "--data-dir",
                str(data_dir),
                "--client-id",
                "privileged",
            ]
        )
        held_out = capsys.readouterr()
        assert held == 0, held_out.err
        payload = json.loads(held_out.out)
        assert payload["ok"] is True
        assert payload["fact_id"]
        assert privileged_token not in held_out.out
    finally:
        if pid_path.is_file():
            import os
            import signal
            import time

            pid = int(pid_path.read_text(encoding="utf-8").strip())
            try:
                os.kill(pid, signal.SIGTERM)
            except OSError:
                pass
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                try:
                    os.kill(pid, 0)
                except OSError:
                    break
                time.sleep(0.02)
